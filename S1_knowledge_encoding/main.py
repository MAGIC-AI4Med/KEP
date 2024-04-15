import logging
import os
import random
from datetime import datetime


import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

# from io import BytesIO
# from petrel_client.client import Client
# from petrel_client.utils.data import DataLoader
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from data.make_dataloader import make_dataloader

try:
    import wandb
except ImportError:
    wandb = None

from torch.utils.tensorboard import SummaryWriter
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from model import CLP_clinical, CLP_BERT

from bert_training.distributed import is_master, init_distributed_device, world_info_from_env
from bert_training.logger import setup_logging
from bert_training.params import parse_args
from bert_training.scheduler import cosine_lr
from bert_training.train import train_one_epoch
import logging
from utils.reid_eval import reid_eval
from loss.adasp_loss import AdaSPLoss
from loss.triplet_loss import TripletLoss
import copy
from bert_training.freeze_scheduler import FreezeChecker, FreezeScheduler

import collections
from utils.openai import load_openai_model
from utils.tokenizer import tokenize

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    args = parse_args()
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.output_dir,args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.output_dir,args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.aws_output_dir,args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''
    
    if args.copy_codebase:
        copy_codebase(args)

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')
    
    random_seed(args.seed, 0)
    if not args.text_eval:
        model = CLP_clinical(bert_model_name = args.pretrained)
    elif args.text_eval == 'transformer':
        model = load_openai_model(
            'ViT-B-32',
            precision='amp',
            device=device,
        )
    elif args.text_eval == 'bert':
        model = CLP_BERT(bert_model_name = args.pretrained,text_head=args.text_head)

    # load XX-Bert
    if args.bert_pretrained != '':
        checkpoint = torch.load(args.bert_pretrained, map_location='cpu')
        state_dict = checkpoint["state_dict"]

        if args.text_eval == 'bert':
            bert_model = collections.OrderedDict()
            for k,v in state_dict.items():
                key = 'text' if args.text_head else 'text.'
                if key in k:
                    new_k = k.split('text.')[-1]
                    bert_model[new_k] = v
            model.load_state_dict(bert_model,strict=True)
            print('Load pretrained bert success from: ',args.bert_pretrained)
        elif args.text_eval == 'transformer':
            model.load_state_dict(state_dict,strict=True)
            print('Load pretrained bert success from: ',args.bert_pretrained)
        
        else:
            filtered_st = copy.deepcopy(state_dict)
            for k,v in state_dict.items():
                if k in ['logit_scale','bert_model.embeddings.position_ids']:
                    filtered_st.pop(k)
            model.load_state_dict(filtered_st,strict=False)
            print('Load pretrained bert success from: ',args.bert_pretrained)

    model.to(device=device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.output_dir,args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    if args.text_eval == 'transformer':
        tokenizer = tokenize
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained,local_files_only=True)

    assert not args.trace, 'Cannot train with traced model'
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    if args.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            # with BytesIO(client.get(args.resume)) as buffer:
            #     checkpoint = torch.load(buffer, map_location=device)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    optimizer.param_groups[0]['capturable'] = True
                    optimizer.param_groups[1]['capturable'] = True
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # initialize datasets
    train_dataloader, val_dataloaders, num_train, num_query = make_dataloader(args)
    num_batches = len(train_dataloader)
    print('num_samples',num_train,'num_batches',num_batches,train_dataloader)

    ## loss function
    if args.ID_loss:
        loss_fn = {'metric_loss': AdaSPLoss(args.device, loss_type = 'adasp'), 'ID_loss': torch.nn.CrossEntropyLoss()}
    elif args.metric_type == 'adasp':
        loss_fn = {'metric_loss': AdaSPLoss(args.device, loss_type = 'adasp')}
    elif args.metric_type == 'triplet':
        loss_fn = {'metric_loss': TripletLoss()}

    total_steps = num_batches * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = os.path.join(args.output_dir,args.logs) and os.path.join(args.output_dir,args.logs).lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    freeze_scheduler = FreezeScheduler()
    for epoch in range(start_epoch, args.epochs):

        logging.info('======================—++++++++++++++++++++++======================')
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        freeze_scheduler.update(args.freeze_bert_epochs, model, epoch)

        freezechecker = FreezeChecker(model)

        train_one_epoch(model, tokenizer, train_dataloader,num_batches,num_train, loss_fn, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        freezechecker.freeze_checker(model)
        
        ## reid evaluation
        if completed_epoch in args.eval_epoch:
            logging.info(f'Start evaluation...')
            reid_eval(args, model, val_dataloaders, num_query, tokenizer, epoch, device, writer)
        
        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (completed_epoch in args.save_epoch):
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))

            if args.save_most_recent:
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path,  f"epoch_latest.pt"))

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.output_dir,args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
