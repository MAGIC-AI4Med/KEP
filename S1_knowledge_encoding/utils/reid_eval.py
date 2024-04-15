import torch
import logging
import numpy as np
from bert_training.train import get_tokenizer
from utils.metrics import R1_mAP_eval

def reid_eval(args, model, val_dataloaders, num_query, tokenizer, epoch, device, tb_writer):
    model.eval()
    # val_loss = []
    step = epoch
    for k, val_loader in val_dataloaders.items():
        evaluator = R1_mAP_eval(num_query[k])
        evaluator.reset()

        for n_iter, (eval_text, did, tid, attr) in enumerate(val_loader):
            with torch.no_grad():
                eval_text = get_tokenizer(eval_text,tokenizer,max_length=args.max_length, ismask=False).to(device=device)
                target = did.to(device)
                try:
                    text_features= model(eval_text)
                except:
                    text_features = model.encode_text(eval_text)
                evaluator.update((text_features, did, tid))
                # if k == 'syn':
                #     val_loss.append(loss_fn(text_features,target,normalize_feature=True).detach().cpu().numpy())
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logging.info("Validation Results for {} - Epoch: {}".format(k, epoch))
        logging.info("mAP: {:.1%}".format(mAP))
        for r in [1,2,3,4,5]:
            logging.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        
        if k == 'syn':
            log_data = {
                "mAP": mAP,
                "r1": cmc[0],
                "r2": cmc[1],
                "r3": cmc[2],
                "r4": cmc[3],
                "r5": cmc[4]
            }
            for name, val in log_data.items():
                name = "val/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
    torch.cuda.empty_cache()