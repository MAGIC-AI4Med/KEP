import argparse
from typing import Tuple

def get_default_params():
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    return {"lr": 5.0e-5, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default='pathology_knowledge_encoder',
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--resume",
        default='',
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument(
        "--bert_pretrained",
        default= '',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--text_eval",
        default= None, 
        type=str,
        help="evaluate clip model.",
    )
    parser.add_argument(
        "--text_head",
        default= False,
        type=bool,
        help="evaluate clip model.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default='../datasets/knowledge_encoder_dataset',
        help="Path to training data",
    )
    parser.add_argument(
        "--freeze-bert-epochs",
        type=int,
        default=0,
        help="epochs to freeze bert",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train for."
    )

    ## no need to adjust
    parser.add_argument(
        "--num-instances",
        type=int,
        default=8,
        help="Number of instances per entity",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default='adasp',
        help="Metric loss type",
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        default=20,
        help="Number of epochs for evaluation",
    )
    parser.add_argument(
        "--eval-epoch",
        type=list,
        default=[10,20,30,40,50,100,200,300,400,500],
        help="Number of epochs for evaluation",
    )
    parser.add_argument(
        "--mrdef-csv-file",
        type=str,
        default='./datasets/MRDEF_name.csv',
        help="Path to json filewith training data",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Number of max length input token."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--aws_output_dir",
        type=str,
        default="./results/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="logs",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log_local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size per GPU."
    )
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=100, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-epoch", type=list, default=[100,300,500], help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--pretrained",
        default= '../pretrained_model/pubmedbert',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--evaluate_perbatch",
        default=False,
        action='store_true',
        help="Use clinical BERT.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='tensorboard',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=None, help="Gradient clip."
    )
    args = parser.parse_args()

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params()
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
    return args
