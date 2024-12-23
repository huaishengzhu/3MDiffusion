import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

import rdkit
from rdkit import Chem
from poly_hgraph import *
from poly_hgraph.dataset import TextDataset
from poly_hgraph.hgnn_clip import HierVAE
sys.path.append("..")
from model.blip2_stage1 import Blip2Stage1
from poly_hgraph.gine_diffusion import *
from diffusion.denoising_diffusion import GaussianDiffusion, Trainer

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)
ATTN_HEAD_DIM=64

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--test', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--latent_size', type=int, default=24)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=20)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--saved_blipmodel', type=str, default='../all_checkpoints/stage1/epoch=49.ckpt', help='path to loade the blip model')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=False)
# parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=False)

parser.add_argument('--text_hidden_dim', type=int, default=250)
parser.add_argument('--mlp_hidden_size', type=int, default=128)
parser.add_argument('--num_MLP_layers', type=int, default=2)

parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

parser.add_argument(
        "--lr", type=float, required=True,
        help="Learning rate",
    )
parser.add_argument("--lr_schedule", type=str, default="cosine")
parser.add_argument("--lr_warmup_steps", type=int, default=1000)
parser.add_argument("--adam_beta1", type=float, default=0.9)
parser.add_argument("--adam_beta2", type=float, default=0.999)
parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
parser.add_argument("--ema_decay", type=float, default=0.9999)
parser.add_argument("--ema_update_every", type=int, default=1)
# Diffusion Hyperparameters
parser.add_argument(
    "--objective",
    type=str,
    default="pred_noise",
    choices=["pred_noise", "pred_x0"],
    help=(
        "Which parameterization to use for the diffusion objective."
    ),
)
parser.add_argument(
    "--loss_type",
    type=str,
    default="l2",
    choices=["l1", "l2", "smooth_l1"],
    help=(
        "Which loss function to use for diffusion."
    ),
)
parser.add_argument(
    "--beta_schedule",
    type=str,
    default="cosine",
    choices=["cosine", "linear"],
    help=(
        "Which noise schedule to use."
    ),
)
parser.add_argument("--p2_loss_weight_gamma", type=float, default=0)
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--sampling_timesteps", type=int, default=250)
parser.add_argument("--normalize_latent", action="store_true", default=False)
# Generation Arguments
parser.add_argument("--save_and_sample_every", type=int, default=1500)
parser.add_argument("--num_samples", type=int, default=16)
parser.add_argument("--ddim_sampling_eta", type=float, default=1)
# Model hyperparemeters
parser.add_argument("--tx_dim", type=int, default=128)
parser.add_argument("--tx_depth", type=int, default=3)
parser.add_argument("--scale_shift", action="store_true", default=False)
parser.add_argument("--disable_dropout", action="store_true", default=False)
parser.add_argument("--class_conditional", action="store_true", default=False)
parser.add_argument("--class_unconditional_prob", type=float, default=.1)
# Accelerate arguments
parser.add_argument("--amp", action="store_true", default=False)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="no",
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU."
    ),
)
# Load and eval model
parser.add_argument("--eval", action="store_true", default=False)
parser.add_argument("--eval_test", action="store_true", default=False)
parser.add_argument("--resume_training", action="store_true", default=False)
parser.add_argument("--gen_data", action="store_true", default=False)
parser.add_argument("--resume_dir", type=str, default=None)
parser.add_argument("--milestone", type=int, default=12)

parser.add_argument("--num_train_steps", type=int, default=60000)
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)


parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--parametrization', type=str, default='eps',
                    help='eps, x')
parser.add_argument("--self_condition", action="store_false", default=True)
parser.add_argument('--score_hidden_size', type=int, default=128)
parser.add_argument('--num_score_layers', type=int, default=2)

# parser = Blip2Stage1.add_model_specific_args(parser)
args = parser.parse_args()
# model_blip = Blip2Stage1(args)
model_blip = Blip2Stage1.load_from_checkpoint(args.saved_blipmodel).cuda()


vocab = [x.strip("\r\n ").split() for x in open(args.vocab + "vocab.txt")] 

with open(args.vocab + "fragment.txt") as f:
    fragments = [x.strip("\r\n ") for x in f]
        
MolGraph.load_fragments([fra for fra in fragments])
args.vocab = PairVocab(vocab)


args.graph_hidden = model_blip.blip2qformer.graph_proj.out_features
args.text_hidden = model_blip.blip2qformer.graph_proj.out_features

bart_model = HierVAE(args)
bart_model.ln_graph = model_blip.blip2qformer.ln_graph
bart_model.graph_proj = model_blip.blip2qformer.graph_proj
bart_model.encoder = model_blip.blip2qformer.graph_encoder


bart_model.load_state_dict(torch.load(args.model))


print("Finish Loading")

torch.manual_seed(args.seed)
random.seed(args.seed)

dataset = DataFolder(args.train, args.batch_size)
args.num_train_steps = args.epochs * len(dataset)

test_tmp = []
test_text = []
count  = 0
with open(args.test) as fin:
    for line in fin.readlines()[1:]:

        test_tmp.append(line.strip().split("\t")[1])
        test_text.append(line.strip().split("\t")[2])

args.train = test_tmp
print(len(args.train))
test_dataset = TextDataset(args.train, test_text, args.batch_size,  model_blip)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

model = DiffusionMLP(
        tx_dim = args.tx_dim,
        tx_depth = args.tx_depth,
        heads = args.tx_dim//ATTN_HEAD_DIM,
        latent_dim = args.latent_size,
        context_dim = args.text_hidden_dim,
        scale_shift = args.scale_shift,
        dropout = 0 if args.disable_dropout else 0.1
    ).cuda()
    
    

args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

diffusion = GaussianDiffusion(
    model,
    timesteps = args.timesteps,           # number of steps
    sampling_timesteps = args.sampling_timesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = args.loss_type,            # L1 or L2
    beta_schedule = args.beta_schedule,
    p2_loss_weight_gamma = args.p2_loss_weight_gamma,
    objective = args.objective,
    ddim_sampling_eta=args.ddim_sampling_eta,
    class_unconditional_prob = args.class_unconditional_prob
).cuda()

trainer = Trainer(
    args=args,
    diffusion=diffusion,
    bart_model = bart_model,
    dataloader = dataset,
    val_dataloader = dataset,
    val_full_dataloader = test_loader,
    train_batch_size = args.batch_size,
    eval_batch_size = args.batch_size,
    gradient_accumulate_every = args.gradient_accumulation_steps,
    train_lr = args.lr,
    train_num_steps = args.num_train_steps,
    lr_schedule = args.lr_schedule,
    num_warmup_steps = args.lr_warmup_steps,
    ema_update_every = args.ema_update_every,
    ema_decay = args.ema_decay,
    adam_betas = (args.adam_beta1, args.adam_beta2),
    adam_weight_decay = args.adam_weight_decay,
    save_and_sample_every = args.save_and_sample_every,
    num_samples = args.num_samples,
    results_folder = args.output_dir,
    amp = args.amp,
    mixed_precision = args.mixed_precision,
)
print("Finish Trainer")
# trainer.load(args.output_dir)
trainer.step =0
trainer.train()
# trainer.evaluate(test_dl)