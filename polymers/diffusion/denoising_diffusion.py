import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from accelerate import Accelerator
import wandb
import diffusion.optimizer as optimizer
from diffusion.torch_utils import compute_grad_norm
from torch_geometric.utils import to_dense_batch
from rdkit import Chem
from rdkit.Chem import rdRascalMCES
from diffusion.evaluate_metric import eval_novelty, eval_diversity
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
import time
# helpers functions
def evaluate_smi(orig_smiles, pred_smiles, part):
    outputs = []
    bad_mols = 0
    smiles = []
    # f_id = open(path)
    # lines = f_id.readlines()
    # idxs = []
    # for line in lines:
    #     idxs.append(int(line.strip()))
    # f_id.close()
    # with open(osp.join(input_file)) as f:
    #     reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    #     for n, line in enumerate(reader):
            # if n not in idxs:
            #     continue
    for i in range(0, len(orig_smiles)):
        try:
            gt_smi = orig_smiles[i]
            ot_smi = pred_smiles[i]
            smiles.append([gt_smi, ot_smi])
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)
            Chem.Kekulize(gt_m)
            Chem.Kekulize(ot_m)
            if ot_m == None: raise ValueError('Bad SMILES')
            outputs.append(("", gt_m, ot_m))
        except:
            bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    sub_smi = []
    count = 0
    count_rd = 0
    count_nv = 0
    for i, (desc, gt_m, ot_m) in enumerate(outputs):
        smi_temp = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=eval('DataStructs.CosineSimilarity'))

        opts = rdRascalMCES.RascalOptions()
        opts.returnEmptyMCES = True
        opts.similarityThreshold = 0.0
        results = rdRascalMCES.FindMCES(gt_m, ot_m, opts)
        
        if len(results) == 0 or results is None:
            sub_smi.append(0)

            # if results is None:
            #     print("Time out")
            continue   
        # if results[0].tier1Sim>=0.5:
        #     count+=1
        
        count+=1
        if smi_temp>=0.5:
            count_rd += 1
            if smi_temp<0.8:
                count_nv += 1
        sub_smi.append(results[0].tier1Sim)
    if not part:
        sort_id = sorted(range(len(sub_smi)), key=lambda k:sub_smi[k])
        f = open("chebi_sorted.txt", "w")
        f.write("description\tground truth\toutput\tsmiliarity\n")

        for id in sort_id:
            f.write( " "+ "\t" + smiles[id][0] + "\t" + smiles[id][1] + "\t" + str(sub_smi[id]) +"\n")
        f.close()
                
    # print("Qualified Samples", count/len(sub_smi))
    print("Qualified Samples", count_rd/count)
    print("Novelty:", count_nv/count_rd)   
    return validity_score, np.mean(sub_smi)
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def evaluate_smi_multi(orig_smiles, pred_smiles, part, n_samples):
    outputs = []
    bad_mols = 0
    good_mols = 0
    smiles = []

    for i in range(0, len(orig_smiles)):
        gt_smi = orig_smiles[i]
        gt_m = Chem.MolFromSmiles(gt_smi)
        Chem.Kekulize(gt_m)
        tmp_mol = []
        for j in range(0, len(pred_smiles[i])):
            try:
                ot_smi = pred_smiles[i][j]
                smiles.append([gt_smi, ot_smi])
                ot_m = Chem.MolFromSmiles(ot_smi)
                if ot_m == None: raise ValueError('Bad SMILES')
                Chem.Kekulize(ot_m)
                good_mols+=1
                tmp_mol.append(ot_m)
            except:
                bad_mols += 1

        outputs.append(("", gt_m, tmp_mol))
    validity_score = good_mols/(good_mols+ bad_mols)
    sub_smi = []
    count_nv = 0
    count_rd = 0
    count = 0
    novelty = []
    diversity = []
    diversity_novel = []
    for i, (desc, gt_m, ot_m) in enumerate(outputs):
        tmp_smi = []
        tmp_smi_novel = []
        for mol in ot_m:
            smi_temp = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(mol), metric=eval('DataStructs.CosineSimilarity'))
            count+=1
            if smi_temp>=0.5:
                count_rd += 1
                tmp_smi.append(mol)
                if smi_temp<0.8:
                    count_nv += 1
                    tmp_smi_novel.append(mol)
        tmp_smi = [gt_m] + tmp_smi 
        tmp_smi_novel = [gt_m] + tmp_smi_novel
        if len(tmp_smi) > 0:
            diversity += eval_diversity(tmp_smi)
            diversity_novel += eval_diversity(tmp_smi_novel)



                
    print("Qualified Samples", count_rd/count)
    print("Novelty:", count_nv/count_rd)
    print("Diversity:", np.mean(diversity))
    print("Validity:", validity_score)
    return validity_score, np.mean(diversity)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        class_unconditional_prob = 0.1, 
    ):
        super().__init__()

        self.diffusion_model = model
        self.class_unconditional_prob = class_unconditional_prob

        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition
        if class_unconditional_prob > 0:
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=class_unconditional_prob)

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        register_buffer('latent_mean', torch.tensor([0]*self.latent_dim))
        register_buffer('latent_scale', torch.tensor(1))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def normalize_latent(self, x_start):
        eps = 1e-5 
                
        return (x_start-self.latent_mean)/(self.latent_scale+eps)
    
    def unnormalize_latent(self, x_start):
        eps = 1e-5 
        
        return x_start*(self.latent_scale+eps)+self.latent_mean

    def diffusion_model_predictions(self, x, mask, t, x_self_cond = None, w = 1):
        if x_self_cond is None:
            x_self_cond = self.diffusion_model.null_embedding(torch.LongTensor([0]).cuda()).repeat(x.shape[0], 1)

        model_output = self.diffusion_model(x, mask, t, x_self_cond)
        if w!=1:
            uncond = self.diffusion_model.null_embedding(torch.LongTensor([0]).cuda()).repeat(model_output.shape[0], 1)
            model_output_un = self.diffusion_model(x, mask, t, uncond)
            model_output = w * model_output + (1 - w) * model_output_un

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, shape, mask, self_cond, w = 1, ddim=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        if ddim:
            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        else:
            times = torch.linspace(-1, sampling_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # latent = torch.randn(shape, device = device)
        latent = torch.randn((shape[0], shape[-1]), device = device)
        # mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
        # mask = torch.tensor(mask, dtype=torch.bool, device=device)
        
        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.diffusion_model_predictions(latent, mask, time_cond, self_cond, w=w)

            if time_next < 0:
                latent = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(latent)

            latent = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return (latent, mask)

    def train_ddim_sample(self, shape, mask, self_cond):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        latent = torch.randn(shape, device = device)
        # mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
        # mask = torch.tensor(mask, dtype=torch.bool, device=device)
        
        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.diffusion_model_predictions(latent, mask, time_cond, self_cond)

            if time_next < 0:
                latent = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(latent)

            latent = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return (latent, mask)

    def train_sample(self, latent, batch_mask, context):
        # TODO Create mask that controls length 
        batch_size, max_seq_len, latent_dim = latent.shape[0], latent.shape[1], latent.shape[2]
        # TODO Implement for p_sample_loop 
        
        sample_fn = self.train_ddim_sample
        return sample_fn((batch_size, max_seq_len, latent_dim), batch_mask, context)

    @torch.no_grad()
    def sample(self, batch_size, max_seq_len, latent_dim, batch_mask, context, w = 1, ddim=True):
        # TODO Create mask that controls length 
        # batch_size, max_seq_len, latent_dim = latent.shape[0], latent.shape[1], latent.shape[2]
        # TODO Implement for p_sample_loop 
        
        sample_fn = self.ddim_sample
        return sample_fn((batch_size, max_seq_len, latent_dim), batch_mask, context, w = w, ddim=ddim)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    #TODO handle masking 
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, context, mask, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        if self.class_unconditional_prob > 0:
            class_unconditional_mask = self.class_unconditional_bernoulli.sample((x_start.shape[0],1)).bool().squeeze()
            context[class_unconditional_mask, :] = self.diffusion_model.null_embedding(torch.LongTensor([0]).cuda())
        x_self_cond = context
            

        # predict and take gradient step
        
        predictions = self.diffusion_model_predictions(x, mask, t, x_self_cond)        
        

        loss = self.loss_fn(predictions.pred_x_start, x_start, reduction = 'none')
        # print(torch.min(x_start), torch.max(x_start))
        # print(torch.min(predictions.pred_x_start), torch.max(predictions.pred_x_start))
        # print(loss.shape)
        # 1/0
        # loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(x_start.shape[0])], 'b 1 -> b 1')

        return loss.mean()

    def forward(self, txt_latent, context, mask, *args, **kwargs):
        b, d, device = *txt_latent.shape, txt_latent.device
        # assert l == max_seq_len, f'length must be {self.max_seq_len}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(txt_latent, context, mask, t, *args, **kwargs)

# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion,
        dataloader,
        val_dataloader,
        val_full_dataloader,
        bart_model,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        save_and_sample_every = 100,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision = 'no',
        split_batches = True,
    ):
        super().__init__()


        set_seeds(42)

        self.args = args

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision,
            log_with='wandb'
        )

        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.accelerator.native_amp = amp
        self.num_timesteps = diffusion.num_timesteps
        self.diffusion = diffusion
        self.mask = None

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        # self.max_seq_len = diffusion.max_seq_len

        # Init Encoder-decoder model
        self.bart_model = bart_model.to("cuda")
        # self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.enc_dec_model)


        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.val_full_dataloader = val_full_dataloader

        # training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask']), self.max_seq_len) for idx in range(self.dataloader.dataset.num_rows)]
        # length_counts = Counter(training_lengths)
        # probs = torch.tensor([length_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.max_seq_len+1)])
        # assert probs[0] == 0, 'Can\'t have examples of length 0'
        # self.length_categorical = torch.distributions.Categorical(probs=probs)

        # if self.diffusion.diffusion_model.class_conditional:
        #     training_labels = [self.dataloader.dataset[idx]['label'] for idx in range(self.dataloader.dataset.num_rows)]
        #     label_counts = Counter(training_labels)
        #     probs = torch.tensor([label_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
        #     self.class_categorical = torch.distributions.Categorical(probs=probs)
        
        # optimizer

        self.opt = optimizer.get_adamw_optimizer(diffusion.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)

        self.opt_decoder = optimizer.get_adamw_optimizer(list(self.bart_model.parameters()), lr = 0.001, betas = adam_betas, weight_decay=adam_weight_decay)
        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_num_steps,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta = ema_decay, update_every = ema_update_every, power=3/4)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion, self.bart_model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(self.diffusion, self.bart_model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self):
        # if not self.accelerator.is_local_main_process:
        #     return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'decoder': self.bart_model.decoder.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-winoise100_train_decoder')   + f'.pt')

    def load(self, file_path=None):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model-winoise100_train_decoder') + f'.pt', map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # For backwards compatibility with earlier models
        self.ema.load_state_dict(data['ema'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        if 'decoder' in data:
            self.bart_model.decoder.load_state_dict(data['decoder'])
            print("Loading Decoder")
        
    def evaluate(self, batch, ini=False, w =1):
        # batch = batch
        # latents, context = self.bart_model.forward_encoder(*batch)
        latent, context = self.bart_model.forward_encoder(*batch)
        mask = torch.ones((context.shape[0], 8), dtype=bool, device=context.device)
        latents, _ = self.ema.ema_model.sample(latent.shape[0], 8, latent.shape[1], mask, context, w = w)
        # latents, _ = self.diffusion.sample(latents, mask, context)

        if self.args.normalize_latent:
            latents = self.ema.ema_model.unnormalize_latent(latents)
        logs = {"cond_text_mse": F.mse_loss(latent, latents).mean().item()
        }
        # print(logs)
        self.accelerator.log(logs, step=self.step)
            # latents = self.diffusion.unnormalize_latent(latents)
        # latents = torch.cat((latents, context.repeat(1, latents.shape[1], 1)), dim=-1)
        return self.bart_model.forward_decoder(latents, context.squeeze())
  
        
    def evaluate_test(self, loader, part=True):
        orig_smiles = []
        dec_smiles = []
        count = 0
        with torch.no_grad():
            with open('./results_condsample_train_chebi.txt', 'w') as fin:
                fin.write("description\tground truth\toutput\n")
                
                for i,batch in enumerate(loader):
                    if count ==3 and part:
                        break
                    orig_smiles = orig_smiles + batch[1]
                    # try:
                    # print(batch)
                    context = batch[0]
                    mask = torch.ones((context.shape[0], 8), dtype=bool, device=context.device)
                    latents, _ = self.ema.ema_model.sample(context.shape[0], 8, self.diffusion.latent_dim, mask, context)
                    # latents, _ = self.diffusion.sample(latents, mask, context)
                    if self.args.normalize_latent:
                        latents = self.ema.ema_model.unnormalize_latent(latents)

                    dec_smiles = dec_smiles + self.bart_model.forward_decoder(latents, context.squeeze())
                    
                    for x, y in zip(orig_smiles, dec_smiles):                        
                        fin.write( " "+ "\t" + x+"\t" + y +"\n")
                    count += 1
        
        validlity, rssmi = evaluate_smi(orig_smiles, dec_smiles, part)
        logs = {"validlity": validlity, 'rssmi': rssmi}
        self.accelerator.log(logs, step=self.step)

    def evaluate_test_multi(self, loader, part=True, n_samples=5, w = 0.9):
        orig_smiles = []
        dec_smiles = []
        count = 0
        with torch.no_grad():
            with open('./results_chebi.txt', 'w') as fin:
                fin.write("description\tground truth\toutput\n")
                print(len(loader), "------------")
                for i,batch in enumerate(loader):
                    if count ==3 and part:
                        break
                    orig_smiles = orig_smiles + batch[1]
                    # try:
                    # print(batch)
                    tmp_smiles = []
                    context = batch[0]
                    for t in range(0, context.shape[0]):
                        tmp_smiles.append([])

                    for t in range(0, n_samples):
                    
                        set_seed(t)
                        mask = torch.ones((context.shape[0], 8), dtype=bool, device=context.device)
                        # start_time = time.time()
                        latents, _ = self.ema.ema_model.sample(context.shape[0], 8, self.diffusion.latent_dim, mask, context, w = w)
                        # latents, _ = self.diffusion.sample(latents, mask, context)
                        if self.args.normalize_latent:
                            latents = self.ema.ema_model.unnormalize_latent(latents)

                        generated_smiles = self.bart_model.forward_decoder(latents, context.squeeze())
                        # end_time = time.time()
                        # print("Running Time:", (end_time - start_time)/len(generated_smiles))
                        for j in range(0, len(generated_smiles)):
                            tmp_smiles[j].append(generated_smiles[j])
                        
                    dec_smiles = dec_smiles + tmp_smiles
                    
                for x, y in zip(orig_smiles, dec_smiles):    
                    fin.write( " "+ "\t" + x)
                    for tmp in y:
                        fin.write("\t" + tmp)
                    fin.write("\n")
                count += 1
        
        validlity, rssmi = evaluate_smi_multi(orig_smiles, dec_smiles, part, n_samples)
        logs = {"validlity": validlity, 'rssmi': rssmi}
        self.accelerator.log(logs, step=self.step)

    def evaluate_uncond(self, num_samples=10000, batch_size = 32):
        ind_sample = np.arange(0, num_samples, dtype=int)
        num_batch = int(np.ceil(len(ind_sample)/batch_size))
        generated_smiles = []
        with torch.no_grad():
            with open('./results_uncondsample_chebi.txt', 'w') as fin:
                for i in range(0, num_batch):
                    if len(ind_sample[batch_size * i : batch_size * (i + 1)]) == 0:
                        break
                    set_seed(i)
                    # sampling_steps = np.random.randint(0, 10)
                    # self.ema.ema_model.sampling_timesteps =  sampling_steps
                    mask = torch.ones((len(ind_sample[batch_size * i : batch_size * (i + 1)]), 8), dtype=bool, device='cuda')
                    latents, _ = self.ema.ema_model.sample(len(ind_sample[batch_size * i : batch_size * (i + 1)]), 8, self.diffusion.latent_dim, mask, None, ddim=False)
                    if self.args.normalize_latent:
                        latents = self.ema.ema_model.unnormalize_latent(latents)

                    generated_smiles += self.bart_model.forward_decoder(latents, None, greedy=False)
                for y in generated_smiles:    
                    fin.write( y+ "\n")

            



    def train_decoder(self, batch):
        context = batch[-1]
        context = torch.from_numpy(np.stack(context)).cuda()
        mask = torch.ones((context.shape[0], 8), dtype=bool, device=context.device)
        latents, _ = self.ema.ema_model.sample(context.shape[0], 8, self.diffusion.latent_dim, mask, context)
        # latents, _ = self.diffusion.sample(latents, mask, context)

        if self.args.normalize_latent:
            latents = self.ema.ema_model.unnormalize_latent(latents)

        self.opt_decoder.zero_grad()
        loss, wacc, iacc, tacc, sacc= self.bart_model.train_decoder(*batch, root_vecs = latents)
        loss.backward()
        self.opt_decoder.step()

        logs = {"Word1": wacc, 'Word2': iacc, 'TopoAcc': tacc, 'Assmacc': sacc}
        self.accelerator.log(logs, step=self.step)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                # if self.step == 0:
                #     self.evaluate(self.val_dataloader, ini=True)
                #TODO center and normalize BART latent space with empirical est. of mean/var.

                total_loss = 0.


                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter)
                    with torch.no_grad():
                        latent, context = self.bart_model.forward_encoder(*data)
                        # print(context.shape)
                        # 1/0
                        mask = torch.ones((latent.shape[0], latent.shape[1]), dtype=bool, device=latent.device)
                        if self.args.normalize_latent:
                            if self.step==0 and grad_accum_step==0:
                                latent_vecs = latent[mask]
                                
                                # Add mean stats to model and EMA wrapper
                                self.diffusion.latent_mean = torch.mean(latent_vecs, dim=0)
                                self.ema.ema_model.latent_mean = self.diffusion.latent_mean

                                # Add var stats to model and EMA wrapper
                                self.diffusion.latent_scale = torch.std(latent_vecs-self.diffusion.latent_mean, unbiased=False)

                                self.ema.ema_model.latent_scale = self.diffusion.latent_scale
                            latent = self.diffusion.normalize_latent(latent)
                            # latent, _ = to_dense_batch(latent, data.batch)

                        

                    with self.accelerator.autocast():
                        loss = self.diffusion(latent, context, mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                    


                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.diffusion.parameters())

                accelerator.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                if not self.args.resume_training:
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # self.train_decoder(data)
                

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    # Log to WandB
                    if self.step % 50 == 0:
                        self.diffusion.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_ema_loss = 0.
                            for grad_accum_step in range(self.gradient_accumulate_every):
                                data = next(self.val_iter)
                                latent, context = self.bart_model.forward_encoder(*data)
                                mask = torch.ones((latent.shape[0], latent.shape[1]), dtype=bool, device=latent.device)

                                if self.args.normalize_latent:
                                    latent = self.diffusion.normalize_latent(latent)
                                    
                                with self.accelerator.autocast():
                                    loss = self.diffusion(latent, context, mask)
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_loss += loss.item()
                                    loss = self.ema.ema_model(latent, context, mask)
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_ema_loss += loss.item()


                            logs = {"loss": total_loss, "val_loss": total_val_loss, "val_ema_loss": total_val_ema_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step, "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), "samples": self.step*self.train_batch_size*self.gradient_accumulate_every}
                            pbar.set_postfix(**logs)
                            accelerator.log(logs, step=self.step)     
                        self.diffusion.train()           


                    if self.step % self.save_and_sample_every == 0:
                        # self.evaluate(self.val_dataloader, ini=False)
                        # self.sample()
                        # if self.diffusion.diffusion_model.class_conditional:
                        #     for class_id in range(self.diffusion.diffusion_model.num_classes):
                        #         if self.args.dataset_name == 'ag_news':
                        #             num_samples = 100
                        #         elif self.args.dataset_name == 'sst':
                        #             num_samples = 500
                        #         self.sample(num_samples=num_samples, class_id=class_id)
                        self.save()
                        # self.evaluate(data)
                        # self.evaluate_test(self.val_full_dataloader)
                        self.diffusion.train() 

                pbar.update(1)

        accelerator.print('training complete')