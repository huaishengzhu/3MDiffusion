import torch
from inspect import isfunction
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from poly_hgraph.transformer.x_transformer import AbsolutePositionalEmbedding, Encoder
from torch_geometric.utils import to_dense_batch

def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x

def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x

def sample_center_gravity_zero_gaussian(size, device):
    # assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected

def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4
def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors

def exists(x):
    return x is not None

def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """

    # return sum_except_batch(
    #         (
    #             torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
    #             + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
    #             - 0.5
    #         )
    #     )

    # return sum_except_batch(
    #         (
    #             torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8).unsqueeze(-1)
    #             + 0.5 * ((q_sigma**2).unsqueeze(-1) + (q_mu - p_mu)**2) / (p_sigma**2).unsqueeze(-1)
    #             - 0.5
    #         ) 
    #     )

    return sum_except_batch(
        (
            torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
            + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
            - 0.5
        ) * node_mask
    )

def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (d * torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8) 
            + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) 
            - 0.5 * d
            )

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(val):
    return val is not None

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        tx_dim,
        tx_depth,
        heads,
        latent_dim = None,
        context_dim = None,
        max_seq_len=1000,
        self_condition = True,
        dropout = 0.1,
        scale_shift = False,
        # class_conditional=False,
        # num_classes=0,
        # class_unconditional_prob=0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.mask = None
        self.context_mask = None
        self.self_condition = self_condition
        self.scale_shift = scale_shift
        # self.class_conditional = class_conditional
        # self.num_classes = num_classes
        # self.class_unconditional_prob = class_unconditional_prob

        self.max_seq_len = max_seq_len

        # time embeddings

        sinu_pos_emb = SinusoidalPosEmb(tx_dim)

        time_emb_dim = tx_dim*4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, tx_dim)
            )

        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)
        
        self.cross = self_condition

        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout = dropout,    # dropout post-attention
            ff_dropout = dropout,       # feedforward dropout
            rel_pos_bias=True,
            ff_glu=True,
            cross_attend=self.cross,
            time_emb_dim=tx_dim*4 if self.scale_shift else None,
        )
        print(self_condition)
        if self_condition:
            self.null_embedding = nn.Embedding(1, tx_dim)
            self.context_proj = nn.Linear(context_dim, tx_dim)
        # if self.class_conditional:
        #     assert num_classes > 0
        #     self.class_embedding = nn.Embedding(num_classes+1, tx_dim)
        
        self.input_proj = nn.Linear(latent_dim + context_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim, latent_dim)

        init_zero_(self.output_proj)

    def forward(self, x, mask, time, x_self_cond = None):
        """
        x: input, [batch, length, latent_dim]
        mask: bool tensor where False indicates masked positions, [batch, length] 
        time: timestep, [batch]
        """
        x = torch.cat((x, x_self_cond.repeat(1, x.shape[1], 1)), dim=-1)

        time_emb = self.time_mlp(time)
        mask = mask.squeeze()
        # self.mask = torch.ones((x.shape[0], x.shape[1]), dtype=bool, device=x.device)
        # if self.mask is None or self.context_mask is None:
        self.context_mask = torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        pos_emb = self.pos_emb(x)

        tx_input = self.input_proj(x) + pos_emb + self.time_pos_embed_mlp(time_emb)

        if self.cross:
            # context, context_mask = [], []
            # if self.self_condition:
            if x_self_cond is None:
                null_context = repeat(self.null_embedding.weight, '1 d -> b 1 d', b=x.shape[0])
                context = null_context
                # context_mask = torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device)
            else: 
                context = self.context_proj(x_self_cond)
                # context_mask.append(mask)
            # if self.class_conditional:
            #     assert exists(class_id)
            #     class_emb = self.class_embedding(class_id)
            #     class_emb = rearrange(class_emb, 'b d -> b 1 d')
            #     context.append(class_emb)
            #     context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device))
            # context = torch.cat(context, dim=1)
            # context_mask = torch.cat(context_mask, dim=1)
            x = self.encoder(tx_input, mask=mask, context=context, context_mask=self.context_mask, time_emb=time_emb)
        else:
            x = self.encoder(tx_input, mask=mask, time_emb=time_emb)

        x = self.norm(x)

        return self.output_proj(x)

class NodeLogits(nn.Module):
    def __init__(
        self,
        node_dim: int,
        output_dim: int = 1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, output_dim),
        )

    def forward(
        self, x
    ) -> torch.Tensor:
        return self.mlp(x)

class DiffusionMLP(nn.Module):
    def __init__(
    self,
    tx_dim,
    tx_depth,
    heads,
    latent_dim = None,
    context_dim = None,
    max_seq_len=100,
    self_condition = True,
    dropout = 0.1,
    scale_shift = False,
    # class_conditional=False,
    # num_classes=0,
    # class_unconditional_prob=0,
):
        super().__init__()

        self.latent_dim = latent_dim
        self.mask = None
        self.context_mask = None
        self.self_condition = self_condition
        self.scale_shift = scale_shift
        # self.class_conditional = class_conditional
        # self.num_classes = num_classes
        # self.class_unconditional_prob = class_unconditional_prob

        # self.max_seq_len = max_seq_len

        # time embeddings
        sinu_pos_emb = SinusoidalPosEmb(tx_dim)
        self.graph_dim = latent_dim
        time_emb_dim = tx_dim*4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, tx_dim)
            )

        # self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)
        
        self.cross = self_condition

        self.encoder = nn.ModuleList([nn.Linear(tx_dim, tx_dim)] + [
            nn.Linear(tx_dim, tx_dim)
            for _ in range(tx_depth - 1)
        ])
        # Encoder(
        #     dim=tx_dim,
        #     depth=tx_depth,
        #     heads=heads,
        #     attn_dropout = dropout,    # dropout post-attention
        #     ff_dropout = dropout,       # feedforward dropout
        #     rel_pos_bias=True,
        #     ff_glu=True,
        #     cross_attend=self.cross,
        #     time_emb_dim=tx_dim*4 if self.scale_shift else None,
        # )
        print(self_condition)
        if self_condition:
            self.null_embedding = nn.Embedding(1, context_dim)
            self.context_proj = nn.Linear(context_dim, tx_dim)
        # if self.class_conditional:
        #     assert num_classes > 0
        #     self.class_embedding = nn.Embedding(num_classes+1, tx_dim)
        
        self.input_proj = nn.Linear(latent_dim + context_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim, latent_dim)

        init_zero_(self.output_proj)

    def forward(self, x, mask, time, x_self_cond = None):
        """
        x: input, [batch, length, latent_dim]
        mask: bool tensor where False indicates masked positions, [batch, length] 
        time: timestep, [batch]
        """
        x = torch.cat((x, x_self_cond), dim=-1)

        time_emb = self.time_mlp(time)
        # mask = mask.squeeze()
        # self.mask = torch.ones((x.shape[0], x.shape[1]), dtype=bool, device=x.device)
        # if self.mask is None or self.context_mask is None:
        # self.context_mask = torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device)
        # time_emb = rearrange(time_emb, 'b d -> b 1 d')

        # pos_emb = self.pos_emb(x)

        x = self.input_proj(x) + self.time_pos_embed_mlp(time_emb)

        for i, layer in enumerate(self.encoder):
                # print(i, x.abs().max().item())
                x = layer(x)
                x = F.relu(x)

        x = self.norm(x)

        return self.output_proj(x)