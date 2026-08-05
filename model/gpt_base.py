from __future__ import annotations
import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dataclasses import dataclass
from typing import Union, List, Optional, Tuple
from typing import Any, ClassVar, Protocol, cast
from torch import Size, Tensor
from .norms import RMSNorm, SeeDNorm, GeoNorm, ProjSpheretNorm, CaySpheretNorm, pSpheretNorm
from .activations import DyT, DyISRU, Derf
from .hyper_conn import hyper_conn_init_func


HC_TYPE_FROM_DEEP_CONN = {"hc": "hc", "mhc": "mhc", "mhclite": "mhc_lite"}


def resolve_hc_type(deep_conn: str) -> str:
    return HC_TYPE_FROM_DEEP_CONN.get(deep_conn, "none")


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    deep_conn: str = "preln"
    hyper_conn_n: int = 4
    angle_upper_bound: float = math.pi / 4
    decay_method: str = "harmonic"
    p: float = 0.5
    scale_residual_init: Optional[bool] = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.scale_residual_init = config.scale_residual_init
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)   # for non-flash path
        self.resid_dropout = nn.Dropout(config.dropout)  # after c_proj
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def reset_parameters(self) -> None:
        init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        if self.c_attn.bias is not None:
            init.zeros_(self.c_attn.bias)

        if self.scale_residual_init is True:
            init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))
        else:
            init.normal_(self.c_proj.weight, mean=0.0, std=0.02)

        if self.c_proj.bias is not None:
            init.zeros_(self.c_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class FFN(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.scale_residual_init = config.scale_residual_init
        self.n_layer = config.n_layer
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.ffn_dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        if self.c_fc.bias is not None:
            init.zeros_(self.c_fc.bias)

        if self.scale_residual_init is True:
            init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))
        else:   
            init.normal_(self.c_proj.weight, mean=0.0, std=0.02)

        if self.c_proj.bias is not None:
            init.zeros_(self.c_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.ffn_dropout(x)
        x = self.c_proj(x)
        return x


class PreLNBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.ln_1.reset_parameters()
        self.ln_2.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class PostLNBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.ln_1.reset_parameters()
        self.ln_2.reset_parameters()

    def forward(self, x):
        x = self.ln_1(self.attn(x) + x)
        x = self.ln_2(self.ffn(x) + x)
        return x


class LNSBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.layernorm_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.layernorm_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.layer_idx = layer_idx
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.layernorm_1.reset_parameters()
        self.layernorm_2.reset_parameters()

    def forward(self, x):
        attn_layer_idx = 2 * self.layer_idx - 1
        ffn_layer_idx = 2 * self.layer_idx
        x = x + self.attn(self.layernorm_1(x) / math.sqrt(attn_layer_idx))
        x = x + self.ffn(self.layernorm_2(x) / math.sqrt(ffn_layer_idx))
        return x


class PeriLNBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.layernorm_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.layernorm_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.layernorm_3 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.layernorm_4 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.layernorm_1.reset_parameters()
        self.layernorm_2.reset_parameters()
        self.layernorm_3.reset_parameters()
        self.layernorm_4.reset_parameters()

    def forward(self, x):
        x = x + self.layernorm_2(self.attn(self.layernorm_1(x)))
        x = x + self.layernorm_4(self.ffn(self.layernorm_3(x)))
        return x


class PreSeeDNormBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.seednorm_1 = SeeDNorm(config.n_embd, bias=config.bias)
        self.seednorm_2 = SeeDNorm(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.seednorm_1.reset_parameters()
        self.seednorm_2.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.seednorm_1(x))
        x = x + self.ffn(self.seednorm_2(x))
        return x


class KEELBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.alpha = 2.0 * config.n_layer
        self.layernorm_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.layernorm_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.layernorm_3 = nn.LayerNorm(config.n_embd, bias=config.bias)
        if self.layer_idx == 1:
            self.layernorm_4 = nn.Identity()
        else:
            self.layernorm_4 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.layernorm_1.reset_parameters()
        self.layernorm_2.reset_parameters()
        self.layernorm_3.reset_parameters()
        reset_fn = getattr(self.layernorm_4, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()

    def forward(self, x):
        if self.layer_idx == 1:
            x = self.attn(self.layernorm_1(x)) + x
            x = self.ffn(self.layernorm_2(x)) + x
            x = self.layernorm_3(x)
            x = self.layernorm_4(x)
        else:
            x = self.layernorm_2(self.alpha * x + self.attn(self.layernorm_1(x)))
            x = self.layernorm_4(self.alpha * x + self.ffn(self.layernorm_3(x)))
        return x


class PreDyTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.dyt_1 = DyT(config.n_embd, bias=config.bias)
        self.dyt_2 = DyT(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.dyt_1.reset_parameters()
        self.dyt_2.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.dyt_1(x))
        x = x + self.ffn(self.dyt_2(x))
        return x


class PreDyISRUBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.dyisru_1 = DyISRU(config.n_embd, bias=config.bias)
        self.dyisru_2 = DyISRU(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.dyisru_1.reset_parameters()
        self.dyisru_2.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.dyisru_1(x))
        x = x + self.ffn(self.dyisru_2(x))
        return x


class PreDerfBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.derf_1 = Derf(config.n_embd, bias=config.bias)
        self.derf_2 = Derf(config.n_embd, bias=config.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.derf_1.reset_parameters()
        self.derf_2.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.derf_1(x))
        x = x + self.ffn(self.derf_2(x))
        return x


class HyperConnBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        hc_type = resolve_hc_type(config.deep_conn)
        init_hc, _, _ = hyper_conn_init_func(hc_type, config.hyper_conn_n)

        self.hc_attn = init_hc(
            dim=config.n_embd,
            layer_index=2 * layer_idx - 1,
        )
        self.hc_ffn = init_hc(
            dim=config.n_embd,
            layer_index=2 * layer_idx,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.ln_1, self.attn, self.ln_2, self.ffn,
                       self.hc_attn, self.hc_ffn):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x):
        branch_input, add_residual = self.hc_attn(x)
        x = add_residual(self.attn(self.ln_1(branch_input)))

        branch_input, add_residual = self.hc_ffn(x)
        x = add_residual(self.ffn(self.ln_2(branch_input)))
        return x


class GeoNormBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.geonorm_1 = GeoNorm(
                            config.n_embd, 
                            angle_upper_bound=config.angle_upper_bound, 
                            layer_idx=2 * layer_idx - 1, 
                            num_layers=2 * config.n_layer, 
                            decay_method=config.decay_method
                        )
        self.geonorm_2 = GeoNorm(
                            config.n_embd, 
                            angle_upper_bound=config.angle_upper_bound, 
                            layer_idx=2 * layer_idx, 
                            num_layers=2 * config.n_layer, 
                            decay_method=config.decay_method
                        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.geonorm_1.reset_parameters()
        self.geonorm_2.reset_parameters()

    def forward(self, x):
        x = self.geonorm_1(x, self.attn(x))
        x = self.geonorm_2(x, self.ffn(x))
        return x


class ProjSpheretNormBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.projspheretnorm_1 = ProjSpheretNorm(
                                    config.n_embd, 
                                    layer_idx=2 * layer_idx - 1, 
                                    num_layers=2 * config.n_layer, 
                                    decay_method=config.decay_method
                                )
        self.projspheretnorm_2 = ProjSpheretNorm(
                                    config.n_embd, 
                                    layer_idx=2 * layer_idx, 
                                    num_layers=2 * config.n_layer, 
                                    decay_method=config.decay_method
                                )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.projspheretnorm_1.reset_parameters()
        self.projspheretnorm_2.reset_parameters()

    def forward(self, x):
        x = self.projspheretnorm_1(x, self.attn(x))
        x = self.projspheretnorm_2(x, self.ffn(x))
        return x


class CaySpheretNormBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.cayspheretnorm_1 = CaySpheretNorm(
                                config.n_embd, 
                                angle_upper_bound=config.angle_upper_bound, 
                                layer_idx=2 * layer_idx - 1, 
                                num_layers=2 * config.n_layer, 
                                decay_method=config.decay_method
                            )
        self.cayspheretnorm_2 = CaySpheretNorm(
                                config.n_embd, 
                                angle_upper_bound=config.angle_upper_bound, 
                                layer_idx=2 * layer_idx, 
                                num_layers=2 * config.n_layer, 
                                decay_method=config.decay_method
                            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.cayspheretnorm_1.reset_parameters()
        self.cayspheretnorm_2.reset_parameters()

    def forward(self, x):
        x = self.cayspheretnorm_1(x, self.attn(x))
        x = self.cayspheretnorm_2(x, self.ffn(x))
        return x


class pSpheretNormBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FFN(config)
        self.pspheretnorm_1 = pSpheretNorm(
                                config.n_embd, 
                                angle_upper_bound=config.angle_upper_bound, 
                                p=config.p, 
                                layer_idx=2 * layer_idx - 1, 
                                num_layers=2 * config.n_layer, 
                                decay_method=config.decay_method
                            )
        self.pspheretnorm_2 = pSpheretNorm(
                                config.n_embd, 
                                angle_upper_bound=config.angle_upper_bound, 
                                p=config.p, 
                                layer_idx=2 * layer_idx, 
                                num_layers=2 * config.n_layer, 
                                decay_method=config.decay_method
                            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.pspheretnorm_1.reset_parameters()
        self.pspheretnorm_2.reset_parameters()

    def forward(self, x):
        x = self.pspheretnorm_1(x, self.attn(x))
        x = self.pspheretnorm_2(x, self.ffn(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.n_embd_sqrt = math.sqrt(self.config.n_embd)

        hc_type = resolve_hc_type(config.deep_conn)
        if hc_type == "none":
            self.expand_stream = None
            self.reduce_stream = None
        else:
            init_hc, expand_stream, reduce_stream = hyper_conn_init_func(
                hc_type,
                config.hyper_conn_n,
            )
            self.expand_stream = expand_stream
            self.reduce_stream = reduce_stream

        self.normalize_input = config.deep_conn in (
            'geonorm', 'projspheretnorm', 'cayspheretnorm', 'pspheretnorm',
        )
        if self.normalize_input is True:
            self.gamma = nn.Parameter(torch.empty(1))
        else:
            self.register_parameter("gamma", None)

        if config.deep_conn == 'preln':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    PreLNBlock(config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'postln':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    PostLNBlock(config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.Identity(),
            ))
        elif config.deep_conn == 'lns':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    LNSBlock(config, i + 1) for i in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'periln':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    PeriLNBlock(config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'preseednorm':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    PreSeeDNormBlock(config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'keel':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    KEELBlock(config, i + 1) for i in range(config.n_layer)
                ]),
                ln_f = nn.Identity(),
            ))
        elif config.deep_conn == 'predyt':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    PreDyTBlock(config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'predyisru':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    PreDyISRUBlock(config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'prederf':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    PreDerfBlock(config) for _ in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'geonorm':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    GeoNormBlock(config, i + 1) for i in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'projspheretnorm':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    ProjSpheretNormBlock(config, i + 1) for i in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'cayspheretnorm':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    CaySpheretNormBlock(config, i + 1) for i in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn == 'pspheretnorm':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    pSpheretNormBlock(config, i + 1) for i in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.deep_conn in ('hc', 'mhc', 'mhclite'):
            assert config.hyper_conn_n > 1, \
                f"deep_conn={config.deep_conn!r} needs hyper_conn_n > 1."
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([
                    HyperConnBlock(config, i + 1) for i in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        else:
            raise ValueError(
                f"Unknown deep_conn: {config.deep_conn!r}. "
                f"Valid options are: preln, postln, lns, periln, "
                f"keel, predyt, predyisru, prederf, "
                f"hc, mhc, mhclite, "
                f"geonorm, projspheretnorm, cayspheretnorm, pspheretnorm"
            )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02)
        init.normal_(self.transformer.wpe.weight, mean=0.0, std=0.02)

        if self.normalize_input is True:
            if self.n_embd_sqrt >= 20.0:
                init.constant_(self.gamma, self.n_embd_sqrt)
            else:
                init.constant_(self.gamma, math.log(math.expm1(self.n_embd_sqrt)))

        for block in self.transformer.h:
            block.reset_parameters()

        reset_fn = getattr(self.transformer.ln_f, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if self.normalize_input:
            x = F.normalize(x, dim=-1) * torch.clamp(F.softplus(self.gamma), min=1.0, max=self.n_embd_sqrt)

        if self.expand_stream is not None:
            x = self.expand_stream(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if self.reduce_stream is not None:
            x = self.reduce_stream(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    #     # first estimate the number of flops we do per iteration.
    #     # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    #     N = self.get_num_params()
    #     cfg = self.config
    #     L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    #     flops_per_token = 6*N + 12*L*H*Q*T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     # express our flops throughput as ratio of A100 bfloat16 peak flops
    #     flops_achieved = flops_per_iter * (1.0/dt) # per second
    #     flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    #     mfu = flops_achieved / flops_promised
    #     return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx