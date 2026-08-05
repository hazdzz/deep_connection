import math
import numbers
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor


class RMSNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        eps: float = 1e-5, 
        bias: bool = False
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.weight * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm


class ScaleNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        eps: float = 1e-5, 
        bias: bool = False
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        C = int(torch.tensor(self.normalized_shape).prod().item())
        self.register_buffer("sqrt_C", torch.tensor(math.sqrt(C)), persistent=False)
        self.eps = eps
        self.gamma = nn.Parameter(torch.empty(1))
        self.softplus = nn.Softplus()
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        s = self.sqrt_C.item()
        init_val = s if s > 20.0 else math.log(math.expm1(s))
        init.constant_(self.gamma, init_val)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        norm = torch.linalg.vector_norm(input, ord=2, dim=dims, keepdim=True)
        gamma = self.softplus(self.gamma)
        scalenorm = gamma * input / (norm + self.eps)
        
        if self.bias is not None:
            scalenorm = scalenorm + self.bias

        return scalenorm


class SeeDNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        alpha_init: float = 1.0, 
        eps: float = 1e-5, 
        bias: bool = False
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.alpha_init = alpha_init
        self.gamma = nn.Parameter(torch.empty(self.normalized_shape))
        self.alpha = nn.Parameter(torch.empty(self.normalized_shape))
        self.beta = nn.Parameter(torch.empty(self.normalized_shape))
        self.eps = eps
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.gamma)
        init.constant_(self.alpha, self.alpha_init)
        init.zeros_(self.beta)

        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)
        rescale = torch.tanh((input * self.beta).sum(dim=-1, keepdim=True))
        dynamic_scale = rescale * self.alpha + self.gamma
        seednorm = dynamic_scale * input_norm

        if self.bias is not None:
            seednorm = seednorm + self.bias

        return seednorm


class GeoNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        angle_upper_bound: float = math.pi * 0.25, 
        layer_idx: int = 1, 
        num_layers: int = 10,
        decay_method: str = 'harmonic', 
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        assert angle_upper_bound > 0.0 and angle_upper_bound <= math.pi * 0.5
        assert layer_idx >= 1
        assert num_layers >= 1
        assert decay_method in ['harmonic', 'linear', 'sqrt', 'lns']
        assert eps > 0.0

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        n = len(self.normalized_shape)
        self.norm_dim = tuple(range(-n, 0))
        self.angle_upper_bound = min(angle_upper_bound, math.pi * 0.5 - eps)
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.decay_method = decay_method
        self.eps = eps
        self.eps_sq = self.eps ** 2
        self.alpha = nn.Parameter(torch.empty(1))
        self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, math.log(math.expm1(1.0)))

    def forward(self, h_in: Tensor, Fh: Tensor) -> Tensor:
        h_in_norm_sq = (h_in * h_in).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        h_in_norm = h_in_norm_sq.sqrt()
        h_in_trs_dot_Fh = (h_in * Fh).sum(dim=self.norm_dim, keepdim=True)

        z = Fh - (h_in_trs_dot_Fh / h_in_norm_sq) * h_in
        z_norm_sq = (z * z).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        z_norm = z_norm_sq.sqrt()
        z_hs = z / z_norm

        if self.decay_method == 'harmonic':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / self.layer_idx
        elif self.decay_method == 'linear':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) * (self.num_layers - self.layer_idx) / self.num_layers
        elif self.decay_method == 'sqrt' or self.decay_method == 'lns':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / math.sqrt(self.layer_idx)

        beta = (alpha * (z_norm / h_in_norm)).clamp(max=self.angle_upper_bound)

        h_out = torch.cos(beta) * h_in + h_in_norm * torch.sin(beta) * z_hs

        return h_out


class ProjSpheretNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        layer_idx: int = 1, 
        num_layers: int = 10, 
        decay_method: str = 'lns', 
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        assert layer_idx >= 1
        assert num_layers >= 1 and num_layers >= layer_idx
        assert decay_method in ['lns', 'sqrt', 'harmonic']
        assert eps > 0.0

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        n = len(self.normalized_shape)
        self.norm_dim = tuple(range(-n, 0))
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.decay_method = decay_method
        self.eps = eps
        self.eps_sq = self.eps ** 2
        self.alpha = nn.Parameter(torch.empty(1))
        self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, math.log(math.expm1(1.0)))

    def forward(self, h_in: Tensor, Fh: Tensor) -> Tensor:
        h_in_norm_sq = (h_in * h_in).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        h_in_norm = h_in_norm_sq.sqrt()
        h_in_trs_dot_Fh = (h_in * Fh).sum(dim=self.norm_dim, keepdim=True)

        z = Fh - (h_in_trs_dot_Fh / h_in_norm_sq) * h_in

        if self.decay_method == 'lns' or self.decay_method == 'sqrt':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / math.sqrt(self.layer_idx)
        elif self.decay_method == 'harmonic':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / self.layer_idx

        h_plus = h_in + alpha * z
        h_plus_norm_sq = (h_plus * h_plus).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        h_plus_norm = h_plus_norm_sq.sqrt()
        h_out = h_in_norm * (h_plus / h_plus_norm)

        return h_out


class CaySpheretNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        angle_upper_bound: float = math.pi * 0.25, 
        layer_idx: int = 1, 
        num_layers: int = 10, 
        decay_method: str = 'lns', 
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        assert angle_upper_bound > 0.0 and angle_upper_bound <= math.pi * 0.5
        assert layer_idx >= 1
        assert num_layers >= 1 and num_layers >= layer_idx
        assert decay_method in ['lns', 'sqrt', 'harmonic']
        assert eps > 0.0

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        n = len(self.normalized_shape)
        self.norm_dim = tuple(range(-n, 0))
        self.angle_upper_bound = min(angle_upper_bound, math.pi * 0.5 - eps)
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.decay_method = decay_method
        self.eps = eps
        self.eps_sq = self.eps ** 2
        self.alpha = nn.Parameter(torch.empty(1))
        self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, math.log(math.expm1(1.0)))

    def forward(self, h_in: Tensor, Fh: Tensor) -> Tensor:
        h_in_norm_sq = (h_in * h_in).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        h_in_norm = h_in_norm_sq.sqrt()
        h_in_trs_dot_Fh = (h_in * Fh).sum(dim=self.norm_dim, keepdim=True)

        z = Fh - (h_in_trs_dot_Fh / h_in_norm_sq) * h_in
        z_norm_sq = (z * z).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        z_norm = z_norm_sq.sqrt()

        if self.decay_method == 'lns' or self.decay_method == 'sqrt':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / math.sqrt(self.layer_idx)
        elif self.decay_method == 'harmonic':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / self.layer_idx

        beta_max = math.tan(self.angle_upper_bound * 0.5) * 2.0
        gamma = torch.minimum(alpha, beta_max * (h_in_norm / z_norm))
        beta = gamma * (z_norm / h_in_norm)
        beta_sq = beta * beta
        h_out = ((4 - beta_sq) * h_in + 4 * gamma * z) / (4 + beta_sq)

        return h_out


class pSpheretNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        angle_upper_bound: float = math.pi * 0.25, 
        p: float = 3.0, 
        layer_idx: int = 1, 
        num_layers: int = 10, 
        decay_method: str = 'lns', 
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        assert angle_upper_bound > 0.0 and angle_upper_bound <= math.pi * 0.5
        assert p > 0.0
        assert layer_idx >= 1
        assert num_layers >= 1 and num_layers >= layer_idx
        assert decay_method in ['lns', 'sqrt', 'harmonic']
        assert eps > 0.0

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        n = len(self.normalized_shape)
        self.norm_dim = tuple(range(-n, 0))
        self.angle_upper_bound = min(angle_upper_bound, math.pi * 0.5 - eps)
        self.p = p
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.decay_method = decay_method
        self.eps = eps
        self.eps_sq = self.eps ** 2
        self.alpha = nn.Parameter(torch.empty(1))
        self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, math.log(math.expm1(1.0)))

    def forward(self, h_in: Tensor, Fh: Tensor) -> Tensor:
        h_in_norm_sq = (h_in * h_in).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        h_in_norm = h_in_norm_sq.sqrt()
        h_in_trs_dot_Fh = (h_in * Fh).sum(dim=self.norm_dim, keepdim=True)

        z = Fh - (h_in_trs_dot_Fh / h_in_norm_sq) * h_in
        z_norm_sq = (z * z).sum(dim=self.norm_dim, keepdim=True).clamp(min=self.eps_sq)
        z_norm = z_norm_sq.sqrt()
        z_hs = z / z_norm

        if self.decay_method == 'lns' or self.decay_method == 'sqrt':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / math.sqrt(self.layer_idx)
        elif self.decay_method == 'harmonic':
            alpha = torch.clamp(self.softplus(self.alpha), max=1.0) / self.layer_idx

        beta = alpha * (z_norm / h_in_norm)
        theta = self.p * torch.arctan(beta / self.p)
        if self.p > 1.0:
            theta = torch.clamp(theta, max=self.angle_upper_bound)

        h_out = torch.cos(theta) * h_in + (h_in_norm * torch.sin(theta)) * z_hs

        return h_out