import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import Size, Tensor
from typing import Union, List, Optional, Tuple


class DyT(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        alpha: float = 1.0, 
        bias: bool = True
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.alpha = nn.Parameter(torch.empty(1))
        self.alpha_init = alpha
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, self.alpha_init)
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        dyt = self.weight * torch.tanh(self.alpha * x)

        if self.bias is not None:
            dyt = dyt + self.bias
        
        return dyt


class DyISRU(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        alpha: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        C = int(torch.tensor(self.normalized_shape).prod().item())
        self.register_buffer("sqrt_C", torch.tensor(math.sqrt(C)), persistent=False)
        self.alpha_init = alpha
        self.beta = nn.Parameter(torch.empty(1))
        self.softplus = nn.Softplus(beta=1.0, threshold=5.0)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        beta_init = 1.0 / (self.alpha_init ** 2)
        init.constant_(self.beta, beta_init + math.log(math.expm1(beta_init)))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        dyisru = self.sqrt_C * x * torch.rsqrt(self.softplus(self.beta) + x ** 2)

        if self.bias is not None:
            dyisru = dyisru + self.bias

        return dyisru


class Derf(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        alpha: float = 1.0,
        shift: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        self.alpha_init = alpha
        self.shift_init = shift
        self.alpha = nn.Parameter(torch.empty(1))
        self.shift = nn.Parameter(torch.empty(1))
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, self.alpha_init)
        init.constant_(self.shift, self.shift_init)
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, x: Tensor) -> Tensor:
        derf = self.weight * torch.erf(self.alpha * x + self.shift)

        if self.bias is not None:
            derf = derf + self.bias

        return derf