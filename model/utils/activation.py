import torch
from torch import Tensor

def fused_add_tanh_sigmoid_multiply(a: Tensor, b: Tensor, n_channels: Tensor):
    n_channels_int = n_channels[0]
    in_act = a + b
    return torch.tanh(in_act[:, :n_channels_int, :]) * torch.sigmoid(in_act[:, n_channels_int:, :])