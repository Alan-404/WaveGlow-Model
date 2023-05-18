import torch 
import torch.nn as nn
from torch import Tensor

from ..utils.activation import fused_add_tanh_sigmoid_multiply

class WN(nn.Module):
    def __init__(self, n_in_channels: int, n_mel_channels: int, n_layers: int, n_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_channels = n_channels
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        self.start = nn.Conv1d(in_channels=n_in_channels, out_channels=n_channels, kernel_size=1)
        self.start = nn.utils.weight_norm(self.start, name='weight')

        self.end = nn.Conv1d(in_channels=n_channels, out_channels=2*n_in_channels, kernel_size=1)
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

        self.cond_layer = nn.Conv1d(in_channels=n_mel_channels, out_channels=2*n_channels*n_layers, kernel_size=1)
        self.cond_layer = nn.utils.weight_norm(self.cond_layer, name='weight')

        for i in range(n_layers):
            dilation = 2**i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = nn.Conv1d(in_channels=n_channels, out_channels=2*n_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            if i < n_layers-1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels

            res_skip_layer = nn.Conv1d(in_channels=n_channels, out_channels=res_skip_channels, kernel_size=1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio: Tensor, spect: Tensor):
        """ 
            audio: Tensor
            - (batch_size, n_half, time / n_group)
            spect: Tensor
            - (batch_size, n_mel_channels * n_group, frames/n_group)
        """
        audio = self.start(audio)
        output = torch.zeros_like(audio)

        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :],
                n_channels_tensor
            )

            res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers-1:
                audio += res_skip_acts[:, :self.n_channels, :]
                output += res_skip_acts[:, self.n_channels:, :]
            else:
                output += res_skip_acts
        
        return self.end(output) 