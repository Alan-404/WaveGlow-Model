import torch
from torch import Tensor
import torch.nn as nn

from .components.invert import Invertible1x1Convolution
from .components.net import WN

class WaveGlow(nn.Module):
    def __init__(self, n_mel_channels: int, n_flows: int, n_group: int, n_early_every: int, n_early_size: int, wn_n_layers: int, wn_n_channels: int, wn_kernel_size: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=n_mel_channels, out_channels=n_mel_channels, kernel_size=1024, stride=256)
        
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size

        self.WN = nn.ModuleList()
        self.convinv = nn.ModuleList()

        n_half = int(n_group/2)

        n_remaining_channels = n_group

        for k in range(n_flows):
            if k % n_early_every == 0 and k > 0:
                n_half = n_half - int(n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size

            self.convinv.append(Invertible1x1Convolution(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels*n_group, wn_n_layers, wn_n_channels, wn_kernel_size))
        
        self.n_remaining_channels = n_remaining_channels

    def forward(self, spect: Tensor, audio: Tensor):
        """ 
            spect: 
            - (batch_size, n_mel_channels, __frames)
            audio:
            - (batch_size, time)
        """
        spect = self.upsample(spect) # (batch_size, n_mel_channels, 256 * __frames + 768)

        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)] # (batch_size, n_mel_channels, frames)

        # squeeze vector
        # unfold: (batch_size, n_mel_channels, frames/n_group, n_group)
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3) # (batch_size, frames/n_group, n_mel_channels, n_group)
        # view: (batch_size, frames/n_group, n_mel_channels * n_group)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1) # (batch_size, n_mel_channels * n_group, frames/n_group)
        
        # unfold: (batch_size, time / n_group, n_group)
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) # (batch_size, n_group, time / n_group)

        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio) # (batch_size, n_group, time / n_group)

            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            # squeeze layer
            audio_0 = audio[:, :n_half, :] # (batch_size, n_half, time / n_group) ... n_half = [n_group/2, (n_group-2)/2, (n_group-2-2)/2, ...]
            audio_1 = audio[:, n_half:, :] # (batch_size, n_half, time / n_group) ... n_half = [n_group/2, (n_group-2)/2, (n_group-2-2)/2, ...]

            output = self.WN[k](audio_0, spect)

            log_s = output[:, n_half:, :]
            t = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + t
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list
    
    def infer(self, spect: Tensor, sigma: float=1.0):
        spect = self.upsample(spect)

        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        if spect.type() == 'torch.cuda.HalfTensor':
            audio = torch.cuda.HalfTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()
        else:
            audio = torch.cuda.FloatTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()

        audio = torch.autograd.Variable(sigma*audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]

            output = self.WN[k](audio_0, spect)

            s = output[:, n_half:, :]
            t = output[:, :n_half, :]
            audio_1 = (audio_1 - t)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1],1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma*z, audio),1)

        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
        
        return audio