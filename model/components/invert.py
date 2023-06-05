import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Invertible1x1Convolution(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        
        W = torch.qr(torch.FloatTensor(channels, channels).normal_())[0]

        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        W = W.view(channels, channels, 1)
        self.conv.weight.data = W

    def forward(self, z: Tensor, reverse: bool = False):
        """ 
            z: Tensor
            - (batch_size, n_group, time / n_group)
        """
        batch_size, group_size, n_of_group = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, "W_inverse"):
                W_inverse = W.float().inverse() # Invert Weights Matrix
                W_inverse = Variable(W_inverse[..., None]) # (channels, channels, 1)
                if z.type() == "torch.cuda.HalfTensor":
                    W_inverse = W_inverse.half() # convert weights been inside to float16
                self.W_inverse = W_inverse
            return F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0) # (batch_size, n_group, time / n_group)
        else:
            log_det_W = batch_size * n_of_group * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W