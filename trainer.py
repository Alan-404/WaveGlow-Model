import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from model.wave_glow import WaveGlow
from model.loss import WaveGlowLoss
import os

class Trainer:
    def __init__(self,
                 n_mel_channels: int=80,
                 n_flows: int = 12, 
                 n_group: int = 8,
                 n_early_every: int = 4,
                 n_early_size: int = 2,
                 wn_n_layers: int = 8,
                 wn_n_channels: int = 256,
                 wn_kernel_size: int =3,
                 checkpoint: str = None,
                 device: str = "cpu") -> None:
        
        self.model = WaveGlow(
            n_mel_channels=n_mel_channels,
            n_flows=n_flows,
            n_group=n_group,
            n_early_every=n_early_every,
            n_early_size=n_early_size,
            wn_n_layers=wn_n_layers,
            wn_n_channels=wn_n_channels,
            wn_kernel_size=wn_kernel_size
        )

        self.device = device

        self.loss_function = WaveGlowLoss()
        self.optimizer = optim.Adam(params=self.model.parameters())

        self.loss_epoch = 0.0
        self.loss_batch = 0.0
        self.epoch = 0

        self.model.to(device)
        self.checkpoint = checkpoint
        if self.checkpoint:
            self.load_model(self.checkpoint)

    def build_dataset(self, inputs: Tensor, labels: Tensor, batch_size: int):
        return DataLoader(dataset=TensorDataset(inputs, labels), batch_size=batch_size, shuffle=True)
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, path)

    def load_model(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
    
    def train_step(self, inputs: Tensor, labels: Tensor):
        self.optimizer.zero_grad()

        outputs = self.model(inputs, labels)
        loss = self.loss_function(outputs)
        loss.backward()
        self.optimizer.step()

        self.loss_batch += loss.item()
        self.loss_epoch += loss.item()

    def fit(self, X_train: Tensor, y_train: Tensor, learning_rate: float = 0.00003, epochs: int = 1, batch_size: int = 1, mini_batch: int = 1, **kwargs):
        dataloader = self.build_dataset(X_train, y_train, batch_size)
        
        for param in self.optimizer.param_groups():
            param['lr'] = learning_rate

        self.model.train()
        for _ in range(epochs):
            count = 0
            for index, data in enumerate(dataloader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                self.train_step(inputs, labels)
                count += 1
                if index%mini_batch == mini_batch-1 and index == len(dataloader)-1:
                    print(f"Epoch: {self.epoch} Batch: {index + 1} Loss: {(self.loss_batch/count):.4f}")
            self.epoch += 1