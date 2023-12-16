from typing import Optional

import torch
from torch import Tensor, nn


def build_model(name: str, out_size: int):
    if name == "LSTM":
        model = Seq2Seq_LSTM(in_channels=7, out_size=out_size)
    else:
        raise NotImplementedError()
    return model


class Seq2Seq_LSTM(nn.Module):
    def __init__(self, in_channels: int, out_size: int) -> None:
        super().__init__()

        self.out_size = out_size

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.encoder = nn.LSTM(256, 256, num_layers=2)
        self.decoder = nn.LSTM(256, 256, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        N, L, C = inputs.shape
        inputs = inputs.transpose(0, 1)     # [N, L, C] -> [L, N, C]
        inputs = self.embedding(inputs)
        _, (h, c) = self.encoder(inputs)
        y = torch.zeros((1, N, 256), device=inputs.device)
        outputs = []
        for t in range(self.out_size):
            y, (h, c) = self.decoder(y, (h, c))
            outputs.append(y)
            # if t < self.out_size - 1 and targets is not None:
            #     y = self.embedding(targets[:, t].unsqueeze(dim=0))
        outputs = torch.cat(outputs)
        outputs = self.fc(outputs)
        outputs = outputs.transpose(0, 1)   # [L, N, C] -> [N, L, C]
        return outputs
