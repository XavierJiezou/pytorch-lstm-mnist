import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 64, num_layers: int = 1, output_size: int = 10):
        super(LSTM, self).__init__()
        self.main = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hn, cn) = self.main(x, None)
        result = self.fc(output[:, -1, :])
        return result


if __name__ == '__main__':
    print(LSTM()(torch.randn(10, 3, 28)))
