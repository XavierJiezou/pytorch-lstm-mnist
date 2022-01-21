import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.main = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        output, (hn, cn) = self.main(x, None)
        result = self.fc(output[:, -1, :])
        return result


if __name__ == '__main__':
    print(LSTM()(torch.randn(10, 28, 28)))
