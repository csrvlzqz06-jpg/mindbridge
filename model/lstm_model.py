import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

class LSTMRiskNet(nn.Cell):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, has_bias=True, batch_first=True)
        self.dropout = nn.Dropout(keep_prob=0.9)
        self.fc = nn.Dense(hidden_size, num_classes)

    def construct(self, x: Tensor):
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # (B, H)
        last = self.dropout(last)
        logits = self.fc(last)         # (B, C)
        return logits
