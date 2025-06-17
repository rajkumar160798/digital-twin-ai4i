import torch.nn as nn
import torch
class BiGRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.bigru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        out, _ = self.bigru(x)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        weighted = (out * attn_weights).sum(dim=1)
        return self.fc(weighted).unsqueeze(1).repeat(1, x.shape[1], 1)