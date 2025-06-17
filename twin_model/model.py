import torch.nn as nn

class GRUDigitalTwin(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(GRUDigitalTwin, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)
