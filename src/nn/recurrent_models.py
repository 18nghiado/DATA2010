import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(in_channel, dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(dim, out_channel)

    def forward(self, x , h):
        print("Herte")
        
        if x.ndim == 1: x = x.unsqueeze(0)
        out, hidden = self.rnn(x,h)
        return self.fc(out[:, -1, :]), hidden  # last timestep
    

class LSTM(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(in_channel, dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(dim, out_channel)

    def forward(self, x, h):
        if x.ndim == 1: x = x.unsqueeze(0)
        out, (hidden, cell) = self.lstm(x,h)
        return self.fc(out[:, -1, :]), hidden  # last timestep
    

class GRU(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, dim: int, num_layers: int = 1):
        super().__init__()
        if x.ndim == 1: x = x.unsqueeze(0)

        self.gru = nn.GRU(in_channel, dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(dim, out_channel)

    def forward(self, x,h):
        out, hidden = self.gru(x,h)
        return self.fc(out[:, -1, :]), hidden  # last timestep