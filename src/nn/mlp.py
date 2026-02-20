import torch

class MLP(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, dim: int, num_layers: int, dropout : int = 0.0):
        super().__init__() 
        self.layers = torch.nn.Sequential()
        self.layers.append(torch.nn.Linear(in_features=in_channel,out_features=dim))
        self.layers.append(torch.nn.ReLU())

        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(in_features=dim,out_features=dim))
            self.layers.append(torch.nn.ReLU())
            if dropout > 0:
                self.layers.append(torch.nn.Dropout(dropout))

        self.layers.append(torch.nn.Linear(in_features=dim,out_features=out_channel))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)
        
