import torch

class MLP(torch.nn.Module):
    r"""
    A multi-layer perceptron (MLP) with ReLU activations and optional dropout.

    Architecture: `Linear → ReLU → [Linear → ReLU → Dropout] × (num_layers - 2) → Linear`

    Args:
        - `in_channel`  (int): Number of input features
        - `out_channel` (int): Number of output features
        - `dim`         (int): Hidden layer dimension
        - `num_layers`  (int): Total number of linear layers (including input and output layers)
        - `dropout`     (float): Dropout probability applied after each hidden layer. Default: `0.0` (disabled)
    """
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
        r"""
        Forward pass through the MLP.

        Args:
            - `x` (torch.Tensor): Input tensor 

        Returns: Output tensor
        """
        return self.layers(x)
        
