import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dims=(128, 64, 32), output_dim=2):
        """
        Configurable MLP with flexible architecture.
        
        Args:
            input_dim: Input dimension (default: 2 for q, p)
            hidden_dims: Tuple of hidden layer sizes
            output_dim: Output dimension
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.Tanh())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim, bias=None))
        
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
