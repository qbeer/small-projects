import torch
from mlp import MLP


class HNN(torch.nn.Module):
    def __init__(self):
        super(HNN, self).__init__()
        self.mlp = MLP()

    def forward(self, x):
        return self.mlp(x)

    def time_derivative(self, x):
        E = self.mlp(x)
        dE = torch.autograd.grad(E.sum(), x, create_graph=True)
        
        return 