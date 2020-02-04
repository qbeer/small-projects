import torch


class MLP(torch.nn.Module):
    def __init__(self, output_dim = 2):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 128),
                                       torch.nn.Tanh(),
                                       torch.nn.Linear(128, 64),
                                       torch.nn.Tanh(),
                                       torch.nn.Linear(64, 32),
                                       torch.nn.Tanh(),
                                       torch.nn.Linear(32, output_dim, bias=None))

    def forward(self, x):
        return self.mlp(x)