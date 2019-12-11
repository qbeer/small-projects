import torch


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(),
                                       torch.nn.Linear(64, 16),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(16, 1, bias=None))

    def forward(self, x):
        return self.mlp(x)