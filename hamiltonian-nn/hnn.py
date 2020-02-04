import torch
from mlp import MLP

class HNN(torch.nn.Module):
    def __init__(self):
        super(HNN, self).__init__()
        self.mlp_backbone = MLP(output_dim=1)

    def forward(self, x):
        H = self.mlp_backbone(x)
        grads = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return grads, H

