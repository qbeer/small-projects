import torch
from model import HNN

hnn = HNN()

data = torch.tensor([[1, -2], [1.1, -2.3]],
                    requires_grad=True,
                    dtype=torch.float32)  # batch of 2 data points

E = hnn.forward(data)
print(E)
hnn.time_derivative(data)