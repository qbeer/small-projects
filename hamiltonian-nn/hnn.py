import torch
from mlp import MLP


class HNN(torch.nn.Module):
    """
    Hamiltonian Neural Network.
    
    Learns a scalar Hamiltonian H(q, p) and computes dynamics via:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
    
    This architecture automatically conserves the learned Hamiltonian.
    """
    
    def __init__(self, input_dim=2, hidden_dims=(128, 64, 32)):
        super(HNN, self).__init__()
        self.mlp_backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)

    def forward(self, x):
        """
        Forward pass computing Hamiltonian gradients.
        
        Args:
            x: Input tensor of shape (batch, 2) containing (q, p) pairs.
               Must have requires_grad=True for gradient computation.
        
        Returns:
            grads: Tensor of shape (batch, 2) containing (∂H/∂q, ∂H/∂p)
            H: Tensor of shape (batch, 1) containing the Hamiltonian values
        """
        H = self.mlp_backbone(x)
        grads = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return grads, H
    
    def time_derivative(self, x):
        """
        Compute time derivatives using Hamilton's equations.
        
        Args:
            x: Input tensor of shape (batch, 2) containing (q, p) pairs.
        
        Returns:
            Tensor of shape (batch, 2) containing (dq/dt, dp/dt)
        """
        grads, _ = self.forward(x)
        dHdq, dHdp = grads[:, 0:1], grads[:, 1:2]
        # Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        return torch.cat([dHdp, -dHdq], dim=1)
    
    def hamiltonian(self, x):
        """
        Compute only the Hamiltonian value (no gradients).
        
        Args:
            x: Input tensor of shape (batch, 2) containing (q, p) pairs.
        
        Returns:
            Tensor of shape (batch, 1) containing Hamiltonian values
        """
        return self.mlp_backbone(x)
