"""
Numerical integrators for Hamiltonian systems.
"""
import numpy as np


def leapfrog_step(q, p, dt, dHdq_func, dHdp_func):
    """
    Single step of the leapfrog (Störmer-Verlet) integrator.
    
    This is a symplectic integrator that preserves the Hamiltonian structure
    and provides better long-term energy conservation than RK methods.
    
    Args:
        q: Position
        p: Momentum
        dt: Time step
        dHdq_func: Function computing ∂H/∂q
        dHdp_func: Function computing ∂H/∂p
    
    Returns:
        q_new, p_new: Updated position and momentum
    """
    # Half step in momentum
    p_half = p - 0.5 * dt * dHdq_func(q, p)
    # Full step in position
    q_new = q + dt * dHdp_func(q, p_half)
    # Half step in momentum
    p_new = p_half - 0.5 * dt * dHdq_func(q_new, p_half)
    return q_new, p_new


def integrate_leapfrog(q0, p0, dt, n_steps, dHdq_func, dHdp_func):
    """
    Integrate Hamiltonian system using leapfrog method.
    
    Args:
        q0: Initial position
        p0: Initial momentum
        dt: Time step
        n_steps: Number of integration steps
        dHdq_func: Function computing ∂H/∂q
        dHdp_func: Function computing ∂H/∂p
    
    Returns:
        trajectory: Array of shape (n_steps+1, 2) containing (q, p) pairs
    """
    trajectory = np.zeros((n_steps + 1, 2))
    trajectory[0] = [q0, p0]
    
    q, p = q0, p0
    for i in range(n_steps):
        q, p = leapfrog_step(q, p, dt, dHdq_func, dHdp_func)
        trajectory[i + 1] = [q, p]
    
    return trajectory


def integrate_hnn_leapfrog(hnn_model, q0, p0, dt, n_steps):
    """
    Integrate using HNN with leapfrog method.
    
    Args:
        hnn_model: Trained HNN model
        q0, p0: Initial conditions
        dt: Time step
        n_steps: Number of steps
    
    Returns:
        trajectory: Array of shape (n_steps+1, 2)
        hamiltonians: Array of shape (n_steps+1,) with learned H values
    """
    import torch
    
    trajectory = np.zeros((n_steps + 1, 2))
    hamiltonians = np.zeros(n_steps + 1)
    trajectory[0] = [q0, p0]
    
    q, p = float(q0), float(p0)
    
    def get_gradients(q_val, p_val):
        x = torch.tensor([[q_val, p_val]], requires_grad=True, dtype=torch.float32)
        grads, H = hnn_model(x)
        return grads[0, 0].item(), grads[0, 1].item(), H[0, 0].item()
    
    # Get initial Hamiltonian
    _, _, hamiltonians[0] = get_gradients(q, p)
    
    for i in range(n_steps):
        # Get gradients at current position
        dHdq, dHdp, _ = get_gradients(q, p)
        
        # Half step in momentum
        p_half = p - 0.5 * dt * dHdq
        
        # Get gradient at half-step
        dHdq_half, dHdp_half, _ = get_gradients(q, p_half)
        
        # Full step in position
        q_new = q + dt * dHdp_half
        
        # Get gradient at new position
        dHdq_new, _, H_new = get_gradients(q_new, p_half)
        
        # Half step in momentum
        p_new = p_half - 0.5 * dt * dHdq_new
        
        q, p = q_new, p_new
        trajectory[i + 1] = [q, p]
        hamiltonians[i + 1] = H_new
    
    return trajectory, hamiltonians
