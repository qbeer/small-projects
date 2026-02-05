"""
Training utilities for Hamiltonian Neural Networks.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_baseline_mlp(model, X, y, epochs=80, lr=1e-4, weight_decay=1e-4, 
                       batch_size=None, verbose=True):
    """
    Train baseline MLP to predict derivatives directly.
    
    Args:
        model: MLP model
        X: Input trajectories (n_traj, n_points, 2)
        y: Target derivatives (n_traj, n_points, 2)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        batch_size: If None, use full trajectories
        verbose: Print progress
    
    Returns:
        losses: List of epoch losses
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_trajectories = X.shape[0]
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        indices = np.random.permutation(n_trajectories)
        
        for ind in indices:
            optimizer.zero_grad()
            
            _X = torch.from_numpy(X[ind])
            _y = torch.from_numpy(y[ind])
            
            _y_hat = model(_X)
            loss = criterion(_y_hat, _y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / n_trajectories
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:4d} | MSE loss: {running_loss:.4f}')
    
    return losses


def train_hnn(model, X, y, epochs=80, lr=1e-4, weight_decay=1e-4, verbose=True):
    """
    Train Hamiltonian Neural Network.
    
    The HNN learns a scalar Hamiltonian H(q, p) such that:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
    
    Args:
        model: HNN model
        X: Input trajectories (n_traj, n_points, 2)
        y: Target derivatives (n_traj, n_points, 2)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        verbose: Print progress
    
    Returns:
        losses: List of (comparison_loss, hnn_loss) tuples per epoch
    """
    criterion_q = nn.MSELoss()
    criterion_p = nn.MSELoss()
    comparison_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_trajectories = X.shape[0]
    losses = []
    
    for epoch in range(epochs):
        running_comp_loss = 0.0
        running_hnn_loss = 0.0
        indices = np.random.permutation(n_trajectories)
        
        for ind in indices:
            optimizer.zero_grad()
            
            _X = torch.from_numpy(X[ind])
            _X.requires_grad_(True)
            _y = torch.from_numpy(y[ind])
            
            pred, energy = model(_X)
            dHdq, dHdp = pred[:, 0:1], pred[:, 1:2]
            q_dot, p_dot = _y[:, 0:1], _y[:, 1:2]
            
            # Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
            q_dot_hat, p_dot_hat = dHdp, -dHdq
            
            # HNN loss: enforce Hamilton's equations
            loss = criterion_q(q_dot, dHdp) + criterion_p(p_dot, -dHdq)
            
            # Comparison loss for logging
            _y_hat = torch.cat([q_dot_hat, p_dot_hat], dim=1)
            comp_loss = comparison_loss(_y_hat, _y)
            
            loss.backward()
            optimizer.step()
            
            running_comp_loss += comp_loss.item()
            running_hnn_loss += loss.item()
        
        losses.append((running_comp_loss, running_hnn_loss))
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:4d} | MSE: {running_comp_loss:.4f} | HNN: {running_hnn_loss:.4f}')
    
    return losses
