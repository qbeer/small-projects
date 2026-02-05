#!/usr/bin/env python
"""
Training and evaluation script for Hamiltonian Neural Networks.

Compares a baseline MLP (direct derivative prediction) against an HNN
(learns Hamiltonian and derives dynamics via Hamilton's equations).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp

from generate_data import DataGenerator, mass_spring_hamiltonian
from mlp import MLP
from hnn import HNN
from training import train_baseline_mlp, train_hnn
from integrators import integrate_hnn_leapfrog


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare HNN vs baseline MLP")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--n-trajectories", type=int, default=25, help="Number of training trajectories")
    parser.add_argument("--use-leapfrog", action="store_true", help="Use symplectic leapfrog integrator")
    parser.add_argument("--output", type=str, default="test.png", help="Output plot filename")
    return parser.parse_args()


def integrate_baseline(model, y0, t_span, t_eval):
    """Integrate baseline MLP using scipy's solve_ivp."""
    def dynamics(t, y):
        y_tensor = torch.tensor(y, requires_grad=True, dtype=torch.float32).view(1, 2)
        dy = model(y_tensor)
        return dy.detach().numpy().reshape(-1)
    
    sol = solve_ivp(dynamics, t_span=t_span, y0=y0, t_eval=t_eval)
    return sol.y.T


def integrate_hnn_rk(model, y0, t_span, t_eval):
    """Integrate HNN using scipy's solve_ivp."""
    def dynamics(t, y):
        y_tensor = torch.tensor(y, requires_grad=True, dtype=torch.float32).view(1, 2)
        dH, H = model(y_tensor)
        dHdq, dHdp = dH[0, 0], dH[0, 1]
        # Hamilton's equations
        q_dot = dHdp.item()
        p_dot = -dHdq.item()
        return [q_dot, p_dot]
    
    sol = solve_ivp(dynamics, t_span=t_span, y0=y0, t_eval=t_eval)
    return sol.y.T


def main():
    args = parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Generate data
    print("Generating training data...")
    gen = DataGenerator(n_trajectories=args.n_trajectories, seed=args.seed)
    X, y = gen.get_dataset(return_derivatives=True)
    
    # Plot sample trajectories
    print("Plotting sample trajectories...")
    colors = ['b', 'r', 'g']
    plt.figure(figsize=(8, 8))
    for i, ind in enumerate(np.random.choice(range(args.n_trajectories), min(3, args.n_trajectories), replace=False)):
        plt.plot(X[ind, :, 0], X[ind, :, 1], f'{colors[i]}x', alpha=0.5, label=f'Trajectory {ind}')
    plt.xlabel('q')
    plt.ylabel('p')
    plt.title('Sample Training Trajectories')
    plt.legend()
    plt.savefig('trajectory-samples.png', dpi=150)
    plt.close()
    
    # Train baseline MLP
    print("\n" + "="*50)
    print("Training Baseline MLP")
    print("="*50)
    baseline_model = MLP(output_dim=2)
    train_baseline_mlp(baseline_model, X, y, epochs=args.epochs, lr=args.lr)
    
    # Train HNN
    print("\n" + "="*50)
    print("Training Hamiltonian Neural Network")
    print("="*50)
    hnn_model = HNN()
    train_hnn(hnn_model, X, y, epochs=args.epochs, lr=args.lr)
    
    # Evaluation
    print("\n" + "="*50)
    print("Evaluating models...")
    print("="*50)
    
    # Get fresh test trajectory
    test_gen = DataGenerator(n_trajectories=1, noise_std=0.0, seed=123)
    X_test, _ = test_gen.get_dataset(return_derivatives=True)
    y0 = X_test[0, 0]
    
    t_span = [0, 100]
    t_eval = np.linspace(0, 100, 1000)
    
    # Integrate models
    print("Integrating baseline model...")
    baseline_traj = integrate_baseline(baseline_model, y0, t_span, t_eval)
    
    if args.use_leapfrog:
        print("Integrating HNN with leapfrog...")
        hnn_traj, learned_H = integrate_hnn_leapfrog(hnn_model, y0[0], y0[1], dt=0.1, n_steps=1000)
    else:
        print("Integrating HNN with RK45...")
        hnn_traj = integrate_hnn_rk(hnn_model, y0, t_span, t_eval)
        # Compute learned Hamiltonian along trajectory
        learned_H = []
        for q, p in hnn_traj:
            x = torch.tensor([[q, p]], dtype=torch.float32)
            H = hnn_model.hamiltonian(x)
            learned_H.append(H.item())
        learned_H = np.array(learned_H)
    
    # Compute energies
    baseline_energy = np.sum(baseline_traj**2, axis=1) / 2
    hnn_true_energy = np.sum(hnn_traj**2, axis=1) / 2
    ground_truth_energy = np.sum(X_test[0]**2, axis=1) / 2
    
    # Plot results
    print("Generating comparison plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Baseline phase space
    axes[0, 0].plot(baseline_traj[:, 0], baseline_traj[:, 1], 'r-', alpha=0.7, label='Baseline')
    axes[0, 0].plot(X_test[0, :, 0], X_test[0, :, 1], 'g--', alpha=0.5, label='Ground Truth')
    axes[0, 0].set_xlabel('q')
    axes[0, 0].set_ylabel('p')
    axes[0, 0].set_title('Baseline: Phase Space')
    axes[0, 0].legend()
    axes[0, 0].set_aspect('equal')
    
    # Baseline energy
    axes[0, 1].plot(baseline_energy, 'r-', label='Baseline')
    axes[0, 1].axhline(y=ground_truth_energy[0], color='g', linestyle='--', label='True E₀')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].set_title('Baseline: True Energy')
    axes[0, 1].legend()
    
    # HNN phase space
    axes[1, 0].plot(hnn_traj[:, 0], hnn_traj[:, 1], 'b-', alpha=0.7, label='HNN')
    axes[1, 0].plot(X_test[0, :, 0], X_test[0, :, 1], 'g--', alpha=0.5, label='Ground Truth')
    axes[1, 0].set_xlabel('q')
    axes[1, 0].set_ylabel('p')
    axes[1, 0].set_title('HNN: Phase Space')
    axes[1, 0].legend()
    axes[1, 0].set_aspect('equal')
    
    # HNN true energy
    axes[1, 1].plot(hnn_true_energy, 'b-', label='HNN (True E)')
    axes[1, 1].axhline(y=ground_truth_energy[0], color='g', linestyle='--', label='True E₀')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('HNN: True Energy (q²+p²)/2')
    axes[1, 1].legend()
    
    # HNN learned Hamiltonian
    axes[1, 2].plot(learned_H, 'm-', label='Learned H')
    axes[1, 2].set_xlabel('Time step')
    axes[1, 2].set_ylabel('H')
    axes[1, 2].set_title('HNN: Learned Hamiltonian')
    axes[1, 2].legend()
    
    # Energy comparison
    axes[0, 2].plot(baseline_energy, 'r-', alpha=0.7, label='Baseline')
    axes[0, 2].plot(hnn_true_energy, 'b-', alpha=0.7, label='HNN')
    axes[0, 2].axhline(y=ground_truth_energy[0], color='g', linestyle='--', label='True E₀')
    axes[0, 2].set_xlabel('Time step')
    axes[0, 2].set_ylabel('Energy')
    axes[0, 2].set_title('Energy Comparison')
    axes[0, 2].legend()
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")
    
    # Print summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Baseline energy drift: {baseline_energy[-1] - baseline_energy[0]:.4f}")
    print(f"HNN true energy std:   {np.std(hnn_true_energy):.4f}")
    print(f"HNN learned H std:     {np.std(learned_H):.4f}")
    print(f"HNN learned H range:   {np.max(learned_H) - np.min(learned_H):.4f}")


if __name__ == "__main__":
    main()
