"""
Tests for Hamiltonian Neural Network components.
"""
import pytest
import numpy as np
import torch

from generate_data import DataGenerator, mass_spring_hamiltonian, generate_mass_spring_trajectories
from mlp import MLP
from hnn import HNN
from integrators import leapfrog_step, integrate_leapfrog, integrate_hnn_leapfrog
from training import train_baseline_mlp, train_hnn


class TestDataGeneration:
    """Tests for data generation."""
    
    def test_trajectory_shape(self):
        """Test that generated trajectories have correct shape."""
        gen = DataGenerator(n_trajectories=10, seed=42)
        X, y = gen.get_dataset(return_derivatives=True)
        
        assert X.shape == (10, 200, 2), f"Expected (10, 200, 2), got {X.shape}"
        assert y.shape == (10, 200, 2), f"Expected (10, 200, 2), got {y.shape}"
    
    def test_trajectory_dtype(self):
        """Test that trajectories are float32."""
        gen = DataGenerator(n_trajectories=5, seed=42)
        X, y = gen.get_dataset(return_derivatives=True)
        
        assert X.dtype == np.float32
        assert y.dtype == np.float32
    
    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        gen1 = DataGenerator(n_trajectories=5, seed=42)
        gen2 = DataGenerator(n_trajectories=5, seed=42)
        
        X1, y1 = gen1.get_dataset(return_derivatives=True)
        X2, y2 = gen2.get_dataset(return_derivatives=True)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_energy_range(self):
        """Test that initial energies are within specified range."""
        gen = DataGenerator(energy_range=(0.5, 0.6), n_trajectories=20, 
                           noise_std=0.0, seed=42)
        X = gen.get_dataset(return_derivatives=False)
        
        # Check initial energy of each trajectory (without noise)
        for i in range(20):
            q0, p0 = X[i, 0]
            E = mass_spring_hamiltonian(q0, p0)
            # Allow some tolerance due to numerical integration
            assert 0.4 < E < 0.7, f"Energy {E} outside expected range"
    
    def test_hamiltonian_conservation_noiseless(self):
        """Test that noiseless trajectories conserve energy."""
        X, _ = generate_mass_spring_trajectories(
            n_trajectories=1, n_points=100, noise_std=0.0, seed=42
        )
        
        energies = [mass_spring_hamiltonian(q, p) for q, p in X[0]]
        energies = np.array(energies)
        
        # Energy should be nearly constant (small numerical error allowed)
        assert np.std(energies) < 0.01, f"Energy std {np.std(energies)} too high"


class TestMLP:
    """Tests for MLP model."""
    
    def test_output_shape(self):
        """Test MLP output shape."""
        model = MLP(input_dim=2, output_dim=2)
        x = torch.randn(10, 2)
        y = model(x)
        
        assert y.shape == (10, 2), f"Expected (10, 2), got {y.shape}"
    
    def test_custom_architecture(self):
        """Test MLP with custom hidden dimensions."""
        model = MLP(input_dim=4, hidden_dims=(64, 32), output_dim=1)
        x = torch.randn(5, 4)
        y = model(x)
        
        assert y.shape == (5, 1)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = MLP()
        x = torch.randn(10, 2, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestHNN:
    """Tests for Hamiltonian Neural Network."""
    
    def test_output_shapes(self):
        """Test HNN output shapes."""
        model = HNN()
        x = torch.randn(10, 2, requires_grad=True)
        grads, H = model(x)
        
        assert grads.shape == (10, 2), f"Expected grads (10, 2), got {grads.shape}"
        assert H.shape == (10, 1), f"Expected H (10, 1), got {H.shape}"
    
    def test_time_derivative(self):
        """Test time derivative computation."""
        model = HNN()
        x = torch.randn(10, 2, requires_grad=True)
        dxdt = model.time_derivative(x)
        
        assert dxdt.shape == (10, 2)
    
    def test_hamiltonian_method(self):
        """Test standalone Hamiltonian computation."""
        model = HNN()
        x = torch.randn(10, 2)
        H = model.hamiltonian(x)
        
        assert H.shape == (10, 1)
    
    def test_gradient_is_computed(self):
        """Test that autograd properly computes gradients."""
        model = HNN()
        x = torch.tensor([[1.0, 0.5]], requires_grad=True)
        grads, H = model(x)
        
        # Gradients should be non-zero for non-zero input
        assert not torch.all(grads == 0)


class TestIntegrators:
    """Tests for numerical integrators."""
    
    def test_leapfrog_mass_spring(self):
        """Test leapfrog integrator on mass-spring system."""
        # For mass-spring: H = (q² + p²)/2, dH/dq = q, dH/dp = p
        dHdq = lambda q, p: q
        dHdp = lambda q, p: p
        
        trajectory = integrate_leapfrog(
            q0=1.0, p0=0.0, dt=0.1, n_steps=1000,
            dHdq_func=dHdq, dHdp_func=dHdp
        )
        
        # Check energy conservation
        energies = np.sum(trajectory**2, axis=1) / 2
        energy_drift = np.abs(energies[-1] - energies[0])
        
        assert energy_drift < 0.01, f"Energy drift {energy_drift} too large"
    
    def test_leapfrog_circular_orbit(self):
        """Test that leapfrog produces circular orbit for harmonic oscillator."""
        dHdq = lambda q, p: q
        dHdp = lambda q, p: p
        
        trajectory = integrate_leapfrog(
            q0=1.0, p0=0.0, dt=0.01, n_steps=628,  # Roughly one period
            dHdq_func=dHdq, dHdp_func=dHdp
        )
        
        # Should return close to starting point
        np.testing.assert_allclose(trajectory[-1], trajectory[0], atol=0.01)
    
    def test_leapfrog_step_reversibility(self):
        """Test that leapfrog step is reversible."""
        dHdq = lambda q, p: q
        dHdp = lambda q, p: p
        
        q0, p0 = 1.0, 0.5
        dt = 0.1
        
        # Forward step
        q1, p1 = leapfrog_step(q0, p0, dt, dHdq, dHdp)
        # Backward step
        q2, p2 = leapfrog_step(q1, p1, -dt, dHdq, dHdp)
        
        np.testing.assert_allclose([q2, p2], [q0, p0], atol=1e-10)


class TestTraining:
    """Tests for training utilities."""
    
    @pytest.fixture
    def small_dataset(self):
        """Generate small dataset for testing."""
        gen = DataGenerator(n_trajectories=5, seed=42)
        return gen.get_dataset(return_derivatives=True)
    
    def test_baseline_training_reduces_loss(self, small_dataset):
        """Test that baseline MLP training reduces loss."""
        X, y = small_dataset
        model = MLP(output_dim=2)
        
        losses = train_baseline_mlp(model, X, y, epochs=20, verbose=False)
        
        assert losses[-1] < losses[0], "Loss should decrease during training"
    
    def test_hnn_training_reduces_loss(self, small_dataset):
        """Test that HNN training reduces loss."""
        X, y = small_dataset
        model = HNN()
        
        losses = train_hnn(model, X, y, epochs=20, verbose=False)
        
        # Check that comparison loss decreases
        assert losses[-1][0] < losses[0][0], "Loss should decrease during training"


class TestEnergyConservation:
    """Integration tests for energy conservation."""
    
    def test_trained_hnn_conserves_learned_hamiltonian(self):
        """Test that trained HNN approximately conserves its learned Hamiltonian."""
        # Generate data
        gen = DataGenerator(n_trajectories=25, seed=42)
        X, y = gen.get_dataset(return_derivatives=True)
        
        # Train HNN
        model = HNN()
        train_hnn(model, X, y, epochs=50, verbose=False)
        
        # Integrate with leapfrog
        q0, p0 = X[0, 0]
        trajectory, hamiltonians = integrate_hnn_leapfrog(
            model, q0, p0, dt=0.1, n_steps=500
        )
        
        # Learned Hamiltonian should be approximately conserved
        h_std = np.std(hamiltonians)
        h_range = np.max(hamiltonians) - np.min(hamiltonians)
        
        # Allow some variation but it should be bounded
        assert h_range < 0.5, f"Hamiltonian range {h_range} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
