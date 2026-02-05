# Hamiltonian Neural Networks

Implementation of Hamiltonian Neural Networks (HNNs) as described in [Greydanus et al., 2019](https://arxiv.org/abs/1906.01563).

## Overview

HNNs learn a scalar Hamiltonian function H(q, p) and derive dynamics through Hamilton's equations:
- dq/dt = ∂H/∂p
- dp/dt = -∂H/∂q

This architectural constraint ensures **automatic energy conservation**, unlike standard neural networks that predict derivatives directly.

## Installation

```bash
# Using pip
pip install torch numpy scipy matplotlib pytest

# Or using pipenv
pipenv install
```

## Usage

### Training

```bash
# Basic training
python train.py

# With symplectic integrator (better energy conservation)
python train.py --use-leapfrog

# Custom settings
python train.py --epochs 100 --lr 1e-3 --seed 42 --output results.png
```

### Running Tests

```bash
pytest test_hnn.py -v
```

## Project Structure

| File | Description |
|------|-------------|
| `mlp.py` | Configurable MLP backbone |
| `hnn.py` | Hamiltonian Neural Network with gradient computation |
| `generate_data.py` | Mass-spring system data generation |
| `integrators.py` | Symplectic (leapfrog) and standard integrators |
| `training.py` | Training utilities for baseline and HNN |
| `train.py` | Main training/evaluation script |
| `test_hnn.py` | Comprehensive test suite |

## Key Improvements

1. **Symplectic Integration**: Added leapfrog integrator that preserves Hamiltonian structure
2. **Modular Architecture**: Separated concerns into distinct modules
3. **Configurable Models**: Flexible hidden dimensions and architecture
4. **Reproducibility**: Seed support throughout
5. **Comprehensive Tests**: Unit and integration tests
6. **Better Visualization**: Shows both true energy and learned Hamiltonian

## Why Energy Oscillates in Original Code

The original `test.png` shows oscillating energy for HNN because:

1. **True vs Learned Hamiltonian**: The plot showed `(q² + p²)/2` (true energy), but HNN learns an *approximate* H that may differ. The learned H is conserved, not necessarily the true one.

2. **Non-symplectic Integration**: Using `solve_ivp` (RK45) breaks the symplectic structure. Even perfect Hamiltonians show energy drift with non-symplectic integrators.

3. **Training Noise**: Data includes Gaussian noise, so the learned Hamiltonian is approximate.

**Solution**: Use `--use-leapfrog` flag to see improved energy conservation with the symplectic integrator.

## References

- [Hamiltonian Neural Networks (Greydanus et al., 2019)](https://arxiv.org/abs/1906.01563)
- [Symplectic Integrators (Leimkuhler & Reich, 2004)](https://www.cambridge.org/core/books/simulating-hamiltonian-dynamics/89E8E8599A9B5E99D52947D1521536FF)
