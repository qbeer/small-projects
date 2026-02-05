import numpy as np
from scipy.integrate import solve_ivp


def mass_spring_hamiltonian(q, p):
    """True Hamiltonian for the mass-spring system: H = (q² + p²) / 2"""
    return (q**2 + p**2) / 2


def mass_spring_diff_eq(t, vec):
    """Hamilton's equations for mass-spring: dq/dt = p, dp/dt = -q"""
    q, p = vec
    return [p, -q]


def generate_mass_spring_trajectories(n_trajectories=25, n_points=200, 
                                       energy_low=0.2, energy_high=1.0,
                                       noise_std=0.1, seed=None):
    """
    Generate mass-spring system trajectories.
    
    Args:
        n_trajectories: Number of trajectories to generate
        n_points: Number of points per trajectory
        energy_low: Minimum initial energy
        energy_high: Maximum initial energy
        noise_std: Standard deviation of Gaussian noise added to trajectories
        seed: Random seed for reproducibility
    
    Returns:
        trajectories: Array of shape (n_trajectories, n_points, 2)
        derivatives: Array of shape (n_trajectories, n_points, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectories = []
    derivatives = []
    
    for _ in range(n_trajectories):
        E = np.random.uniform(energy_low, energy_high)
        phi = np.random.uniform(0, 2 * np.pi)  # Full circle, not just half
        r = np.sqrt(2. * E)
        q0 = r * np.cos(phi)
        p0 = r * np.sin(phi)
        
        sol = solve_ivp(
            mass_spring_diff_eq, 
            t_span=[0, 100], 
            y0=[q0, p0], 
            first_step=0.1, 
            max_step=0.1,
            method='RK45'
        )
        
        trajectory = sol.y.T[:n_points]
        derivative = np.array([mass_spring_diff_eq(0, vec) for vec in trajectory])
        
        # Add noise
        noisy_trajectory = trajectory + np.random.randn(n_points, 2) * noise_std
        
        trajectories.append(noisy_trajectory)
        derivatives.append(derivative)
    
    return np.array(trajectories, dtype=np.float32), np.array(derivatives, dtype=np.float32)


class DataGenerator:
    """
    Data generator for Hamiltonian system trajectories.
    """
    DATASETS = {"mass_spring": generate_mass_spring_trajectories}
    
    def __init__(self, dataset_name="mass_spring", energy_range=(0.2, 1.0), 
                 n_trajectories=25, noise_std=0.1, seed=None):
        """
        Initialize data generator.
        
        Args:
            dataset_name: Name of the dataset to generate
            energy_range: Tuple of (min_energy, max_energy)
            n_trajectories: Number of trajectories
            noise_std: Noise standard deviation
            seed: Random seed
        """
        self.data_gen = self.DATASETS[dataset_name]
        self.energy_range = energy_range
        self.n_trajectories = n_trajectories
        self.noise_std = noise_std
        self.seed = seed
    
    def get_dataset(self, return_derivatives=False):
        """
        Generate dataset.
        
        Args:
            return_derivatives: If True, return (X, y) tuple, else just X
        
        Returns:
            X: Trajectories array
            y: Derivatives array (if return_derivatives=True)
        """
        X, y = self.data_gen(
            n_trajectories=self.n_trajectories,
            energy_low=self.energy_range[0],
            energy_high=self.energy_range[1],
            noise_std=self.noise_std,
            seed=self.seed
        )
        if return_derivatives:
            return X, y
        return X
