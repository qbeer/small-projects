import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def mass_spring_system(energy_low, energy_high):
    trajectories = []
    derivatives = []
    for i in range(25):
        trajectory = []
        E = np.random.uniform(energy_low, energy_high)
        phi = np.random.uniform(0, np.pi)
        r = np.sqrt(2. * E)
        x = r * np.cos(phi)
        p = r * np.sin(phi)
        sol = solve_ivp(mass_spring_diff_eq, t_span=[0, 100], y0=[x, p], first_step=0.1, max_step=0.1)
        # adding random noise to the trajectory
        trajectory = sol.y.T[:200] 
        derivative = [mass_spring_diff_eq(0, vec) for vec in trajectory]
        trajectories.append(trajectory + np.random.randn(200, 2) * 0.1)
        derivatives.append(derivative)
    return np.array(trajectories, dtype=np.float32), np.array(derivatives, dtype=np.float32)
        

def mass_spring_diff_eq(t, vec):
    x, p = vec
    return [p, -x]

class DataGenerator:
    DATASETS = {"mass_spring" : mass_spring_system}
    def __init__(self, dataset_name = "mass_spring", energy_range = [.2, 1.]):
        self.data_gen = self.DATASETS[dataset_name]
        self.energy_range = energy_range
    def get_dataset(self, return_X_y = False):
        X, y = self.data_gen(self.energy_range[0], self.energy_range[1])
        if return_X_y:
            return X, y
        return X