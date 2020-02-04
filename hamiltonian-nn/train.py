from generate_data import DataGenerator
from mlp import MLP
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from hnn import HNN
from scipy.integrate import solve_ivp

gen = DataGenerator()
X, y = gen.get_dataset(return_X_y=True)

colors=['bx', 'rx', 'gx']

for i, ind in enumerate(np.random.choice(range(0, 25), 3)):
    plt.plot(X[ind, :, 0], X[ind, :, 1], colors[i])
plt.savefig('trajectory-samples.png')

net = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

EPOCHS = 80

for epoch in range(EPOCHS):
    running_loss = 0.0
    for ind in np.random.choice(range(25), 25, replace=False):
        optimizer.zero_grad()

        _X = torch.from_numpy(X[ind])
        _y = torch.from_numpy(y[ind])

        _y_hat = net(_X)
    
        loss = criterion(_y_hat, _y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print('ITER %4d | MSE loss : %8.4f |' % ((epoch + 1) * 25,  running_loss))

print('************************************')

hnn_net = HNN()
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
comparison_loss = nn.MSELoss()
optimizer = optim.Adam(hnn_net.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(EPOCHS):
    running_loss = 0.0
    hnn_loss = 0.0
    for ind in np.random.choice(range(25), 25, replace=False):
        optimizer.zero_grad()

        _X = torch.from_numpy(X[ind])
        _X.requires_grad_()
        _y = torch.from_numpy(y[ind])

        pred, energy = hnn_net(_X)
        dHdq, dHdp = torch.split(pred, 1, dim=1)
        q_dot, p_dot = torch.split(_y, 1, dim=1)

        q_dot_hat, p_dot_hat = dHdp, -dHdq

        _y_hat = torch.cat([q_dot_hat, p_dot_hat], axis=1)
    
        loss = criterion1(p_dot, -dHdq) + criterion2(q_dot, dHdp)
        comp_loss = comparison_loss(_y_hat, _y)
        loss.backward()
        optimizer.step()

        running_loss += comp_loss.item()
        hnn_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print('ITER %4d | MSE loss : %8.4f | HNN loss : %8.4f' % ((epoch + 1) * 25,  running_loss, hnn_loss))

X, y = gen.get_dataset(return_X_y=True)

energy = y[0][:, 0]**2 / 2. - y[0][:, 1] ** 2 / 2.

total_energy = y[0][:, 0]**2 / 2. + y[0][:, 1]**2 / 2

real_start = torch.from_numpy(X[0][0])

def baseline(t, y):
    y = torch.tensor(y, requires_grad=True, dtype=torch.float32).view(1, 2)
    dy = net(y)
    return dy.data.numpy().reshape(-1)

def hnn(t, y):
    y = torch.tensor(y, requires_grad=True, dtype=torch.float32).view(1, 2)
    dH, H = hnn_net(y)
    dHdq, dHdp = torch.split(dH, 1, dim=1)
    q_dot_hat, p_dot_hat = dHdp, -dHdq
    dy = torch.cat([q_dot_hat, p_dot_hat], axis=1)
    return dy.data.numpy().reshape(-1)

baseline_model_output = solve_ivp(baseline, t_span=[0, 100], y0=real_start, t_eval=np.linspace(0, 100, 1000)).y.T

real_start = torch.from_numpy(X[0][0])

hnn_model_output = solve_ivp(hnn, t_span=[0, 100], y0=real_start, t_eval=np.linspace(0, 100, 1000)).y.T


baseline_total_energy = np.sum(baseline_model_output ** 2 / 2, axis=1)
hnn_total_energy = np.sum(hnn_model_output ** 2 / 2, axis=1)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for ind, ax in enumerate(axes.flatten()):
    if ind == 0:
        ax.plot(baseline_model_output[:, 0], baseline_model_output[:, 1], 'rx')
        ax.plot(X[0][:, 0], X[0][:, 1], 'gx')
        ax.set_title('Baseline p-q')
    if ind == 2:
        ax.plot(hnn_model_output[:, 0], hnn_model_output[:, 1], 'bx')
        ax.plot(X[0][:, 0], X[0][:, 1], 'gx')
        ax.set_title('HNN p-q')
    if ind == 1:
        ax.plot(range(baseline_total_energy.size), baseline_total_energy, 'r-')
        ax.plot(range(total_energy.size), total_energy, 'g-')
        ax.set_title('Baseline E')
    if ind == 3:
        ax.plot(range(hnn_total_energy.size), hnn_total_energy, 'b-')
        ax.plot(range(total_energy.size), total_energy, 'g-')
        ax.set_title('HNN E')
    plt.savefig('test.png')
