## Implementation of 2019 Chemical Science code in Python
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Simulation settings
experiments = int(1e5)
duration = 100
attempts = duration
time = np.linspace(0, duration, attempts + 1)
N = 100
d = 100e-4
V = (4/3) * np.pi * (d/2)**3
J = 1.23

# Probability a droplet remains crystal-free
P0t = np.exp(-J * V * time * 3600)
cutoff = P0t[1]

def run_experiment(N, attempts, cutoff):
    droplets = N
    result = np.zeros(attempts + 1, dtype=int)
    for step in range(1, attempts + 1):
        new_crystals = np.sum(np.random.rand(droplets) > cutoff)
        result[step] = new_crystals
        droplets -= new_crystals
    return result

# Monte Carlo Simulation
crystals = Parallel(n_jobs=-1)(
    delayed(run_experiment)(N, attempts, cutoff) 
    for _ in range(experiments)
)
crystals = np.array(crystals)
N0 = N - np.cumsum(crystals, axis=1)
f_survival = N0 / N
bins = 0.5 * np.array([1 - 0.997, 1 - 0.95, 1 - 0.68, 1, 1 + 0.68, 1 + 0.95, 1 + 0.997])
envelopes = np.quantile(f_survival, bins, axis=0)

# Plot P0t
plt.figure(figsize=(8,6))
plt.plot(time, P0t, 'k--', label='P0t')

# Shaded regions
plt.fill_between(time, envelopes[0], envelopes[6], color=(0.6, 0.6, 0.6, 0.4), label='3σ')
plt.fill_between(time, envelopes[1], envelopes[5], color=(0.6, 0.6, 0.6, 0.4),  label='2σ')
plt.fill_between(time, envelopes[2], envelopes[4], color=(0.6, 0.6, 0.6, 0.4), label='1σ')

plt.xlabel('Time (hours)')
plt.ylabel('Survival (no crystal)')
plt.legend()
plt.show()
