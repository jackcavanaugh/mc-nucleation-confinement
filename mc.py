## Implementation of 2019 Chemical Science code in Python
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Simulation settings
experiments = int(1e3)
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

#for i in range(0, 5):
#    plt.plot(time, f_survival[i], '.', color=(i/10, i/10, i/10, 0.4), label='Expi')

colors = ['r', 'g', 'b']

ExpPlot = 5; # Number of individual survival curves to plot
for i, y in enumerate(f_survival[:ExpPlot]):
    plt.plot(time, y, '.', c=(np.random.rand(), np.random.rand()/10, np.random.rand()))

# Plot shaded envelopes for quantiles based on 1-3 standard deviations from mean
plt.fill_between(time, envelopes[0], envelopes[6], color=(0.6, 0.6, 0.6, 1), label='3σ')
plt.fill_between(time, envelopes[1], envelopes[5], color=(0.7, 0.7, 0.7, 1),  label='2σ')
plt.fill_between(time, envelopes[2], envelopes[4], color=(0.8, 0.8, 0.8, 1), label='1σ')

plt.xlabel('Time (hours)')
plt.ylabel('Survival (no crystal)')
plt.legend()
plt.show()
