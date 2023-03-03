import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pickle

def simulation(N, t_end, D, delta_t, run_simulation=True):
    if not run_simulation:
        with open("1.2.pkl", 'rb') as fp:
            return pickle.load(fp)

    delta_x = 1 / N

    grid = np.zeros((N, N))
    grid[-1] = 1

    constant = D * delta_t / delta_x ** 2

    grid_list = [grid.copy()]

    for i, t in enumerate(np.arange(0, t_end + delta_t, delta_t)):
        grid[1:-1] += constant * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid)[1:-1]
        if i % 100 == 0:
            grid_list.append(grid.copy())

        print(f"{t:.2f}/{t_end}", end='\r')

    grid_list.append(grid.copy())

    with open("1.2.pkl", 'wb') as fp:
        pickle.dump(grid_list, fp)

    return grid_list

def analytical(D, t, N):
    line = np.zeros(N)
    step_size = 1 / N

    for n in range(N):
        y = n * step_size
        for i in range(10000):
            line[n] += erfc((1 - y + 2 * i) / (2 * (D * t) ** 0.5)) - erfc((1 + y + 2 * i) / (2 * (D * t) ** 0.5))

    return line


delta_t = 0.00001
animation_frames = 100

grid_list = simulation(100, 1, 1, delta_t, run_simulation=False)

for t in [1e-20, 0.001, 0.01, 0.1, 1]:
    analytical_list = analytical(1, t, 100)
    plt.plot(np.linspace(0, 1, 100), grid_list[int(t/(animation_frames*delta_t))+1][:,0], label=f"Simulation t={t:.3f}")
    plt.plot(np.linspace(0, 1, 100), analytical_list, label=f"Analytical t={t:.3f}", linestyle='--')
plt.legend()
plt.xlabel("y")
plt.ylabel("Concentration")
plt.savefig("experimental_analytical.png")
plt.clf()

# print(len(grid_list))

# for t in [0, 0.001, 0.01, 0.1, 1]:
#     plt.imshow(grid_list[int(t/(animation_frames*delta_t))], origin='lower')
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.savefig(f"animation_{t}.png")
