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

def plot(grid_list, analytical_list, t, N):
    plt.plot(np.linspace(0, 1, N), grid_list[int(t/100)][:,0], label="Simulation")
    plt.plot(np.linspace(0, 1, 25), analytical_list, label="Analytical")
    plt.legend()
    plt.savefig(f"1.2_{t*delta_t:.3f}.png")
    plt.clf()

delta_t = 0.00001

grid_list = simulation(100, 1, 1, delta_t, run_simulation=False)

# for t in [1e-20, 0.001 / delta_t, 0.01 / delta_t, 0.1 / delta_t, 1 / delta_t]:
#     plot(grid_list, analytical(1, t * delta_t, 25), t, 100)

i = 1
print(len(grid_list))

while np.sum(abs(grid_list[i] - grid_list[i-1])) > 1e-6:
    plt.imshow(grid_list[i], origin='lower')
    plt.draw()
    plt.pause(0.001)
    plt.clf()
    i += 1
