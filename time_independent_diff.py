import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

def jacobi_iteration():
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1


    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0

    delta = 1
    delta_list = []

    while delta > 1e-5:
        print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')

        grid[1:-1] = 0.25 * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))[1:-1]
        grid_list.append(grid.copy())
        counter += 1

        delta = max(abs(grid_list[-1] - grid_list[-2]).flatten())
        delta_list.append(delta)

    print()
    print(counter)

    return delta_list

def gauss_seidel():
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1

    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0

    delta = 1
    delta_list = []

    while delta > 1e-5:
        print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')
        new_grid = np.zeros((N, N))
        new_grid[-1] = 1
        for y in range(1, N-1):
            new_grid[y][0] = 0.25 * (grid[y + 1][0] + grid[y - 1][0] + grid[y][1] + grid[y][-1])
            for x in range(1, N-1):
                new_grid[y][x] = 0.25 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1])
            new_grid[y][-1] = 0.25 * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][-2] + new_grid[y][0])

        grid = new_grid.copy()
        grid_list.append(grid.copy())

        delta = max(abs(grid_list[-1] - grid_list[-2]).flatten())
        delta_list.append(delta)

        counter += 1
    print()
    print(counter)

    return delta_list

def successive_over_relaxation(omega, N=100):
    grid = np.zeros((N, N))
    grid[-1] = 1

    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0

    delta = 1
    delta_list = []

    while delta > 1e-5 and delta < 1e5 and counter < 1e4:
        # print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')
        new_grid = np.zeros((N, N))
        new_grid[-1] = 1
        for y in range(1, N-1):
            new_grid[y][0] = 0.25 * (grid[y + 1][0] + grid[y - 1][0] + grid[y][1] + grid[y][-1]) + (1 - omega) * grid[y][0]
            for x in range(1, N-1):
                new_grid[y][x] = (1 - omega) * grid[y][x] + omega * 0.25 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1])
            new_grid[y][-1] = 0.25 * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][-2] + new_grid[y][0]) + (1 - omega) * grid[y][-1]

        grid = new_grid.copy()
        grid_list.append(grid.copy())

        delta = max(abs(grid_list[-1] - grid_list[-2]).flatten())
        delta_list.append(delta)

        counter += 1
    # print()
    # print(counter)
    # print(delta)

    return delta_list

def sink_simulation(x1, x2, y1, y2):
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1


    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0
    while max(abs(grid_list[-1] - grid_list[-2]).flatten()) > 1e-4 and counter < 10000:
        grid[y1:y2, x1:x2] = 0
        print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')

        grid[1:-1] = 0.25 * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))[1:-1]
        grid_list.append(grid.copy())
        counter += 1
    print()
    print(counter)

    return grid_list[-1]


# jacobi_list = jacobi_iteration()
# gauss_list = gauss_seidel()
# plt.plot(jacobi_list, label='Jacobi')
# plt.plot(gauss_list, label='Gauss-Seidel')
# for omega in np.arange(1.4, 2, 0.1):
#     print(omega)
#     sor_list = successive_over_relaxation(omega)
#     plt.plot(sor_list, label=f'SOR {omega:.1f}')


# plt.legend()
# plt.yscale('log')
# plt.xlabel("Iteration")
# plt.ylabel("Delta")
# plt.tight_layout()
# plt.savefig("jacobi_gauss_sor.png")


min_iterations = 1e5
for N in [25, 50, 75, 100]:
    print(f"N = {N}")
    for omega in np.arange(1.6, 2, 0.01):
        print(f"omega = {omega}", end='\r')
        delta_list = successive_over_relaxation(omega, N)
        iterations = len(delta_list)
        if iterations < min_iterations and delta_list[-1] < 1:
            min_iterations = iterations
            min_omega = omega

    print(f"min_omega = {min_omega}, min_iterations = {min_iterations}")
