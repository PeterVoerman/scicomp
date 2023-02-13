import numpy as np
import matplotlib.pyplot as plt

def jacobi_iteration():
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1


    grid_list = [np.zeros((N, N)), grid.copy()]

    while max(abs(grid_list[-1] - grid_list[-2]).flatten()) > 1e-6:
        grid[1:-1] = 0.25 * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))[1:-1]
        grid_list.append(grid.copy())

    return grid_list[-1]

def gauss_seidel():
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1

    grid_list = [np.zeros((N, N)), grid.copy()]

    while max(abs(grid_list[-1] - grid_list[-2]).flatten()) > 1e-6:
        new_grid = np.zeros((N, N))
        new_grid[-1] = 1
        for y in range(1, N-2):
            new_grid[0, y] = 0.25 * (grid[0, y + 1] + grid[0, y - 1] + grid[1, y] + grid[-1, y])
            for x in range(1, N-1):
                new_grid[x, y] = 0.25 * (grid[x, y + 1] + new_grid[x, y - 1] + grid[x + 1, y] + new_grid[x - 1, y])

        print(new_grid[0, 1])
        grid = new_grid.copy()
        grid_list.append(grid.copy())

        plt.imshow(grid, origin='lower')
        plt.show()
    return grid_list[-1]

plt.imshow(gauss_seidel(), origin='lower')
plt.show()