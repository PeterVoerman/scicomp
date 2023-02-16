import numpy as np
import matplotlib.pyplot as plt

def jacobi_iteration():
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1


    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0
    while max(abs(grid_list[-1] - grid_list[-2]).flatten()) > 1e-4:
        print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')

        grid[1:-1] = 0.25 * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))[1:-1]
        grid_list.append(grid.copy())
        counter += 1
    print()
    print(counter)

    return grid_list[-1]

def gauss_seidel():
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1

    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0
    while max(abs(grid_list[-1] - grid_list[-2]).flatten()) > 1e-4:
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

        counter += 1
    print()
    print(counter)

    return grid_list[-1]

def successive_over_relaxation(omega):
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1

    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0
    while max(abs(grid_list[-1] - grid_list[-2]).flatten()) > 1e-4:
        print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')
        new_grid = np.zeros((N, N))
        new_grid[-1] = 1
        for y in range(1, N-1):
            new_grid[y][0] = 0.25 * (grid[y + 1][0] + grid[y - 1][0] + grid[y][1] + grid[y][-1]) + (1 - omega) * grid[y][0]
            for x in range(1, N-1):
                new_grid[y][x] = (1 - omega) * grid[y][x] + omega * 0.25 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1])
            new_grid[y][-1] = 0.25 * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][-2] + new_grid[y][0]) + (1 - omega) * grid[y][-1]

        grid = new_grid.copy()
        grid_list.append(grid.copy())

        counter += 1
    print()
    print(counter)

    return grid_list[-1]

jacobi_iteration()
gauss_seidel()
successive_over_relaxation(1.7)