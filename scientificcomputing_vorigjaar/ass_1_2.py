# Scientific computing Exercise 1.2
# Peter Voerman and Nik brouw

# The results of one of the simulations are stored in "1.2.pkl"
# Change the variable below to True in order to run the entire simulation again

run_again = False

import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

N = 100
t_end = 1

D = 1
delta_t = 0.00001
delta_x = 1 / N

constant = D * delta_t / delta_x ** 2

grid = np.zeros((N, N))

output = "1.2.pkl"

for i in range(N):
    grid[0][i] = 1

grid_list = [grid]

def calculate_diffusion():

    counter = 0
    for t in np.arange(0, t_end, delta_t):
        counter += 1
        print(f"{t:.2f}/{t_end}", end='\r')

        new_grid = np.zeros((N, N))
        for i in range(N):
            new_grid[0][i] = 1

        for y in range(1, N - 1):
            for x in range(1, N - 1):
                between_brackets = grid[y][x+1] + grid[y][x-1] + grid[y-1][x] + grid[y+1][x] - 4 * grid[y][x]

                new_grid[y][x] = grid[y][x] + constant * between_brackets

            new_grid[y][0] = grid[y][x] + constant * (grid[y][1] + grid[y][-2] + grid[y-1][0] + grid[y+1][0] - 4 * grid[y][0])
            new_grid[y][-1] = grid[y][x] + constant * (grid[y][1] + grid[y][-2] + grid[y-1][-1] + grid[y+1][-1] - 4 * grid[y][-1])

        grid = new_grid

        if counter % 100 == 0:
            grid_list.append(grid)

    
    with open(output, 'wb') as fp:
        pickle.dump(grid_list, fp)

    return grid_list

def analytical_solution(t, D = 1):
    solution_list = []

    for y in np.arange(0, 1.05, 0.05):
        solution = 0

        for i in range(100000):
            solution += math.erfc((1 - y + 2 * i) / (2 * math.sqrt(D * t))) - math.erfc((1 + y + 2 * i) / (2 * math.sqrt(D * t)))

        solution_list.append(solution)

    return solution_list

if not run_again:
    with open(output, 'rb') as fp:
        grid_list = pickle.load(fp)
else:
    grid_list = calculate_diffusion()

times = [1, 50, 150, 500, 1000]

for time in times:
    analytical = analytical_solution(time / 1000)

    experimental = []
    for row in grid_list[time]:
        experimental.insert(0, row[50])

    plt.plot(range(0, 105, 5), analytical, label=f"Analytical, t={time}")
    plt.plot(range(0, 100), experimental, label=f"Experimental, t={time}")
plt.title("Simulation compared to analytical solution of the diffusion equation")
plt.legend()
plt.show()

times = [0, 1, 10, 100, 1000]

for time in times:
    plt.imshow(grid_list[time])
    plt.title(f"t={time/1000}")
    plt.show()

for grid in grid_list:
    plt.imshow(grid)
    plt.draw()
    plt.pause(0.001)
    plt.clf()