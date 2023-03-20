import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

def jacobi_iteration(delta_min):
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1


    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0

    delta = 1
    delta_list = []

    while delta > delta_min:
        print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')

        grid[1:-1] = 0.25 * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))[1:-1]
        grid_list.append(grid.copy())
        counter += 1

        delta = max(abs(grid_list[-1] - grid_list[-2]).flatten())
        delta_list.append(delta)

    print()
    print(counter)

    return delta_list

def gauss_seidel(delta_min):
    N = 100

    grid = np.zeros((N, N))
    grid[-1] = 1

    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0

    delta = 1
    delta_list = []

    while delta > delta_min:
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

def sink_check(sinks, x, y):
    if sinks == []:
        return False
    
    for sink in sinks:
        if x >= sink[0][0] and x <= sink[0][1] and y >= sink[1][0] and y <= sink[1][1]:
            return True
    return False

def successive_over_relaxation(omega, delta_min, N=100, sinks = [], plot_grid=False):
    grid = np.zeros((N, N))
    grid[-1] = 1

    grid_list = [np.zeros((N, N)), grid.copy()]
    counter = 0

    delta = 1
    delta_list = []

    while delta > delta_min and delta < 1e2 and counter < 1e3:
        print(f'{delta:.7f}', end='\r')
        # print(f"{max(abs(grid_list[-1] - grid_list[-2]).flatten()):.7f}", end='\r')
        new_grid = np.zeros((N, N))
        new_grid[-1] = 1
        # dit jaar
        # for y in range(1, N-1):
        #     new_grid[y][0] = 0.25 * omega * (grid[y + 1][0] + grid[y - 1][0] + grid[y][1] + grid[y][-1]) + (1 - omega) * grid[y][0] if sink_check(sinks, 0, y) == False else 0
            
        #     for x in range(1, N-1):
        #         new_grid[y][x] = omega * 0.25 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1]) + (1 - omega) * grid[y][x] if sink_check(sinks, x, y) == False else 0
            
        #     new_grid[y][-1] = 0.25 * omega * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][-2] + new_grid[y][0]) + (1 - omega) * grid[y][-1] if sink_check(sinks, x + 1, y) == False else 0
        
        # vorig jaar
        for y in range(1, N - 1):
            new_grid[y][0] = omega / 4 * (grid[y + 1][0] + new_grid[y - 1][0] + grid[y][1] + grid[y][-2]) + (1 - omega) * grid[y][0] if sink_check(sinks, 0, y) == False else 0

            for x in range(1, N - 1):
                new_grid[y][x] = omega / 4 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1]) + (1 - omega) * grid[y][x] if sink_check(sinks, x, y) == False else 0

            
            new_grid[y][-1] = omega / 4 * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][1] + new_grid[y][-2]) + (1 - omega) * grid[y][-1] if sink_check(sinks, x + 1, y) == False else 0
 

        grid = new_grid.copy()
        grid_list.append(grid.copy())

        delta = max(abs(grid_list[-1] - grid_list[-2]).flatten())
        delta_list.append(delta)

        counter += 1

    if plot_grid:
        return grid_list[-1]
    return delta_list


delta_min = 1e-4
# jacobi_list = jacobi_iteration(delta_min)
# gauss_list = gauss_seidel(delta_min)
# plt.plot(jacobi_list, label='Jacobi')
# plt.plot(gauss_list, label='Gauss-Seidel')

best = [0 for i in range(1000)]
best_omega = 0
for omega in np.arange(1.9, 2, 0.01):
    print(omega)
    sor_list = successive_over_relaxation(omega, delta_min)
    
    if len(sor_list) < len(best) and sor_list[-1] < 1:
        best = sor_list
        best_omega = omega
    plt.plot(sor_list, label=f'SOR {omega:.2f}')

print(f'omega: {best_omega:.2f}, iterations: {len(best)}')


plt.legend()
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Delta")
plt.tight_layout()
plt.savefig("jacobi_gauss_sor.png")

quit()

min_iterations = 1e5
for N in [25, 50, 75, 100]:
    print(f"N = {N}")
    for omega in np.arange(1.4, 1.6, 0.01):
        print(f"omega = {omega}", end='\r')
        delta_list = successive_over_relaxation(omega, delta_min=1e-4, N=N)
        iterations = len(delta_list)
        if iterations < min_iterations and delta_list[-1] < 1:
            min_iterations = iterations
            min_omega = omega

    print(f"min_omega = {min_omega}, min_iterations = {min_iterations}")

# for this to work, SOR function must return the final grid.
sinks = [[(20,40),(70,80)], [(70, 80), (80, 96)]] # a sink = [(x1, x2), (y1, y2)]
grid = successive_over_relaxation(1.5, 1e-5, 100, sinks, True)
plt.imshow(grid, origin='lower', cmap='magma')
plt.savefig("sinks.png")
plt.clf()


print(omega_list)
for omega in np.arange(1.1, 1.9, 0.1):
    print(omega)
    sor_list = successive_over_relaxation(omega, delta_min, 100, sinks)
    plt.plot(sor_list, label=f'SOR {omega:.1f}')


plt.legend()
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Delta")
plt.tight_layout()
plt.savefig("jacobi_gauss_sor_sinks.png")