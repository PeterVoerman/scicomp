import numpy as np
import matplotlib.pyplot as plt
import time
# plt.rcParams.update({'font.size': 22})


start = time.time()

def successive_over_relaxation(grid, cluster, t, omega = 1, delta_precision = 1e-4):
    counter = 0
    delta = 1
    max_counter = 1e4



    N = len(grid)

    while delta > delta_precision and delta < 1e5 and counter < max_counter:
        # print(f"counter = {counter}, delta = {delta:.2E}     ", end='\r')s
        new_grid = grid.copy()
        new_grid[-1] = 1

        # for y in range(1, N-1):
        #     new_grid[y][0] = 0.25 * omega * (grid[y + 1][0] + grid[y - 1][0] + grid[y][1] + grid[y][-1]) + (1 - omega) * grid[y][0] if (y,0) not in cluster else 0
        #     for x in range(1, N-1):
        #         new_grid[y][x] = (1 - omega) * grid[y][x] + omega * 0.25 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1]) if (y,x) not in cluster else 0
        #     new_grid[y][-1] = 0.25 * omega * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][-2] + new_grid[y][0]) + (1 - omega) * grid[y][-1] if (y,N-1) not in cluster else 0
        

        for y in range(1, N - 1):
            new_grid[y][0] = omega / 4 * (grid[y + 1][0] + new_grid[y - 1][0] + grid[y][1] + grid[y][-2]) + (1 - omega) * grid[y][0] if (y,0) not in cluster else 0
            for x in range(1, N - 1):
                new_grid[y][x] = omega / 4 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1]) + (1 - omega) * grid[y][x] if (y,x) not in cluster else 0           
            new_grid[y][-1] = omega / 4 * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][1] + new_grid[y][-2]) + (1 - omega) * grid[y][-1] if (y,N-1) not in cluster else 0
 
        
        delta = max(abs(new_grid - grid).flatten())

        grid = new_grid.copy()

        counter += 1

    diverge = False
    if delta >= 1e5:
        diverge = True

    
    return grid, counter, diverge

def find_neighbors(cluster, N):
    neighbors = set([])

    for coord in cluster:
        if coord[0] != 0:
            neighbors.add((coord[0]-1,coord[1]))
        if coord[0] != N-1:
            neighbors.add((coord[0]+1,coord[1]))
        if coord[1] != 0:
            neighbors.add((coord[0],coord[1]-1))
        if coord[1] != N-1:
            neighbors.add((coord[0],coord[1]+1))

    for coord in cluster:
        if coord in neighbors:
            neighbors.remove(coord)

    return neighbors

def run_simulation(grid, cluster, eta, omega = 1, growth_steps=100, delta_precision = 1e-4, max_iterations=np.inf):
    total_counter = 0
    return_after_plot = False



    for t in range(growth_steps):
        print(f"t = {t}, total_counter = {total_counter}", end='\r')
        if total_counter > max_iterations:
            break

        grid, counter, diverge = successive_over_relaxation(grid, cluster, t, omega, delta_precision)
        total_counter += counter

        if diverge:
            break

        probability_sum = 0
        probability_list = []

        growth_candidates = list(find_neighbors(cluster, len(grid)))
        for coord in growth_candidates:
            probability_sum += grid[coord[0]][coord[1]] ** eta
            probability_list.append(grid[coord[0]][coord[1]] ** eta)

        probability_list = [x / probability_sum for x in probability_list]
        negative = False
        for i in range(len(probability_list)):
            if probability_list[i] < 0:
                negative = True
                probability_list[i] = 0
        
        # correct for values dipping below zero ever so slightly
        if negative:
            probsum = np.sum(probability_list)
            probability_list = [x / probsum for x in probability_list]

        new_coords = np.random.choice(range(len(growth_candidates)), 1, p=probability_list)[0]
        cluster.append(growth_candidates[new_coords])
        
        # make image
        if (t + 1) % 10 == 0:
            plotgrid = grid.copy()
            for coord in cluster:
                plotgrid[coord[0]][coord[1]] = None
                
                # check whether top has been reached
                if coord[0] == 99:
                    return_after_plot = True

            plt.imshow(plotgrid, origin='lower', cmap='gist_rainbow')
            filename_t = t + 1
            if filename_t < 100:
                filename_t = '0' + str(filename_t)
            plt.savefig(f'plots/DLA/eta_{eta}/eta_{eta}_t_{filename_t}.png')
            plt.clf()

            # return after top has been reached
            if return_after_plot:
                return grid, cluster

    return grid, cluster, total_counter, diverge


def make_eta_plot(eta, growth_steps, omega = 1, max_it=np.inf):

    N = 100 
    delta_precision = 1e-5
    delta_precision_pkl = 1e-5

    grid = np.zeros((N,N))
    grid = np.array([np.array([x/N for y in range(N)]) for x in range(N)])

    cluster = [(0, N // 2)]

    grid, cluster, total_iterations, diverge = run_simulation(grid, cluster, eta, omega, growth_steps, delta_precision, max_iterations=max_it)
    print()

    return total_iterations, diverge

    # for coord in cluster:
    #     grid[coord[0]][coord[1]] = None

    # plt.imshow(grid, origin='lower', cmap='gist_rainbow')
    # plt.savefig(f'plots/eta_{eta}_t_{growth_steps}.png')
    # plt.clf()





# eta_list = [0.5, 1, 1.5, 2, 2.5]
# growth_steps = 750
# for eta in eta_list:
#     print(f"eta = {eta}")
#     a = make_eta_plot(eta, growth_steps)


# optimizing omega
iterations = []
omega_list = np.arange(1,2.05,0.05)
omega_list_plot = []

for omega in omega_list:
    print(f'omega: {omega:.2f}')
    iters, diverge = make_eta_plot(eta=1,growth_steps=50, omega=omega)
    if not diverge:
        iterations.append(iters)
        omega_list_plot.append(omega)
    else:
        print(f'--------------------diverged')


plt.plot(omega_list_plot, iterations)
plt.xlabel(r'$\omega$')
plt.ylabel('iterations')
plt.savefig("DLA_omega_comparison_lower_delta.png")

end = time.time()
print()
print(f'total time: {end - start:.2f}')