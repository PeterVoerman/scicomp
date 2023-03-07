import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

def successive_over_relaxation(grid, cluster, omega = 1):
    counter = 0
    delta = 1

    N = len(grid)

    while delta > 1e-3 and delta < 1e5 and counter < 1e4:
        new_grid = grid.copy()
        new_grid[-1] = 1
        for y in range(1, N-1):
            new_grid[y][0] = 0.25 * omega * (grid[y + 1][0] + grid[y - 1][0] + grid[y][1] + grid[y][-1]) + (1 - omega) * grid[y][0] if (y,0) not in cluster else 0
            for x in range(1, N-1):
                new_grid[y][x] = (1 - omega) * grid[y][x] + omega * 0.25 * (grid[y + 1][x] + new_grid[y - 1][x] + grid[y][x + 1] + new_grid[y][x - 1]) if (y,x) not in cluster else 0
            new_grid[y][-1] = 0.25 * omega * (grid[y + 1][-1] + new_grid[y - 1][-1] + grid[y][-2] + new_grid[y][0]) + (1 - omega) * grid[y][-1] if (y,N-1) not in cluster else 0
        
        delta = max(abs(new_grid - grid).flatten())

        grid = new_grid.copy()

        counter += 1
    
    return grid, counter

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

def run_simulation(grid, cluster, eta, growth_steps=100, max_iterations=np.inf):
    total_counter = 0

    for t in range(growth_steps):
        print(f"t = {t}, total_counter = {total_counter}", end='\r')
        if total_counter > max_iterations:
            break

        grid, counter = successive_over_relaxation(grid, cluster)
        total_counter += counter

        probability_sum = 0
        probability_list = []

        growth_candidates = list(find_neighbors(cluster, len(grid)))
        for coord in growth_candidates:
            probability_sum += grid[coord[0]][coord[1]] ** eta
            probability_list.append(grid[coord[0]][coord[1]] ** eta)

        probability_list = [x / probability_sum for x in probability_list]

        new_coords = np.random.choice(range(len(growth_candidates)), 1, p=probability_list)[0]
        cluster.append(growth_candidates[new_coords])

    return grid, cluster

def make_eta_plot(eta):

    N = 100 
    growth_steps = 750
    eta = 1

    grid = np.zeros((N,N))
    grid = np.array([np.array([x/N for y in range(N)]) for x in range(N)])

    cluster = [(0, N // 2)]

    grid, cluster = run_simulation(grid, cluster, eta, growth_steps, max_iterations=1e4)
    print()

    for coord in cluster:
        grid[coord[0]][coord[1]] = None

    plt.imshow(grid, origin='lower', cmap='gist_rainbow')
    plt.savefig(f'eta_{eta}.png')
    plt.clf()

for eta in [0, 0.5, 1, 1.5, 2]:
    print(f"eta = {eta}")
    make_eta_plot(eta)



end = time.time()
print()
print(end - start)