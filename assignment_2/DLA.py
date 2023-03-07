import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

start = time.time()

def successive_over_relaxation(grid, cluster, t, omega = 1, delta_precision = 1e-4, make_pickle = False, pickle_name = ''):
    counter = 0
    delta = 1
    max_counter = 1e4

    if make_pickle:
        max_counter = 1e6

    N = len(grid)

    while delta > delta_precision and delta < 1e5 and counter < max_counter:
        # print(f"counter = {counter}, delta = {delta:.2E}     ", end='\r')s
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

    if make_pickle:
        with open(f'N{N}delta{delta_precision}.pkl', 'wb') as save_file:
            pickle.dump(grid, save_file)
    
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

def run_simulation(grid, cluster, eta, omega = 1, growth_steps=100, delta_precision = 1e-4, max_iterations=np.inf):
    total_counter = 0

    for t in range(growth_steps):
        print(f"t = {t}, total_counter = {total_counter}", end='\r')
        if total_counter > max_iterations:
            break

        grid, counter = successive_over_relaxation(grid, cluster, t, omega, delta_precision)
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

def make_pickle(N, delta_precision):
    grid = np.zeros((N,N))
    grid, counter = successive_over_relaxation(grid, [], 0, omega = 1, delta_precision = delta_precision, make_pickle = True, pickle_name = f'N{N}')
    return

def make_eta_plot(eta):

    N = 100 
    growth_steps = 100
    # eta = 1
    omega = 1
    delta_precision = 1e-5
    delta_precision_pkl = 1e-5

    # grid = np.zeros((N,N))
    # grid = np.array([np.array([x/N for y in range(N)]) for x in range(N)])

    with open(f"N{N}delta{delta_precision_pkl}.pkl", 'rb') as save_file:
        grid = pickle.load(save_file)

    

    cluster = [(0, N // 2)]

    grid, cluster = run_simulation(grid, cluster, eta, omega, growth_steps, delta_precision)
    print()

    for coord in cluster:
        grid[coord[0]][coord[1]] = None

    plt.imshow(grid, origin='lower', cmap='gist_rainbow')
    plt.savefig(f'eta_{eta}_t_{growth_steps}.png')
    plt.clf()


# make starting grid and save in pickle file
# make_pickle(100, 1e-5)
# quit()


# plot starting grid that is loaded via pickle to check
# N=100
# delta_precision = 1e-5
# with open(f"N{N}delta{delta_precision}.pkl", 'rb') as save_file:
#     grid = pickle.load(save_file)

# plt.imshow(grid, origin='lower', cmap='gist_rainbow')
# plt.show()
# quit()

for eta in [0, 0.5, 1, 1.5, 2, 2.5]:
    print(f"eta = {eta}")
    make_eta_plot(eta)



end = time.time()
print()
print(end - start)