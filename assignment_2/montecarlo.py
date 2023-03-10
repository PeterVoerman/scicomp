import numpy as np
import matplotlib.pyplot as plt



def generate_step(x, y, N, cluster):
    direction = np.random.randint(0, 4)
    if direction == 0 and [x+1,y] not in cluster:
        x += 1
    elif direction == 1 and [x-1,y] not in cluster:
        x -= 1

    elif direction == 2 and [x,y+1] not in cluster:
        y += 1
        if y == N:
            return None, None
    elif direction == 3 and [x,y-1] not in cluster:
        y -= 1
        if y == -1:
            return None, None
        
    x %= N
        
    return x, y

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


def generate_tree(N, cluster, grid, growth_steps, prob, animation_gap=1e10):
    counter = 0

    while len(cluster) < growth_steps:
        print(len(cluster), end='\r')
        in_cluster = False

        neighbors = find_neighbors(cluster, N)
        x, y = np.random.randint(0, N), N-1

        while not in_cluster:
            x, y = generate_step(x, y, N, cluster)

            if not x:
                break

            grid[y][x] = 1


            if (x, y) in neighbors:
                if np.random.random() < prob:
                    in_cluster = True
                    cluster.append((x, y))

            # if counter % animation_gap == 0:
            #     plt.imshow(grid, origin='lower')
            #     plt.draw()
            #     plt.pause(0.001)
            #     plt.clf()

            grid[y][x] = 0
            counter += 1

        if in_cluster:
            grid[y][x] = 1

    return cluster
        
    
N = 100
growth_steps = 300
neighbors_list = []

prob_list = np.arange(0.05, 1 + 0.05, 0.05)

for prob in prob_list:
    grid = np.zeros((N,N))
    cluster = [(N//2, 0)]
    grid[0][N//2] = 1
    neighbors = 0


    print(prob)
    
    cluster = generate_tree(N, cluster, grid, growth_steps, prob)

    for point in cluster:
        grid[point[1]][point[0]] = 1
        if (point[0] + 1, point[1]) in cluster:
            neighbors += 1
        if (point[0] - 1, point[1]) in cluster:
            neighbors += 1
        if (point[0], point[1] + 1) in cluster:
            neighbors += 1
        if (point[0], point[1] - 1) in cluster:
            neighbors += 1
      

    # average neighbors for a cell in the cluster
    neighbors_list.append(neighbors/growth_steps)


    plt.imshow(grid, origin='lower')
    plt.savefig(f'montecarloplots/p_{prob:.2f}.png')
    plt.clf()

    

plt.plot(prob_list, neighbors_list)
plt.xlabel(r'$p_s$')
plt.ylabel('Average neighbors per cell')
plt.savefig(f'montecarloplots/neighbors.png')
plt.clf()

