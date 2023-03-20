import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import sparse

from matplotlib import colors

import time

def generate_matrix(grid):
    matrix = np.zeros((np.count_nonzero(grid) + 1, np.count_nonzero(grid) + 1))

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y, x] != None:
                neighbor_count = 0
                element = grid[y, x]

                if x - 1 >= 0:
                    matrix[element, grid[y, x - 1]] = 1
                    neighbor_count += 1
                if x + 1 < len(grid):
                    matrix[element, grid[y, x + 1]] = 1
                    neighbor_count += 1
                if y - 1 >= 0:
                    matrix[element, grid[y - 1, x]] = 1
                    neighbor_count += 1
                if y + 1 < len(grid):
                    matrix[element, grid[y + 1, x]] = 1
                    neighbor_count += 1

                matrix[element, element] = -neighbor_count

    return matrix

def generate_sparse_matrix(grid, mask=None):
    row = []
    col = []
    data = []

    if mask is None:
        mask = np.ones(grid.shape)

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            element = grid[y, x]
            neighbor_count = 0
            if mask[y][x] == 1:
                if x - 1 >= 0 and mask[y][x - 1] == 1:
                    row.append(element)
                    col.append(grid[y, x - 1])
                    data.append(1)
                    neighbor_count += 1
                if x + 1 < len(grid) and mask[y][x + 1] == 1:
                    row.append(element)
                    col.append(grid[y, x + 1])
                    data.append(1)
                    neighbor_count += 1
                if y - 1 >= 0 and mask[y - 1][x] == 1:
                    row.append(element)
                    col.append(grid[y - 1, x])
                    data.append(1)
                    neighbor_count += 1
                if y + 1 < len(grid) and mask[y + 1][x] == 1:
                    row.append(element)
                    col.append(grid[y + 1, x])
                    data.append(1)
                    neighbor_count += 1

            row.append(element)
            col.append(element)
            data.append(-neighbor_count)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data, dtype=np.float64)

    return sparse.csr_matrix((data, (row, col)))
    

# Square
L = 50
grid = np.arange(0, L ** 2).reshape(L, L)

# start_non_sparse = time.time()
# matrix = generate_matrix(grid)
# result = linalg.eig(matrix)
# end_non_sparse = time.time()

# print("Non-sparse time: ", end_non_sparse - start_non_sparse)

# start_sparse = time.time()
# matrix = generate_sparse_matrix(grid)
# result = sparse.linalg.eigs(matrix)
# end_sparse = time.time()

# print("Sparse time: ", end_sparse - start_sparse)

# plt.imshow(np.abs(result[1][:,0]).reshape(L, L))
# plt.show()


# Circle
L = 50
grid = np.arange(0, L ** 2).reshape(L, L)

mask = np.zeros((L, L))
for y in range(L):
    for x in range(L):
        if (x - L / 2) ** 2 + (y - L / 2) ** 2 <= (L / 2) ** 2:
            mask[y, x] = 1


matrix = generate_sparse_matrix(grid, mask=mask)
result = sparse.linalg.eigs(matrix)

# plt.imshow(np.abs(result[1][:,5]).reshape(L, L))
# plt.show()

index = 1
eigenvector = np.abs(result[1][:, index].reshape(L, L))
eigenvalue = np.abs(result[0][index])

# print(np.amin(eigenvector), np.amax(eigenvector))

divnorm=colors.TwoSlopeNorm(vmin=-np.amax(eigenvector.copy()), vcenter=0., vmax=np.amax(eigenvector.copy()))

for t in np.arange(0, 10, 0.05):
    u = eigenvector * (np.cos(eigenvalue ** 0.5 * t) + np.sin(eigenvalue ** 0.5 * t))
    plt.imshow(u, animated=True, norm=divnorm)
    plt.draw()
    plt.pause(0.01)
    plt.clf()

