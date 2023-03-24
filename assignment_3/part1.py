import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import sparse
import time

from matplotlib import colors
import matplotlib

matplotlib.rcParams['font.size'] = 16

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
                if x + 1 < len(grid[0]) and mask[y][x + 1] == 1:
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
L = 100
grid = np.arange(0, L ** 2).reshape(L, L)
matrix = generate_sparse_matrix(grid)
result = sparse.linalg.eigs(matrix)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(np.abs(result[1][:, 0]).reshape(L, L))
ax[0, 1].imshow(np.abs(result[1][:, 1]).reshape(L, L))
ax[1, 0].imshow(np.abs(result[1][:, 2]).reshape(L, L))
ax[1, 1].imshow(np.abs(result[1][:, 3]).reshape(L, L))
ax[0, 0].set_title(f"Eigenvalue: {np.real(result[0][0]):.4f}")
ax[0, 1].set_title(f"Eigenvalue: {np.real(result[0][1]):.4f}")
ax[1, 0].set_title(f"Eigenvalue: {np.real(result[0][2]):.4f}")
ax[1, 1].set_title(f"Eigenvalue: {np.real(result[0][3]):.4f}")
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
plt.tight_layout()
plt.savefig("plots/square.png")
plt.clf()



# Circle
L = 100
grid = np.arange(0, L ** 2).reshape(L, L)

mask = np.zeros((L, L))
for y in range(L):
    for x in range(L):
        if (x - L / 2) ** 2 + (y - L / 2) ** 2 <= (L / 2) ** 2:
            mask[y, x] = 1


matrix = generate_sparse_matrix(grid, mask=mask)
result = sparse.linalg.eigs(matrix)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(np.abs(result[1][:, 0]).reshape(L, L))
ax[0, 1].imshow(np.abs(result[1][:, 1]).reshape(L, L))
ax[1, 0].imshow(np.abs(result[1][:, 2]).reshape(L, L))
ax[1, 1].imshow(np.abs(result[1][:, 3]).reshape(L, L))
ax[0, 0].set_title(f"Eigenvalue: {np.real(result[0][0]):.4f}")
ax[0, 1].set_title(f"Eigenvalue: {np.real(result[0][1]):.4f}")
ax[1, 0].set_title(f"Eigenvalue: {np.real(result[0][2]):.4f}")
ax[1, 1].set_title(f"Eigenvalue: {np.real(result[0][3]):.4f}")
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
plt.tight_layout()
plt.savefig("plots/circle.png")
plt.clf()


# Rectangle
L1 = 75
L2 = 150
grid = np.arange(0, L1 * L2).reshape(L1, L2)
matrix = generate_sparse_matrix(grid)
result = sparse.linalg.eigs(matrix)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(np.abs(result[1][:, 0]).reshape(L1, L2))
ax[0, 1].imshow(np.abs(result[1][:, 1]).reshape(L1, L2))
ax[1, 0].imshow(np.abs(result[1][:, 2]).reshape(L1, L2))
ax[1, 1].imshow(np.abs(result[1][:, 3]).reshape(L1, L2))
ax[0, 0].set_title(f"Eigenvalue: {np.real(result[0][0]):.4f}")
ax[0, 1].set_title(f"Eigenvalue: {np.real(result[0][1]):.4f}")
ax[1, 0].set_title(f"Eigenvalue: {np.real(result[0][2]):.4f}")
ax[1, 1].set_title(f"Eigenvalue: {np.real(result[0][3]):.4f}")
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
plt.tight_layout()
plt.savefig("plots/rectangle.png")
plt.clf()


# Time plot
non_sparse_time_list = []
sparse_time_list = []
x_list_non_sparse = []

for L in range(5, 105, 5):
    print(f"L = {L}", end="\r")

    grid = np.arange(0, L ** 2).reshape(L, L)

    if L <= 60:
        start_non_sparse = time.time()
        matrix = generate_matrix(grid)
        result = linalg.eig(matrix)
        end_non_sparse = time.time()
        x_list_non_sparse.append(L)
        non_sparse_time_list.append(end_non_sparse - start_non_sparse)

    start_sparse = time.time()
    matrix = generate_sparse_matrix(grid)
    result = sparse.linalg.eigs(matrix)
    end_sparse = time.time()

    
    sparse_time_list.append(end_sparse - start_sparse)

plt.plot(x_list_non_sparse, non_sparse_time_list, label="Non-sparse")
plt.plot(range(5, 105, 5), sparse_time_list, label="Sparse")
plt.legend()
plt.xlabel("L")
plt.ylabel("Time (s)")
plt.xlim(5, 100)
plt.ylim(0, max(non_sparse_time_list))
plt.tight_layout()
plt.savefig("plots/time.png")
plt.clf()


# Eigenfrequency plot
square_list = []
circle_list = []
rectangle_list = []
L_list = []

for L in range(10, 105, 5):
    L_list.append(L)

    print(f"L = {L}", end="\r")

    grid = np.arange(0, L ** 2).reshape(L, L)
    matrix = generate_sparse_matrix(grid)
    result = sparse.linalg.eigs(matrix)

    square_list.append(np.real(result[0]))

    grid = np.arange(0, L ** 2).reshape(L, L)
    mask = np.zeros((L, L))
    for y in range(L):
        for x in range(L):
            if (x - L / 2) ** 2 + (y - L / 2) ** 2 <= (L / 2) ** 2:
                mask[y, x] = 1
    
    matrix = generate_sparse_matrix(grid, mask)
    result = sparse.linalg.eigs(matrix)

    circle_list.append(np.real(result[0]))

    L1 = L
    L2 = 2 * L
    grid = np.arange(0, L1 * L2).reshape(L1, L2)
    matrix = generate_sparse_matrix(grid)
    result = sparse.linalg.eigs(matrix)

    rectangle_list.append(np.real(result[0]))

counter = 0
for freq in square_list:
    plt.scatter([L_list[counter]] * len(freq), freq, s=2, c="b")
    counter += 1

plt.xticks(range(0, 105, 20))
plt.xlabel("L")
plt.ylabel("Eigenfrequency")
plt.savefig("plots/square_freqs.png")
plt.clf()

counter = 0
for freq in circle_list:
    plt.scatter([L_list[counter]] * len(freq), freq, s=2, c="b")
    counter += 1

plt.xticks(range(0, 105, 20))
plt.xlabel("L")
plt.ylabel("Eigenfrequency")
plt.savefig("plots/circle_freqs.png")
plt.clf()

counter = 0
for freq in rectangle_list:
    plt.scatter([L_list[counter]] * len(freq), freq, s=2, c="b")
    counter += 1

plt.xticks(range(0, 105, 20))
plt.xlabel("L")
plt.ylabel("Eigenfrequency")
plt.savefig("plots/rectangle_freqs.png")
plt.clf()


L = 100
grid = np.arange(0, L ** 2).reshape(L, L)

mask = np.zeros((L, L))
for y in range(L):
    for x in range(L):
        if (x - L / 2) ** 2 + (y - L / 2) ** 2 <= (L / 2) ** 2:
            mask[y, x] = 1


matrix = generate_sparse_matrix(grid, mask=mask)
result = sparse.linalg.eigs(matrix)

index = 0
eigenvector = np.abs(result[1][:, index].reshape(L, L))
eigenvalue = np.abs(result[0][index])
divnorm=colors.TwoSlopeNorm(vmin=-np.amax(eigenvector.copy()), vcenter=0., vmax=np.amax(eigenvector.copy()))

for t in np.arange(0, 5, 0.05):
    u = eigenvector * (np.cos(eigenvalue ** 0.5 * t) + np.sin(eigenvalue ** 0.5 * t))
    plt.imshow(u, animated=True, norm=divnorm)
    plt.xticks([])
    plt.yticks([])
    plt.draw()
    plt.pause(0.01)
    plt.clf()

index = 1
eigenvector = np.abs(result[1][:, index].reshape(L, L))
eigenvalue = np.abs(result[0][index])
divnorm=colors.TwoSlopeNorm(vmin=-np.amax(eigenvector.copy()), vcenter=0., vmax=np.amax(eigenvector.copy()))

counter = 0
vector_list = []
t_list = []
for t in np.arange(0, 5, 0.05):
    u = eigenvector * (np.cos(eigenvalue ** 0.5 * t) + np.sin(eigenvalue ** 0.5 * t))
    plt.imshow(u, animated=True, norm=divnorm)
    plt.xticks([])
    plt.yticks([])
    
    if counter in [6, 17, 28, 39, 50]:
        vector_list.append(u)
        t_list.append(t)

    counter += 1
    plt.draw()
    plt.pause(0.01)
    plt.clf()

matplotlib.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    axs[i].imshow(vector_list[i], norm=divnorm)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title(f"t = {t_list[i]:.2f}")
plt.tight_layout()
plt.savefig("plots/animation.png")
plt.clf()

index = 5
eigenvector = np.abs(result[1][:, index].reshape(L, L))
eigenvalue = np.abs(result[0][index])
divnorm=colors.TwoSlopeNorm(vmin=-np.amax(eigenvector.copy()), vcenter=0., vmax=np.amax(eigenvector.copy()))

for t in np.arange(0, 5, 0.05):
    u = eigenvector * (np.cos(eigenvalue ** 0.5 * t) + np.sin(eigenvalue ** 0.5 * t))
    plt.imshow(u, animated=True, norm=divnorm)
    plt.xticks([])
    plt.yticks([])
    plt.draw()
    plt.pause(0.01)
    plt.clf()

