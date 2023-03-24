import numpy as np
import matplotlib.pyplot as plt

def generate_matrix(grid, domain=[]):
    n_elements = len(grid) * len(grid[0])
    matrix = np.zeros((n_elements, n_elements))

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y, x] != None:
                neighbor_count = 0
                element = grid[y, x]

                if x - 1 >= 0 and grid[y, x - 1] in domain:
                    matrix[element, grid[y, x - 1]] = 1
                    neighbor_count += 1
                if x + 1 < len(grid) and grid[y, x + 1] in domain:
                    matrix[element, grid[y, x + 1]] = 1
                    neighbor_count += 1
                if y - 1 >= 0 and grid[y - 1, x] in domain:
                    matrix[element, grid[y - 1, x]] = 1
                    neighbor_count += 1
                if y + 1 < len(grid) and grid[y + 1, x] in domain:
                    matrix[element, grid[y + 1, x]] = 1
                    neighbor_count += 1

                matrix[element, element] = -neighbor_count

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            element = grid[y][x]
            if element not in domain and domain != []:
                matrix[element] = 0

    np.fill_diagonal(matrix, 4)
    return matrix



def circle_mask(x_length, y_length):
    mask = np.zeros((x_length, y_length))
    for y in range(y_length):
        for x in range(x_length):
            if (x - x_length / 2) ** 2 + (y - y_length / 2) ** 2 <= (x_length / 2) ** 2:
                mask[y, x] = 1
    return mask

x_length = 100
y_length = 100

mask = circle_mask(x_length, y_length)
grid = np.zeros((y_length, x_length), dtype=int)

index = 0
domain = []
for y in range(y_length):
    for x in range(x_length):
        if mask[y][x]:
            domain.append(index)
        grid[y][x] = index
        index += 1

matrix = generate_matrix(grid, domain)

n_elements = len(grid) * len(grid[0])

# circle with radius 2 centered in origin, generate b vector with a source on (0.6,1.2)
b = np.zeros(n_elements)
b[int(0.8 * len(grid) * len(grid[0]) + 0.65 * len(grid[0]))] = 1

solution = np.linalg.solve(matrix, b)

solution_grid = np.zeros((len(grid[0]), len(grid)))
for y in range(len(grid[0])):
    for x in range(len(grid)):
        # take absolute value due to imaginary numbers
        solution_grid[y][x] = np.abs((solution[y * len(grid) + x]))
plt.imshow(solution_grid, origin="lower")
plt.show()


