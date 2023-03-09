import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


def diffusion_solver(grid, x_length, y_length, filename, sparce=True, circle_list=[]):
    n_elements = x_length * y_length
    matrix = np.zeros((n_elements, n_elements))

    for y in range(0, y_length):
        for x in range(0, x_length):
            element = grid[y][x]
            if y > 0:
                matrix[grid[y-1][x]][element] = -1
                matrix[element][element] -=1
            if y < y_length - 1 :
                matrix[grid[y+1][x]][element] = -1
                matrix[element][element] -=1
            if x > 0:
                matrix[grid[y][x-1]][element] = -1
                matrix[element][element] -=1
            if x < x_length - 1:
                matrix[grid[y][x+1]][element] = -1
                matrix[element][element] -=1

            matrix[element][element] = 4

    for y in range(y_length):
        for x in range(x_length):
            element = grid[y][x]
            if circle_list != [] and element not in circle_list:
                matrix[element] = 0

    np.fill_diagonal(matrix, 4)

    b = np.zeros(n_elements)

    b[int(0.3*x_length*y_length + 0.6 * y_length)] = 10

    solution = linalg.solve(matrix, b)

    solution_grid = np.zeros((y_length, x_length))

    for y in range(y_length):
        for x in range(x_length):
            solution_grid[y][x] = (solution[y * x_length + x])
    plt.imshow(solution_grid)
    plt.savefig(f"plots_3/diffusion/{filename}")

    return solution


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

mask = create_circular_mask(100, 100)

x_length = 50
y_length = 50

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
circle_list = []

for y in range(y_length):
    for x in range(x_length):
        if mask[y][x]:
            circle_list.append(counter)
        grid[y][x] = counter
        counter += 1

solution = diffusion_solver(grid, x_length, y_length, "circle", circle_list=circle_list)