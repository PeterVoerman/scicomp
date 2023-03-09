import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import colors

import time


def wave_solver(grid, x_length, y_length, filename, sparce=True, circle_list=[], freqs=False):
    n_elements = x_length * y_length
    matrix = np.zeros((n_elements, n_elements))

    for y in range(0, y_length):
        for x in range(0, x_length):
            element = grid[y][x]
            if y > 0:
                matrix[grid[y-1][x]][element] = 1
                matrix[element][element] -=1
            if y < y_length - 1 :
                matrix[grid[y+1][x]][element] = 1
                matrix[element][element] -=1
            if x > 0:
                matrix[grid[y][x-1]][element] = 1
                matrix[element][element] -=1
            if x < x_length - 1:
                matrix[grid[y][x+1]][element] = 1
                matrix[element][element] -=1

            matrix[element][element] = -4

    if circle_list != []:
        for y in range(y_length):
            for x in range(x_length):
                element = grid[y][x]
                if element not in circle_list:
                    matrix[element] = 0

    if sparce:
        solution = eigs(matrix)
    else:
        solution = linalg.eig(matrix)

    if not freqs:
        for L in range(min(10, len(solution[1][0]))):

            solution_grid = np.zeros((y_length, x_length))

            for y in range(y_length):
                for x in range(x_length):
                    solution_grid[y][x] = abs(solution[1][y * x_length + x][L])

            plt.title(f"f={solution[0][L]}")
            plt.imshow(solution_grid)
            plt.savefig(f"plots_3/{filename}{L+1}")

        return solution
    else:
        return solution[0]

# Calculate the time difference between sparce and non-sparce matrices
x_length = 60
y_length = 60

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
for y in range(y_length):
    for x in range(x_length):
        grid[y][x] = int(counter)
        counter += 1


start = time.time()
wave_solver(grid, x_length, y_length, "square", sparce=False, freqs=True)
end = time.time()

print(f"The calculation took {end-start} seconds without a sparce matrix")

start = time.time()
wave_solver(grid, x_length, y_length, "square", sparce=True, freqs=True)
end = time.time()

print(f"The calculation took {end-start} seconds with a sparce matrix")

# Save the images
x_length = 100
y_length = 100

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
for y in range(y_length):
    for x in range(x_length):
        grid[y][x] = int(counter)
        counter += 1

wave_solver(grid, x_length, y_length, "square")

x_length = 130
y_length = 65

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
for y in range(y_length):
    for x in range(x_length):
        grid[y][x] = int(counter)
        counter += 1


wave_solver(grid, x_length, y_length, "rectangle")

def create_circular_mask(diameter):

    center = int(diameter/2)
    radius = center

    mask = [[False for i in range(diameter)] for i in range(diameter)]
    
    for y in range(diameter):
        for x in range(diameter):
            if np.sqrt((x-center) ** 2 + (y-center) ** 2) <= radius:
                mask[y][x] = True

    return mask

mask = create_circular_mask(100)

x_length = 100
y_length = 100

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
circle_list = []

for y in range(y_length):
    for x in range(x_length):
        if mask[y][x]:
            circle_list.append(counter)
        grid[y][x] = counter
        counter += 1
        

solution = wave_solver(grid, x_length, y_length, "circle", circle_list=circle_list)


# Plot the eigenvalues for different grid sizes
for length in [25, 50, 75, 100, 125, 150, 175]:
    print(length)
    grid = np.zeros((length, length), dtype=int)

    counter = 0
    for y in range(length):
        for x in range(length):
            grid[y][x] = int(counter)
            counter += 1

    freqs = wave_solver(grid, length, length, "square", freqs=True)

    plt.scatter([length for i in range(len(freqs))], freqs)
plt.xlabel("Box size")
plt.ylabel("Eigenvalues")
plt.show()


# Animate the wave
freq = abs((solution[0][5]))
print(freq)
vector = abs(solution[1][:,5])

# for t in np.arange(0, 10, 0.1):
#     u = vector * 100*(np.cos((freq)**0.5* t) + 0.1*np.sin(freq ** 0.5 * t))
#     print(max(u), min(u))
#     solution_grid = np.zeros((y_length, x_length))

#     for y in range(y_length):
#         for x in range(x_length):
#             solution_grid[y][x] = (u[y * x_length + x])

#     plt.imshow(solution_grid)
#     plt.draw()
#     plt.pause(0.001)
#     plt.clf()

# plt.show()

fig = plt.figure()

t = 0
u = vector * (np.cos((freq)**0.5* t) + np.sin(freq ** 0.5 * t))

divnorm=colors.TwoSlopeNorm(vmin=-max(vector), vcenter=0., vmax=max(vector))

def f(u):
    solution_grid = np.zeros((y_length, x_length))

    for y in range(y_length):
        for x in range(x_length):
            solution_grid[y][x] = (u[y * x_length + x])
    return solution_grid

im = plt.imshow(f(u), animated=True, norm=divnorm)


def updatefig(*args):
    global u, t
    t += 0.05
    u = vector * (np.cos((freq)**0.5* t) + np.sin(freq ** 0.5 * t))
    im.set_array(f(u))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()