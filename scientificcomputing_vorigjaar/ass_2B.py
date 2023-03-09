# Scientific computing Exercise 2B&C
# Peter Voerman and Nik brouw

import numpy as np
import matplotlib.pyplot as plt
import random

N = 100

grid = np.zeros((N,N))

cluster = [[50, N-1]]

sticking_probability = 0.1

while len(cluster) < 750:
	found = False

	x = np.random.randint(0, 100)
	y = 0

	while not found:
		direction = np.random.choice(["e", "n", "w", "s"])

		if direction == "e" and [x+1,y] not in cluster:
			x += 1
		if direction == "w" and [x-1,y] not in cluster:
			x -= 1
		if direction == "n" and [x,y+1] not in cluster:
			y += 1
		if direction == "s" and [x,y-1] not in cluster:
			y -= 1

		x %= 100

		if y >= 100 or y < 0:
			found = True
			continue

		if [x + 1, y] in cluster or [x - 1, y] in cluster or [x, y + 1] in cluster or [x, y - 1] in cluster:
			if random.random() < sticking_probability:
				cluster.append([x, y]) 
				print(len(cluster))
				found = True

for point in cluster:
	grid[point[1]][point[0]] = 1

plt.title("A DLA cluster generated using the Monte Carlo method")
plt.imshow(grid)
plt.show()

		