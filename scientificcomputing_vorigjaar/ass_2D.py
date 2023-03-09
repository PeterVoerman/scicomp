# Scientific computing Exercise 2D
# Peter Voerman and Nik brouw

# The results of one of the simulations are stored in "2.pkl", "2withnoise.pkl and "2fromcorner.pkl"
# Change the variable below to True in order to run the entire simulation again

run_again = False

# When running the simulation again, change the variable below to True in order to add noise to the system
noise = False

# Change the filename in order to save/view a different simulation
filename = "2withnoise.pkl"

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

N = 100
t_end = 5000

Du = 0.16
Dv = 0.08
delta_t = 1
delta_x = 1

f = 0.035
k = 0.06

def calculate_diffusion():
    u_grid = [[0.5 for i in range(N)] for i in range(N)]
    v_grid = np.zeros((N, N))

    for x in range(45, 55):
        for y in range(45, 55):
            v_grid[y][x] = 0.25

    u_grid_list = [u_grid]
    v_grid_list = [v_grid]


    counter = 0
    for t in np.arange(0, t_end, delta_t):
        counter += 1
        print(f"{t:.2f}/{t_end}", end='\r')

        new_u_grid = np.zeros((N, N))
        new_v_grid = np.zeros((N, N))

        for y in range(1, N - 1):
            for x in range(1, N - 1):
                new_u_grid[y][x] = delta_t * (Du * ((u_grid[y][x+1]+u_grid[y][x-1]+u_grid[y+1][x]+u_grid[y-1][x]-4*u_grid[y][x])/delta_x**2)-u_grid[y][x]*v_grid[y][x]**2+f*(1-u_grid[y][x]))+u_grid[y][x]
                new_v_grid[y][x] = delta_t * (Dv * ((v_grid[y][x+1]+v_grid[y][x-1]+v_grid[y+1][x]+v_grid[y-1][x]-4*v_grid[y][x])/delta_x**2)+u_grid[y][x]*v_grid[y][x]**2-(f+k)*v_grid[y][x])+v_grid[y][x]

        for x in range(1, N - 1):
            new_u_grid[0][x] = delta_t * (Du*(((4 / 3) * (u_grid[0][x+1]+u_grid[0][x-1]+u_grid[1][x])-4*u_grid[0][x])/delta_x**2)-u_grid[0][x]*v_grid[0][x]**2+f*(1-u_grid[0][x]))+u_grid[0][x]
            new_v_grid[0][x] = delta_t * (Dv*(((4 / 3) * (v_grid[0][x+1]+v_grid[0][x-1]+v_grid[1][x])-4*v_grid[0][x])/delta_x**2)+u_grid[0][x]*v_grid[0][x]**2-(f+k)*v_grid[0][x])+v_grid[0][x]

            new_u_grid[-1][x] = delta_t * (Du*(((4 / 3) * (u_grid[-1][x+1]+u_grid[-1][x-1]+u_grid[-2][x])-4*u_grid[-1][x])/delta_x**2)-u_grid[-1][x]*v_grid[-1][x]**2+f*(1-u_grid[-1][x]))+u_grid[-1][x]
            new_v_grid[-1][x] = delta_t * (Dv*(((4 / 3) * (v_grid[-1][x+1]+v_grid[-1][x-1]+v_grid[-2][x])-4*v_grid[-1][x])/delta_x**2)+u_grid[-1][x]*v_grid[-1][x]**2-(f+k)*v_grid[-1][x])+v_grid[-1][x]

        for y in range(1, N - 1):
            new_u_grid[y][0] = delta_t * (Du*(((4 / 3) * (u_grid[y][1]+u_grid[y+1][0]+u_grid[y-1][0])-4*u_grid[y][0])/delta_x**2)-u_grid[y][0]*v_grid[y][0]**2+f*(1-u_grid[y][0]))+u_grid[y][0]
            new_v_grid[y][0] = delta_t * (Dv*(((4 / 3) * (v_grid[y][1]+v_grid[y+1][0]+v_grid[y-1][0])-4*v_grid[y][0])/delta_x**2)+u_grid[y][0]*v_grid[y][0]**2-(f+k)*v_grid[y][0])+v_grid[y][0]
   
            new_u_grid[y][-1] = delta_t * (Du*(((4 / 3) * (u_grid[y][-2]+u_grid[y+1][-1]+u_grid[y-1][-1])-4*u_grid[y][-1])/delta_x**2)-u_grid[y][-1]*v_grid[y][-1]**2+f*(1-u_grid[y][-1]))+u_grid[y][-1]
            new_v_grid[y][-1] = delta_t * (Dv*(((4 / 3) * (v_grid[y][-2]+v_grid[y+1][-1]+v_grid[y-1][-1])-4*v_grid[y][-1])/delta_x**2)+u_grid[y][-1]*v_grid[y][-1]**2-(f+k)*v_grid[y][-1])+v_grid[y][-1]
        
        new_u_grid[0][0] = delta_t * (Du*((2 * (u_grid[0][1]+u_grid[1][0])-4*u_grid[y][-1])/delta_x**2)-u_grid[y][-1]*v_grid[y][-1]**2+f*(1-u_grid[y][-1]))+u_grid[y][-1]
        new_v_grid[0][0] = delta_t * (Dv*((2 * (v_grid[0][1]+v_grid[1][0])-4*v_grid[y][-1])/delta_x**2)+u_grid[y][-1]*v_grid[y][-1]**2-(f+k)*v_grid[y][-1])+v_grid[y][-1]
        
        new_u_grid[0][-1] = delta_t * (Du*((2 * (u_grid[0][-2]+u_grid[1][-1])-4*u_grid[y][-1])/delta_x**2)-u_grid[y][-1]*v_grid[y][-1]**2+f*(1-u_grid[y][-1]))+u_grid[y][-1]
        new_v_grid[0][-1] = delta_t * (Dv*((2 * (v_grid[0][-2]+v_grid[1][-1])-4*v_grid[y][-1])/delta_x**2)+u_grid[y][-1]*v_grid[y][-1]**2-(f+k)*v_grid[y][-1])+v_grid[y][-1]
        
        new_u_grid[-1][0] = delta_t * (Du*((2 * (u_grid[-1][1]+u_grid[-2][0])-4*u_grid[y][-1])/delta_x**2)-u_grid[y][-1]*v_grid[y][-1]**2+f*(1-u_grid[y][-1]))+u_grid[y][-1]
        new_v_grid[-1][0] = delta_t * (Dv*((2 * (v_grid[-1][1]+v_grid[-2][0])-4*v_grid[y][-1])/delta_x**2)+u_grid[y][-1]*v_grid[y][-1]**2-(f+k)*v_grid[y][-1])+v_grid[y][-1]
        
        new_u_grid[-1][-1] = delta_t * (Du*((2 * (u_grid[-1][-2]+u_grid[-2][-1])-4*u_grid[y][-1])/delta_x**2)-u_grid[y][-1]*v_grid[y][-1]**2+f*(1-u_grid[y][-1]))+u_grid[y][-1]
        new_v_grid[-1][-1] = delta_t * (Dv*((2 * (v_grid[-1][-2]+v_grid[-2][-1])-4*v_grid[y][-1])/delta_x**2)+u_grid[y][-1]*v_grid[y][-1]**2-(f+k)*v_grid[y][-1])+v_grid[y][-1]
        


        for y in range(0, N):
            for x in range(0, N):
                if noise:
                    new_u_grid[y][x] += 0.02 * random.random() - 0.01
                    new_v_grid[y][x] += 0.02 * random.random() - 0.01

                if new_u_grid[y][x] < 0:
                    new_u_grid[y][x] = 0
                if new_v_grid[y][x] < 0:
                    new_v_grid[y][x] = 0
                if new_u_grid[y][x] > 1:
                    new_u_grid[y][x] = 1
                if new_v_grid[y][x] > 1:
                    new_v_grid[y][x] = 1


        u_grid = new_u_grid
        v_grid = new_v_grid

        if counter % 1 == 0:
            u_grid_list.append(u_grid)
            v_grid_list.append(v_grid)

    
    with open(filename, 'wb') as fp:
        pickle.dump([u_grid_list,v_grid_list], fp)

    return u_grid_list, v_grid_list

if not run_again:
    with open(filename, 'rb') as fp:
        u_grid_list, v_grid_list = pickle.load(fp)
else:
    u_grid_list, v_grid_list = calculate_diffusion()


times = [0, 10, 100, 500, 1000, 2500, 4999]


plt.rcParams.update({'font.size': 25})
for time in times:
    plt.imshow(u_grid_list[time])
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
    plt.title(f"Concentration of U, t={time}")
    plt.savefig(f"plots_2/{filename}u{time}.png")

counter = 0
for grid in u_grid_list:
    if counter % 25 == 0:
        plt.imshow(grid)
        plt.draw()
        plt.pause(0.001)
        plt.clf()
    counter += 1

for time in times:
    plt.imshow(v_grid_list[time])
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
    plt.title(f"Concentration of V, t={time}")
    plt.savefig(f"plots_2/{filename}v{time}.png")

# counter = 0
# for grid in v_grid_list:
#     if counter % 25 == 0:
#         plt.imshow(grid)
#         plt.draw()
#         plt.pause(0.001)
#         plt.clf()
#     counter += 1