import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math

N = 1000
c_squared = 1
L = 1
delta_t = 0.001
t_max = 1
delta_x = L/N

boundary1 = 0
boundary2 = 0

u = [math.sin(2*math.pi*x) for x in np.arange(0,L,delta_x)]

u[0] = boundary1
u[-1] = boundary2

string = [u.copy(),u.copy()]

for dt in np.arange(0,t_max,delta_t):
    print(f"{dt}/{t_max}", end="\r")
    u = np.zeros(N)
    for i in range(1,len(u)-1):
        u[i] = c_squared * (delta_t**2/delta_x**2)*(string[-1][i+1] + string[-1][i-1]-2*string[-1][i]) - string[-2][i] + 2*string[-1][i]
    string.append(u)


string_x = [x for x in np.arange(0,L,delta_x)]
for i in range(0, len(string), 10):
    plt.plot(string_x, string[i])
    plt.ylim(-1,1)
    plt.xlim(0,L)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()