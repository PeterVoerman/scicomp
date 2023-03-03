import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

N = 100
dt = 1
dx = 1
Du = 0.16
Dv = 0.08
f = 0.035
k = 0.06

tend = 100000
plot_interval = 100

u = np.full((N+2, N+2), 0.5)
v = np.zeros((N+2, N+2))
v[46:57, 46:57] = 0.25

fig, axs = plt.subplots(1, 2)

u_list = []
v_list = []

for t in np.arange(0, tend, dt):
    print(f"{t/tend*100:.2f}%", end="\r")

    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    v[0, :] = v[1, :]
    v[-1, :] = v[-2, :]
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]

    laplace_u = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / dx**2
    laplace_v = (v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4 * v[1:-1, 1:-1]) / dx**2

    u[1:-1, 1:-1] += dt * (Du * laplace_u - u[1:-1, 1:-1] * v[1:-1, 1:-1]**2 + f * (1 - u[1:-1, 1:-1]))
    v[1:-1, 1:-1] += dt * (Dv * laplace_v + u[1:-1, 1:-1] * v[1:-1, 1:-1]**2 - (f + k) * v[1:-1, 1:-1])

    if t % plot_interval == 0:
        u_list.append(u.copy())
        v_list.append(v.copy())

def animate(i):
    axs[0].clear()
    axs[0].imshow(u_list[i])
    axs[1].clear()
    axs[1].imshow(v_list[i])

anim = animation.FuncAnimation(fig, animate, frames=len(u_list), interval=100)
plt.show()

        