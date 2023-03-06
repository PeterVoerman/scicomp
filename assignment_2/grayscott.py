import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def run_simulation(initial_u, initial_v, filename):
    dt = 1
    dx = 1
    Du = 0.16
    Dv = 0.08
    f = 0.035
    k = 0.06

    tend = 1e6
    plot_interval = 100

    u = initial_u.copy()
    v = initial_v.copy()
    

    fig, axs = plt.subplots(1, 2)

    u_list = [u.copy()]
    v_list = [v.copy()]

    t = 0
    diff = 1

    while diff > 1e-6 and t < tend:
        print(f"t={t}, diff={diff:.2E}", end="\r")

        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]

        laplace_u = (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4 * u) / dx**2
        laplace_v = (np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0) + np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 4 * v) / dx**2


        u[1:-1, 1:-1] += dt * (Du * laplace_u[1:-1, 1:-1] - u[1:-1, 1:-1] * v[1:-1, 1:-1]**2 + f * (1 - u[1:-1, 1:-1]))
        v[1:-1, 1:-1] += dt * (Dv * laplace_v[1:-1, 1:-1]  + u[1:-1, 1:-1] * v[1:-1, 1:-1]**2 - (f + k) * v[1:-1, 1:-1])

        if t % plot_interval == 0:
            u_list.append(u.copy())
            v_list.append(v.copy())

        t += dt
        diff = max(abs(u_list[-1] - u_list[-2]).flatten())

    def animate(i):
        axs[0].clear()
        axs[0].imshow(u_list[i], origin="lower")
        axs[1].clear()
        axs[1].imshow(v_list[i], origin="lower")

    anim = animation.FuncAnimation(fig, animate, frames=len(u_list), interval=100)
    anim.save(f"{filename}.mp4", fps=30)


N = 100
u = np.full((N+2, N+2), 0.5)
v = np.zeros((N+2, N+2))
v[46:56, 46:56] = 0.25

N = 100
x, y = np.meshgrid(np.arange(N+2), np.arange(N+2))
center = (N//2, N//2)
radius = N//10
u = np.full((N+2, N+2), 0.5)
v = np.where((x-center[0])**2 + (y-center[1])**2 < radius**2, 0.25, 0.0)

# u += 0.05 * np.random.normal(0, 1, u.shape)
# v += 0.05 * np.random.normal(0, 1, v.shape)

filename = "circle"
run_simulation(u, v, filename)