import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import time

matplotlib.rcParams['font.size'] = 16

def run_simulation(initial_u, initial_v, filename, plot_frames=[], save_animation=False, min_delta=1e-4):
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
    fig.set_size_inches(10, 5)

    u_list = [u.copy()]
    v_list = [v.copy()]

    t = 0
    diff = 1

    laplace_u = np.zeros_like(u)
    laplace_v = np.zeros_like(v)

    while diff > min_delta and t < tend:
        print(f"t={t}, diff={diff:.2E}", end="\r")

        laplace_u[1:-1, 1:-1] = (np.roll(u, -1, axis=0)[1:-1, 1:-1] + np.roll(u, 1, axis=0)[1:-1, 1:-1] + np.roll(u, -1, axis=1)[1:-1, 1:-1] + np.roll(u, 1, axis=1)[1:-1, 1:-1] - 4 * u[1:-1, 1:-1]) / dx**2
        laplace_v[1:-1, 1:-1] = (np.roll(v, -1, axis=0)[1:-1, 1:-1] + np.roll(v, 1, axis=0)[1:-1, 1:-1] + np.roll(v, -1, axis=1)[1:-1, 1:-1] + np.roll(v, 1, axis=1)[1:-1, 1:-1] - 4 * v[1:-1, 1:-1]) / dx**2

        laplace_u[0, 1:-1] = (u[1, 1:-1] + np.roll(u, -1, axis=1)[0, 1:-1] + np.roll(u, 1, axis=1)[0, 1:-1] - 3 * u[0, 1:-1]) / dx**2
        laplace_u[-1, 1:-1] = (u[-2, 1:-1] + np.roll(u, -1, axis=1)[-1, 1:-1] + np.roll(u, 1, axis=1)[-1, 1:-1] - 3 * u[-1, 1:-1]) / dx**2
        laplace_u[1:-1, 0] = (np.roll(u, -1, axis=0)[1:-1, 0] + np.roll(u, 1, axis=0)[1:-1, 0] + u[1:-1, 1] - 3 * u[1:-1, 0]) / dx**2
        laplace_u[1:-1, -1] = (np.roll(u, -1, axis=0)[1:-1, -1] + np.roll(u, 1, axis=0)[1:-1, -1] + u[1:-1, -2] - 3 * u[1:-1, -1]) / dx**2

        laplace_v[0, 1:-1] = (v[1, 1:-1] + np.roll(v, -1, axis=1)[0, 1:-1] + np.roll(v, 1, axis=1)[0, 1:-1] - 3 * v[0, 1:-1]) / dx**2
        laplace_v[-1, 1:-1] = (v[-2, 1:-1] + np.roll(v, -1, axis=1)[-1, 1:-1] + np.roll(v, 1, axis=1)[-1, 1:-1] - 3 * v[-1, 1:-1]) / dx**2
        laplace_v[1:-1, 0] = (np.roll(v, -1, axis=0)[1:-1, 0] + np.roll(v, 1, axis=0)[1:-1, 0] + v[1:-1, 1] - 3 * v[1:-1, 0]) / dx**2
        laplace_v[1:-1, -1] = (np.roll(v, -1, axis=0)[1:-1, -1] + np.roll(v, 1, axis=0)[1:-1, -1] + v[1:-1, -2] - 3 * v[1:-1, -1]) / dx**2

        laplace_u[0, 0] = (u[1, 0] + u[0, 1] - 2 * u[0, 0]) / dx**2
        laplace_u[-1, 0] = (u[-2, 0] + u[-1, 1] - 2 * u[-1, 0]) / dx**2
        laplace_u[0, -1] = (u[1, -1] + u[0, -2] - 2 * u[0, -1]) / dx**2
        laplace_u[-1, -1] = (u[-2, -1] + u[-1, -2] - 2 * u[-1, -1]) / dx**2

        laplace_v[0, 0] = (v[1, 0] + v[0, 1] - 2 * v[0, 0]) / dx**2
        laplace_v[-1, 0] = (v[-2, 0] + v[-1, 1] - 2 * v[-1, 0]) / dx**2
        laplace_v[0, -1] = (v[1, -1] + v[0, -2] - 2 * v[0, -1]) / dx**2
        laplace_v[-1, -1] = (v[-2, -1] + v[-1, -2] - 2 * v[-1, -1]) / dx**2

        u += dt * (Du * laplace_u - u * v**2 + f * (1 - u))
        v += dt * (Dv * laplace_v + u * v**2 - (f + k) * v)


        if t % plot_interval == 0:
            u_list.append(u.copy())
            v_list.append(v.copy())

        t += dt
        diff = max(abs(u_list[-1] - u_list[-2]).flatten())

    plot_frames.append(len(u_list) - 1)
    for frame in plot_frames:
        if frame >= len(u_list):
            continue
        axs[0].clear()
        axs[0].imshow(u_list[frame], origin="lower")
        axs[0].set_title(f"U, t = {frame * plot_interval * dt}")
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[1].clear()
        axs[1].imshow(v_list[frame], origin="lower")
        axs[1].set_title("V")
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        plt.savefig(f"{filename}_{frame}.png")

    def animate(i):
        axs[0].clear()
        axs[0].imshow(u_list[i], origin="lower")
        axs[1].clear()
        axs[1].imshow(v_list[i], origin="lower")

    if save_animation:
        anim = animation.FuncAnimation(fig, animate, frames=len(u_list), interval=100)
        anim.save(f"{filename}.mp4", fps=30)


N = 100
u = np.full((N, N), 0.5)
v = np.zeros((N, N))
v[45:55, 45:55] = 0.25

frames = [0, 1, 5, 10, 20, 40, 80, 150, 300, 500, 1000]

filename = "normal_reaction/normal_reaction"
run_simulation(u, v, filename, plot_frames=frames, save_animation=False, min_delta=1e-4)
