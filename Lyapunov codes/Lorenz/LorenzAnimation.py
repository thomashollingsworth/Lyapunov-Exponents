"""Creating a graphic to viusalise the chaotic nature of the Lorenz System"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Lorenz as lorenz
from matplotlib.animation import PillowWriter


def gen_initial_vals(centre: np.ndarray, num: float, size: float) -> np.ndarray:
    """Creates a number of initial values
    randomly distributed in a (hyper)cube around the desired central value.

    Args:
        centre (np.ndarray): central value (given as a vector/array)
        num (float): number of points (including central point)
        size (float): dimension of (hyper)cube around central point

    Returns:
        np.ndarray: each initial point as a row vector in an array
    """

    initial = np.tile(centre, (num - 1, 1))
    random_array = size * np.random.uniform(
        low=-1.0, high=1.0, size=(num - 1, len(centre))
    )
    initial_vals = random_array + initial
    initial_vals = np.vstack((centre, initial_vals))
    return initial_vals


def create_trajectories(centre: np.ndarray, num: float, size: float, iterations: float)->np.ndarray:
    """Integrates the trajectories of a cluster of points in the Lorenz system

    Args:
        centre (np.ndarray): central trajectory
        num (float): number of points in cluster
        size (float): initial size of cluster of points (cube)
        iterations (float): number of times outputted by solve_ivp  

    Returns:
        np.ndarray: (x,y,z) trajectories for all of the points in the cluster shape(num,3,iterations) for 3D
    """
    x_0 = gen_initial_vals(centre, num, size)
    trajectories = np.zeros((num, len(centre), iterations))
    for i, row in enumerate(x_0):
        t, *pos = lorenz.run_lorenz(row, 0, 2.5, iterations)
        trajectories[i] = pos
    return trajectories


def create_animation(centre: np.ndarray, num: float, size: float, iterations: float):

    # Initialising the Background
    fig = plt.figure()
    fig.set_facecolor("k")

    # syntax for 3-D projection
    ax = plt.axes(projection="3d")
    ax.set_axis_off()
    ax.set_facecolor("k")
    t, x, y, z = lorenz.run_lorenz(centre, 0, 60, iterations * 40)
    ax.plot3D(x, y, z, "white", lw=0.3)

    points = create_trajectories(centre, num, size, iterations)
    zero_points = points[:, :, 0]
    scat = [
        ax.scatter(
            zero_points[:, 0],
            zero_points[:, 1],
            zero_points[:, 2],
            color="red",
            marker="o",
            s=2,
        )
    ]

    def update(frame):
        updated_points = points[:, :, frame]

        scat[0].remove()
        scat[0] = ax.scatter(
            updated_points[:, 0],
            updated_points[:, 1],
            updated_points[:, 2],
            color="red",
            marker="o",
            s=2,
        )

        return (scat[0],)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=10)
    plt.tight_layout()
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save("scatter.gif", writer=writer)
    plt.show()


create_animation([10, 0, 10], 100, 1, 250)
