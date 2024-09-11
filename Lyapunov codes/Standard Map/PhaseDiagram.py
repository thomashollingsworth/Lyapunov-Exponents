"""Plotting phase diagram of Standard Map"""

import numpy as np
import StandardMap as smap
import matplotlib.pyplot as plt

# Plot phase diagram given trajectories


def smap_phase_diagram(
    trajectories: np.ndarray, K: float, savefig=False, namefig: str = "untitled"
):
    """Takes a series of input phase space points and plots the standard map trajectories of each for a certain number of iterations

    Args:
        trajectories (np.ndarray): phase space trajectories (p,theta), shape (m,2,n) where m is number of samples and n is number of iterations
        K (float): K parameter in standard map (only needed for adding info to plot title)
        savefig (Bool): Do you want to save the phase diagram. Defaults to False
        namefig (str): What name do you want to give to figure. Defaults to "untitled"
    """

    # For easier visualisation
    colours = plt.cm.plasma(np.linspace(0, 1, np.shape(trajectories)[0]))

    # Formatting plot
    plt.figure()
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.title(f"Standard Map: K = {K}")
    plt.xlabel("p", fontsize=10)
    plt.xticks(fontsize=8)

    plt.ylabel("$\Theta$", fontsize=10)
    plt.yticks(fontsize=8)

    i, j, k = np.shape(trajectories)  # i:num samples,j:2 (2D),k:num iterations 'n'

    for count, trajectory in enumerate(
        np.vsplit(trajectories, i)
    ):  # Separates each trajectory to plot separately
        newtrajectory = np.reshape(trajectory, (j, k))  # convenience reshaping
        p_trajectory, theta_trajectory = np.vsplit(newtrajectory, j)
        plt.plot(
            p_trajectory[:5000],
            theta_trajectory[:5000],
            ".",
            markersize=1,
            color=colours[count],
        )
    if savefig:
        plt.savefig(f"{namefig}_K={K}.png", dpi=300)


# Creating a variety of plots at interesting K values
"""
K_values = [0.5, 1, 2]
for K in K_values:
    smap_phase_diagram(
        smap.run_smap(smap.generate_grid_x0(15), K, 500),
        K,
        savefig=True,
        namefig="SMap",
    )
"""
