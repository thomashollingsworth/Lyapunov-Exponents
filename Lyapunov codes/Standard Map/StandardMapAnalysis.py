"""Analysing Algorithm for calculating Lyapunov spectrum of Standard Map"""

import numpy as np
import StandardMapCalcs as smapcalcs
import StandardMap as smap
import matplotlib.pyplot as plt
import PhaseDiagram as diagram
from scipy.optimize import curve_fit


# How do the LE estimates converge with number of renormalisation steps


def iteration_convergence(step_size: int, num_steps: int, K: int, x0: np.ndarray):
    """Performs one or multiple runs of the LE algorithm
    - Produces a phase diagram of each phase space trajectory
    - Produces a graph of lyapunov estimates vs number of steps


    Args:
        step_size (int): number of iterations before renormalisation
        num_steps (int): numnber of renormalisation steps
        K (int): K parameter for standard map
        x0 (np.ndarray): initial conditions shape (m,2) where m is number of samples
    """

    trajectories, final_vals, dense_vals = smapcalcs.calc_lyapunov_smap(
        x0, K, step_size, num_steps, dense_output=True
    )

    # Produce phase diagram
    diagram.smap_phase_diagram(
        trajectories, K, savefig=True, namefig="ConvergencePhase"
    )

    # Create plot of estimates vs num_steps
    plt.figure()

    colours = plt.cm.plasma(
        np.linspace(0, 1, np.shape(dense_vals)[0])
    )  # match colours to phase diagram

    steps_data = np.arange(num_steps)  # for plotting

    for count, exponents in enumerate(
        np.vsplit(dense_vals, np.shape(dense_vals)[0])
    ):  # Separates the results of each run to plot LEs separately

        newexponents = np.reshape(exponents, (num_steps, 2))  # convenience reshaping
        lyapunov1, lyapunov2 = np.hsplit(newexponents, 2)
        plt.plot(steps_data, lyapunov1, color=colours[count])
        plt.plot(steps_data, lyapunov2, color=colours[count])

    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines["left"].set_position(("data", 0))
    plt.title(f"Convergence with Iterations,K={K}")

    plt.xlabel("Iteration Steps", fontsize=10)
    plt.xticks(fontsize=8)

    plt.ylabel("Lyapunov Exponents", fontsize=10)
    plt.yticks(fontsize=8)
    plt.ylim(-1, 1)  # optional

    # Add a (mirror) line at y=0
    xvals = np.linspace(0, num_steps, 100000)
    yvals = np.zeros_like(xvals)

    plt.plot(xvals, yvals, "k", linewidth=0.75)
    plt.savefig(f"SMapConvergence_K={K}.png", dpi=300)
    plt.show()


def iteration_error(step_size: int, num_steps: int, K: int, x0: np.ndarray):
    """- Iterates NON-CHAOTIC trajectories on the standard map
    - Both Lyapunov Exponents should symmetrically (about x-axis) approach 0
    - Can calculate error in Lyapunov Exponent and plot a log log curve
    - Do linear fit to determine how error depends on number of iterations


    Args:
        step_size (int): number of iterations before renormalisation
        num_steps (int): numnber of renormalisation steps
        K (int): K parameter for standard map
        x0 (np.ndarray): initial conditions shape (m,2) where m is number of samples
    """

    trajectories, final_vals, dense_vals = smapcalcs.calc_lyapunov_smap(
        x0, K, step_size, num_steps, dense_output=True
    )
    steps_data = np.arange(num_steps).astype(float)

    # Produce phase diagram, easy way to check whether trajectories are chaotic or not
    diagram.smap_phase_diagram(
        trajectories, K, savefig=True, namefig="NewCheckChaotic1"
    )

    def linear(x, m, c):  # Linear curve fit equation
        return m * x + c

    # Formatting Plot

    colours = plt.cm.plasma(
        np.linspace(0, 1, np.shape(dense_vals)[0])
    )  # match colours to phase diagram

    plt.figure()

    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)

    plt.title(f"Log(Error) vs Iteration for Non-Chaotic Trajectory")

    plt.xlabel("Log(Iteration Steps)", fontsize=10)
    plt.xticks(fontsize=8)

    plt.ylabel("Log(Largest Lyapunov Exponent)", fontsize=10)
    plt.yticks(fontsize=8)

    for count, exponents in enumerate(
        np.vsplit(dense_vals, np.shape(dense_vals)[0])
    ):  # Separates the results of each sample to plot separately

        newexponents = np.reshape(exponents, (num_steps, 2))  # convenience reshaping
        lyapunov1, lyapunov2 = np.hsplit(newexponents, 2)

        loglyapunov = np.log(
            lyapunov1[2:]
        ).flatten()  # only log positive lyapunov exponent
        logsteps = np.log(steps_data[2:]).flatten()  # avoiding log(0) error

        # Perform the curve fit and find errors on fitted parameters, ignore the first quarter iterations to get a better fit

        popt, pcov = curve_fit(
            linear, logsteps[len(logsteps) // 4 :], loglyapunov[len(loglyapunov) // 4 :]
        )
        perr = np.sqrt(np.diag(pcov))

        # Logging the results of the linear fit
        print(f"Sample {count}:")
        print(f"m ={round(popt[0], 3)} +/- {round(perr[0], 3)}")
        print(f"c = {round(popt[1], 3)} +/- {round(perr[1], 3)}")

        xvals = np.linspace(
            logsteps[0], logsteps[-1] * 1.05, 1000
        )  # for plotting best fit line

        plt.plot(
            logsteps,
            loglyapunov,
            "x",
            markersize=3,
            color=colours[count],
            label=f"Sample {count+1}",
        )
        plt.plot(xvals, linear(xvals, popt[0], popt[[1]]), "--", color=colours[count])

    # plt.legend(loc="best", frameon=False, fontsize=8)
    plt.savefig("NewLogError_Smap.png", dpi=300)
    plt.show()


iteration_error(10, 100, 0.5, smap.generate_random_x0(1))

# How do LE estimates depend on the initial 'displacement cube' i.e. Q_0


def variation_with_Q(step_size: int, num_steps: int, K: int, num_samples: int):
    """Performs multiple runs of the LE algorithm along the same trajectory but with different Q_0 initial displacement conditions
    - Produces a graph to show the variations

    Args:
        step_size (int): number of iterations before renormalisation
        num_steps (int): numnber of renormalisation steps
        K (int): K parameter for standard map
        num_samples (int): number of samples to run along the same random x0 trajectory

    """

    # Create a series of identical start points in phase space

    initial = np.array([4, 3])  # Starting in chaotic region
    x0 = np.tile(initial, (5, 1))

    trajectories, final_vals, dense_vals = smapcalcs.calc_lyapunov_smap(
        x0, K, step_size, num_steps, dense_output=True
    )

    # Produce phase diagram to check whether trajectories are chaotic or not
    diagram.smap_phase_diagram(
        trajectories, K, savefig=True, namefig="ChaoticQ0VariationPhase"
    )

    # Create plot of estimates vs num_steps
    plt.figure()

    colours = plt.cm.plasma(
        np.linspace(0, 1, np.shape(dense_vals)[0])
    )  # match colours to phase diagram

    steps_data = np.arange(num_steps)  # for plotting

    for count, exponents in enumerate(
        np.vsplit(dense_vals, np.shape(dense_vals)[0])
    ):  # Separates the results of each run to plot LEs separately

        newexponents = np.reshape(exponents, (num_steps, 2))  # convenience reshaping
        lyapunov1, lyapunov2 = np.hsplit(newexponents, 2)
        plt.plot(
            steps_data, lyapunov1, color=colours[count]
        )  # For better visual only plot one exponent

    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines["left"].set_position(("data", 0))
    plt.title(f"Largest Lyapunov Exponent: Varying $Q_0$")

    plt.xlabel("Iteration Steps", fontsize=10)
    plt.xticks(fontsize=8)

    plt.ylabel("Lyapunov Exponents", fontsize=10)
    plt.yticks(fontsize=8)

    plt.savefig(f"ChaoticQ0VariationPlot.png", dpi=300)
    plt.show()
