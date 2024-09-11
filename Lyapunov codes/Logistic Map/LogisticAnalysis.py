"""Investigating the efficiency and accuracy of algorithm for calculating LE of Logistic Map """

import numpy as np
import matplotlib.pyplot as plt
import LogisticCalcs as logistic
import scipy.stats as stats
from scipy.optimize import curve_fit


# How does the LE estimate converge with number of iterations


def iteration_convergence(total_iterations: int):
    """Performs a few runs of the LE algorithm and plots the estimates against number of iterations
    Also plots the magnitude of the difference between subsequent LE estimates

    Args:
        total_iterations (int): total iterations for algorithm to run for
    """

    lyapunov_estimates = logistic.logistic_lyapunov(
        3, total_iterations, 200, dense_output=True
    )[-1]

    iterations = np.arange(total_iterations)

    # Make subplots for LLE estimate and difference between subsequent estimates
    plt.subplot(2, 1, 1)
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.suptitle("LE Estimates against Iterations")
    plt.xlabel("Iterations", fontsize=10)
    plt.xticks(fontsize=8)

    plt.ylabel("LE Estimates ", fontsize=10)
    plt.yticks(fontsize=8)

    # plot convergence of each run
    for i in range(lyapunov_estimates.shape[0]):
        plt.plot(iterations, lyapunov_estimates[i, :])

    plt.subplot(2, 1, 2)
    ax2 = plt.gca()
    ax2.spines[["top", "right"]].set_visible(False)
    plt.suptitle("LE Estimates for Logistic Map")
    plt.xlabel("Iterations ", fontsize=10)
    plt.xticks(fontsize=8)

    plt.ylabel("Convergence of Estimates", fontsize=10)
    plt.yticks(fontsize=8)

    difference = np.abs(
        np.diff(lyapunov_estimates, axis=1)
    )  # Absolute difference between LLE Estimate after each time-step

    for j in range(lyapunov_estimates.shape[0]):
        plt.plot(iterations[1:], difference[j, :])

    plt.tight_layout(h_pad=2)
    plt.savefig("Time Convergence Logistic", dpi=300)

    plt.show()


# How does the estimated LE vary when initial position is varied


def x0_convergence(total_iterations: int, num_samples: int):
    """Plots a histogram of LE estimates with different (random) intial x0 positions

    Args:
        total_iterations (int): number of iterations to perform (after transient decays)
        num_samples (int): number of samples to run
    """

    lyapunov_estimates = logistic.logistic_lyapunov(num_samples, total_iterations, 100)[
        1
    ]
    mean = np.mean(lyapunov_estimates)
    stdev = np.std(lyapunov_estimates)

    # Normalised Histogram of results

    plt.hist(
        lyapunov_estimates,
        bins=50,
        density=True,
        facecolor="#2ab0ff",
        edgecolor="#169acf",
        linewidth=0.5,
    )

    # Fitting a normal curve
    xvals = np.linspace(np.min(lyapunov_estimates), np.max(lyapunov_estimates), 1000)
    yvals = stats.norm.pdf(xvals, mean, stdev)

    plt.plot(xvals, yvals, "k--", linewidth=0.75)
    plt.text(
        np.min(lyapunov_estimates)
        + 0.1 * (np.max(lyapunov_estimates) - np.min(lyapunov_estimates)),
        0.8 * (1 / (stdev * np.sqrt(2 * np.pi))),
        f"$\mu$: {mean:.3g}\n$\sigma$ : {stdev:.1g}",
    )
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_yticks([])
    ax1.yaxis.set_tick_params(labelleft=False)
    plt.xticks(fontsize=8)
    plt.xlabel("LLE Estimate ", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.title("LE Estimates Distribution", y=1.05)
    plt.savefig("Logistic x0 error.png", dpi=300)

    plt.show()

    # Checking that the distribution is approx. normal
    shapiro_test = stats.shapiro(lyapunov_estimates)

    print(
        f"Test Statistic: {shapiro_test[0]}\n P value: {shapiro_test[1]}\n Mean: {mean}\n StDev: {stdev}"
    )


# Observe how st. dev. in LE estimates for different x0 values varies with number of iterations


def stdev_convergence(num_samples: int, step_size: int, num_steps: int):
    """- Create a distribution of LE estimates for a given number of iterations
    - Calculate st.dev of this distribution
    - Create plots of st.dev vs iteration number

    Args:
        num_samples (int): number of samples to make up distribution
        step_size (int): number of iterations to do between calculating st.dev
        num_steps (int): total number of steps to investigate

    """

    lyapunov_estimates = logistic.logistic_lyapunov(
        num_samples, num_steps * step_size, 100, dense_output=True
    )[-1][:, ::step_size]

    # Create arrays to store stdev and iterations

    iterations = np.arange(num_steps) * step_size
    stdev = np.std(lyapunov_estimates, axis=0)

    # Additionally make a log log plot and do a linear fit to get an approximate power law scaling

    logiterations = np.log(iterations[1:])  # avoiding log(0) error
    logdeviations = np.log(stdev[1:])

    def linear(x, m, c):  # Linear curve fit equation
        return m * x + c

    # Perform the curve fit and find errors on fitted parameters
    popt, pcov = curve_fit(linear, logiterations, logdeviations)
    perr = np.sqrt(np.diag(pcov))

    # Logging the results of the linear fit
    print("m =", round(popt[0], 3), "+/-", round(perr[0], 3))
    print("c =", round(popt[1], 3), "+/-", round(perr[1], 3))

    xvals = np.linspace(
        0.95 * logiterations[0], logiterations[-1] * 1.05, 1000
    )  # for plotting best fit line

    """To make the log log plot clearer, only plot elements of the array 
    with indices corresponding to powers of 2 so points are spread more evenly on the plot"""

    indices = 2 ** np.arange(int(np.log2(len(logdeviations))))
    plotlogdeviations = logdeviations[indices]
    plotlogiterations = logiterations[indices]

    # Plotting the results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(iterations, stdev)
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.xlabel("Iterations", fontsize=10)
    plt.xticks(fontsize=8)
    plt.ylabel("Deviation", fontsize=10)
    plt.yticks(fontsize=8)
    plt.suptitle("Deviation of LE Estimates vs Iterations")

    plt.subplot(2, 1, 2)
    plt.plot(plotlogiterations, plotlogdeviations, "kx", label="Data", markersize=3)
    plt.plot(xvals, linear(xvals, popt[0], popt[[1]]), "--")
    ax2 = plt.gca()
    ax2.spines[["top", "right"]].set_visible(False)
    plt.xlabel("log(Iterations)", fontsize=10)
    plt.xticks(fontsize=8)
    plt.ylabel("log(Deviation)", fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout(h_pad=2)
    plt.savefig("Logistic Deviation vs Iterations.png", dpi=300)
    plt.show()
