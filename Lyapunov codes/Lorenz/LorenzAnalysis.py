"""Investigating the accuracy and efficiency of LLE algorithm on the Lorenz System"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import LorenzCalcs as calcs
import scipy.stats as stats
import time
from scipy.optimize import curve_fit
import pandas as pd

# How does renormalisation time step affect result?


def timestep_convergence():
    """Plots the convergence of LLE estimate with time for different renormalisation time-step sizes"""
    step_sizes = [100, 200, 300]  # What time-step sizes to investigate
    results = {}  # stores calculated data
    for i, step in enumerate(step_sizes):
        results[step] = calcs.calc_largest_lyapunov_lorenz(
            step,
            int(90000 / step),
            [1, 0, 1],  # Each run has same initial conditions
        )

    # Make subplots for LLE estimate and difference between subsequent estimates
    plt.subplot(2, 1, 1)
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.suptitle("LLE Estimates for Different Time Steps")
    plt.xlabel("$\it{Computing}$ $\it{Time}$ ", fontsize=9)
    plt.ylim(0.89, 0.91)

    plt.ylabel("$\it{LLE}$ $\it{Estimate}$ ", fontsize=9)

    for key, value in results.items():
        plt.plot(value[0], value[2], label=f"T={key}")

    plt.subplot(2, 1, 2)
    ax2 = plt.gca()
    ax2.spines[["top", "right"]].set_visible(False)
    plt.suptitle("LLE Estimates for Lorenz System")
    plt.xlabel("$\it{Computing}$ $\it{Time}$ ", fontsize=9)

    plt.ylabel("$\it{Estimate}$ $\it{Convergence}$ ", fontsize=9)
    plt.ylim(0, 0.003)

    for key, value in results.items():
        difference = np.abs(
            np.diff(value[2])
        )  # Absolute difference between LLE Estimate after each time-step
        plt.plot(value[0][1:], difference, label=f"T={key}")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        frameon=False,
        ncol=4,
        fontsize=8,
    )
    plt.tight_layout(h_pad=2)

    plt.show()


# How does different initial conditions affect result?


def x0_convergence(num_samples, num_steps, T=300):
    """Plots a histogram of LLE estimates with different (random) intial x0 positions

    Args:
        num_samples (float): number of samples to run
        num_steps (float): number of renormalisation time-steps before LLE is evaluated
        T (float, optional): Size of time-step. Defaults to 300.
    """
    results = np.zeros(num_samples)
    t1 = time.monotonic()
    for i in range(num_samples):
        random_x0 = np.random.uniform(0, 20, (3,))

        results[i] = calcs.calc_largest_lyapunov_lorenz(T, num_steps, random_x0)[2][-1]
        print(f"Iteration {i+1} complete")
    t2 = time.monotonic()
    plt.hist(results, bins=16)
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_yticks([])
    ax1.yaxis.set_tick_params(labelleft=False)
    plt.xlabel("$\it{LLE}$ $\it{Estimate}$ ", fontsize=9)
    plt.ylabel("$\it{Frequency}$", fontsize=9)
    plt.title("LLE Distribution: Different Initial Conditions")
    plt.savefig("Different x0 Histogram.png", dpi=300)
    plt.show()

    # Tracking time to run test and checking that the distribution is approx. normal
    shapiro_test = stats.shapiro(results)
    mean = np.mean(results)
    std = np.std(results)

    print(
        f"Time to run test: {t2-t1}\nTest Statistic: {shapiro_test[0]}\n P value: {shapiro_test[1]}\n Mean: {mean}\n StDev: {std}"
    )


def u0_convergence(num_samples, num_steps, T=300):
    """Plots a histogram of LLE estimates with different (random) intial u0 positions

    Args:
        num_samples (float): number of samples to run
        num_steps (float): number of renormalisation time-steps before LLE is evaluated
        T (float, optional): Size of time-step. Defaults to 300.
    """
    results = np.zeros(num_samples)
    t1 = time.monotonic()
    for i in range(num_samples):
        results[i] = calcs.calc_largest_lyapunov_lorenz(
            T, num_steps, [1, 0, 1], random_u0=True
        )[2][-1]
        print(f"Iteration {i+1} complete")
    t2 = time.monotonic()
    plt.hist(results, bins=16, color="r")
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_yticks([])
    ax1.yaxis.set_tick_params(labelleft=False)
    plt.xlabel("$\it{LLE}$ $\it{Estimate}$ ", fontsize=9)
    plt.ylabel("$\it{Frequency}$", fontsize=9)
    plt.title("LLE Distribution: Different Initial Conditions")
    plt.savefig("Different u0 Histogram.png", dpi=300)
    plt.show()

    # Tracking time to run test and checking that the distribution is approx. normal
    shapiro_test = stats.shapiro(results)
    mean = np.mean(results)
    std = np.std(results)
    print(
        f"Time to run test: {t2-t1}\nTest Statistic: {shapiro_test[0]}\n P value: {shapiro_test[1]}\n Mean: {mean}\n StDev: {std}"
    )


def x0_u0_convergence(num_samples, num_steps, T=300):
    """Plots a histogram of LLE estimates with different (random) intial u0 and x0 positions

    Args:
        num_samples (float): number of samples to run
        num_steps (float): number of renormalisation time-steps before LLE is evaluated
        T (float, optional): Size of time-step. Defaults to 300.
    """
    results = np.zeros(num_samples)

    for i in range(num_samples):
        random_x0 = np.random.uniform(0, 20, (3,))

        results[i] = calcs.calc_largest_lyapunov_lorenz(
            T,
            num_steps,
            random_x0,
            random_u0=True,
            method="LSODA",  # Choose method here
        )[2][-1]
        print(f"Iteration {i+1} complete")

    mean = np.mean(results)
    stdev = np.std(results)
    plt.hist(
        results,
        bins=20,
        density=True,
        facecolor="#34b663",
        edgecolor="#31854f",
        linewidth=0.5,
    )

    # Fitting normal curve
    xvals = np.linspace(np.min(results), np.max(results), 1000)
    yvals = stats.norm.pdf(xvals, mean, stdev)

    plt.plot(xvals, yvals, "k--", linewidth=0.75)
    plt.text(
        np.min(results) + 0.1 * (np.max(results) - np.min(results)),
        0.8 * (1 / (stdev * np.sqrt(2 * np.pi))),
        f"$\mu$: {mean:.3g}\n$\sigma$ : {stdev:.1g}",
    )
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_yticks([])
    ax1.yaxis.set_tick_params(labelleft=False)
    plt.xlabel("LLE Estimate", fontsize=10)
    plt.xticks(fontsize=8)
    plt.ylabel("Frequency", fontsize=10)
    plt.title("LLE Estimates Distribution", y=1.05)
    plt.savefig("LSODA Different x0 and u0 Histogram.png", dpi=300)
    plt.show()

    # Tracking time to run test and checking that the distribution is approx. normal
    shapiro_test = stats.shapiro(results)

    print(
        f"Test Statistic: {shapiro_test[0]}\n P value: {shapiro_test[1]}\n Mean: {mean}\n StDev: {stdev}"
    )


# How does run time affect accuracy of result?


def spatial_convergence_with_time(num_samples=100):
    """- Generate a distribution of LLE estiamtes from different initial conditions (u and x)
    - Calculate the standard deviation of this distribution
    - Repeat for various different algorithm run times
    - Produce a plot to show how spatial deviations is affected by algorithm run time"""
    num_steps = np.arange(
        20, 500, 20
    )  # Will be used to create different runs of algorithm
    deviations = np.zeros(len(num_steps))  # Will store the st.dev. of each run
    run_times = np.zeros(len(num_steps))  # Stores avg. runtime for each step size

    for i, step in enumerate(num_steps):

        results = np.zeros(num_samples)
        timecount = 0
        for j in range(num_samples):

            random_x0 = np.random.uniform(0, 20, (3,))

            solution = calcs.calc_largest_lyapunov_lorenz(
                10, step, random_x0, random_u0=True, method="DOP853"
            )
            results[j] = solution[2][-1]  # Stores final LLE Estimate in results array
            timecount += solution[0][-1]  # Add runtime to the counter
            print(f"Step:{step}, iteration no.:{j}")
        avgtime = timecount / num_samples
        std = np.std(
            results
        )  # Calculates standard deviation of distribution of LLE estimates
        mean = np.mean(results)

        run_times[i] = avgtime
        deviations[i] = std
        print(f"std:{std},time:{avgtime}")

    # Print final value and error
    print(f"FINAL VALUE\n {mean} ± {std}")

    # Additionally make a log log plot and do a linear fit to get an approximate power law scaling

    logtimes = np.log(run_times)
    logdeviations = np.log(deviations)

    def linear(x, m, c):  # Linear curve fit equation
        return m * x + c

    # Perform the curve fit and find errors on fitted parameters
    popt, pcov = curve_fit(linear, logtimes, logdeviations)
    perr = np.sqrt(np.diag(pcov))
    xvals = np.linspace(
        0.95 * logtimes[0], logtimes[-1] * 1.05, 1000
    )  # Plotting line of best fit

    # Logging the results of the linear fit and the final
    print("m =", round(popt[0], 3), "+/-", round(perr[0], 3))
    print("c =", round(popt[1], 3), "+/-", round(perr[1], 3))

    # Plotting the results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(run_times, deviations)
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.xlabel("$\it{Computational}$ $\it{Time}$ ", fontsize=9)
    plt.ylabel("$\it{Deviation}$ $\it{of}$ $\it{Samples}$", fontsize=9)
    plt.suptitle("Effect of runtime on spatial deviations of LLE Estimate")

    plt.subplot(2, 1, 2)
    plt.plot(logtimes, logdeviations, "kx", label="Data", markersize=3)
    plt.plot(xvals, linear(xvals, popt[0], popt[[1]]))
    ax2 = plt.gca()
    ax2.spines[["top", "right"]].set_visible(False)
    plt.xlabel("$\it{log(Computational}$ $\it{Time)}$ ", fontsize=9)
    plt.xticks(fontsize=8)
    plt.ylabel("$\it{log(Deviation}$ $\it{of}$ $\it{Samples)}$", fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout(h_pad=2)
    plt.savefig("Spatial Variation vs Time.png", dpi=300)


# How do the solve_ivp parameters affect accuracy and run time


def ivp_method(num_samples=100, step_size=300, num_steps=20):
    """- Do a sample of LLE runs with different intial conditions and calculate deviation of sample
    - Repeat for different integration methods
    - Compare time to complete and error for the different methods
    - Record results in a dataframe
    """

    results = pd.DataFrame({}, index=["Time", "Mean", "Deviation"])  # Storing results
    methods = [
        "RK23",
        "RK45",
        "DOP853",
        "Radau",
        "LSODA",
    ]  # Different methods to investigate

    for i, method in enumerate(methods):

        LLEvals = np.zeros(num_samples)
        timecount = 0
        for j in range(num_samples):

            random_x0 = np.random.uniform(0, 20, (3,))

            solution = calcs.calc_largest_lyapunov_lorenz(
                step_size, num_steps, random_x0, random_u0=True, method=method
            )
            LLEvals[j] = solution[2][-1]  # Stores final LLE Estimate in LLEvals array
            timecount += solution[0][-1]  # Add runtime to the counter
            print(f"Method:{method}, iteration no.:{j}")
        avgtime = timecount / num_samples
        std = np.std(
            LLEvals
        )  # Calculates standard deviation of distribution of LLE estimates
        mean = np.mean(LLEvals)
        results[method] = [avgtime, mean, std]

    results.to_csv("Lorenz_LLE_Methods.csv")  # Save results as csv


def ivp_params(num_samples=30, step_size=200, num_steps=20):
    "Investigating how accuracy and runtime is affected by atol and rtol"

    atol_values = np.logspace(-12, -6, 6)  # range of atol values
    rtol_values = np.logspace(-5, -1, 6)  # range of rtol values

    # 2D arrays that store mean LLE estimate, deviation of results and average runtime for atol and rtol
    log_deviation_matrix = np.zeros((len(atol_values), len(rtol_values)))
    runtime_matrix = np.zeros((len(atol_values), len(rtol_values)))
    mean_matrix = np.zeros((len(atol_values), len(rtol_values)))

    # Iterate over tolerance values and calculate Lyapunov exponents
    for i, atol in enumerate(atol_values):
        for j, rtol in enumerate(rtol_values):

            results = np.zeros(num_samples)  # Will store LLE estimates
            timecount = 0

            for k in range(num_samples):
                random_x0 = np.random.uniform(0, 20, (3,))
                solution = calcs.calc_largest_lyapunov_lorenz(
                    step_size,
                    num_steps,
                    random_x0,
                    random_u0=True,
                    method="RK45",
                    atol=atol,
                    rtol=rtol,
                )
                results[k] = solution[2][-1]
                timecount += solution[0][-1]  # Add runtime to the counter
                print(
                    f"Atol,Rtol:{atol,rtol}, iteration no.:{k}"
                )  # For logging progress

            # Calculating desired quantities
            mean = np.mean(results)
            avgtime = timecount / num_samples
            std = np.std(results)

            # Storing in array and dataframe

            runtime_matrix[i, j] = avgtime
            log_deviation_matrix[i, j] = np.log(std)
            mean_matrix[i, j] = mean

    # Convert mean_matrix to a dataframe and save

    mean_data = pd.DataFrame(
        mean_matrix,
        index=[f"atol={atol:.2e}" for i, atol in enumerate(atol_values)],
        columns=[f"rtol={rtol:.2e}" for i, rtol in enumerate(rtol_values)],
    )
    mean_data.to_csv("LLEs for RK45.csv")  # Save results as csv
    plt.figure()
    plt.title("Runtimes")
    plt.xlabel("rtol")
    plt.ylabel("atol")
    contour_levels1 = np.linspace(
        np.min(runtime_matrix), np.max(runtime_matrix), 100
    )  # Increased resolution
    plt.contourf(
        np.log10(rtol_values),
        np.log10(atol_values),
        runtime_matrix,
        levels=contour_levels1,
        cmap="viridis",
    )
    plt.colorbar(label="Avg. Runtime")
    plt.savefig("RK45 Time for Params.png", dpi=300)

    plt.figure()
    plt.title("Error in Estimate")
    plt.xlabel("rtol")
    plt.ylabel("atol")
    contour_levels2 = np.linspace(
        np.min(log_deviation_matrix), np.max(log_deviation_matrix), 100
    )
    plt.contourf(
        np.log10(rtol_values),
        np.log10(atol_values),
        log_deviation_matrix,
        levels=contour_levels2,
        cmap="viridis",
    )
    plt.colorbar(label="Log(sample deviation)")

    plt.savefig("RK45 Error for Params.png", dpi=300)
