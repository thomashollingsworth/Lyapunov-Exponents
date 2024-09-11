"""Analysing the accuracy and complexity of the different QR decomposition algorithms"""

import numpy as np
import scipy as scipy
import QRDecomposition as qr
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit


"""Complexity Calculations:
- Calculate the average time to decompose an n by n random matrix 
- Make log plot of time vs n to determine complexity scaling"""


def time_for_n(n: np.ndarray, iterations: int, func) -> np.ndarray:
    """Calculates the average time to perform decomposition of a random n by n matrix for different n values

    Args:
        n (np.ndarray): array of n values (must be integers)
        iterations (int): number of iterations to average over
        func (function): function being tested

    Returns:
        np.ndarray: array of avg. times
    """
    times = np.zeros_like(n).astype(float)
    for i, nvals in enumerate(n):
        count = np.zeros(iterations)

        for j in range(iterations):
            random_matrix = np.random.random((nvals, nvals))
            t1 = time.monotonic()

            a, b = func(random_matrix)
            t2 = time.monotonic()
            count[j] = t2 - t1

        times[i] = np.mean(count)
        print(f"n={nvals},method={func},time={times[i]}")
    return times


# ------------------------------------------------------------------------------------------------------

# PLOTTING RESULTS


def plot_large_n():
    """Plotting time taken to perform QR decomposition for a variety of methods for large matrix sizes"""
    nvals = np.arange(1, 1800, 50)

    times_classic = time_for_n(nvals, 10, qr.gram_schmidt_classic)
    times_modified = np.log(time_for_n(nvals, 10, qr.gram_schmidt_modified))
    times_householder = np.log(time_for_n(nvals, 2, qr.householder))
    times_numpy = np.log(time_for_n(nvals, 50, qr.qr_numpy))
    times_scipy = np.log(time_for_n(nvals, 50, qr.qr_scipy))

    plt.plot(nvals, times_modified, label="Modified Gram-Schmidt")
    plt.plot(nvals, times_classic, label="Classic Gram-Schmidt")
    plt.plot(nvals, times_householder, label="Householder Reflections")
    plt.plot(nvals, times_numpy, label="Numpy Algorithm")
    plt.plot(nvals[1:], times_scipy[1:], label="Scipy Algorithm")
    plt.title("QR Decomposition Time, scaling with Matrix Size")
    plt.xlabel("$\it{log(Matrix}$ $\it{size)}$")
    plt.ylabel("$\it{log(Time/s)}$")
    plt.legend(loc="best", frameon=False, fontsize=8)
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.savefig("Finalised Time vs Matrix Size.png", dpi=300)
    plt.show()
    return None


def plot_small_n():
    """Plotting time taken to perform QR decomposition for a variety of methods for small matrix sizes"""
    nvals = np.arange(1, 6)

    times_classic = time_for_n(nvals, 1000, qr.gram_schmidt_classic)
    times_modified = time_for_n(nvals, 1000, qr.gram_schmidt_modified)
    times_householder = time_for_n(nvals, 10000, qr.householder)
    times_numpy = time_for_n(nvals, 1000, qr.qr_numpy)
    times_scipy = time_for_n(nvals, 1000, qr.qr_scipy)

    plt.plot(nvals[1:], times_modified[1:], label="Modified Gram-Schmidt")
    plt.plot(nvals[1:], times_classic[1:], label="Classic Gram-Schmidt")
    plt.plot(nvals[1:], times_householder[1:], label="Householder Reflections")
    plt.plot(nvals[1:], times_numpy[1:], label="Numpy Algorithm")
    plt.plot(nvals[1:], times_scipy[1:], label="Scipy Algorithm")
    plt.title("QR Decomposition Time for small Matrices")
    plt.xlabel("$\it{Matrix}$ $\it{size}$ $\it{(n)}$")
    plt.xticks(nvals[1:])
    plt.yscale("log")
    plt.ylabel("$\it{Time/s}$")
    plt.legend(loc="best", frameon=False, fontsize=8)
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("Finalised Time vs Matrix Size, Small n.png", dpi=300)
    plt.show()
    return None


def calc_scaling_relation():
    """Using curve fit on scipy algorithm to estimate the complexity"""
    log_nvals = np.linspace(1, 9.5, 40)
    nvals = np.round(np.exp(log_nvals)).astype(int)
    times_scipy = np.log(time_for_n(nvals, 4, qr.qr_scipy))

    def linear(x, m, c):  # Linear curve fit equation
        return m * x + c

    # Perform the curve fit and find errors on fitted parameters
    popt, pcov = curve_fit(linear, log_nvals[32:], times_scipy[32:])
    perr = np.sqrt(np.diag(pcov))

    # Report the results
    print("m =", round(popt[0], 3), "+/-", round(perr[0], 3))
    print("c =", round(popt[1], 3), "+/-", round(perr[1], 3))

    # Illustrate the fit on a graph
    plt.plot(log_nvals, times_scipy, "kx", label="Data", markersize=3)
    plt.plot(log_nvals, linear(log_nvals, *popt), "r:", label="Linear Fit")
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.title("Scipy QR algorithm: Time vs Matrix Size")
    plt.xlabel("$\it{log(Matrix}$ $\it{size)}$ ")
    plt.ylabel("$\it{log(Time/s)}$")
    plt.text(7.8, -1.75, f"m = {round(popt[0], 2)} +/- {round(perr[0], 3)}")
    plt.savefig("LogLog plot for Scipy Algorithm.png", dpi=300)
    plt.show()
    return None


# ------------------------------------------------------------------------------------------------------

"""Accuracy Calculations:
- Calculate Q and R using different algorithms for a range of small n by n matrices
- Evaluate a reconstruction error, for initial matrix A: Error= (A-QR)
- Evaluate an orthogonality error: Error= (Q(Q.T)-I)
- Compare the errors for different algorithms and different n values
"""


def reconstruction_error(A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    """Calculates reconstruction error using Frobenius Norm"""
    return np.linalg.norm(A - np.dot(Q, R), "fro")


def orthogonality_error(Q: np.ndarray) -> float:
    """Calculates orthogonality error using Frobenius Norm"""
    return np.linalg.norm(np.eye(np.shape(Q)[0]) - np.dot(Q, Q.T), "fro")


def error_for_n(n: np.ndarray, iterations: int, func) -> tuple[np.ndarray]:
    """Calculates orthogonality error and reconstruction error
    for range of matrix sizes (n)

    Args:
        n (np.ndarray): matrix sizes
        iterations (int): number of iterations used to calc an average error
        func (function): function used to perform the QR decomposition

    Returns:
        tuple[np.ndarray]: avg orthogonality error and reconstruction error at different n
    """
    orthogonality_errors = np.zeros_like(n).astype(float)
    reconstruction_errors = np.zeros_like(n).astype(float)
    for i, nval in enumerate(n):
        orthogonality_count = 0
        reconstruction_count = 0
        for j in range(iterations):

            A = np.random.random((nval, nval))

            Q, R = func(A)
            orthogonality_count += orthogonality_error(Q)
            reconstruction_count += reconstruction_error(A, Q, R)

        orthogonality_count /= iterations
        reconstruction_count /= iterations
        orthogonality_errors[i] = orthogonality_count
        reconstruction_errors[i] = reconstruction_count

    return orthogonality_errors, reconstruction_errors


# ------------------------------------------------------------------------------------------------------

# PLOTTING RESULTS


def error_plot():
    nvals = np.arange(2, 21)
    ortho_gsc, recon_gsc = error_for_n(nvals, 500, qr.gram_schmidt_classic)
    ortho_gsm, recon_gsm = error_for_n(nvals, 500, qr.gram_schmidt_modified)
    ortho_hh, recon_hh = error_for_n(nvals, 500, qr.householder)
    ortho_np, recon_np = error_for_n(nvals, 500, qr.qr_numpy)
    ortho_sp, recon_sp = error_for_n(nvals, 500, qr.qr_scipy)

    plt.subplot(2, 1, 1)
    plt.plot(nvals, ortho_gsm, label="Modified Gram-Schmidt")
    plt.plot(nvals, ortho_gsc, label="Classic Gram-Schmidt")
    plt.plot(nvals, ortho_hh, label="Householder Reflections")
    plt.plot(nvals, ortho_np, label="Numpy Algorithm")
    plt.plot(nvals, ortho_sp, label="Scipy Algorithm")
    plt.suptitle("Orthogonalisation and Reconstruction Error for Matrix Dimension 'n'")
    plt.xticks(nvals[::2], fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("$\it{Matrix}$ $\it{size}$ $\it{(n)}$", fontsize=9)
    plt.ylabel("$\it{Orthogonalisation}$ $\it{Error}$", fontsize=9)
    plt.yscale("log")
    ax1 = plt.gca()
    ax1.spines[["top", "right"]].set_visible(False)

    plt.subplot(2, 1, 2)
    plt.plot(nvals, recon_gsm, label="Modified Gram-Schmidt")
    plt.plot(nvals, recon_gsc, label="Classic Gram-Schmidt")
    plt.plot(nvals, recon_hh, label="Householder Reflections")
    plt.plot(nvals, recon_np, label="Numpy Algorithm")
    plt.plot(nvals, recon_sp, label="Scipy Algorithm")
    plt.xticks(nvals[::2], fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel("$\it{Reconstruction}$ $\it{Error}$", fontsize=9)
    plt.yscale("log")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.175),
        frameon=False,
        ncol=3,
        fontsize=8,
    )
    ax2 = plt.gca()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("QR Decomposition Error.png", dpi=300)
    plt.show()
    return None
