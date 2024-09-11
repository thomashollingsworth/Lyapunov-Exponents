"""Performs the standard map on an input array of initial conditions
- Design code so that multiple runs can be performed simultaneously in a vectorised fashion"""

import numpy as np


def standard_map(xn: np.ndarray, K: float) -> np.ndarray:
    """Takes an array of different (p_n,theta_n) pairs and returns an array of
    (p_(n+1),theta_(n+1)) pairs by performing one standard map iteration

    p_(n+1) =   p_n + Ksin(theta_n)% (pi * 2)
    theta_(n+1) =   theta_n + p_(n+1)% (pi * 2)


    Args:
        x0 (np.ndarray): an array of points in phase space (p_n,theta_n): size (n,2), where n is number of samples to run simultaneously
        K (float): K parameter in standard map

    Returns:
        np.ndarray: an array (n,2) of (p_(n+1),theta_(n+1)) pairs
    """
    # update p values
    xn[:, 0] = (xn[:, 0] + K * np.sin(xn[:, 1])) % (np.pi * 2)
    # update theta values
    xn[:, 1] = (xn[:, 1] + xn[:, 0]) % (np.pi * 2)
    return xn


def run_smap(x0: np.ndarray, K: float, n: int) -> np.ndarray:
    """Performs n iterations of the logistic map for an array of initial values
    allowing for a vectorised approach

    Args:
        x0 (np.ndarray): 'm' initial (p_0,theta_0) pairs in (m,2) array (m is no. of samples)
        K (float): K parameter in standard map
        n (int): number of iterations

    Returns:
        np.ndarray: an array of all the (p,theta) iterations for each initial phase space point
                    array has size (m,2,n)
    """
    i, j = np.shape(x0)
    x_values = np.zeros((i, j, n))
    x_values[:, :, 0] = x0
    x = x0
    for _ in range(n - 1):
        x = standard_map(x, K)
        x_values[:, :, _ + 1] = x

    return x_values


# Functions to quickly create a set of initial conditions


def generate_grid_x0(sqrt_num_samples: int) -> np.ndarray:
    """Creates an x0 array of initial points spread evenly across phase space

    Args:
        sqrt_num_samples (int): square root of number of samples

    Returns:
        np.ndarray: a (n,2) array of phase space points where n is number of samples
    """
    pvals = np.linspace(0, 2 * np.pi, sqrt_num_samples)
    thetavals = np.linspace(0, 2 * np.pi, sqrt_num_samples)
    p, theta = np.meshgrid(pvals, thetavals)

    return np.transpose(np.vstack((p.flatten(), theta.flatten())))


def generate_random_x0(num_samples: int) -> np.ndarray:
    """Creates an x0 array of initial points spread uniformly and randomly across phase space

    Args:
        num_samples (int): number of samples

    Returns:
        np.ndarray: a (n,2) array of phase space points where n is number of samples
    """

    return np.random.uniform(0, 2 * np.pi, (num_samples, 2))
