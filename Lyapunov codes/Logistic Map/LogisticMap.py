""" Running the logistic map """

import numpy as np


def logistic_map(x_n: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Takes an input x array returns an array of the next x values in logistic map
    allows for repeat calcs to be handled in a vectorised fashion
    x_n->x_(n+1)

    Args:
        x_n (np.ndarray): initial x value (x_n)
        r (np.ndarray): r parameter in logistic equation

    Returns:
        np.ndarray: next x value (x_(n+1))
    """
    return r * x_n * (1 - x_n)


def run_logistic_map(x0: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    """Performs n iterations of the logistic map for an array of initial values
    allowing for a vectorised approach

    Args:
        x0 (np.ndarray): start x values
        r (np.ndarray): r parameters in logistic equation
        n (int): number of iterations

    Returns:
        np.ndarray: an array of all the x values x_0->x_n for each initial x value (each row)
    """
    x_values = np.zeros((len(x0), n))
    x_values[:, 0] = x0
    x = x0
    for i in range(n - 1):
        x = logistic_map(x, r)
        x_values[:, i + 1] = x

    return x_values
