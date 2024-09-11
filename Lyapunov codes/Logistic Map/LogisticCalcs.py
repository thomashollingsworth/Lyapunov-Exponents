"""Calculating the Lyapunov Exponent of the Logistic Map
- Choose a start value x0 and a value for r
- Use logistic map to iteratively determine x values
- Iterate sufficiently to avoid transient behaviour before recording x values
- For each subsequent x value record the value of the Jacobian (derivative for 1d)
- Lyapunov exponent is given by the mean value of the log of these derivatives as number of iterations tends to infinity
"""

import numpy as np
import matplotlib.pyplot as plt
import LogisticMap as logistic
import time


# Repetitively estimate Lyapunov exponent with options to vary parameters r,x0 and to vary runtime
def logistic_lyapunov(
    repeats: int,
    num_iterations: int,
    num_eqm_iterations: int,
    rvals="default",
    x0vals="default",
    dense_output=False,
) -> np.ndarray:
    """Performs repeat estimates of the Lyapunov Exponent of the Logistic Map with options to vary parameters

    Args:
        repeats (int): number of repeat estimates to perform
        num_iterations (int): number of iterations to do (post transient)
        num_eqm_iterations (int): number of iterations deemed sufficient for transient to have decayed
        rvals (np.ndarray, optional): Array of r values for each repeat (must have correct length). Defaults to 3.6 for all repeats.
        x0vals (np.ndarray, optional): Array of x0 values for each repeat (must have correct length). Defaults to random uniform on (0-1).
        dense_output (bool,optional): Do you want to store the lyapunov estimate at each iteration. Defaults to False.
    Returns:
        tuple: array of calculated x values,array of final lyapunov estimates, time to complete and optionally an array of LE estimates for each iteration
    """

    if (
        rvals == "default"
    ):  # Provides a default (chaotic) r value of 3.7 for each repeat
        rvals = 3.7 * np.ones(repeats)
    if x0vals == "default":  # Provides a defualt of random x0 values in range (0-1)
        x0vals = np.random.uniform(0, 1, (repeats))

    num_total = num_iterations + num_eqm_iterations
    t1 = time.monotonic()
    # Select only the xvalues corresponding to iterations after transient has decayed
    xdata = logistic.run_logistic_map(x0vals, rvals, num_total)[:, num_eqm_iterations:]

    # Handling calc. of lyapunov estimate in vectorised fashion

    derivatives = np.multiply(-2 * xdata + 1, np.reshape(rvals, (repeats, 1)))
    log_derivatives = np.log(np.abs(derivatives))
    # note this will encounter errors if x=0.5

    final_lyapunov_exponents = np.mean(log_derivatives, axis=1)  # The final values

    t2 = (
        time.monotonic()
    )  # Stop time here, the remaining code is optional and is only used to show how the estimate evolves with iterations
    runtime = t2 - t1
    if dense_output:
        # Now store the LE estimates after each iteration
        dense_lyapunov_exponents = np.zeros_like(
            xdata
        )  # Want to store the estimated lyapunov exponent after each iteration

        for i in range(num_iterations):
            dense_lyapunov_exponents[:, i] = np.mean(log_derivatives[:, :i], axis=1)
        # Additionally return dense array of LEs
        return xdata, final_lyapunov_exponents, runtime, dense_lyapunov_exponents
    else:
        return xdata, final_lyapunov_exponents, runtime
