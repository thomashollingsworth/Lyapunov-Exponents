""""Calculating the Largest Lyapunov Exponent (LLE) of the Lorenz System"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import Lorenz as lorenz
import time


def random_initial_displacement():
    """Creates an initial random unit length displacement vector 'u'."""
    u = np.random.random(3)
    u = u / np.linalg.norm(u)
    return u


def calc_largest_lyapunov_lorenz(
    t_step: float,
    num_steps: float,
    start_pos: np.ndarray,
    random_u0: bool = False,
    sigma: float = 10,
    beta: float = 8 / 3,
    rho: float = 28,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    method: str = "RK45",
) -> tuple[np.ndarray]:
    """Calculates a rolling average of the largest lyapunov exponent (LLE) for the Lorenz System
    tracks real time taken per evaluation and 'simulation' time for dynamics of system

    Args:
        t_step (float): Time step between re-normalising displacement vector and updating lyapunov estimate
        num_steps (float): number of time steps to run
        start_pos (np.ndarray): starting point before transient behaviour has finished
        random_u0(bool,optional): Should the initial displacement 'u0' be random, Defaults to False.
        sigma (float, optional): Parameter in Lorenz Equation, Defaults to 10.
        beta (float, optional): Parameter in Lorenz Equation, Defaults to 8/3.
        rho (float, optional): Parameter in Lorenz Equation, Defaults to 28.
        rtol(float,optional): Relative tolerance for integrator
        atol(float,optional): Absolute tolerance for integrator
        method(str,optional): Integration method, Defaults to RK45

    Returns:
        tuple[np.ndarray]: real/computer time,simulation time,LLE estimate at that time
    """

    # Make sure initial x values are on the strange attractor by doing an initial integration

    unused, x, y, z = lorenz.run_lorenz(
        start_pos,
        0,
        50,
        100,
        sigma=sigma,
        beta=beta,
        rho=rho,
        rtol=rtol,
        atol=atol,
        method=method,
    )

    x0 = [x[-1], y[-1], z[-1]]  # initial x point on attractor

    # Incorportate an option to choose a random initial displacement or not
    if random_u0:
        u0 = random_initial_displacement()
    else:
        u0 = [0, 1, 0]  # an arbitrary unit displacement

    x_and_u0 = np.concatenate((x0, u0))
    lyapunov_sum = 0  # keeps track of sum ln(|u|)
    lyapunov_estimates = np.zeros(
        num_steps
    )  # calculates the new lyapunov estimate for each time step

    comp_times_sum = 0  # keeps track of the time taken to calc LLE at each step
    comp_times = np.zeros(
        num_steps
    )  # records time to calc as a fucntion of number of steps

    for i in range(num_steps):
        t1 = time.monotonic()
        solution = lorenz.run_lorenz_plus(
            x_and_u0,
            0,
            t_step,
            sigma=sigma,
            beta=beta,
            rho=rho,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        new_x, new_u = (
            solution.y[:3, -1],
            solution.y[3:, -1],
        )  # Will start next interval where the previous ended
        mag_u = np.linalg.norm(new_u)
        new_u = new_u / mag_u  # re-normalise vector

        x_and_u0 = np.concatenate((new_x, new_u))

        log_mag_u = np.log(mag_u)
        lyapunov_sum += log_mag_u
        lyapunov_estimates[i] = lyapunov_sum / ((i + 1) * t_step)
        t2 = time.monotonic()
        comp_times_sum += t2 - t1
        comp_times[i] = comp_times_sum

    simulation_times = (np.arange(num_steps) + 1) * t_step
    return comp_times, simulation_times, lyapunov_estimates
