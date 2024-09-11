"""Modelling Lorenz Attractor"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate


def lorenz_derivatives(
    t: np.ndarray,
    x: np.ndarray,
    sigma: float = 10,
    beta: float = 8 / 3,
    rho: float = 28,
) -> np.ndarray:
    """Returns dx/dt, dy/dt, dz/dt for a point (x,y,z) in the Lorenz system

    Args:
        t (np.ndarray): time (necessary for passing to scipy solve_ivp)
        x (np.ndarray): the (x,y,z) values of the point
        sigma (float, optional): Parameter in Lorenz Equation, Defaults to 10.
        beta (float, optional): Parameter in Lorenz Equation, Defaults to 8/3.
        rho (float, optional): Parameter in Lorenz Equation,Defaults to 28.

    Returns:
        np.ndarray: The derivatives [dx/dt,dy/dt,dz/dt]
    """
    x_dot = sigma * (x[1] - x[0])
    y_dot = x[0] * (rho - x[2]) - x[1]
    z_dot = x[0] * x[1] - beta * x[2]
    return [x_dot, y_dot, z_dot]


def lorenz_derivatives_plus(
    t: np.ndarray,
    x_and_u: np.ndarray,
    sigma: float = 10,
    beta: float = 8 / 3,
    rho: float = 28,
) -> np.ndarray:
    """Returns derivatives for a point x and a displacement vector 'u' (linearised)

    Args:
        t (np.ndarray): time (necessary for passing to scipy solve_ivp)
        x_and_u (np.ndarray): (x,y,z,ux,uy,uz) values of point and components of 'u' vector
        sigma (float, optional): Parameter in Lorenz Equation, Defaults to 10.
        beta (float, optional): Parameter in Lorenz Equation, Defaults to 8/3.
        rho (float, optional): Parameter in Lorenz Equation, Defaults to 28.

    Returns:
        np.ndarray: The derivatives [dx/dt,dy/dt,dz/dt,d(ux)/dt...]
    """
    x, y, z, ux, uy, uz = x_and_u
    ux_dot = sigma * (uy - ux)
    uy_dot = (rho - z) * ux - uy - x * uz
    uz_dot = y * ux + x * uy - beta * uz
    x_dot, y_dot, z_dot = lorenz_derivatives(t, [x, y, z], sigma, beta, rho)

    return [x_dot, y_dot, z_dot, ux_dot, uy_dot, uz_dot]


def run_lorenz(
    x0: np.ndarray,
    t0: float,
    tf: float,
    iterations: float,
    sigma: float = 10,
    beta: float = 8 / 3,
    rho: float = 28,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    method: str = "RK45",
) -> tuple[np.ndarray]:
    """Integrates a trajectory following Lorenz Equations

    Args:
        x0 (np.ndarray): intitial position
        t0 (float): start time of integration
        tf (float): stop time of integration
        iterations (float): number of times outputted by solve_ivp
        sigma (float, optional): . Parameter in Lorenz Equation, Defaults to 10.
        beta (float, optional): Parameter in Lorenz Equation, Defaults to 8/3.
        rho (float, optional): Parameter in Lorenz Equation, Defaults to 28.
        method (str, optional): Method for integration, Defaults to RK45
    Returns:
        tuple[np.ndarray]: arrays for t,x,y,z
    """
    solution = scipy.integrate.solve_ivp(
        fun=lorenz_derivatives,
        t_span=(t0, tf),
        y0=x0,
        args=(sigma, beta, rho),
        t_eval=np.linspace(t0, tf, iterations),
        rtol=rtol,
        atol=atol,
        method=method,
    )

    return solution.t, solution.y[0], solution.y[1], solution.y[2]


def run_lorenz_plus(
    x_and_u0: np.ndarray,
    t0: float,
    tf: float,
    iterations=1000,
    sigma: float = 10,
    beta: float = 8 / 3,
    rho: float = 28,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    method: str = "RK45",
) -> list[np.ndarray]:
    """Integrates a trajectory x and displacement vector u for Lorenz system

    Args:
        x_and_u0 (np.ndarray): intitial position and displacement vector
        t0 (float): start time of integration
        tf (float): stop time of integration
        iterations (float): number of times outputted by solve_ivp
        sigma (float, optional): . Parameter in Lorenz Equation, Defaults to 10.
        beta (float, optional): Parameter in Lorenz Equation, Defaults to 8/3.
        rho (float, optional): Parameter in Lorenz Equation, Defaults to 28.
        rtol(float,optional): Relative tolerance for integrator
        atol(float,optional): Absolute tolerance for integrator
        method (str, optional): Method for integration, Defaults to RK45

    Returns:
        result of the scipy integration (contains info on t,x,y,z,ux,uy,uz) see scipy solve_ivp documentation for handling
    """
    solution = scipy.integrate.solve_ivp(
        fun=lorenz_derivatives_plus,
        t_span=(t0, tf),
        y0=x_and_u0,
        args=(sigma, beta, rho),
        t_eval=np.linspace(t0, tf, iterations),
        rtol=rtol,
        atol=atol,
        method=method,
    )

    return solution
