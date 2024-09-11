"""Calculating the Spectrum of Lyapunov Exponents for the Standard Map"""

import numpy as np
import StandardMap as smap
import scipy as scipy


def calc_lyapunov_smap(
    x0: np.ndarray, K: float, step_size: int, num_steps: int, dense_output=False
) -> np.ndarray:
    """Estimates the lyapunov spectrum of the standard map, able to handle multiple initial conditions at once

    Args:
        x0 (np.ndarray): 'm' initial (p_0,theta_0) pairs in (m,2) array (m is no. of samples)
        K (float): K parameter in standard map
        step_size (int): number of iterations per renormalisation step
        num_steps (int): number of renormalisation steps to take
        dense_output (bool,optional): option to return lyapunov estimates at each renormalisation step. Defaults to False

    Returns:
        multiple np.ndarray: - points: (m,2,n) array that stores (p,theta) trajectories for all m samples over n iterations
                            - final_lyapunov_exponents: (m,2) array that stores final lyapunov spectrum estimates for all m samples
                            - Optional, dense_lyapunov_exponents: if dense_output=True, (m,num_steps,2) array that stores lyapunov spectrum estimates of all m samples after each renormalisation step
    """

    num_iter = num_steps * step_size
    num_samples = np.shape(x0)[0]
    points = smap.run_smap(x0, K, num_iter)

    """J matrix will store all relevant Jacobians and will have size: (m,n,(2,2))
    - m is number of samples, dictated by size of x0
    - n is number of iterations
    - The 2x2 matrix is the Jacobian of the standard map evaluated at each iteration step 'n':
        [[1,Kcos(theta_n)],[1,1+Kcos(theta_n)]]
    
    -Only theta values are required to construct the J matrix
    """
    # Separate out the theta values for each sample and iteration and reshape to make compatible with J_matrix

    p_vals, theta_vals = np.hsplit(points, 2)

    reshaped_theta_vals = np.reshape(theta_vals, (num_samples, num_iter))

    Kcos_theta = K * np.cos(reshaped_theta_vals)

    J_matrix = np.ones((num_samples, num_iter, 2, 2))

    J_matrix[:, :, 1, 1] = 1 + Kcos_theta
    J_matrix[:, :, 0, 1] = Kcos_theta

    """-Unable to find a nice vectorised way to take the matrix product of a series of matrices along a certain axis
     - Also can't easily perform batch QR decompositions so will have to rely on looping 
            -> Will have to resort to separating each trajectory sample and dealing with them individually

   
    - Taking the matrix product of a list of matrices can be performed using a for loop and np.dot or using np.linalg.multidot
    - np.linalg.multidot out-performs the for loop when matrix dimension is large ~1000 but for small matrices the for loop performs better"""

    """- Separate and iterate over each sample
    - Subdivide each sample into steps
    - Take the matrix product of jacobians in each step
    - Recursively Apply this product to a set of initial conditions (orthogonal 2x2 matrix
    - Perform Q,R decomposition after each step and record the diagonal of the R matrix
    - Calculate Lyapunov estimate from the mean of the log of R diagonals for each step"""

    # Array to store log(diag(R)) at each step

    results = np.zeros((num_samples, num_steps, 2))

    for count_sample, sample in enumerate(J_matrix):

        # generate initial conditions (random orthogonal 2x2 matrix adjust depending on dimension of map)

        rand_matrix = np.random.random((2, 2))
        Q, unused = scipy.linalg.qr(rand_matrix)

        steps = np.vsplit(sample, num_steps)  # split sample trajectory into steps

        for count_step, step in enumerate(steps):

            # Have to ensure matrices are multiplied in the correct order

            product = step[-1]
            for matrix in step[-2::-1]:
                product = product @ matrix

            Q, R = scipy.linalg.qr((product @ Q))

            exponents = np.log(np.abs(np.diag(R)))

            results[count_sample, count_step, :] = exponents

    final_lyapunov_exponents = np.mean(results, axis=1)

    # add option to show lyapunov estimates at each step

    if dense_output:
        # Now store the LE estimates after each iteration
        dense_lyapunov_exponents = np.zeros_like(results)
        for i in range(num_steps):
            estimate = np.mean(results[:, : (i + 1), :], axis=1)
            dense_lyapunov_exponents[:, i, :] = estimate

        return points, final_lyapunov_exponents, dense_lyapunov_exponents
    else:
        return points, final_lyapunov_exponents
