"""A variety of functions that perform QR decomposition of a matrix
 A->QR where A is a general matrix, Q is orthonormal and R is upper triangular"""

import numpy as np

import scipy as scipy


# Gram-Schmidt Procedure, Classic and Modified


def gram_schmidt_classic(A: np.ndarray) -> tuple[np.ndarray]:
    """Performs QR decomposition on input matrix using classic Gram-Schmidt procedure
    (orthonormalises w.r.t the columns of the input matrix)

    Args:
        A (np.ndarray): Input matrix

    Returns:
        tuple[np.ndarray]: Q and R matrices (orthonormal and upper triangular)

    """
    m, n = np.shape(A)
    Q = np.zeros((m, n))
    for i in range(n):
        column = A[:, i]
        for j in range(n - 1):
            column = np.subtract(column, np.dot(column, Q[:, j]) * Q[:, j])
        column /= np.linalg.norm(column)
        Q[:, i] = column
    R = np.matmul(Q.T, A)

    # Testing function
    if __name__ == "__main__":
        print(
            f"Testing Classic GS Method \nA=\n{A} \n Q=\n{np.round(Q)} \n QQ.T=\n{np.round(np.matmul(Q.T,Q))}  \n R=\n{np.round(R)} \n QR=\n {np.round(np.matmul(Q,R))}\n Done "
        )

    return Q, R


def gram_schmidt_modified(A: np.ndarray) -> tuple[np.ndarray]:
    """Performs QR decomposition on input matrix using modified Gram-Schmidt procedure
    (orthonormalises w.r.t the columns of the input matrix)

    Args:
        A (np.ndarray): Input matrix

    Returns:
        tuple[np.ndarray]: Q and R matrices (orthonormal and upper triangular)

    """
    m, n = np.shape(A)

    Q = np.copy(A.astype(float))

    for i in range(n):
        Q[:, i] = np.divide(Q[:, i], np.linalg.norm(Q[:, i]))

        for j in range(i + 1, n):
            Q[:, j] = np.subtract(Q[:, j], np.dot(Q[:, j], Q[:, i]) * Q[:, i])

    R = np.matmul(Q.T, A)

    # Testing function
    if __name__ == "__main__":
        print(
            f"Testing Modified GS Method \nA=\n{A} \n Q=\n{np.round(Q)} \n QQ.T=\n{np.round(np.matmul(Q.T,Q))}  \n R=\n{np.round(R)} \n QR=\n {np.round(np.matmul(Q,R))}\n Done "
        )

    return Q, R


# Householder Reflections


def householder(A: np.ndarray) -> tuple[np.ndarray]:
    """Performs QR decomposition on input matrix using householder reflection procedure
    (orthonormalises w.r.t the columns of the input matrix)

    Args:
        A (np.ndarray): Input Matrix

    Returns:
        tuple[np.ndarray]: Q and R matrices (orthonormal and upper triangular)

    """
    m, n = np.shape(A)
    R = np.copy(A.astype(float))
    Q = np.eye(m)
    for i in range(n - 1):
        magnitude = np.linalg.norm(R[i:, i])
        adjustment_vector = np.zeros_like(R[i:, [i]])
        adjustment_vector[0] = magnitude
        reflection_vector = R[i:, [i]] - adjustment_vector
        reflection_vector = reflection_vector / np.linalg.norm(reflection_vector)
        Q_i = np.eye(m - i) - 2 * np.dot(reflection_vector, reflection_vector.T)
        Q_i = np.block([[np.eye(i), np.zeros((i, m - i))], [np.zeros((m - i, i)), Q_i]])

        R = np.dot(Q_i, R)
        Q = np.dot(Q, Q_i.T)

    # Testing function
    if __name__ == "__main__":
        print(
            f"Testing Householder Method \n A=\n{A} \n Q=\n{np.round(Q)} \n QQ.T=\n{np.round(np.matmul(Q.T,Q))}  \n R=\n{np.round(R)} \n QR=\n {np.round(np.matmul(Q,R))}\n Done "
        )

    return Q, R


# Numpy inbuilt function


def qr_numpy(A: np.ndarray) -> tuple[np.ndarray]:
    Q, R = np.linalg.qr(A)
    # Testing function
    if __name__ == "__main__":
        print(
            f"Testing In-Built Numpy Method \n A=\n{A} \n Q=\n{np.round(Q)} \n QQ.T=\n{np.round(np.matmul(Q.T,Q))}  \n R=\n{np.round(R)} \n QR=\n {np.round(np.matmul(Q,R))}\n Done "
        )
    return Q, R


def qr_scipy(A: np.ndarray) -> tuple[np.ndarray]:
    Q, R = scipy.linalg.qr(A)
    # Testing function
    if __name__ == "__main__":
        print(
            f"Testing In-Built Scipy Method \n A=\n{A} \n Q=\n{np.round(Q)} \n QQ.T=\n{np.round(np.matmul(Q.T,Q))}  \n R=\n{np.round(R)} \n QR=\n {np.round(np.matmul(Q,R))}\n Done "
        )
    return Q, R



