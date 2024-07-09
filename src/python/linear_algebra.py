import numpy as np


def power_method(A, x, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        y = A @ x
        x = y / np.linalg.norm(y)
        if np.linalg.norm(A @ x - x) < tol:
            break
    return x, np.linalg.norm(y) / np.linalg.norm(x)


def inverse_power_method(A, x, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        y = np.linalg.solve(A, x)
        x = y / np.linalg.norm(y)
        if np.linalg.norm(A @ x - x) < tol:
            break
    return x, np.linalg.norm(y) / np.linalg.norm(x)


def prefactored_solve(AB, x):
    """
    Solve the linear system A x = , where A = L U."""
    A, B = AB
    y = np.linalg.solve(L, x)
    return np.linalg.solve(U, y)
