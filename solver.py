import numpy as np

def normal_solver(A, y):
    if np.linalg.matrix_rank(A) == A.shape[1]:
        theta = np.linalg.solve((A.T @ A), (A.T @ y))
        SSE = np.power((A @ theta - y), 2).sum()
        return theta, SSE
    else:
        return None, None

def lstsq_solver(A, y):
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    SSE = np.power((A @ theta - y),2).sum()

    return theta, SSE