from __future__ import division

import copy
import time

import numpy as np


def qr_solvable(A):
    return not any(A[i][j] for i in range(1, len(A)) for j in range(i - 1))


def timeit(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        print('%-20r %7.5f sec' % (method.__name__, te - ts))
        return result

    return timed


@timeit
def upper_tri_solver(a, b):
    n = len(b)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(n))) / a[i][i]
    return x


@timeit
def lower_tri_solver(a, b):
    n = len(b)
    x = [0] * n
    for i in range(n):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(n))) / a[i][i]
    return x


@timeit
def QR_Givenz(H):
    m, n = len(H), len(H[0])

    if n > m:
        raise ValueError("m must be grater than n")

    Q = np.eye(n, n)
    for k in range(n - 1):
        alpha = np.sqrt(H[k][k] ** 2 + H[k + 1][k] ** 2)
        c = H[k][k] / alpha
        s = -H[k + 1][k] / alpha
        H[k][k] = alpha
        H[k + 1][k] = 0
        for j in range(k + 1, n):
            H[k][j] = c * H[k][j] + s * H[k + 1][j]
            H[k + 1][j] = -s * H[k][j] + c * H[k + 1][j]

        Q[:, k] = c * Q[:, k] + s * Q[:, k + 1]
        Q[:, k + 1] = -s * Q[:, k] + c * Q[:, k + 1]

    return Q, H


# A = input("Enter Matrix A(n,n): ")
# B = input("Enter matrix B(1,n): ")

# n = 3
# A = np.random.uniform(-100, 100, (n, n))
# B = np.random.uniform(-10, 10, n)

A = [[2, 1, 3], [0, 2, 1], [0, 0, 7]]
B = [1, 2, 6]

# check if QR Decomposition method can solve the equation
if not qr_solvable(A):
    raise ValueError("Matrix must be upper Hessenberg")

# find L and U
Q, R = QR_Givenz(copy.deepcopy(A))

# round L and U elements to 5 digits
Q, R = np.around([Q, R], 5)

# solve LY = B equation
Y = lower_tri_solver(Q, B)

# solve UX = Y equation
X = upper_tri_solver(R, Y)

# find exact solution of AX = B
real = np.dot(np.linalg.inv(A), B)

# round results to 5 digits
real, X = np.around([real, X], 5)

assert np.allclose(X, real, rtol=0, atol=1e-1)

print(
    """
    QR Decomposition(Givenz): {0}

    Matrix Solution         : {1}
    """.format(X, real))
