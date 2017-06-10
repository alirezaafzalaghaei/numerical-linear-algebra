from __future__ import division

import copy
import time

import numpy as np



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
def QR_Gramschmitt(A):
    m, n = len(A), len(A[0])
    if n > m:
        raise ValueError("m must be grater than n")
    elif np.linalg.matrix_rank(A) != n:
        raise ValueError('Matrix rank must be equal to %d' % n)
    R = np.zeros((n, n))
    Q = np.zeros((m, n))
    A = np.array(A)
    for k in range(n):
        for j in range(k):
            R[j][k] = np.dot(Q[:, j].T, A[:, k])
        qt = A[:, k] - sum(R[j][k] * Q[:, j] for j in range(k))
        R[k][k] = np.linalg.norm(qt)
        Q[:, k] = qt / R[k][k]
    return Q, R


# A = input("Enter Matrix A(n,n): ")
# B = input("Enter matrix B(1,n): ")

# n = 3
# A = np.random.uniform(-100, 100, (n, n))
# B = np.random.uniform(-10, 10, n)

# A = [[2, 1,3], [0, 2,1],[0,0,7]]
# B = [1,2,6]
A = [[1, 2], [-1, 1], [0, 1]]


# calculate Q and R
Q, R = QR_Gramschmitt(copy.deepcopy(A))

# round Q and R elements to 5 digits
Q, R = np.around(Q, 5), np.around(R, 5)

QR = np.dot(Q, R)

assert np.allclose(A, QR, rtol=0, atol=1e-1)

print(QR)
print('A = QR')
