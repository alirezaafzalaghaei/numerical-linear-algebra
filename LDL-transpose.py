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
def lu_solvable(A):
    return all(not np.linalg.det(np.array(A)[0:k, 0:k]) == 0 for k in range(1, len(A) + 1))


@timeit
def upper_tri_solver(a, b):
    n = len(b)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(n))) / a[i][i]
    return x


@timeit
def diagonal_solver(a, b):
    return [b[i] / a[i] for i in range(len(b))]


@timeit
def lower_tri_solver(a, b):
    n = len(b)
    x = [0] * n
    for i in range(n):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(n))) / a[i][i]
    return x


@timeit
def symmetric_possitve_difinite(D):
    return all(i > 0 for i in D)


@timeit
def LDLt(A):
    n = len(A)
    L = [x[:] for x in [[0] * n] * n]
    D = [0] * n
    for i in range(n):
        L[i][i] = 1
    for i in range(n):
        D[i] = A[i][i] - sum(D[k] * L[i][k] ** 2 for k in range(n - 1))
        for j in range(i + 1, n):
            L[j][i] = (A[i][j] - sum(D[k] * L[i][k] * L[j][k] for k in range(n - 1))) / D[i]
    return D, L


# A = input("Enter Matrix A(n,n): ")
# B = input("Enter matrix B(1,n): ")

n = 3
A = np.random.uniform(-1, 1, (n, n))
A = A + A.T
B = np.random.uniform(-10, 10, n)

A = [[1, 3, 2], [3, 4, -1], [2, -1, 7]]
# B = [2, 3, 4]

# in LDL-transpose method matrix must be symmetric
if not np.array_equal(np.array(A), np.array(A).transpose()):
    raise ValueError("Matrix must be symmetric!")

# check if LU Decomposition method can solve the equation
if not lu_solvable(A):
    raise ValueError("Determinant must be greater than zero!")

# find L and U
D, L = LDLt(copy.deepcopy(A))

# round L and U elements to 5 digits
L = np.around(L, 5)
D = np.around(D, 5)

print(L)
print(D)

# solve LY = B equation
Y = lower_tri_solver(L, B)

# solve DZ = Y equation
Z = diagonal_solver(D, Y)

# solve LtX = Z equation
X = upper_tri_solver(np.array(L).T, Z)

# find exact solution of AX = B
real = np.dot(np.linalg.inv(A), B)

# round results to 5 digits
real, X = np.around([real, X], 5)

assert np.allclose(X, real, rtol=0, atol=1e-1)

print(
    """
    LU Decomposition: {0}

    Matrix Solution : {1}
    """.format(X, real))
