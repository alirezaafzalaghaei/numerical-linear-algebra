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
def QR_HouseHolder(A):
    A = np.array(A)
    Q = np.eye(len(A))
    for i in range(len(A) - 1):
        x = A[i:, i]
        x = x[np.newaxis, :].T

        k = (-np.sign(x[0]) * np.linalg.norm(x))[0]

        e = np.zeros(len(x))
        e = e[np.newaxis, :].T
        e[0] = 1

        xt = x - k * e

        Ht = np.eye(len(x)) - 2 / np.linalg.norm(xt) ** 2 * np.dot(xt, xt.T)

        H = np.eye(len(A))
        H[i:i + len(x), i:i + len(x)] = Ht

        A = np.dot(H, A)
        Q = np.dot(Q, H)
    return Q, A


# A = input("Enter Matrix A(n,n): ")
# B = input("Enter matrix B(1,n): ")


# A = [[2, 1,3], [0, 2,1],[0,0,7]]
# B = [1,2,6]
A = [[1, 2], [-1, 1], [0, 1]]

# calculate Q and R
Q, R = QR_HouseHolder(copy.deepcopy(A))


# round Q and R elements to 5 digits
Q, R = np.around(Q, 5), np.around(R, 5)

#
QR = np.dot(Q, R)

assert np.allclose(A, QR, rtol=0, atol=1e-1)
