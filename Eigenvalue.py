import numpy as np


def eig(A, max_iter=100):
    i = 0
    while i < max_iter:
        q, r = np.linalg.qr(A)
        A = np.dot(r, q)
        i += 1
    return list(map(lambda i:A[i][i],range(len(A))))


A = [[3, -1, 0], [-1, 2, -1], [1, -1, 4]]

w,v = np.linalg.eig(A)
_eig = eig(A)
assert np.allclose(sorted(_eig), sorted(w), rtol=0, atol=1e-1)
