import numpy as np


def eig(A, max_iter=100):
    n = len(A)
    x = [1] * n
    i, eps = 0, 1

    x = np.dot(A, x)
    indx = x.tolist().index(max(np.abs(x)))
    _lambda1 = x[indx]
    x = x / _lambda1

    while i < max_iter and eps > 1e-7:
        x = np.dot(A, x)
        indx = x.tolist().index(max(np.abs(x)))
        _lambda = x[indx]
        eps = abs(_lambda - _lambda1)
        _lambda1 = _lambda
        x = x / _lambda
        i += 1

    return _lambda, i, x



A = [[3, -1, 0], [-1, 2, -1], [1, -1, 4]]
_eig, iter, vec = eig(A)

w, v = np.linalg.eig(A)

assert np.allclose(max(w), _eig, rtol=0, atol=1e-1)
