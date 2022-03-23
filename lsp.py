import numpy as np


def lstsq_ne(A, b):
    at = A.T
    ata = np.linalg.inv(at @ A)
    x = ata @ at @ b
    cost = np.linalg.norm(A @ x - b) ** 2
    var = cost * ata / (np.shape(A)[0] - np.shape(A)[1])
    return x, cost, var


def lstsq_svd(A, b, rcond=None):
    u, s, v = np.linalg.svd(A)
    sm = max(s)
    for i in range(len(s)):
        if rcond != None and s[i] * rcond < sm:
            s[i] = 0
        if s[i] != 0:
            s[i] = 1 / s[i]
    vs = v.shape[0]
    smat = np.zeros((vs, u.shape[0]))
    smat[:vs, :vs] = np.diag(s)
    Aplus = (v.T @ smat) @ u.T
    x = Aplus @ b
    cost = np.linalg.norm(A @ x - b) ** 2
    var = cost * (v.T @ np.diag(s ** 2) @ v) / (np.shape(A)[0] - np.shape(A)[1])
    return x, cost, var


def lstsq(A, b, method, **kwargs):
    if method == "ne":
        return lstsq_ne(A, b)
    if method == "svd":
        return lstsq_svd(A, b, **kwargs)

