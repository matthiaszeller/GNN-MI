from typing import Iterable

import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy.spatial import distance_matrix
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def get_laplacian(A: sparse.spmatrix):
    # Degree matrix
    D = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(D)

    L = D - A
    return L


def quadratic_forms_expm(L: sparse.spmatrix, x: np.ndarray, tau_0: float, kmax: int) -> np.ndarray:
    """
    Compute quadratic forms z_i^T L z_i, where the z_i vectors are a filtered version of the x signal.
    Define the matrix exponential M_i = exp(-tau_0 * i * L),
    then z_i = M_i x.
    """
    e = expm_multiply(L, x, start=-tau_0 * kmax, stop=-tau_0, num=kmax, endpoint=True)
    quadratic_forms = np.array([
        z.T @ L @ z for z in e
    ])
    return quadratic_forms


def matrix_exponentials(L: sparse.spmatrix, tau_0: float, kmax: int):
    """
    Compute the matrix exponentials exp(-tau_0 * k * L) with all values of k in (1, 2, ..., kmax)
    Done by first computing matrix eponential E_0 = exp(-tau_0 L), then E_i = E_0^i = E_{i-1} @ E_0
    """
    assert tau_0 > 0
    # Turns out that in our case, matrix exponential is faster with the dense version than the sparse one
    E0 = expm(-tau_0 * L.todense())
    yield E0
    # Compute powers of E0
    Ek = E0
    for k in range(2, kmax+1):
        Ek = Ek @ E0
        yield Ek


if __name__ == '__main__':
    import setup
    import torch
    import matplotlib.pyplot as plt

    # Test matrix exponentials
    n = 100
    L = np.random.randn(n, n)
    zero_r = np.random.choice(n, size=int(n**2*0.8))
    zero_c = np.random.choice(n, size=int(n**2*0.8))
    L[zero_r, zero_c] = 0.0
    L = sparse.coo_matrix(L)

    plt.spy(L, markersize=1); plt.show()

    tau_0 = 0.05
    kmax = 10
    k = 1
    x = np.random.randn(n)
    r = expm_multiply(L, x, start=-tau_0*kmax, stop=-tau_0, num=kmax, endpoint=True)

    taus = np.linspace(-tau_0 * kmax, -tau_0, kmax, endpoint=True)
    for i, tau in enumerate(taus):
        E = expm(tau * L)
        xtilde = E @ x
        error = np.linalg.norm(xtilde - r[i])
        print(f'tau = {tau}, error = {error}')
    # for Ek in matrix_exponentials(L, tau_0, kmax):
    #     true_Ek = expm(-tau_0 * k * L.tocsc())
    #     inf_error = np.abs(true_Ek - Ek).max()
    #     print(f'k={k}, inf_error = {inf_error}')
    #     k += 1
