from typing import Iterable

import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy.spatial import distance_matrix
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, eigsh


def get_laplacian(A: sparse.spmatrix):
    # Degree matrix
    D = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(D)

    L = D - A
    return L


def get_laplacian_spectral_bounds(L: sparse.spmatrix, n_eigs: int):
    """
    Get `n_eigs` largest eigenvalues of the Laplacian and `n_eigs` smallest eigenvalues of Laplacian.
    Since the laplacian is SPD,
    """
    assert (L - L.T != 0).sum() == 0
    # Largest eigenvalues
    emax = eigsh(L, k=n_eigs, return_eigenvectors=False)
    # Smallest
    emin = eigsh(L, k=n_eigs, return_eigenvectors=False, sigma=-0.1)
    return np.array([emin, emax])


def quadratic_forms_expm(L: sparse.spmatrix, x: np.ndarray, taus: Iterable[float], check_spd: bool = False) -> np.ndarray:
    """
    Compute quadratic forms z_i^T L z_i, where the z_i vectors are a filtered version of the x signal.
    Define the matrix exponential M_i = exp(-tau_0 * i * L),
    then z_i = M_i x.
    Avoid explicit matrix exponential computation by using specialized algorithm.
    The tau values should be easy to compute from a uniform grid, since we rely on scipy expm_multiply which computes
    on a uniform grid.
    """
    if x.ndim != 1:
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.flatten()
        else:
            raise ValueError(f'x should be a vector')
    if check_spd:
        eigs = get_laplacian_spectral_bounds(L, n_eigs=1)
        if not (eigs > 0).all():
            raise ValueError('L is not SPD')
    # Just to make sure the behaviour is as expected, we later use (-1) * tau
    # maths would remain correct with tau < 0, but user might make a mistake without knowing it
    if not all(tau > 0 for tau in taus):
        raise ValueError('tau values should be positive')
    # We could sort them without error but there would be a mismatch in the order btw input<->output
    if not (np.sort(taus) == np.array(taus)).all():
        raise ValueError('tau values should be sorted')

    # See how many steps required to go from smallest to largest value of tau
    taus = np.array(taus)
    mult = taus / taus[0]
    if not (np.round(mult) == mult).all():
        raise ValueError('does not fit on uniform grid, all elements must be integer multiply of first element')
    # we could take a higher value than 10, just stay safe
    if len(taus) > 10:
        raise ValueError(f'need more than 10 values on the uniform grid, expm_multiply could likely'
                         f'loose precision')

    mult = mult.astype(int)
    kmax = mult[-1]
    exp_tauL_x = expm_multiply(- taus[0] * L, x, start=1, stop=kmax, num=kmax, endpoint=True)
    # Pick on uniform grid only needed values
    exp_tauL_x = exp_tauL_x[mult-1]

    quadratic_forms = np.array([
        z.T @ L @ z for z in exp_tauL_x
    ])
    # xtildes = np.array([
    #     M @ x
    #     for M in matrix_exponentials(L, tau_0, kmax)
    # ])
    # quadratic_forms = np.array([
    #     xtilde.T @ L @ xtilde
    #     for xtilde in xtildes
    # ])

    return quadratic_forms


def matrix_exponentials(L: sparse.spmatrix, tau_0: float, kmax: int):
    """
    DEPRECATED, too slow.
    Compute the matrix exponentials exp(-tau_0 * k * L) with all values of k in (1, 2, ..., kmax)
    Done by first computing matrix eponential E_0 = exp(-tau_0 L), then E_i = E_0^i = E_{i-1} @ E_0
    """
    def sparsify(A: np.ndarray, thres=1e-16):
        A[A < thres] = 0.0

    assert tau_0 > 0
    # Turns out that in our case, matrix exponential is faster with the dense version than the sparse one
    E0 = expm(-tau_0 * L.todense())
    sparsify(E0)
    yield E0
    # Compute powers of E0
    Ek = E0
    for k in range(2, kmax+1):
        Ek = Ek @ E0
        sparsify(Ek)
        yield Ek


if __name__ == '__main__':
    import setup
    import torch
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_scipy_sparse_matrix
    from data_augmentation import add_edge_weights

    # Test matrix exponentials
    # n = 100
    # L = np.random.randn(n, n)
    # zero_r = np.random.choice(n, size=int(n**2*0.8))
    # zero_c = np.random.choice(n, size=int(n**2*0.8))
    # L[zero_r, zero_c] = 0.0
    # L = sparse.coo_matrix(L)
    p = setup.get_dataset_path('CoordToCnc_perimeters')
    file = next(p.glob('*.pt'))
    data = add_edge_weights(torch.load(file))
    A = to_scipy_sparse_matrix(data.edge_index, data.edge_weight)
    L = get_laplacian(A)
    x = data.x.numpy()
    # Submatrix
    sub_idx = 1000
    L = sp.sparse.coo_matrix(L.todense()[:sub_idx, :sub_idx])
    x = x[:sub_idx]

    tau_0 = 1
    kmax = 100
    k = 1

    taus = [1, 3, 5, 10]
    quads = quadratic_forms_expm(L, x, taus)

    for i, tau in enumerate(taus):
        # For some reason, scipy.linalg.expm with L.todense is faster than scipy.sparse.linalg.expm
        E = expm(- tau * L.todense())
        xtilde = E @ x
        q = xtilde.T @ L @ xtilde
        error = np.linalg.norm(q - quads[i]) / quads[i]
        print(f'tau = {tau}, error = {error}')
    # for Ek in matrix_exponentials(L, tau_0, kmax):
    #     true_Ek = expm(-tau_0 * k * L.tocsc())
    #     inf_error = np.abs(true_Ek - Ek).max()
    #     print(f'k={k}, inf_error = {inf_error}')
    #     k += 1
