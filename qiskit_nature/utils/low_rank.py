# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear algebra utilities."""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import scipy.linalg
import scipy.optimize


def low_rank_decomposition(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    *,
    final_rank: Optional[int] = None,
    spin_basis: bool = False,
    compress: bool = False,
    method: str = "L-BFGS-B",
    options: Optional[dict] = None,
    validate: bool = True,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r"""Low rank decomposition of a molecular Hamiltonian.

    The low rank decomposition acts on a Hamiltonian of the form

    ..math::

        H = \sum_{pq} h_{pq} a^\dagger_p a_q
            + \frac12 \sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_r a_s a_q
            + \text{constant}.

    The Hamiltonian is decomposed into the form

    ..math::

        H = \sum_{pq} \kappa_{pq} a^\dagger_p a_q + \frac12 \sum_t \sum_{ij} Z^{(t)}_{ij} n^{(t)}_i n^{t}_j

    where

    ..math::

        n^{(t)}_i = \sum_{pq} U^{(t)}_{pi} a^\dagger_p a^\dagger_q U^{(t)}_{qi}.

    Here :math:`U^(t)_{ij}` and :math:`Z^(t)_{ij}` are tensors that are output by the decomposition.
    Each matrix :math:`U^(t)` is guaranteed to be unitary so that the :math:`n^{(t)}_i` are
    number operators in a rotated basis.
    The value :math:`t` is the "final rank" of the decomposition and can be chosen by the user.
    The default behavior is to use a full rank of :math:`N^2`, which yields an exact decomposition.
    Specifying a smaller final rank will truncate smaller terms from the decompsition, introducing
    some error.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957

    Args:
        one_body_tensor: The one-body tensor.
        two_body_tensor: The two-body tensor.
        final_rank: The desired number of terms to keep in the decomposition
            of the two-body tensor.
            The default behavior is to include all terms, which yields an
            exact decomposition.
        spin_basis: Whether the tensors are specified in the spin-orbital basis.
            If so, the interaction must be spin-symmetric.
        validate: Whether to check that the input tensors have the correct symmetries.
        atol: Absolute numerical tolerance for input validation.

    Returns:
        The corrected one-body tensor, leaf tensors, core tensors, and constant term
    """
    if spin_basis:
        # tensors are specified in the spin-orbital basis, so reduce to
        # spatial orbital basis assuming a spin-symmetric interaction
        n_modes, _ = one_body_tensor.shape
        one_body_tensor = one_body_tensor[: n_modes // 2, : n_modes // 2]
        two_body_tensor = two_body_tensor[
            : n_modes // 2, : n_modes // 2, : n_modes // 2, : n_modes // 2
        ]
        # spin basis uses physicist indexing, need to convert to chemist
        two_body_tensor = np.transpose(two_body_tensor, (2, 1, 0, 3))

    sign = 1 if spin_basis else -1
    corrected_one_body_tensor = one_body_tensor + sign * 0.5 * np.einsum("prqr", two_body_tensor)
    if compress:
        leaf_tensors, core_tensors = low_rank_compressed_two_body_decomposition(
            two_body_tensor,
            final_rank=final_rank,
            method=method,
            options=options,
            validate=validate,
            atol=atol,
        )
    else:
        leaf_tensors, core_tensors = low_rank_two_body_decomposition(
            two_body_tensor, final_rank=final_rank, validate=validate, atol=atol
        )

    if spin_basis:
        # expand back to spin-orbital basis
        corrected_one_body_tensor = np.kron(np.eye(2), corrected_one_body_tensor)
        leaf_tensors = np.kron(np.kron(np.ones((1, 1, 1)), np.eye(2)), leaf_tensors)
        core_tensors = -np.kron(np.ones((1, 2, 2)), core_tensors)

    return corrected_one_body_tensor, leaf_tensors, core_tensors


# TODO add truncation threshold option
# TODO add support for complex orbitals
def low_rank_two_body_decomposition(
    two_body_tensor: np.ndarray,
    *,
    final_rank: Optional[int] = None,
    validate: bool = True,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Low rank decomposition of a two-body tensor.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957

    Args:
        two_body_tensor: The two-body tensor to decompose. The tensor indices
            should be ordered according to the "chemist" convention.
        final_rank: The desired number of terms to keep in the decomposition.
            The default behavior is to include all terms, which yields an
            exact decomposition.
        validate: Whether to check that the input tensors have the correct symmetries.
        atol: Absolute numerical tolerance for input validation.
    """
    n_modes, _, _, _ = two_body_tensor.shape
    full_rank = n_modes**2
    if final_rank is None:
        final_rank = full_rank
    reshaped_tensor = np.reshape(two_body_tensor, (n_modes**2, n_modes**2))

    if validate:
        if not np.all(np.isreal(reshaped_tensor)):
            raise ValueError("Two-body tensor must be real.")
        if not np.allclose(reshaped_tensor, reshaped_tensor.T, atol=atol):
            raise ValueError("Two-body tensor must be symmetric.")

    outer_eigs, outer_vecs = np.linalg.eigh(reshaped_tensor)
    indices = np.argsort(np.abs(outer_eigs))[-1 : -final_rank - 1 : -1]
    leaf_tensors = []
    core_tensors = []
    for i in indices:
        mat = np.reshape(outer_vecs[:, i], (n_modes, n_modes))
        inner_eigs, inner_vecs = np.linalg.eigh(mat)
        core_tensor = outer_eigs[i] * np.outer(inner_eigs, inner_eigs)
        leaf_tensors.append(inner_vecs)
        core_tensors.append(core_tensor)

    return np.array(leaf_tensors), np.array(core_tensors)


def low_rank_z_representation(
    leaf_tensors: np.ndarray, core_tensors: np.ndarray
) -> tuple[np.ndarray, float]:
    one_body_correction = 0.25 * (
        np.einsum("tij,tpi,tqi->pq", core_tensors, leaf_tensors, leaf_tensors.conj())
        + np.einsum("tij,tpj,tqj->pq", core_tensors, leaf_tensors, leaf_tensors.conj())
    )
    constant_correction = 0.125 * (np.einsum("ijj->", core_tensors) - np.sum(core_tensors))
    return one_body_correction, constant_correction


def low_rank_optimal_core_tensors(
    two_body_tensor: np.ndarray, leaf_tensors: np.ndarray, cutoff_threshold: float = 1e-8
) -> np.ndarray:
    """Compute optimal low rank core tensors given fixed leaf tensors.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957

    Args:
        two_body_tensor: The two-body tensor to decompose. The tensor indices
            should be ordered according to the "chemistry" convention.
        leaf_tensors: The leaf tensors of the low rank decomposition.
        cutoff_threshold: Eigenvalues smaller than this value will be ignored
            when solving the least-squares problem.
    """
    n_modes, _, _, _ = two_body_tensor.shape
    n_tensors, _, _ = leaf_tensors.shape

    dim = n_tensors * n_modes**2
    target = np.einsum(
        "pqrs,tpk,tqk,trl,tsl->tkl",
        two_body_tensor,
        leaf_tensors,
        leaf_tensors,
        leaf_tensors,
        leaf_tensors,
    )
    target = np.reshape(target, (dim,))
    coeffs = np.zeros((n_tensors, n_modes, n_modes, n_tensors, n_modes, n_modes))
    for i in range(n_tensors):
        for j in range(i, n_tensors):
            metric = (leaf_tensors[i].T @ leaf_tensors[j]) ** 2
            coeffs[i, :, :, j, :, :] = np.einsum("kl,mn->kmln", metric, metric)
            coeffs[j, :, :, i, :, :] = np.einsum("kl,mn->kmln", metric.T, metric.T)
    coeffs = np.reshape(coeffs, (dim, dim))

    eigs, vecs = np.linalg.eigh(coeffs)
    pseudoinverse = np.zeros_like(eigs)
    pseudoinverse[eigs > cutoff_threshold] = eigs[eigs > cutoff_threshold] ** -1
    solution = vecs @ (vecs.T @ target * pseudoinverse)

    return np.reshape(solution, (n_tensors, n_modes, n_modes))


def low_rank_compressed_two_body_decomposition(
    two_body_tensor,
    *,
    final_rank: Optional[int] = None,
    method="L-BFGS-B",
    options: Optional[dict] = None,
    validate: bool = True,
    atol: float = 1e-8,
):
    leaf_tensors, _ = low_rank_two_body_decomposition(
        two_body_tensor, final_rank=final_rank, validate=validate, atol=atol
    )
    n_tensors, n_modes, _ = leaf_tensors.shape

    def fun(x):
        leaf_tensors = _params_to_leaf_tensors(x, n_tensors, n_modes)
        core_tensors = low_rank_optimal_core_tensors(two_body_tensor, leaf_tensors)
        diff = two_body_tensor - np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        return 0.5 * np.sum(diff**2)

    def jac(x):
        leaf_tensors = _params_to_leaf_tensors(x, n_tensors, n_modes)
        core_tensors = low_rank_optimal_core_tensors(two_body_tensor, leaf_tensors)
        diff = two_body_tensor - np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        grad_leaf = -4.0 * np.einsum(
            "pqrs,tqk,tkl,trl,tsl->tpk",
            diff,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        leaf_logs = _params_to_leaf_logs(x, n_tensors, n_modes)
        return np.ravel([_gradient(log, grad) for log, grad in zip(leaf_logs, grad_leaf)])

    x0 = _leaf_tensors_to_params(leaf_tensors)
    result = scipy.optimize.minimize(fun, x0, method=method, jac=jac, options=options)
    leaf_tensors = _params_to_leaf_tensors(result.x, n_tensors, n_modes)
    core_tensors = low_rank_optimal_core_tensors(two_body_tensor, leaf_tensors)

    return leaf_tensors, core_tensors


def _leaf_tensors_to_params(leaf_tensors: np.ndarray):
    _, n_modes, _ = leaf_tensors.shape
    leaf_logs = [np.real(scipy.linalg.logm(mat)) for mat in np.real(leaf_tensors)]
    triu_indices = np.triu_indices(n_modes, k=1)
    return np.ravel([leaf_log[triu_indices] for leaf_log in leaf_logs])


def _params_to_leaf_logs(params: np.ndarray, n_tensors: int, n_modes: int):
    leaf_logs = np.zeros((n_tensors, n_modes, n_modes))
    triu_indices = np.triu_indices(n_modes, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_logs[i][triu_indices] = params[i * param_length : (i + 1) * param_length]
        leaf_logs[i] -= leaf_logs[i].T
    return leaf_logs


def _params_to_leaf_tensors(params: np.ndarray, n_tensors: int, n_modes: int):
    leaf_logs = _params_to_leaf_logs(params, n_tensors, n_modes)
    return np.array([_expm_antisymmetric(mat) for mat in leaf_logs])


def _expm_antisymmetric(mat: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mat)
    return np.real(vecs @ np.diag(np.exp(1j * eigs)) @ vecs.T.conj())


def _gradient(mat: np.ndarray, grad_factor: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mat)
    eig_i, eig_j = np.meshgrid(eigs, eigs, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        coeffs = -1j * (np.exp(1j * eig_i) - np.exp(1j * eig_j)) / (eig_i - eig_j)
    coeffs[eig_i == eig_j] = np.exp(1j * eig_i[eig_i == eig_j])
    grad = vecs.conj() @ (vecs.T @ grad_factor @ vecs.conj() * coeffs) @ vecs.T
    grad -= grad.T
    n_modes, _ = mat.shape
    triu_indices = np.triu_indices(n_modes, k=1)
    return np.real(grad[triu_indices])
