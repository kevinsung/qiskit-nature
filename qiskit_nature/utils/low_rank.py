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

from typing import Optional

import numpy as np


def low_rank_decomposition(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    *,
    final_rank: Optional[int] = None,
    spin_basis: bool = False,
    validate: bool = True,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low rank decomposition of a molecular Hamiltonian.

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
        The corrected one-body tensor, leaf tensors, and core tensors
    """
    if spin_basis:
        # tensors are specified in the spin-orbital basis, so reduce to
        # spatial orbital basis assuming a spin-symmetric interaction
        n_modes, _ = one_body_tensor.shape
        one_body_tensor = one_body_tensor[: n_modes // 2, : n_modes // 2]
        two_body_tensor = two_body_tensor[
            : n_modes // 2, : n_modes // 2, : n_modes // 2, : n_modes // 2
        ]
        two_body_tensor = np.transpose(two_body_tensor, (2, 1, 0, 3))

    sign = 1 if spin_basis else -1
    corrected_one_body_tensor = one_body_tensor + sign * 0.5 * np.einsum("prqr", two_body_tensor)
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
    if final_rank is None:
        final_rank = n_modes**2
    reshaped_tensor = np.reshape(two_body_tensor, (n_modes**2, n_modes**2))

    if validate:
        if not np.all(np.isreal(reshaped_tensor)):
            raise ValueError("Two-body tensor must be real.")
        if not np.allclose(reshaped_tensor, reshaped_tensor.T, atol=atol):
            raise ValueError("Two-body tensor must be symmetric.")

    outer_eigs, outer_vecs = np.linalg.eigh(reshaped_tensor)
    leaf_tensors = []
    core_tensors = []
    for i in range(final_rank):
        mat = np.reshape(outer_vecs[:, -i - 1], (n_modes, n_modes))
        inner_eigs, inner_vecs = np.linalg.eigh(mat)
        core_tensor = outer_eigs[-i - 1] * np.outer(inner_eigs, inner_eigs)
        leaf_tensors.append(inner_vecs)
        core_tensors.append(core_tensor)

    return np.array(leaf_tensors), np.array(core_tensors)


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
