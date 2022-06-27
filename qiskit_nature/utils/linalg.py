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

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import XGate, XXPlusYYGate

# HACK: Sphinx fails to handle "ellipsis"
# See https://github.com/python/typing/issues/684
if TYPE_CHECKING:
    _SliceAtom = Union[int, slice, "ellipsis"]
else:
    _SliceAtom = Union[int, slice, type(Ellipsis)]

_Slice = Union[_SliceAtom, Tuple[_SliceAtom, ...]]


def apply_matrix_to_slices(
    target: np.ndarray, matrix: np.ndarray, slices: Sequence[_Slice]
) -> np.ndarray:
    """Apply a matrix to slices of a target tensor.

    Args:
        target: The tensor containing the slices on which to apply the matrix.
        matrix: The matrix to apply to slices of the target tensor.
        slices: The slices of the target tensor on which to apply the matrix.

    Returns:
        The tensor resulting from applying the matrix to the slices of the target tensor.
    """
    result = target.copy()
    for i, slice_i in enumerate(slices):
        result[slice_i] *= matrix[i, i]
        for j, slice_j in enumerate(slices):
            if j != i:
                result[slice_i] += target[slice_j] * matrix[i, j]
    return result


def givens_matrix(a: complex, b: complex) -> np.ndarray:
    r"""Compute the Givens rotation to zero out a row entry.

    Returns a :math:`2 \times 2` unitary matrix G that satisfies

    .. math::
        G
        \begin{pmatrix}
            a \\
            b
        \end{pmatrix}
        =
        \begin{pmatrix}
            r \\
            0
        \end{pmatrix}

    where :math:`r` is a complex number.

    References:
        - `<https://en.wikipedia.org/wiki/Givens_rotation#Stable_calculation>`_
        - `<https://www.netlib.org/lapack/lawnspdf/lawn148.pdf>`_

    Args:
        a: A complex number representing the first row entry
        b: A complex number representing the second row entry

    Returns:
        The Givens rotation matrix.
    """
    # Handle case that a is zero
    if np.isclose(a, 0.0):
        cosine = 0.0
        sine = 1.0
    # Handle case that b is zero and a is nonzero
    elif np.isclose(b, 0.0):
        cosine = 1.0
        sine = 0.0
    # Handle case that a and b are both nonzero
    else:
        hypotenuse = np.hypot(abs(a), abs(b))
        cosine = abs(a) / hypotenuse
        sign_a = a / abs(a)
        sine = sign_a * b.conjugate() / hypotenuse

    return np.array([[cosine, sine], [-sine.conjugate(), cosine]])


def fermionic_gaussian_decomposition_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> tuple[list[tuple[Gate, tuple[Qubit, ...]]], np.ndarray]:
    """Matrix decomposition to prepare a fermionic Gaussian state under the Jordan-Wigner Transform.

    Reference: `arXiv:1711.05395`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395

    Args:
        register: The register containing the qubits to use
        transformation_matrix: The transformation matrix describing
            the fermionic Gaussian state

    Returns:
        - givens rotation decomposition
        - left unitary left over from the decomposition
    """
    n, _ = transformation_matrix.shape

    # transform matrix to align with exposition in arXiv:1711.05395
    left = transformation_matrix[:, :n]
    right = transformation_matrix[:, n:]
    current_matrix = np.block([right.conj(), left.conj()]).astype(complex, copy=False)

    # compute left_unitary
    left_unitary = np.eye(n, dtype=complex)
    for j in range(n - 1):
        # Zero out entries in column k
        for i in range(n - 1 - j):
            # Zero out entry in row l if needed
            if not np.isclose(current_matrix[i, j], 0.0):
                givens_mat = givens_matrix(current_matrix[i + 1, j], current_matrix[i, j])
                current_matrix = apply_matrix_to_slices(current_matrix, givens_mat, [i + 1, i])
                left_unitary = apply_matrix_to_slices(left_unitary, givens_mat, [i + 1, i])

    # decompose matrix into Givens rotations and particle-hole transformations
    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    for i in range(n):
        # zero out the columns in row i
        for j in range(n - 1 - i, n):
            if not np.isclose(current_matrix[i, j], 0.0):
                if j == n - 1:
                    # particle-hole transformation
                    decomposition.append((XGate(), (register[-1],)))
                    _swap_columns(current_matrix, n - 1, 2 * n - 1)
                else:
                    # compute Givens rotation
                    givens_mat = givens_matrix(current_matrix[i, j + 1], current_matrix[i, j])
                    theta = np.arccos(np.real(givens_mat[0, 0]))
                    phi = np.angle(givens_mat[0, 1])
                    # add operations
                    decomposition.append(
                        (XXPlusYYGate(2 * theta, phi - np.pi / 2), (register[j], register[j + 1]))
                    )
                    # update matrix
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat,
                        [(Ellipsis, j + 1), (Ellipsis, j)],
                    )
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat.conj(),
                        [(Ellipsis, n + j + 1), (Ellipsis, n + j)],
                    )

    for i in range(n):
        left_unitary[i] *= current_matrix[i, n + i].conj()

    return decomposition, left_unitary


def _swap_columns(matrix: np.ndarray, i: int, j: int) -> None:
    """Swap columns of a matrix, mutating it."""
    column_i = matrix[:, i].copy()
    column_j = matrix[:, j].copy()
    matrix[:, i], matrix[:, j] = column_j, column_i


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
