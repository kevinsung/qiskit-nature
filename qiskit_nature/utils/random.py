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

"""Methods to sample random objects."""

from typing import Any, Optional

import numpy as np
from qiskit.quantum_info import random_hermitian
from qiskit.utils import algorithm_globals
from qiskit_nature.operators.second_quantization import (
    QuadraticHamiltonian as LegacyQuadraticHamiltonian,
)
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian


# TODO see if type of seed can be specified instead of using Any
def parse_random_seed(seed: Any) -> np.random.Generator:
    """Parse a random number generator seed and return a Generator.

    Args:
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The np.random.Generator instance
    """
    if seed is None:
        return algorithm_globals.random
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def random_antisymmetric_matrix(dim: int, seed: Any = None) -> np.ndarray:
    """Return a random antisymmetric matrix.

    Args:
        dim: The width and height of the matrix.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled antisymmetric matrix.
    """
    rng = parse_random_seed(seed)
    mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return mat - mat.T


def random_quadratic_hamiltonian(
    n_orbitals: int, num_conserving: bool = False, seed: Any = None
) -> QuadraticHamiltonian:
    """Generate a random instance of QuadraticHamiltonian.

    Args:
        n_orbitals: the number of orbitals
        num_conserving: whether the Hamiltonian should conserve particle number
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled QuadraticHamiltonian
    """
    rng = parse_random_seed(seed)
    hermitian_part = np.array(random_hermitian(n_orbitals, seed=rng))
    antisymmetric_part = (
        None if num_conserving else random_antisymmetric_matrix(n_orbitals, seed=rng)
    )
    constant = rng.standard_normal()
    return QuadraticHamiltonian(
        hermitian_part=hermitian_part, antisymmetric_part=antisymmetric_part, constant=constant
    )


def random_two_body_tensor(
    n_orbitals: int, rank: Optional[int] = None, seed: Any = None
) -> np.ndarray:
    """Sample a random two-body tensor.

    Args:
        n_orbitals: The number of orbitals.
        rank: Rank of the sampled tensor. The default behavior is to use
            the maximum rank, which is `n_orbitals * (n_orbitals + 1) // 2`.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled two-body tensor.
    """
    rng = parse_random_seed(seed)
    if rank is None:
        rank = n_orbitals * (n_orbitals + 1) // 2
    cholesky_vecs = rng.standard_normal((rank, n_orbitals, n_orbitals))
    cholesky_vecs += cholesky_vecs.transpose((0, 2, 1))
    return np.einsum("ipr,iqs->prqs", cholesky_vecs, cholesky_vecs)


# pylint: disable=invalid-name
def random_legacy_quadratic_hamiltonian(
    n_orbitals: int, num_conserving: bool = False, seed: Any = None
) -> LegacyQuadraticHamiltonian:
    """Generate a random instance of QuadraticHamiltonian.

    Args:
        n_orbitals: the number of orbitals
        num_conserving: whether the Hamiltonian should conserve particle number
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`

    Returns:
        The sampled QuadraticHamiltonian
    """
    rng = parse_random_seed(seed)
    hermitian_part = np.array(random_hermitian(n_orbitals, seed=rng))
    antisymmetric_part = (
        None if num_conserving else random_antisymmetric_matrix(n_orbitals, seed=rng)
    )
    constant = rng.standard_normal()
    return LegacyQuadraticHamiltonian(
        hermitian_part=hermitian_part, antisymmetric_part=antisymmetric_part, constant=constant
    )