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

import itertools
from typing import Any

import numpy as np
from qiskit.quantum_info import random_hermitian
from qiskit.utils import algorithm_globals

from qiskit_nature.operators.second_quantization import QuadraticHamiltonian


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
    n_orbitals: int, real: bool = False, chemist: bool = False, seed: Any = None
) -> np.ndarray:
    """Sample a random two-body tensor.

    Args:
        n_orbitals: The number of orbitals.
        real: Whether the tensor should be real-valued.
        chemist: Whether the tensor should use the "chemist" indexing convention.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled two-body tensor.
    """
    if chemist:
        return _random_two_body_tensor_chemist(n_orbitals, real, seed)
    return _random_two_body_tensor_physicist(n_orbitals, real, seed)


def _random_two_body_tensor_chemist(n_orbitals: int, real: bool, seed: Any) -> np.ndarray:
    rng = parse_random_seed(seed)
    two_body_tensor = np.zeros(
        (n_orbitals, n_orbitals, n_orbitals, n_orbitals),
        dtype=float if real else complex,
    )
    for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
        coeff = rng.standard_normal()
        if not real and len(set((p, q, r, s))) > 2:
            coeff += 1j * rng.standard_normal()
        two_body_tensor[p, q, r, s] = coeff
        two_body_tensor[r, s, p, q] = coeff
        two_body_tensor[q, p, s, r] = coeff.conjugate()
        two_body_tensor[s, r, q, p] = coeff.conjugate()
        if real:
            two_body_tensor[q, p, r, s] = coeff
            two_body_tensor[s, r, p, q] = coeff
            two_body_tensor[p, q, s, r] = coeff
            two_body_tensor[r, s, q, p] = coeff
    return two_body_tensor


def _random_two_body_tensor_physicist(  # pylint: disable=invalid-name
    n_orbitals: int, real: bool, seed: Any
) -> np.ndarray:
    rng = parse_random_seed(seed)
    two_body_tensor = np.zeros(
        (n_orbitals, n_orbitals, n_orbitals, n_orbitals),
        dtype=float if real else complex,
    )
    for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
        coeff = rng.standard_normal()
        if not real and len(set((p, q, r, s))) > 2:
            coeff += 1j * rng.standard_normal()
        two_body_tensor[p, q, r, s] = coeff
        two_body_tensor[q, p, s, r] = coeff
        two_body_tensor[s, r, q, p] = coeff.conjugate()
        two_body_tensor[r, s, p, q] = coeff.conjugate()
        if real:
            two_body_tensor[r, q, p, s] = coeff
            two_body_tensor[s, p, q, r] = coeff
            two_body_tensor[p, s, r, q] = coeff
            two_body_tensor[q, r, s, p] = coeff
    return two_body_tensor
