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

"""Test linear algebra utilities."""

import itertools
from test import QiskitNatureTestCase
from test.random import random_two_body_tensor

import numpy as np
from ddt import data, ddt, unpack
from qiskit.quantum_info import random_hermitian

from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.utils import (
    givens_matrix,
    low_rank_decomposition,
    low_rank_two_body_decomposition,
)


@ddt
class TestGivensMatrix(QiskitNatureTestCase):
    """Tests for computing Givens rotation matrix."""

    @unpack
    @data((0, 1 + 1j), (1 + 1j, 0), (1 + 2j, 3 - 4j))
    def test_givens_matrix(self, a: complex, b: complex):
        """Test computing Givens rotation matrix."""
        givens_mat = givens_matrix(a, b)
        product = givens_mat @ np.array([a, b])
        np.testing.assert_allclose(product[1], 0.0, atol=1e-8)


class TestLowRank(QiskitNatureTestCase):
    """Tests for low rank decomposition utilities."""

    def test_low_rank_two_body_decomposition(self):
        """Test low rank two-body decomposition."""
        n_orbitals = 5
        one_body_tensor = np.array(random_hermitian(n_orbitals))
        two_body_tensor = random_two_body_tensor(n_orbitals, real=True, chemist=True)
        leaf_tensors, core_tensors = low_rank_two_body_decomposition(two_body_tensor)

        one_body_integrals = OneBodyElectronicIntegrals(ElectronicBasis.SO, one_body_tensor)
        two_body_integrals = TwoBodyElectronicIntegrals(ElectronicBasis.SO, 0.5 * two_body_tensor)
        electronic_energy = ElectronicEnergy([one_body_integrals, two_body_integrals])
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]

        actual = FermionicOp.zero(register_length=n_orbitals)
        for p, q in itertools.product(range(n_orbitals), repeat=2):
            coeff = one_body_tensor[p, q]
            actual += FermionicOp([([("+", p), ("-", q)], coeff)])
        for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
            coeff = 0.0
            for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
                coeff += np.einsum(
                    "i,i,ij,j,j",
                    leaf_tensor[p],
                    leaf_tensor[q],
                    core_tensor,
                    leaf_tensor[r],
                    leaf_tensor[s],
                )
            actual += FermionicOp([([("+", p), ("+", r), ("-", s), ("-", q)], 0.5 * coeff)])

        self.assertTrue(actual.normal_ordered().approx_eq(expected.normal_ordered(), atol=1e-8))

    def test_low_rank_decomposition(self):
        """Test low rank decomposition."""
        n_orbitals = 5
        one_body_tensor = np.array(random_hermitian(n_orbitals))
        two_body_tensor = random_two_body_tensor(n_orbitals, real=True, chemist=True)

        one_body_integrals = OneBodyElectronicIntegrals(ElectronicBasis.SO, one_body_tensor)
        two_body_integrals = TwoBodyElectronicIntegrals(ElectronicBasis.SO, 0.5 * two_body_tensor)
        electronic_energy = ElectronicEnergy([one_body_integrals, two_body_integrals])
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]

        corrected_one_body_tensor, leaf_tensors, core_tensors = low_rank_decomposition(
            one_body_tensor, two_body_tensor
        )
        actual = FermionicOp.zero(register_length=n_orbitals)
        for p, q in itertools.product(range(n_orbitals), repeat=2):
            coeff = corrected_one_body_tensor[p, q]
            actual += FermionicOp([([("+", p), ("-", q)], coeff)])
        for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
            coeff = 0.0
            for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
                coeff += np.einsum(
                    "i,i,ij,j,j",
                    leaf_tensor[p],
                    leaf_tensor[q],
                    core_tensor,
                    leaf_tensor[r],
                    leaf_tensor[s],
                )
            actual += FermionicOp([([("+", p), ("-", q), ("+", r), ("-", s)], 0.5 * coeff)])

        self.assertTrue(actual.normal_ordered().approx_eq(expected.normal_ordered(), atol=1e-8))
