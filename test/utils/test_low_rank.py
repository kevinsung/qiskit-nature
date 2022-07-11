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

"""Test low rank utilities."""

import itertools
from test import QiskitNatureTestCase
from test.random import random_two_body_tensor

import numpy as np
from qiskit.quantum_info import random_hermitian

from qiskit_nature.hdf5 import load_from_hdf5
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.utils.low_rank import (
    _low_rank_compressed_two_body_decomposition,
    _low_rank_optimal_core_tensors,
    _low_rank_two_body_decomposition,
    low_rank_decomposition,
    low_rank_z_representation,
)


class TestLowRank(QiskitNatureTestCase):
    """Tests for low rank decomposition utilities."""

    def test_low_rank_decomposition(self):
        """Test low rank decomposition."""
        n_orbitals = 3
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
        for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
            num_ops = []
            for i in range(n_orbitals):
                num_op = FermionicOp.zero(register_length=n_orbitals)
                for p, q in itertools.product(range(n_orbitals), repeat=2):
                    num_op += FermionicOp(
                        [([("+", p), ("-", q)], leaf_tensor[p, i] * leaf_tensor[q, i].conj())]
                    )
                num_ops.append(num_op)
            for i, j in itertools.product(range(n_orbitals), repeat=2):
                actual += 0.5 * core_tensor[i, j] * num_ops[i] @ num_ops[j]

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    def test_low_rank_decomposition_spin(self):
        """Test low rank decomposition with spin."""
        result = load_from_hdf5("test/transformers/second_quantization/electronic/H2_sto3g.hdf5")
        energy = result.get_property("ElectronicEnergy")
        expected = energy.second_q_ops()["ElectronicEnergy"]

        one_body_tensor = energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()
        two_body_tensor = 2 * energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin()
        n_orbitals, _ = one_body_tensor.shape

        corrected_one_body_tensor, leaf_tensors, core_tensors = low_rank_decomposition(
            one_body_tensor, two_body_tensor, spin_basis=True
        )
        actual = FermionicOp.zero(register_length=n_orbitals)
        for p, q in itertools.product(range(n_orbitals), repeat=2):
            coeff = corrected_one_body_tensor[p, q]
            actual += FermionicOp([([("+", p), ("-", q)], coeff)])
        for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
            num_ops = []
            for i in range(n_orbitals):
                num_op = FermionicOp.zero(register_length=n_orbitals)
                for p, q in itertools.product(range(n_orbitals), repeat=2):
                    num_op += FermionicOp(
                        [([("+", p), ("-", q)], leaf_tensor[p, i] * leaf_tensor[q, i].conj())]
                    )
                num_ops.append(num_op)
            for i, j in itertools.product(range(n_orbitals), repeat=2):
                actual += 0.5 * core_tensor[i, j] * num_ops[i] @ num_ops[j]

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    def test_low_rank_decomposition_truncation(self):
        """Test low rank decomposition truncation."""
        result = load_from_hdf5("test/transformers/second_quantization/electronic/H2_sto3g.hdf5")
        energy = result.get_property("ElectronicEnergy")
        expected = energy.second_q_ops()["ElectronicEnergy"]

        one_body_tensor = energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()
        two_body_tensor = 2 * energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin()
        n_orbitals, _ = one_body_tensor.shape

        final_rank = 3
        corrected_one_body_tensor, leaf_tensors, core_tensors = low_rank_decomposition(
            one_body_tensor, two_body_tensor, spin_basis=True, final_rank=final_rank
        )
        self.assertEqual(len(leaf_tensors), final_rank)
        self.assertEqual(len(core_tensors), final_rank)

        actual = FermionicOp.zero(register_length=n_orbitals)
        for p, q in itertools.product(range(n_orbitals), repeat=2):
            coeff = corrected_one_body_tensor[p, q]
            actual += FermionicOp([([("+", p), ("-", q)], coeff)])
        for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
            num_ops = []
            for i in range(n_orbitals):
                num_op = FermionicOp.zero(register_length=n_orbitals)
                for p, q in itertools.product(range(n_orbitals), repeat=2):
                    num_op += FermionicOp(
                        [([("+", p), ("-", q)], leaf_tensor[p, i] * leaf_tensor[q, i].conj())]
                    )
                num_ops.append(num_op)
            for i, j in itertools.product(range(n_orbitals), repeat=2):
                actual += 0.5 * core_tensor[i, j] * num_ops[i] @ num_ops[j]

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    def test_low_rank_decomposition_z_representation(self):
        """Test low rank decomposition equation "Z" representation."""
        n_orbitals = 3
        one_body_tensor = np.array(random_hermitian(n_orbitals))
        two_body_tensor = random_two_body_tensor(n_orbitals, real=True, chemist=True)

        one_body_integrals = OneBodyElectronicIntegrals(ElectronicBasis.SO, one_body_tensor)
        two_body_integrals = TwoBodyElectronicIntegrals(ElectronicBasis.SO, 0.5 * two_body_tensor)
        electronic_energy = ElectronicEnergy([one_body_integrals, two_body_integrals])
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]

        corrected_one_body_tensor, leaf_tensors, core_tensors = low_rank_decomposition(
            one_body_tensor, two_body_tensor
        )
        one_body_correction, constant = low_rank_z_representation(leaf_tensors, core_tensors)
        corrected_one_body_tensor += one_body_correction
        actual = constant * FermionicOp.one(register_length=n_orbitals)
        for p, q in itertools.product(range(n_orbitals), repeat=2):
            coeff = corrected_one_body_tensor[p, q]
            actual += FermionicOp([([("+", p), ("-", q)], coeff)])
        for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
            num_ops = []
            for i in range(n_orbitals):
                num_op = FermionicOp.zero(register_length=n_orbitals)
                for p, q in itertools.product(range(n_orbitals), repeat=2):
                    num_op += FermionicOp(
                        [([("+", p), ("-", q)], leaf_tensor[p, i] * leaf_tensor[q, i].conj())]
                    )
                num_ops.append(num_op)
            for i, j in itertools.combinations(range(n_orbitals), 2):
                z1 = (  # pylint: disable=invalid-name
                    FermionicOp.one(register_length=n_orbitals) - 2 * num_ops[i]
                )
                z2 = (  # pylint: disable=invalid-name
                    FermionicOp.one(register_length=n_orbitals) - 2 * num_ops[j]
                )
                actual += 0.125 * (core_tensor[i, j]) * z1 @ z2
                actual += 0.125 * (core_tensor[j, i]) * z1 @ z2

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    def test_low_rank_decomposition_z_representation_spin(self):
        """Test low rank decomposition equation "Z" representation with spin."""
        result = load_from_hdf5("test/transformers/second_quantization/electronic/H2_sto3g.hdf5")
        energy = result.get_property("ElectronicEnergy")
        expected = energy.second_q_ops()["ElectronicEnergy"]

        one_body_tensor = energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()
        two_body_tensor = 2 * energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin()
        n_orbitals, _ = one_body_tensor.shape

        corrected_one_body_tensor, leaf_tensors, core_tensors = low_rank_decomposition(
            one_body_tensor, two_body_tensor, spin_basis=True
        )

        one_body_correction, constant = low_rank_z_representation(leaf_tensors, core_tensors)
        corrected_one_body_tensor += one_body_correction
        actual = constant * FermionicOp.one(register_length=n_orbitals)
        for p, q in itertools.product(range(n_orbitals), repeat=2):
            coeff = corrected_one_body_tensor[p, q]
            actual += FermionicOp([([("+", p), ("-", q)], coeff)])
        for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
            num_ops = []
            for i in range(n_orbitals):
                num_op = FermionicOp.zero(register_length=n_orbitals)
                for p, q in itertools.product(range(n_orbitals), repeat=2):
                    num_op += FermionicOp(
                        [([("+", p), ("-", q)], leaf_tensor[p, i] * leaf_tensor[q, i].conj())]
                    )
                num_ops.append(num_op)
            for i, j in itertools.combinations(range(n_orbitals), 2):
                z1 = (  # pylint: disable=invalid-name
                    FermionicOp.one(register_length=n_orbitals) - 2 * num_ops[i]
                )
                z2 = (  # pylint: disable=invalid-name
                    FermionicOp.one(register_length=n_orbitals) - 2 * num_ops[j]
                )
                actual += 0.125 * (core_tensor[i, j]) * z1 @ z2
                actual += 0.125 * (core_tensor[j, i]) * z1 @ z2

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    def test_low_rank_decomposition_optimal_core_tensors(self):
        """Test low rank decomposition optimal core tensors."""
        n_orbitals = 4
        one_body_tensor = np.array(random_hermitian(n_orbitals))
        two_body_tensor = random_two_body_tensor(n_orbitals, real=True, chemist=True)

        _, leaf_tensors, core_tensors = low_rank_decomposition(one_body_tensor, two_body_tensor)
        core_tensors = _low_rank_optimal_core_tensors(
            two_body_tensor, leaf_tensors, cutoff_threshold=1e-8
        )

        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)

    def test_low_rank_two_body_decomposition(self):
        """Test low rank two-body decomposition."""
        n_orbitals = 4
        two_body_tensor = random_two_body_tensor(n_orbitals, real=True, chemist=True)
        leaf_tensors, core_tensors = _low_rank_two_body_decomposition(two_body_tensor)

        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)

    def test_low_rank_compressed_two_body_decomposition(self):
        """Test low rank compressed two-body decomposition."""
        n_orbitals = 4
        two_body_tensor = random_two_body_tensor(n_orbitals, real=True, chemist=True)
        leaf_tensors, core_tensors = _low_rank_compressed_two_body_decomposition(two_body_tensor)

        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)

    def test_low_rank_decomposition_compressed_spin(self):
        """Test compressed low rank decomposition with spin."""
        result = load_from_hdf5(
            "test/transformers/second_quantization/electronic/BeH_sto3g_reduced.hdf5"
        )
        energy = result.get_property("ElectronicEnergy")
        expected = energy.second_q_ops()["ElectronicEnergy"]

        one_body_tensor = energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()
        two_body_tensor = 2 * energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin()
        n_orbitals, _ = one_body_tensor.shape

        corrected_one_body_tensor, leaf_tensors, core_tensors = low_rank_decomposition(
            one_body_tensor, two_body_tensor, final_rank=4, spin_basis=True, optimize=True
        )
        actual = FermionicOp.zero(register_length=n_orbitals)
        for p, q in itertools.product(range(n_orbitals), repeat=2):
            coeff = corrected_one_body_tensor[p, q]
            actual += FermionicOp([([("+", p), ("-", q)], coeff)])
        for leaf_tensor, core_tensor in zip(leaf_tensors, core_tensors):
            num_ops = []
            for i in range(n_orbitals):
                num_op = FermionicOp.zero(register_length=n_orbitals)
                for p, q in itertools.product(range(n_orbitals), repeat=2):
                    num_op += FermionicOp(
                        [([("+", p), ("-", q)], leaf_tensor[p, i] * leaf_tensor[q, i].conj())]
                    )
                num_ops.append(num_op)
            for i, j in itertools.product(range(n_orbitals), repeat=2):
                actual += 0.5 * core_tensor[i, j] * num_ops[i] @ num_ops[j]

        diff = (actual - expected).normal_ordered().simplify()
        self.assertLess(diff.induced_norm(), 1e-3)
