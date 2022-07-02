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

"""Test low rank Trotter step."""

from test import QiskitNatureTestCase

import numpy as np
import scipy.sparse.linalg
from qiskit.quantum_info import (
    random_hermitian,
    random_statevector,
    state_fidelity,
)

from qiskit_nature.circuit.library.low_rank_trotter import SimulateTrotterLowRank
from qiskit_nature.hdf5 import load_from_hdf5
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from test.random import random_two_body_tensor


class TestAsymmetricLowRankTrotterStep(QiskitNatureTestCase):
    """Tests for asymmetric low rank Trotter step."""

    def test_asymmetric_low_rank_trotter_step_random(self):
        """Test asymmetric low rank Trotter step."""
        time = 0.1
        rng = np.random.default_rng()

        # generate Hamiltonian
        n_orbitals = 4
        one_body_tensor = np.array(random_hermitian(n_orbitals, seed=rng))
        two_body_tensor = random_two_body_tensor(n_orbitals, real=True, chemist=True, seed=rng)

        one_body_integrals = OneBodyElectronicIntegrals(ElectronicBasis.SO, one_body_tensor)
        two_body_integrals = TwoBodyElectronicIntegrals(ElectronicBasis.SO, 0.5 * two_body_tensor)
        electronic_energy = ElectronicEnergy([one_body_integrals, two_body_integrals])
        hamiltonian = electronic_energy.second_q_ops()["ElectronicEnergy"]

        # generate random initial state
        n_modes = hamiltonian.register_length
        initial_state = random_statevector(2**n_modes, seed=rng)

        # simulate exact evolution
        hamiltonian_sparse = JordanWignerMapper().map(hamiltonian).to_spmatrix()
        exact_state = scipy.sparse.linalg.expm_multiply(
            -1j * time * hamiltonian_sparse, np.array(initial_state)
        )

        # make sure time is not too small
        assert state_fidelity(exact_state, initial_state) < 0.97

        # simulate Trotter evolution
        circuit = SimulateTrotterLowRank(one_body_tensor, two_body_tensor, time, n_steps=10)
        final_state = initial_state.evolve(circuit)
        fidelity = state_fidelity(final_state, exact_state)
        self.assertGreater(fidelity, 0.99)

    def test_asymmetric_low_rank_trotter_step_h2(self):
        """Test asymmetric low rank Trotter step."""
        time = 1.0

        # load Hamiltonian
        result = load_from_hdf5("test/transformers/second_quantization/electronic/H2_sto3g.hdf5")
        energy = result.get_property("ElectronicEnergy")
        one_body_tensor = energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()
        two_body_tensor = 2 * energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin()
        hamiltonian = energy.second_q_ops()["ElectronicEnergy"]

        # generate random initial state
        n_modes, _ = one_body_tensor.shape
        initial_state = random_statevector(2**n_modes)

        # simulate exact evolution
        hamiltonian_sparse = JordanWignerMapper().map(hamiltonian).to_spmatrix()
        exact_state = scipy.sparse.linalg.expm_multiply(
            -1j * time * hamiltonian_sparse, np.array(initial_state)
        )

        # make sure time is not too small
        assert state_fidelity(exact_state, initial_state) < 0.97

        # simulate Trotter evolution
        circuit = SimulateTrotterLowRank(
            one_body_tensor, two_body_tensor, time, n_steps=5, spin_basis=True
        )
        final_state = initial_state.evolve(circuit)
        fidelity = state_fidelity(final_state, exact_state)
        self.assertGreater(fidelity, 0.999)
