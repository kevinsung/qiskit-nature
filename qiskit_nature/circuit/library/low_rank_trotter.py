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

"""Trotter Hamiltonian simulation via low rank decomposition."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Qubit
from qiskit.circuit.library import RZGate, RZZGate

from qiskit_nature.circuit.library import BogoliubovTransform
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import QuadraticHamiltonian
from qiskit_nature.utils import low_rank_decomposition, low_rank_z_representation


class SimulateTrotterLowRank(QuantumCircuit):
    r"""Trotter Hamiltonian simulation via low rank decomposition.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    """

    def __init__(
        self,
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        time: float,
        *,
        n_steps: int = 1,
        final_rank: Optional[int] = None,
        spin_basis: bool = False,
        qubit_converter: Optional[QubitConverter] = None,
        **circuit_kwargs,
    ) -> None:
        r"""
        Args:
            one_body_tensor: The one-body tensor of the Hamiltonian.
            two_body_tensor: The two-body tensor of the Hamiltonian.
            time: Simulation time.
            qubit_converter: The qubit converter. The default behavior is to create
                one using the call `QubitConverter(JordanWignerMapper())`.
            circuit_kwargs: Keyword arguments to pass to the QuantumCircuit initializer.
        """
        if qubit_converter is None:
            qubit_converter = QubitConverter(JordanWignerMapper())

        n, _ = one_body_tensor.shape
        register = QuantumRegister(n)
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_converter.mapper, JordanWignerMapper):
            trotter_step = AsymmetricLowRankTrotterStepJW(
                one_body_tensor, two_body_tensor, final_rank=final_rank, spin_basis=spin_basis
            )
            operations = _simulate_trotter(register, trotter_step, time, n_steps)
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError(
                "Currently, only the Jordan-Wigner Transform is supported. "
                "Please use "
                "qiskit_nature.mappers.second_quantization.JordanWignerMapper "
                "to construct the qubit mapper."
            )


# TODO add higher-order Trotter formulas
def _simulate_trotter(
    register: QuantumRegister,
    trotter_step: AsymmetricLowRankTrotterStepJW,
    time: float,
    n_steps: int,
) -> Iterator[tuple[Instruction, Sequence[Qubit]]]:
    step_time = time / n_steps
    for _ in range(n_steps):
        yield from trotter_step.trotter_step(register, step_time)


class AsymmetricLowRankTrotterStepJW:
    r"""Trotter Hamiltonian simulation via low rank decomposition.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    """

    def __init__(
        self,
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        final_rank: Optional[int] = None,
        spin_basis: bool = False,
    ) -> None:
        r"""
        Args:
            one_body_tensor: The one-body tensor of the Hamiltonian.
            two_body_tensor: The two-body tensor of the Hamiltonian.
            time: Simulation time.
        """
        self.spin_basis = spin_basis
        corrected_one_body_tensor, leaf_tensors, core_tensors = low_rank_decomposition(
            one_body_tensor, two_body_tensor, final_rank=final_rank, spin_basis=spin_basis
        )
        one_body_correction, constant_correction = low_rank_z_representation(
            leaf_tensors, core_tensors
        )
        corrected_one_body_tensor += one_body_correction
        if spin_basis:
            # tensors are specified in the spin-orbital basis, so reduce to
            # spatial orbital basis assuming a spin-symmetric interaction
            n_modes, _ = one_body_tensor.shape
            n_modes //= 2
            corrected_one_body_tensor = corrected_one_body_tensor[:n_modes, :n_modes]
            leaf_tensors = leaf_tensors[:, :n_modes, :n_modes]
            core_tensors = core_tensors[:, :n_modes, :n_modes]
        self.one_body_tensor = corrected_one_body_tensor
        self.leaf_tensors = leaf_tensors
        self.core_tensors = core_tensors
        self.constant = constant_correction

    def trotter_step(
        self, register: QuantumRegister, time: float
    ) -> Iterator[tuple[Instruction, Sequence[Qubit]]]:
        if self.spin_basis:
            yield from self._trotter_step_spin(register, time)
        else:
            yield from self._trotter_step_spatial(register, time)

    def _trotter_step_spin(
        self, register: QuantumRegister, time: float
    ) -> Iterator[tuple[Instruction, Sequence[Qubit]]]:
        n_qubits = len(register)
        n_modes = n_qubits // 2

        # compute basis change that diagonalizes one-body term
        transformation_matrix, orbital_energies, _ = QuadraticHamiltonian(
            self.one_body_tensor
        ).diagonalizing_bogoliubov_transform()

        # change to the basis in which the one-body term is diagonal
        bog_circuit = BogoliubovTransform(transformation_matrix.T.conj())
        yield bog_circuit, register[:n_modes]
        yield bog_circuit, register[n_modes:]

        # simulate the one-body terms
        for i in range(n_modes):
            rz_gate = RZGate(-orbital_energies[i] * time)
            yield rz_gate, (register[i],)
            yield rz_gate, (register[n_modes + i],)

        # simulate the two-body terms
        prior_transformation_matrix = transformation_matrix
        for leaf_tensor, core_tensor in zip(self.leaf_tensors, self.core_tensors):
            # change basis
            merged_transformation_matrix = prior_transformation_matrix @ leaf_tensor.conj()
            bog_circuit = BogoliubovTransform(merged_transformation_matrix)
            yield bog_circuit, register[:n_modes]
            yield bog_circuit, register[n_modes:]
            # simulate two-body terms
            for i, j in itertools.combinations(range(n_qubits), 2):
                yield RZZGate(
                    0.25
                    * (
                        core_tensor[i % n_modes, j % n_modes]
                        + core_tensor[j % n_modes, i % n_modes]
                    )
                    * time
                ), (
                    register[i],
                    register[j],
                )
            # update prior basis change matrix
            prior_transformation_matrix = leaf_tensor.T

        # undo final basis change
        bog_circuit = BogoliubovTransform(prior_transformation_matrix)
        yield bog_circuit, register[:n_modes]
        yield bog_circuit, register[n_modes:]

    def _trotter_step_spatial(
        self, register: QuantumRegister, time: float
    ) -> Iterator[tuple[Instruction, Sequence[Qubit]]]:
        n_qubits = len(register)

        # compute basis change that diagonalizes one-body term
        transformation_matrix, orbital_energies, _ = QuadraticHamiltonian(
            self.one_body_tensor
        ).diagonalizing_bogoliubov_transform()

        # change to the basis in which the one-body term is diagonal
        yield BogoliubovTransform(transformation_matrix.T.conj()), register

        # simulate the one-body terms
        for qubit, energy in zip(register, orbital_energies):
            yield RZGate(-energy * time), (qubit,)

        # simulate the two-body terms
        prior_transformation_matrix = transformation_matrix
        for leaf_tensor, core_tensor in zip(self.leaf_tensors, self.core_tensors):
            # change basis
            merged_transformation_matrix = prior_transformation_matrix @ leaf_tensor.conj()
            yield BogoliubovTransform(merged_transformation_matrix), register
            # simulate two-body terms
            for i, j in itertools.combinations(range(n_qubits), 2):
                yield RZZGate(0.25 * (core_tensor[i, j] + core_tensor[j, i]) * time), (
                    register[i],
                    register[j],
                )
            # update prior basis change matrix
            prior_transformation_matrix = leaf_tensor.T

        # undo final basis change
        yield BogoliubovTransform(prior_transformation_matrix), register
