# %%
import os
import pytest
import numpy as np
import jax.numpy as jnp
import jax
import haiku as hk

import supergrad
from supergrad.helper.compute_density_evolve import DensityEvolve
from supergrad.quantum_system import KronObj
from supergrad.utils.operators import sigmax, sigmaz, destroy
from supergrad.utils.dissipation_channels import (
    amplitude_damping_operator,
    phase_damping_operator
)
from supergrad.scgraph.graph import SCGraph

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def _initialize_density_evo_haiku(density_evo):
    """Initialize DensityEvolve within Haiku transform context and return params."""
    @hk.transform
    def init_system():
        density_evo._init_quantum_system()
        return

    rng = jax.random.PRNGKey(0)
    init_system.init(rng)
    return density_evo.ls_params()


class DensityEvolveHelper(supergrad.Helper):
    def _init_quantum_system(self):
        # Create a simple 2-qubit graph
        self.graph = SCGraph()
        self.graph.add_node('q0', system_type='transmon', ec=0.2, ej=5.0)
        self.graph.add_node('q1', system_type='transmon', ec=0.2, ej=5.0)
        self.graph.add_edge('q0', 'q1', capacitive_coupling={'strength': 0.1})

        self.density_evo = DensityEvolve(
            graph=self.graph,
            truncated_dim=2,
            add_random=False,
            solver='ode_expm',
            options={'astep': 100, 'trotter_order': 1}
        )
        self.density_evo._init_quantum_system()
        self.hilbertspace = self.density_evo.hilbertspace

    def get_dims(self):
        return self.hilbertspace.dim

    def _get_graph(self):
        return self.graph

    def _get_density_evo(self):
        return self.density_evo

    # Common methods that tests might need
    def create_default(self):
        """Create default density matrices."""
        return self.density_evo._create_default_density_matrices()

    def sim_pre(self, rho, U):
        """Apply similarity transform before evolution."""
        return self.density_evo._sim_pre(rho, U)

    def sim_post(self, rho, U):
        """Apply similarity transform after evolution."""
        return self.density_evo._sim_post(rho, U)

    def ensure_trace_preservation(self, rho):
        """Ensure trace preservation."""
        return self.density_evo._ensure_trace_preservation(rho)

    def ensure_hermiticity(self, rho):
        """Ensure hermiticity."""
        return self.density_evo._ensure_hermiticity(rho)

    def prepare_initial(self, rho_list=None):
        """Prepare initial density matrices."""
        return self.density_evo._prepare_initial_density_matrices(rho_list=rho_list)

    # Methods for integration tests
    def construct_hamiltonian_and_pulseshape(self):
        """Construct Hamiltonian and pulse shape."""
        return self.density_evo.construct_hamiltonian_and_pulseshape(self.ls_params())

    def product_basis(self, tlist, rho_list):
        """Product basis evolution."""
        return self.density_evo.product_basis(self.ls_params(), tlist=tlist, rho_list=rho_list)

    def eigen_basis(self, tlist, rho_list):
        """Eigen basis evolution."""
        return self.density_evo.eigen_basis(self.ls_params(), tlist=tlist, rho_list=rho_list)

    def product_basis_trajectory(self, tlist, rho_list):
        """Product basis trajectory evolution."""
        return self.density_evo.product_basis_trajectory(self.ls_params(), tlist=tlist, rho_list=rho_list)

    def eigen_basis_trajectory(self, tlist, rho_list):
        """Eigen basis trajectory evolution."""
        return self.density_evo.eigen_basis_trajectory(self.ls_params(), tlist=tlist, rho_list=rho_list)


def test_density_evolve_dims_assignment():
    """Test that dims is properly set."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    dims = helper.get_dims(params)
    assert dims is not None
    assert len(dims) == 2
    assert all(dim == 2 for dim in dims)


def test_create_default_density_matrices():
    """Test creation of default computational-basis density matrices."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_list = helper.create_default(params)
    # Check shape: should be (B, D, D) where B=4 (2 qubits, 2 states each), D=4 (2x2)
    expected_shape = (4, 4, 4)
    assert rho_list.shape == expected_shape, f"Expected {expected_shape}, got {rho_list.shape}"

    for i in range(rho_list.shape[0]):
        rho = rho_list[i]
        assert jnp.isclose(jnp.trace(rho), 1.0), f"Trace should be 1, got {jnp.trace(rho)}"
        assert jnp.allclose(rho, rho.conj().T), "Density matrix should be Hermitian"


def test_sim_pre_sim_post():
    """Test similarity transform methods."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create a simple density matrix |0⟩⟨0| for 2-qubit system
    rho = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho = rho.at[0, 0].set(1.0)

    # Create a simple unitary (Pauli X on first qubit)
    U = jnp.kron(sigmax(), jnp.eye(2))

    rho_pre = helper.sim_pre(params, rho, U)
    rho_post = helper.sim_post(params, rho, U)
    assert rho_pre.shape == (4, 4), f"Expected (4,4), got {rho_pre.shape}"
    assert rho_post.shape == (4, 4), f"Expected (4,4), got {rho_post.shape}"


def test_trace_preservation():
    """Test trace preservation."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create a density matrix with wrong trace
    rho = jnp.array([[2, 0], [0, 1]], dtype=jnp.complex128)
    rho_fixed = helper.ensure_trace_preservation(params, rho)

    assert jnp.isclose(jnp.trace(rho_fixed), 1.0)
    assert jnp.allclose(rho_fixed, rho / jnp.trace(rho))


def test_hermiticity_preservation():
    """Test hermiticity preservation."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create a non-Hermitian matrix
    rho = jnp.array([[1, 1j], [0, 1]], dtype=jnp.complex128)
    rho_fixed = helper.ensure_hermiticity(params, rho)

    assert jnp.allclose(rho_fixed, jnp.conj(rho_fixed).T)


def test_prepare_initial_density_matrices():
    """Test initial density matrix preparation."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Test with default
    helper.prepare_initial(params)
    rho_list = helper._get_density_evo().rho_list

    assert rho_list is not None
    for i in range(rho_list.shape[0]):
        rho = rho_list[i]
        assert jnp.isclose(jnp.trace(rho), 1.0)
        assert jnp.allclose(rho, jnp.conj(rho).T)


def test_prepare_initial_density_matrices_with_custom_rho():
    """Test initial density matrix preparation with custom rho_list."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create custom density matrices
    custom_rho = jnp.zeros((2, 4, 4), dtype=jnp.complex128)
    custom_rho = custom_rho.at[0, 0, 0].set(1.0)
    custom_rho = custom_rho.at[1, 3, 3].set(1.0)

    helper.prepare_initial(params, custom_rho)

    assert jnp.allclose(helper._get_density_evo().rho_list, custom_rho), "Custom rho_list should be preserved"


def _create_dissipation_test_graph():
    """Create a simple test graph for dissipation tests."""
    graph = SCGraph()
    graph.add_node('q0', system_type='transmon', ec=0.2, ej=5.0)
    return graph


def _create_dissipation_collapse_operators():
    """Create amplitude and phase damping collapse operators."""
    # Note: dims must be tuple (2,) to match Hamiltonian format, not list [2]
    c_op_amp = amplitude_damping_operator(qubit_index=0, rate=1.0, dims=(2,))
    c_op_phase = phase_damping_operator(qubit_index=0, rate=1.0, dims=(2,))

    return [c_op_amp], [c_op_phase]


def test_amplitude_damping_ground_state():
    """Ground state should be unaffected by amplitude damping."""
    graph = _create_dissipation_test_graph()
    c_ops_amp, _ = _create_dissipation_collapse_operators()
    density_evo = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_amp
    )
    density_evo.pulse_endtime = 5.0
    density_evo.tlist = np.linspace(0, 5.0, 100)

    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |0⟩⟨0| (ground state)
    rho0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)

    # Use DensityEvolve with proper tlist for amplitude damping
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5
    result = density_evo.product_basis(params, tlist=times, rho_list=rho0)

    # Should remain unchanged (ground state is stable)
    assert jnp.allclose(result, rho0, atol=1e-2), f"Ground state changed: {result}"
    # Verify trace preservation
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace not preserved: {jnp.trace(result)}"


def test_amplitude_damping_excited_state():
    """Excited state should decay to ground state."""
    graph = _create_dissipation_test_graph()
    c_ops_amp, _ = _create_dissipation_collapse_operators()
    density_evo = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_amp
    )
    density_evo.pulse_endtime = 5.0
    density_evo.tlist = np.linspace(0, 5.0, 100)

    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |1⟩⟨1| (excited state)
    rho0 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)

    # Use DensityEvolve with proper tlist for amplitude damping
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5
    result = density_evo.product_basis(params, tlist=times, rho_list=rho0)

    # Verify that amplitude damping occurred, rho11 should decay to a small value
    initial_excited_pop = rho0[1, 1]
    final_excited_pop = result[1, 1]
    decay_ratio = final_excited_pop / initial_excited_pop
    assert decay_ratio < 0.1, f"Amplitude damping should reduce excited state by >90%, got {decay_ratio:.3f}"


def test_amplitude_damping_superposition():
    """Superposition should decay to ground state."""
    graph = _create_dissipation_test_graph()
    c_ops_amp, _ = _create_dissipation_collapse_operators()
    density_evo = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_amp
    )
    density_evo.pulse_endtime = 5.0
    density_evo.tlist = np.linspace(0, 5.0, 100)

    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |+⟩⟨+| (superposition)
    rho0 = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.complex128)

    # Use DensityEvolve with proper tlist for amplitude damping
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5
    result = density_evo.product_basis(params, tlist=times, rho_list=rho0)

    # Verify that amplitude damping occurred, coherence should be lost
    initial_coherence = jnp.abs(rho0[0, 1])
    final_coherence = jnp.abs(result[0, 1])
    coherence_reduction = final_coherence / initial_coherence if initial_coherence > 0 else 0
    assert coherence_reduction < 0.5, f"Coherence should be significantly reduced, got {coherence_reduction:.3f}"


def test_amplitude_damping_vs_no_dissipation():
    """Verify that amplitude damping actually affects evolution."""
    graph = _create_dissipation_test_graph()
    c_ops_amp, _ = _create_dissipation_collapse_operators()
    density_evo_amp = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_amp
    )
    density_evo_no_diss = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=[]
    )
    density_evo_amp.pulse_endtime = 5.0
    density_evo_amp.tlist = np.linspace(0, 5.0, 100)
    density_evo_no_diss.pulse_endtime = 5.0
    density_evo_no_diss.tlist = np.linspace(0, 5.0, 100)

    params_amp = _initialize_density_evo_haiku(density_evo_amp)
    params_no_diss = _initialize_density_evo_haiku(density_evo_no_diss)

    # Initial state |1⟩⟨1| (excited state)
    rho0 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5

    result_no_diss = density_evo_no_diss.product_basis(params_no_diss, tlist=times, rho_list=rho0)
    result_amp = density_evo_amp.product_basis(params_amp, tlist=times, rho_list=rho0)

    # Results should be different, the excited state should decay more significantly
    assert not jnp.allclose(result_no_diss, result_amp, atol=1e-2), "Amplitude damping has no effect!"
    excited_decay_with_diss = result_amp[1, 1] / rho0[1, 1]
    excited_decay_without_diss = result_no_diss[1, 1] / rho0[1, 1]
    assert excited_decay_with_diss < excited_decay_without_diss, "Amplitude damping should cause more excited state decay"


def test_phase_damping_ground_state():
    """Ground state should be unaffected by phase damping."""
    graph = _create_dissipation_test_graph()
    _, c_ops_phase = _create_dissipation_collapse_operators()
    density_evo = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_phase
    )
    density_evo.pulse_endtime = 5.0
    density_evo.tlist = np.linspace(0, 5.0, 100)

    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |0⟩⟨0| (ground state)
    rho0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)

    # Use DensityEvolve with proper tlist for phase damping
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5
    result = density_evo.product_basis(params, tlist=times, rho_list=rho0)

    # Should remain unchanged
    assert jnp.allclose(result, rho0, atol=1e-2), f"Ground state changed: {result}"
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace not preserved: {jnp.trace(result)}"


def test_phase_damping_excited_state():
    """Excited state should be unaffected by phase damping."""
    graph = _create_dissipation_test_graph()
    _, c_ops_phase = _create_dissipation_collapse_operators()
    density_evo = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_phase
    )
    density_evo.pulse_endtime = 5.0
    density_evo.tlist = np.linspace(0, 5.0, 100)

    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |1⟩⟨1| (excited state)
    rho0 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)

    # Use DensityEvolve with proper tlist for phase damping
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5
    result = density_evo.product_basis(params, tlist=times, rho_list=rho0)

    # Should remain unchanged
    assert jnp.allclose(result, rho0, atol=1e-2), f"Excited state changed: {result}"
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace not preserved: {jnp.trace(result)}"


def test_phase_damping_superposition():
    """Superposition should lose coherence due to phase damping."""
    graph = _create_dissipation_test_graph()
    _, c_ops_phase = _create_dissipation_collapse_operators()
    density_evo = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_phase
    )
    density_evo.pulse_endtime = 5.0
    density_evo.tlist = np.linspace(0, 5.0, 100)

    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |+⟩⟨+| (superposition)
    rho0 = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.complex128)

    # Use DensityEvolve with proper tlist for phase damping
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5
    result = density_evo.product_basis(params, tlist=times, rho_list=rho0)

    # Should preserve trace
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace not preserved: {jnp.trace(result)}"
    # Should preserve diagonal elements
    assert jnp.isclose(result[0, 0], 0.5, atol=1e-2), f"Ground state population changed: {result[0, 0]}"
    assert jnp.isclose(result[1, 1], 0.5, atol=1e-2), f"Excited state population changed: {result[1, 1]}"
    # Should lose coherence
    initial_coherence = jnp.abs(rho0[0, 1])
    final_coherence = jnp.abs(result[0, 1])
    coherence_reduction = final_coherence / initial_coherence if initial_coherence > 0 else 0
    assert coherence_reduction < 0.1, f"Coherence should be significantly reduced, got {coherence_reduction:.3f}"


def test_phase_damping_coherent_state():
    """Complex coherent state should lose coherence due to phase damping."""
    graph = _create_dissipation_test_graph()
    _, c_ops_phase = _create_dissipation_collapse_operators()
    density_evo = DensityEvolve(
        graph=graph,
        truncated_dim=2,
        add_random=False,
        solver='ode_expm',
        options={'astep': 100, 'trotter_order': 1},
        c_ops=c_ops_phase
    )
    density_evo.pulse_endtime = 5.0
    density_evo.tlist = np.linspace(0, 5.0, 100)

    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |ψ⟩⟨ψ| where |ψ⟩ = (|0⟩ + i|1⟩)/√2
    # This gives rho = [[0.5, -0.5i], [0.5i, 0.5]]
    rho0 = jnp.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=jnp.complex128)

    # Use DensityEvolve with proper tlist for phase damping
    times = np.linspace(0, 5.0, 100)  # 100 time points from 0 to 5
    result = density_evo.product_basis(params, tlist=times, rho_list=rho0)

    # Should preserve trace (no energy loss)
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace not preserved: {jnp.trace(result)}"
    # Should preserve diagonal elements
    assert jnp.isclose(result[0, 0], 0.5, atol=1e-2), f"Ground state population changed: {result[0, 0]}"
    assert jnp.isclose(result[1, 1], 0.5, atol=1e-2), f"Excited state population changed: {result[1, 1]}"
    # Should lose coherence
    initial_coherence = jnp.abs(rho0[0, 1])
    final_coherence = jnp.abs(result[0, 1])
    coherence_reduction = final_coherence / initial_coherence if initial_coherence > 0 else 0
    assert coherence_reduction < 0.1, f"Coherence should be significantly reduced, got {coherence_reduction:.3f}"


def test_hamiltonian_construction():
    """Test Hamiltonian construction."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    ham_static, ham_components, pulse_endtime = helper.construct_hamiltonian_and_pulseshape(params)

    assert isinstance(ham_static, KronObj)
    assert isinstance(ham_components, list)
    assert isinstance(pulse_endtime, (int, float, jnp.ndarray)), f"Expected int/float/array, got {type(pulse_endtime)}"


def test_product_basis_evolution():
    """Test product basis evolution."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create initial density matrix |00⟩⟨00| for 2-qubit system (4x4)
    rho0 = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho0 = rho0.at[0, 0].set(1.0)

    # Use proper tlist for evolution
    times = np.linspace(0, 1.0, 50)  # 50 time points from 0 to 1

    result = helper.product_basis(params, tlist=times, rho_list=rho0)

    assert result is not None
    assert result.shape == (4, 4), f"Expected (4,4), got {result.shape}"
    # Verify it's a valid density matrix
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace should be 1, got {jnp.trace(result)}"
    assert jnp.allclose(result, result.conj().T), "Should be Hermitian"


def test_eigen_basis_evolution():
    """Test eigen basis evolution."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create initial density matrix |00⟩⟨00| for 2-qubit system (4x4)
    rho0 = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho0 = rho0.at[0, 0].set(1.0)

    # Use proper tlist for evolution
    times = np.linspace(0, 1.0, 50)  # 50 time points from 0 to 1

    result = helper.eigen_basis(params, tlist=times, rho_list=rho0)

    assert result is not None
    assert result.shape == (4, 4), f"Expected (4,4), got {result.shape}"
    # Verify it's a valid density matrix
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace should be 1, got {jnp.trace(result)}"
    assert jnp.allclose(result, result.conj().T), "Should be Hermitian"


def test_product_basis_trajectory_evolution():
    """Test trajectory evolution."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create initial density matrix |00⟩⟨00| for 2-qubit system (4x4)
    rho0 = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho0 = rho0.at[0, 0].set(1.0)
    tlist = jnp.linspace(0, 1, 10)

    states, comp_states = helper.product_basis_trajectory(params, tlist=tlist, rho_list=rho0)

    assert states is not None
    assert comp_states is not None
    assert states.shape == (10, 4, 4), f"Expected (10,4,4), got {states.shape}"
    assert comp_states.shape == (10, 4), f"Expected (10,4), got {comp_states.shape}"

    # Verify trajectory properties
    for t in range(states.shape[0]):
        assert jnp.isclose(jnp.trace(states[t]), 1.0), f"Trace should be 1 at time {t}"
        assert jnp.allclose(states[t], states[t].conj().T), f"Should be Hermitian at time {t}"


def test_eigen_basis_single_density_matrix_shape():
    """Test eigen_basis with single density matrix (D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_single = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho_single = rho_single.at[0, 0].set(1.0)  # |00⟩⟨00|

    # Use proper tlist for evolution
    times = np.linspace(0, 1.0, 50)  # 50 time points from 0 to 1

    result = helper.eigen_basis(params, tlist=times, rho_list=rho_single)
    assert result.shape == (4, 4), f"Expected (4,4), got {result.shape}"

    # Verify it's a valid density matrix
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace should be 1, got {jnp.trace(result)}"
    assert jnp.allclose(result, result.conj().T), "Should be Hermitian"


def test_product_basis_single_density_matrix_shape():
    """Test product_basis with single density matrix (D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_single = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho_single = rho_single.at[0, 0].set(1.0)  # |00⟩⟨00|

    # Use proper tlist for evolution
    times = np.linspace(0, 1.0, 50)  # 50 time points from 0 to 1

    result = helper.product_basis(params, tlist=times, rho_list=rho_single)
    assert result.shape == (4, 4), f"Expected (4,4), got {result.shape}"

    # Verify it's a valid density matrix
    assert jnp.isclose(jnp.trace(result), 1.0, atol=1e-2), f"Trace should be 1, got {jnp.trace(result)}"
    assert jnp.allclose(result, result.conj().T), "Should be Hermitian"


def test_eigen_basis_batch_density_matrices_shape():
    """Test eigen_basis with batch density matrices (B,D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_batch = jnp.zeros((2, 4, 4), dtype=jnp.complex128)
    rho_batch = rho_batch.at[0, 0, 0].set(1.0)  # |00⟩⟨00|
    rho_batch = rho_batch.at[1, 3, 3].set(1.0)  # |11⟩⟨11|

    # Use proper tlist for evolution
    times = np.linspace(0, 1.0, 50)  # 50 time points from 0 to 1

    result = helper.eigen_basis(params, tlist=times, rho_list=rho_batch)
    assert result.shape == (2, 4, 4), f"Expected (2,4,4), got {result.shape}"

    # Verify all density matrices are valid
    for i in range(result.shape[0]):
        assert jnp.isclose(jnp.trace(result[i]), 1.0, atol=1e-2), (
            f"Trace should be 1 for batch {i}, got {jnp.trace(result[i])}"
        )
        assert jnp.allclose(result[i], result[i].conj().T), f"Should be Hermitian for batch {i}"


def test_product_basis_batch_density_matrices_shape():
    """Test product_basis with batch density matrices (B,D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_batch = jnp.zeros((2, 4, 4), dtype=jnp.complex128)
    rho_batch = rho_batch.at[0, 0, 0].set(1.0)  # |00⟩⟨00|
    rho_batch = rho_batch.at[1, 3, 3].set(1.0)  # |11⟩⟨11|

    # Use proper tlist for evolution
    times = np.linspace(0, 1.0, 50)  # 50 time points from 0 to 1

    result = helper.product_basis(params, tlist=times, rho_list=rho_batch)
    assert result.shape == (2, 4, 4), f"Expected (2,4,4), got {result.shape}"

    # Verify all density matrices are valid
    for i in range(result.shape[0]):
        assert jnp.isclose(jnp.trace(result[i]), 1.0, atol=1e-2), (
            f"Trace should be 1 for batch {i}, got {jnp.trace(result[i])}"
        )
        assert jnp.allclose(result[i], result[i].conj().T), f"Should be Hermitian for batch {i}"


def test_eigen_basis_trajectory_single_density_matrix_shape():
    """Test eigen_basis_trajectory with single density matrix (D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_single = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho_single = rho_single.at[0, 0].set(1.0)  # |00⟩⟨00|
    tlist = jnp.linspace(0, 1, 5)

    traj, sliced_traj = helper.eigen_basis_trajectory(params, tlist=tlist, rho_list=rho_single)
    assert traj.shape == (5, 4, 4), f"Expected (5,4,4), got {traj.shape}"
    assert sliced_traj.shape == (5, 4), f"Expected (5,4), got {sliced_traj.shape}"

    # Verify trajectory properties
    for t in range(traj.shape[0]):
        assert jnp.isclose(jnp.trace(traj[t]), 1.0), f"Trace should be 1 at time {t}"
        assert jnp.allclose(traj[t], traj[t].conj().T), f"Should be Hermitian at time {t}"


def test_product_basis_trajectory_single_density_matrix_shape():
    """Test product_basis_trajectory with single density matrix (D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_single = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho_single = rho_single.at[0, 0].set(1.0)  # |00⟩⟨00|
    tlist = jnp.linspace(0, 1, 5)

    traj, sliced_traj = helper.product_basis_trajectory(params, tlist=tlist, rho_list=rho_single)
    assert traj.shape == (5, 4, 4), f"Expected (5,4,4), got {traj.shape}"
    assert sliced_traj.shape == (5, 4), f"Expected (5,4), got {sliced_traj.shape}"

    # Verify trajectory properties
    for t in range(traj.shape[0]):
        assert jnp.isclose(jnp.trace(traj[t]), 1.0), f"Trace should be 1 at time {t}"
        assert jnp.allclose(traj[t], traj[t].conj().T), f"Should be Hermitian at time {t}"


def test_eigen_basis_trajectory_batch_density_matrices_shape():
    """Test eigen_basis_trajectory with batch density matrices (B,D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_batch = jnp.zeros((2, 4, 4), dtype=jnp.complex128)
    rho_batch = rho_batch.at[0, 0, 0].set(1.0)  # |00⟩⟨00|
    rho_batch = rho_batch.at[1, 3, 3].set(1.0)  # |11⟩⟨11|
    tlist = jnp.linspace(0, 1, 5)

    traj, sliced_traj = helper.eigen_basis_trajectory(params, tlist=tlist, rho_list=rho_batch)
    assert traj.shape == (2, 5, 4, 4), f"Expected (2,5,4,4), got {traj.shape}"
    assert sliced_traj.shape == (2, 5, 4), f"Expected (2,5,4), got {sliced_traj.shape}"

    # Verify trajectory properties
    for b in range(traj.shape[0]):
        for t in range(traj.shape[1]):
            assert jnp.isclose(jnp.trace(traj[b, t]), 1.0), f"Trace should be 1 for batch {b}, time {t}"
            assert jnp.allclose(traj[b, t], traj[b, t].conj().T), f"Should be Hermitian for batch {b}, time {t}"


def test_product_basis_trajectory_batch_density_matrices_shape():
    """Test product_basis_trajectory with batch density matrices (B,D,D) input."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_batch = jnp.zeros((2, 4, 4), dtype=jnp.complex128)
    rho_batch = rho_batch.at[0, 0, 0].set(1.0)  # |00⟩⟨00|
    rho_batch = rho_batch.at[1, 3, 3].set(1.0)  # |11⟩⟨11|
    tlist = jnp.linspace(0, 1, 5)

    traj, sliced_traj = helper.product_basis_trajectory(params, tlist=tlist, rho_list=rho_batch)
    assert traj.shape == (2, 5, 4, 4), f"Expected (2,5,4,4), got {traj.shape}"
    assert sliced_traj.shape == (2, 5, 4), f"Expected (2,5,4), got {sliced_traj.shape}"

    # Verify trajectory properties
    for b in range(traj.shape[0]):
        for t in range(traj.shape[1]):
            assert jnp.isclose(jnp.trace(traj[b, t]), 1.0), f"Trace should be 1 for batch {b}, time {t}"
            assert jnp.allclose(traj[b, t], traj[b, t].conj().T), f"Should be Hermitian for batch {b}, time {t}"


def test_trajectory_shape_consistency_shape():
    """Test trajectory shape consistency between single and batch inputs."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    rho_single = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho_single = rho_single.at[0, 0].set(1.0)

    rho_batch = rho_single[None, :, :]  # (1,4,4)
    tlist = jnp.linspace(0, 1, 5)

    # Test eigen basis trajectory
    traj_single, sliced_single = helper.eigen_basis_trajectory(params, tlist=tlist, rho_list=rho_single)
    traj_batch, sliced_batch = helper.eigen_basis_trajectory(params, tlist=tlist, rho_list=rho_batch)

    assert jnp.allclose(traj_single, traj_batch[0]), "Single and batch trajectories should match"
    assert jnp.allclose(sliced_single, sliced_batch[0]), "Single and batch sliced trajectories should match"

    # Test product basis trajectory
    traj_single_prod, sliced_single_prod = helper.product_basis_trajectory(params, tlist=tlist, rho_list=rho_single)
    traj_batch_prod, sliced_batch_prod = helper.product_basis_trajectory(params, tlist=tlist, rho_list=rho_batch)

    assert jnp.allclose(traj_single_prod, traj_batch_prod[0]), "Single and batch trajectories should match"
    assert jnp.allclose(sliced_single_prod, sliced_batch_prod[0]), "Single and batch sliced trajectories should match"


def test_default_density_matrices_shape():
    """Test that default density matrices have correct shape."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Test default density matrix creation
    rho_list = helper.create_default(params)

    assert rho_list.ndim == 3, "Should be 3D batch"
    assert rho_list.shape[0] == 4, "Should have 4 density matrices (2 qubits)"
    assert rho_list.shape[1:] == (4, 4), "Each density matrix should be (4,4)"

    # Verify all are valid density matrices
    for i in range(rho_list.shape[0]):
        rho = rho_list[i]
        assert jnp.isclose(jnp.trace(rho), 1.0, atol=1e-2), f"Trace should be 1 for density matrix {i}, got {jnp.trace(rho)}"
        assert jnp.allclose(rho, rho.conj().T), f"Should be Hermitian for density matrix {i}"


def test_similarity_transform_shape_handling():
    """Test that similarity transforms handle all supported shapes correctly."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    # Create test unitary
    U = jnp.kron(sigmax(), jnp.eye(2))  # Pauli X on first qubit

    # Test single density matrix (D,D)
    rho_single = jnp.zeros((4, 4), dtype=jnp.complex128)
    rho_single = rho_single.at[0, 0].set(1.0)

    rho_pre_single = helper.sim_pre(params, rho_single, U)
    rho_post_single = helper.sim_post(params, rho_single, U)

    assert rho_pre_single.shape == (4, 4), f"Expected (4,4), got {rho_pre_single.shape}"
    assert rho_post_single.shape == (4, 4), f"Expected (4,4), got {rho_post_single.shape}"

    # Test batch density matrices (B,D,D)
    rho_batch = jnp.zeros((2, 4, 4), dtype=jnp.complex128)
    rho_batch = rho_batch.at[0, 0, 0].set(1.0)
    rho_batch = rho_batch.at[1, 3, 3].set(1.0)

    rho_pre_batch = helper.sim_pre(params, rho_batch, U)
    rho_post_batch = helper.sim_post(params, rho_batch, U)

    assert rho_pre_batch.shape == (2, 4, 4), f"Expected (2,4,4), got {rho_pre_batch.shape}"
    assert rho_post_batch.shape == (2, 4, 4), f"Expected (2,4,4), got {rho_post_batch.shape}"

    # Test trajectory data (T,D,D)
    traj = jnp.zeros((3, 4, 4), dtype=jnp.complex128)
    for t in range(3):
        traj = traj.at[t, 0, 0].set(1.0)

    traj_pre = helper.sim_pre(params, traj, U)
    traj_post = helper.sim_post(params, traj, U)

    assert traj_pre.shape == (3, 4, 4), f"Expected (3,4,4), got {traj_pre.shape}"
    assert traj_post.shape == (3, 4, 4), f"Expected (3,4,4), got {traj_post.shape}"

    # Test batch trajectory data (B,T,D,D)
    traj_batch = jnp.zeros((2, 3, 4, 4), dtype=jnp.complex128)
    for b in range(2):
        for t in range(3):
            traj_batch = traj_batch.at[b, t, 0, 0].set(1.0)

    traj_pre_batch = helper.sim_pre(params, traj_batch, U)
    traj_post_batch = helper.sim_post(params, traj_batch, U)

    assert traj_pre_batch.shape == (2, 3, 4, 4), f"Expected (2,3,4,4), got {traj_pre_batch.shape}"
    assert traj_post_batch.shape == (2, 3, 4, 4), f"Expected (2,3,4,4), got {traj_post_batch.shape}"


def test_invalid_input_shapes():
    """Test error handling for invalid input shapes."""
    helper = DensityEvolveHelper()
    params = helper.ls_params()

    invalid_shapes = [
        jnp.ones(4),  # 1D
        jnp.ones((2, 2, 2, 2)),  # 4D
        jnp.ones((2, 3)),  # Wrong dimensions
    ]

    for invalid_rho in invalid_shapes:
        with pytest.raises((ValueError, TypeError)):
            helper.eigen_basis(params, rho_list=invalid_rho)

        with pytest.raises((ValueError, TypeError)):
            helper.product_basis(params, rho_list=invalid_rho)


def test_invalid_solver():
    """Test invalid solver handling."""
    graph = SCGraph()
    graph.add_node('q0', system_type='transmon', ec=0.2, ej=5.0)

    with pytest.raises(NotImplementedError):
        DensityEvolve(graph=graph, solver='invalid_solver')


def test_invalid_compensation_option():
    """Test invalid compensation option handling."""
    graph = SCGraph()
    graph.add_node('q0', system_type='transmon', ec=0.2, ej=5.0)

    with pytest.raises(NotImplementedError):
        DensityEvolve(graph=graph, compensation_option='invalid_option')


def test_unsupported_density_matrix_shape():
    """Test handling of unsupported density matrix shapes."""
    graph = SCGraph()
    graph.add_node('q0', system_type='transmon', ec=0.2, ej=5.0)

    density_evo = DensityEvolve(graph=graph, truncated_dim=2)
    _initialize_density_evo_haiku(density_evo)

    # Test with invalid rho_list shape
    invalid_rho = jnp.ones((2, 2, 2, 2))  # 4D array

    with pytest.raises(ValueError):
        density_evo._prepare_initial_density_matrices(rho_list=invalid_rho)


def test_unsupported_similarity_transform_shape():
    """Test handling of unsupported similarity transform shapes."""
    graph = SCGraph()
    graph.add_node('q0', system_type='transmon', ec=0.2, ej=5.0)

    density_evo = DensityEvolve(graph=graph, truncated_dim=2)
    _initialize_density_evo_haiku(density_evo)

    # Test with invalid trajectory shape
    invalid_trajectory = jnp.ones((2, 3, 4))  # Non-square last two dimensions

    with pytest.raises(ValueError):
        density_evo._sim_pre(invalid_trajectory, jnp.eye(2))