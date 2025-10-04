# %%
import os
import numpy as np
import jax.numpy as jnp
import jax
import haiku as hk
import qutip as qt

from supergrad.helper.compute_density_evolve import DensityEvolve
from supergrad.quantum_system import KronObj
from supergrad.scgraph.graph import SCGraph
from supergrad.utils.operators import sigmax, sigmaz, sigmay
from supergrad.utils.qutip_interface import to_qutip_operator, to_qutip_collapse_operators

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


def create_spin_relaxation_system(tunneling_rate, relaxation_rate):
    """Create a simple spin-1/2 system for relaxation testing."""
    graph = SCGraph()
    graph.add_node('q0', system_type='transmon', ec=0.2, ej=5.0)
    sigmax_collapse = KronObj([jnp.sqrt(relaxation_rate) * sigmax()], dims=(2,), locs=[0])

    # Create custom DensityEvolve that overrides the Hamiltonian to simulate spin-1/2
    class CustomDensityEvolve(DensityEvolve):
        def _init_quantum_system(self):
            super()._init_quantum_system()

            def custom_idling_hamiltonian():
                return KronObj([tunneling_rate * sigmax()], dims=(2,), locs=[0])

            self.hilbertspace.idling_hamiltonian_in_prod_basis = custom_idling_hamiltonian

    return graph, sigmax_collapse, CustomDensityEvolve


def test_supergrad_lindblad_lcam_with_qutip():
    """Compare the result of SuperGrad Lindblad solver (with LCAM) against QuTiP for a spin-1/2 system.
    """
    # Setup parameters
    tunneling_rate = 2 * np.pi * 0.1  # H = 2π × 0.1 × σX
    relaxation_rate = 0.05            # c_op = √0.05 × σX
    times = np.linspace(0.0, 10.0, 100)

    graph, sigmax_collapse, CustomDensityEvolve = create_spin_relaxation_system(tunneling_rate, relaxation_rate)
    density_evo = CustomDensityEvolve(
        graph=graph,
        truncated_dim=2,
        c_ops=[sigmax_collapse],
        solver='ode_expm',
        options={
            'custom_vjp': True
        }
    )

    # Initialize SuperGrad system
    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |0⟩⟨0|
    initial_state = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)

    # SuperGrad evolution - get full trajectory
    rho_traj_supergrad_tuple = density_evo.product_basis_trajectory(
        params, tlist=times, rho_list=initial_state)
    rho_traj_supergrad = rho_traj_supergrad_tuple[0]

    # Create QuTiP data
    sigmax_ham = KronObj([tunneling_rate * sigmax()], dims=(2,), locs=[0])
    H_qutip = to_qutip_operator(sigmax_ham)
    c_ops_qutip = to_qutip_collapse_operators([sigmax_collapse])
    psi0 = qt.basis(2, 0)

    # QuTiP evolution
    result_qutip_expect = qt.mesolve(H_qutip, psi0, times, c_ops_qutip, [qt.sigmaz(), qt.sigmay()])

    # Compare expectation value trajectories
    exp_z_traj_supergrad = np.array([jnp.trace(rho @ sigmaz()) for rho in rho_traj_supergrad])
    exp_y_traj_supergrad = np.array([jnp.trace(rho @ sigmay()) for rho in rho_traj_supergrad])

    exp_z_traj_qutip = result_qutip_expect.expect[0]
    exp_y_traj_qutip = result_qutip_expect.expect[1]

    print("Sample output comparison (LCAM):")
    print(f"σZ expectation values - SuperGrad: {exp_z_traj_supergrad[:10]}")
    print(f"σZ expectation values - QuTiP: {exp_z_traj_qutip[:10]}")
    print(f"σY expectation values - SuperGrad: {exp_y_traj_supergrad[:10]}")
    print(f"σY expectation values - QuTiP: {exp_y_traj_qutip[:10]}")
    print(f"\nFinal σZ expectation - SuperGrad: {exp_z_traj_supergrad[-1]}")
    print(f"Final σZ expectation - QuTiP: {exp_z_traj_qutip[-1]}")
    print(f"Final σY expectation - SuperGrad: {exp_y_traj_supergrad[-1]}")
    print(f"Final σY expectation - QuTiP: {exp_y_traj_qutip[-1]}")

    z_diff = np.abs(exp_z_traj_supergrad - exp_z_traj_qutip)
    y_diff = np.abs(exp_y_traj_supergrad - exp_y_traj_qutip)
    # Check if results are close (within numerical tolerance)
    z_close = np.allclose(exp_z_traj_supergrad, exp_z_traj_qutip, rtol=1e-3, atol=1e-3)
    y_close = np.allclose(exp_y_traj_supergrad, exp_y_traj_qutip, rtol=1e-3, atol=1e-3)

    print(f"\nσZ trajectories close: {z_close}")
    print(f"σY trajectories close: {y_close}")

    # Only pass if results are within tolerance
    assert z_close, f"σZ trajectories do not match within tolerance. Max diff: {np.max(z_diff):.6f}"
    assert y_close, f"σY trajectories do not match within tolerance. Max diff: {np.max(y_diff):.6f}"


def test_supergrad_lindblad_tad_with_qutip():
    """Compare the result of SuperGrad Lindblad solver (with LCAM) against QuTiP for a spin-1/2 system.
    """
    # Setup parameters
    tunneling_rate = 2 * np.pi * 0.1  # H = 2π × 0.1 × σX
    relaxation_rate = 0.05            # c_op = √0.05 × σX
    times = np.linspace(0.0, 10.0, 100)

    graph, sigmax_collapse, CustomDensityEvolve = create_spin_relaxation_system(tunneling_rate, relaxation_rate)
    density_evo = CustomDensityEvolve(
        graph=graph,
        truncated_dim=2,
        c_ops=[sigmax_collapse],
        solver='ode_expm',
        options={
            'custom_vjp': None
        }
    )

    # Initialize SuperGrad system
    params = _initialize_density_evo_haiku(density_evo)

    # Initial state |0⟩⟨0|
    initial_state = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)

    # SuperGrad evolution - get full trajectory
    rho_traj_supergrad_tuple = density_evo.product_basis_trajectory(
        params, tlist=times, rho_list=initial_state)
    rho_traj_supergrad = rho_traj_supergrad_tuple[0]

    # Create QuTiP data
    sigmax_ham = KronObj([tunneling_rate * sigmax()], dims=(2,), locs=[0])
    H_qutip = to_qutip_operator(sigmax_ham)
    c_ops_qutip = to_qutip_collapse_operators([sigmax_collapse])
    psi0 = qt.basis(2, 0)

    # QuTiP evolution
    result_qutip_expect = qt.mesolve(H_qutip, psi0, times, c_ops_qutip, [qt.sigmaz(), qt.sigmay()])

    # Compare expectation value trajectories
    exp_z_traj_supergrad = np.array([jnp.trace(rho @ sigmaz()) for rho in rho_traj_supergrad])
    exp_y_traj_supergrad = np.array([jnp.trace(rho @ sigmay()) for rho in rho_traj_supergrad])

    exp_z_traj_qutip = result_qutip_expect.expect[0]
    exp_y_traj_qutip = result_qutip_expect.expect[1]

    print("Sample output comparison (TAD):")
    print(f"σZ expectation values - SuperGrad: {exp_z_traj_supergrad[:10]}")
    print(f"σZ expectation values - QuTiP: {exp_z_traj_qutip[:10]}")
    print(f"σY expectation values - SuperGrad: {exp_y_traj_supergrad[:10]}")
    print(f"σY expectation values - QuTiP: {exp_y_traj_qutip[:10]}")
    print(f"\nFinal σZ expectation - SuperGrad: {exp_z_traj_supergrad[-1]}")
    print(f"Final σZ expectation - QuTiP: {exp_z_traj_qutip[-1]}")
    print(f"Final σY expectation - SuperGrad: {exp_y_traj_supergrad[-1]}")
    print(f"Final σY expectation - QuTiP: {exp_y_traj_qutip[-1]}")

    z_diff = np.abs(exp_z_traj_supergrad - exp_z_traj_qutip)
    y_diff = np.abs(exp_y_traj_supergrad - exp_y_traj_qutip)
    # Check if results are close (within numerical tolerance)
    z_close = np.allclose(exp_z_traj_supergrad, exp_z_traj_qutip, rtol=1e-3, atol=1e-3)
    y_close = np.allclose(exp_y_traj_supergrad, exp_y_traj_qutip, rtol=1e-3, atol=1e-3)

    print(f"\nσZ trajectories close: {z_close}")
    print(f"σY trajectories close: {y_close}")

    # Only pass if results are within tolerance
    assert z_close, f"σZ trajectories do not match within tolerance. Max diff: {np.max(z_diff):.6f}"
    assert y_close, f"σY trajectories do not match within tolerance. Max diff: {np.max(y_diff):.6f}"