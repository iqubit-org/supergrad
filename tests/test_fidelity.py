import numpy as np
from supergrad.utils import (gates, compute_fidelity_with_1q_rotation_axis)


def test_fidelity_vz():
    ut = gates.cz_gate(2)
    angle = np.array([0.78129602, 1.42407966, -2.30043467, 1.3672363])
    # angle = np.array([0.7, 1.4, -2.3, 1.54])
    sim_u = np.diag(np.exp(1j * angle))
    rng = np.random.default_rng(2983461)
    sim_u += rng.uniform(-1e-4, 1e-4, 4 * 4).reshape((4, 4))
    for method in ["gd", "lbfgs", "simple_vz", "gd_init2", "lbfgs_init2"]:
        fid, _ = compute_fidelity_with_1q_rotation_axis(u_target=ut,
                                                        u_computed=sim_u,
                                                        opt_method=method,
                                                        compensation_option="only_vz")
        assert fid > 0 and fid < 1
