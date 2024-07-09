import sys
import matplotlib.pyplot as plt
import numpy as np

from supergrad.helper import Evolve
from supergrad.scgraph import SCGraph
from supergrad.utils.sgm_format import read_sgm_data
from supergrad.utils.gates import cz_gate
from supergrad.utils import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.optimize import scipy_minimize, adam_opt

g: SCGraph = read_sgm_data("transmon_2q1c.json")
evo = Evolve(g, solver="ode_expm", options={'astep': 2000, 'trotter_order': 2})
params = g.convert_graph_to_parameters()
p1 = params["nodes"]["q2"]["pulse"]["p1"]
params_init = {"nodes":
    {
        "q0": {"compensation": params["nodes"]["q0"]["compensation"]},
        "q1": {"compensation": params["nodes"]["q1"]["compensation"]},
        "q2": {"pulse": {"p1": {
            "amp": p1["amp"], "t_ramp": p1["t_ramp"], "t_plateau": p1["t_plateau"]
        }}}}}


# Note this gate natively is CZ^-1
# Target cannot be the original CZ is not possible near the working point
target_unitary = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

b_plot_pulse = False

if b_plot_pulse:
    evo.init_quantum_system({})
    pulse_lst, time_max = evo.graph.convert_graph_to_pulse_lst(evo.hilbertspace, convert_amp=False)

    ar_t = np.linspace(0, time_max, 1001, endpoint=True)
    ar_y = pulse_lst[0][1](ar_t)

    plt.plot(ar_t, ar_y)
    plt.show()
    sys.exit(0)


def infidelity(params, target_unitary):
    evo.init_quantum_system(params)
    u = evo.final_state(basis="eigen")
    fidelity, res_unitary = compute_fidelity_with_1q_rotation_axis(
        target_unitary, u, compensation_option='no_comp')
    return 1 - fidelity


res = scipy_minimize(infidelity,
                     params_init,
                     args=(target_unitary,),
                     method='l-bfgs-b',
                     logging=True,
                     options={'maxiter': 2000})

print(res)
