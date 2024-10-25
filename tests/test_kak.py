import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlin
from supergrad.utils.gates import sigmax, sigmay, sigmaz, tensor, identity
from supergrad.utils.kak import kak1_decomposition, construct_u_from_k_1q
from functools import partial

u1 = jnp.array([
    [+0.5387199 - 0.2632373j, +0.0834077 - 0.4712990j, +0.0470534 + 0.4486978j, +0.3624394 - 0.2765886j],
    [-0.0598106 + 0.1107563j, +0.4818738 - 0.1523462j, +0.4490990 - 0.5878578j, +0.2960287 - 0.3063400j],
    [+0.1010169 + 0.0385598j, +0.3778473 + 0.4729944j, -0.1299617 + 0.2377876j, -0.3637780 - 0.6450187j],
    [+0.3683266 + 0.6908596j, -0.3818637 - 0.0565374j, +0.4160828 + 0.0512987j, -0.2273659 - 0.1028611j],
]
)
u2 = jnp.array([
    [-0.5951617 + 0.5204996j, -0.0644475 + 0.3336411j, -0.4468223 + 0.0788024j, -0.0186007 - 0.2306230j],
    [-0.3566298 - 0.2241272j, +0.1121142 + 0.4476695j, +0.3264768 + 0.2963006j, -0.3676850 + 0.5291797j],
    [-0.3646397 - 0.0972916j, +0.4656507 - 0.6549090j, -0.0794616 + 0.1034174j, -0.4209065 - 0.1329068j],
    [+0.0737978 - 0.2226456j, +0.1085991 - 0.1184158j, -0.6278179 + 0.4342744j, +0.3783264 + 0.4396435j],
])

list_q = [
    {
        "u": u1,
        "option": {"all_positive": True, "canonical": True},
        "k": jnp.array([+0.6155514, +0.4149090, +0.2490519])
    },
    {
        "u": u2,
        "option": {"all_positive": True, "canonical": True},
        "k": jnp.array([+0.9771531, +0.5205528, +0.0043738])
    },
    {
        "u": u1,
        "option": {"all_positive": False, "canonical": True},
        "k": jnp.array([+0.6155514, +0.4149090, +0.2490519])
    },
    {
        "u": u2,
        "option": {"all_positive": False, "canonical": True},
        "k": jnp.array([+0.5936432, +0.5205528, -0.0043738])
    },

]
def get_coef_xx(anglex1, u):
    ux = jlin.expm(1j * anglex1 * tensor(sigmax(), identity(2))) @ u
    k, a1, a0, b1, b0 = kak1_decomposition(ux, canonical=True, all_positive=False)
    return k[0]


def test_kak():
    tol_same = 1e-6
    for q in list_q:
        k, a1, a0, b1, b0 = kak1_decomposition(q["u"], **q["option"])
        uc = construct_u_from_k_1q(k, a1, a0, b1, b0)
        err = jnp.abs(jnp.abs(jnp.trace(q["u"] @ uc.conj().T)) - 4)
        # Check if the answer is correct
        assert jnp.abs(k - q["k"]).max() < tol_same
        # Check if U can be reconstructed from results correctly
        assert err < tol_same

def test_kak_grad():
    tol_same = 1e-6
    for u in [u1, u2]:
        fg = jax.value_and_grad(partial(get_coef_xx, u=u))
        # Gradient of input angle should be 0 ( not relevant to 2Q part)
        kxx, gkxx = fg(0.1)
        assert jnp.abs(gkxx) < tol_same