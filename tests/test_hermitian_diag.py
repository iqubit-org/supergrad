import numpy as np
import jax.scipy.linalg as jla
import pytest

from supergrad.quantum_system.kronobj import KronObj

rng = np.random.default_rng(seed=42)
dims = [2, 4, 3, 5, 2]
locs_list = [[[2]], [[2, 3]], [[4, 1]], [[2, 4, 0]], [[0], [1, 2], [3, 1, 2]]]
conjugate_list = ['dag', 'transpose', 'conjugate']


def random_complex_hermitian(dim):
    a = rng.random((dim, dim)) + 1j * rng.random((dim, dim))
    return (a + a.conj().T) / 2


@pytest.mark.parametrize('locs', locs_list)
def test_diag_representation(locs):
    mats = []
    for loc in locs:
        dims_cfg = [dims[i] for i in loc]
        mats.append([random_complex_hermitian(dim) for dim in dims_cfg])
    kr = KronObj(mats, dims, locs, _nested_inpt=True)
    diag_kr = kr.diagonalize_operator()
    assert all(diag_kr.diag_status)
    assert np.allclose(kr.full(), diag_kr.full())
    mixed_kr = kr + diag_kr
    assert np.allclose(mixed_kr.full(), 2 * kr.full())


@pytest.mark.parametrize('locs', locs_list)
@pytest.mark.parametrize('cj_type', conjugate_list)
def test_diag_conjugate(locs, cj_type):
    mats = []
    for loc in locs:
        dims_cfg = [dims[i] for i in loc]
        mats.append([random_complex_hermitian(dim) for dim in dims_cfg])
    kr = KronObj(mats, dims, locs, _nested_inpt=True)
    diag_kr = kr.diagonalize_operator()
    mixed_kr = kr + diag_kr
    for i, k in enumerate([diag_kr, mixed_kr], 1):
        if cj_type == 'dag':
            assert np.allclose(np.conj(kr.full() * i).T, k.dag().full())
        elif cj_type == 'transpose':
            assert np.allclose(np.transpose(kr.full() * i),
                               k.transpose().full())
        elif cj_type == 'conjugate':
            assert np.allclose(np.conj(kr.full()) * i, k.conjugate().full())


# Skip the last one with Trotter
@pytest.mark.parametrize('locs', locs_list[:-1])
def test_diag_exp_vecmul(locs):
    mats = []
    for loc in locs:
        dims_cfg = [dims[i] for i in loc]
        mats.append([random_complex_hermitian(dim) for dim in dims_cfg])
    kr = KronObj(mats, dims, locs, _nested_inpt=True)
    diag_kr = kr.diagonalize_operator()
    psi = rng.random(kr.shape[0])[:, np.newaxis]
    assert np.allclose(diag_kr.expm(psi, 1), kr.expm(psi, 1))
    assert np.allclose(kr.expm(psi, 1), jla.expm(kr.full()) @ psi)
    assert np.allclose(diag_kr.expm(psi, 1), jla.expm(diag_kr.full()) @ psi)
