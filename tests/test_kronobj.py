import numpy as np
import jax.scipy.linalg as jla

import supergrad
from supergrad.quantum_system.kronobj import KronObj


def random_complex_matrix(dim):
    return rng.random((dim, dim)) + 1j * rng.random((dim, dim))


rng = np.random.default_rng(seed=42)
dims = [3, 5, 6, 4, 2]
mat0 = random_complex_matrix(dims[0])
mat1 = random_complex_matrix(dims[1])
mat3 = random_complex_matrix(dims[3])
kron1 = KronObj([mat0, mat1, mat3], dims, [0, 1, 3])
manual1 = supergrad.tensor(mat0, mat1, np.eye(dims[2]), mat3, np.eye(dims[4]))

mat2 = random_complex_matrix(dims[2])
mat4 = random_complex_matrix(dims[4])
kron2 = KronObj([mat2, mat4], dims, [2, 4])
manual2 = supergrad.tensor(np.eye(dims[0]), np.eye(dims[1]), mat2,
                           np.eye(dims[3]), mat4)


def test_kronecker_product():

    assert np.allclose(manual1, kron1.full())
    assert np.allclose(manual2, kron2.full())


def test_addition():

    assert np.allclose(manual1 + manual2, (kron1 + kron2).full())
    assert np.allclose(manual2 + manual1, (kron2 + kron1).full())
    assert np.allclose(manual1 - manual2, (kron1 - kron2).full())
    assert np.allclose(manual2 - manual1, (kron2 - kron1).full())


def test_dag():

    assert np.allclose(
        np.conj(manual1 + manual2).T, (kron1 + kron2).dag().full())


def test_scalar_multiplication():
    flt = rng.random()
    assert np.allclose(flt * (manual1 + manual2),
                       (flt * (kron1 + kron2)).full())
    assert np.allclose(flt * (manual1 + manual2),
                       ((kron1 + kron2) * flt).full())


def test_vector_multiplication():
    psi = rng.random(kron1.shape[0])[:, np.newaxis]
    assert np.allclose((manual1 + manual2) @ psi, ((kron1 + kron2) @ psi))


def test_matrix_exponential():
    res = jla.expm((manual1 + manual2))
    res0 = (kron1 + kron2).expm()
    assert np.allclose(res0, res)


def test_exponential_matrix_vecmul():
    psi = rng.random(kron2.shape[0])[:, np.newaxis]
    mixed_kron = kron1 + kron2
    assert np.allclose(kron1.expm(psi, 1), jla.expm(manual1) @ psi)
    assert np.allclose(kron2.expm(psi, 1), jla.expm(manual2) @ psi)
    assert np.allclose(mixed_kron.expm(psi, 1),
                       jla.expm(manual1 + manual2) @ psi)


def test_matmul_between_kronobj():
    assert np.allclose((kron1 @ kron2).full(), manual1 @ manual2)
    assert np.allclose(((kron1 + kron2) @ (kron1 + kron1)).full(),
                       (manual1 + manual2) @ (manual1 + manual1))
