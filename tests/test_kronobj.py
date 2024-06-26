import numpy as np
import scipy.linalg as sp

from supergrad.quantum_system.kronobj import KronObj

rng = np.random.default_rng(seed=42)
dims = [3, 5, 6, 4, 2]
mat0 = rng.random((dims[0], dims[0]))
mat1 = rng.random((dims[1], dims[1]))
mat3 = rng.random((dims[3], dims[3]))
kron1 = KronObj([mat0, mat1, mat3], dims, [0, 1, 3])
manual1 = np.kron(mat0, mat1)
manual1 = np.kron(manual1, np.eye(dims[2]))
manual1 = np.kron(manual1, mat3)
manual1 = np.kron(manual1, np.eye(dims[4]))

mat4 = rng.random((dims[0], dims[0]))
mat5 = rng.random((dims[2], dims[2]))
kron2 = KronObj([mat4, mat5], dims, [0, 2])
manual2 = np.kron(mat4, np.eye(dims[1]))
manual2 = np.kron(manual2, mat5)
manual2 = np.kron(manual2, np.eye(dims[3]))
manual2 = np.kron(manual2, np.eye(dims[4]))


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
    res = sp.expm((manual1 + manual2))
    res0 = (kron1 + kron2).expm()
    assert np.allclose(res0, res)


def test_matmul_between_kronobj():
    assert np.allclose((kron1 @ kron2).full(), manual1 @ manual2)
    assert np.allclose(((kron1 + kron2) @ (kron1 + kron1)).full(),
                       (manual1 + manual2) @ (manual1 + manual1))
