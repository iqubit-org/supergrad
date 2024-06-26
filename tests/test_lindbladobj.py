import numpy as np
import scipy.linalg as sp

from supergrad.quantum_system import KronObj, LindbladObj

from jax import config

config.update('jax_enable_x64', True)

rng = np.random.default_rng(seed=42)
dims = [3, 2, 4]
mat0 = rng.random((dims[0], dims[0]))
mat0_1 = rng.random((dims[0], dims[0]))
mat1 = rng.random((dims[1], dims[1]))
mat2 = rng.random((dims[2], dims[2]))
kron1 = KronObj([mat0, mat1], dims, [0, 1])
manual1 = np.kron(mat0, mat1)
manual1 = np.kron(manual1, np.eye(dims[2]))

mat22 = rng.random((dims[0], dims[0]))
mat3 = rng.random((dims[2], dims[2]))
kron2 = KronObj([mat22, mat3], dims, [0, 2])
manual2 = np.kron(mat22, np.eye(dims[1]))
manual2 = np.kron(manual2, mat3)

mat4 = rng.random((dims[1], dims[1]))
kron3 = KronObj([mat4], dims, [1])
manual3 = np.kron(np.eye(dims[0]), mat4)
manual3 = np.kron(manual3, np.eye(dims[2]))

ident = np.eye(manual1.shape[0], dtype=complex)

density = rng.random((np.prod(dims), np.prod(dims)))

lind1 = LindbladObj()
lind1.add_liouvillian(kron1)
lind2 = LindbladObj()
lind2.add_lindblad_operator(kron2)
lind3 = LindbladObj()
chi3 = 0.3
lind3.add_lindblad_operator(kron2, kron1 + kron3, chi3)

# Convert Hamiltonian to Liouvillian
manual_lind1 = -1.0j * (np.kron(ident, manual1) -
                        np.kron(manual1.transpose(), ident))
# Convert collapse opeartor to dissipator
ad_b = manual2.conjugate().transpose() @ manual2
manual_lind2 = np.kron(manual2.conjugate(), manual2) - 0.5 * np.kron(
    ident, ad_b) - 0.5 * np.kron(ad_b.transpose(), ident)

ad_b = manual2.conjugate().transpose() @ (manual1 + manual3)
manual_lind3 = np.kron((manual1 + manual3).conjugate(), manual2) * np.exp(
    1j * chi3) - 0.5 * np.kron(ident, ad_b) - 0.5 * np.kron(
        ad_b.transpose(), ident)


def test_liouvillian():
    assert np.allclose(manual_lind1, lind1.full().reshape(manual_lind1.shape))


def test_dissipator():
    assert np.allclose(manual_lind2, lind2.full().reshape(manual_lind2.shape))
    assert np.allclose(manual_lind3, lind3.full().reshape(manual_lind3.shape))


def test_tensor_multiplication():
    res = (manual_lind1 @ density.ravel('F')).reshape(np.prod(dims),
                                                      np.prod(dims)).T
    res0 = lind1 @ density
    assert np.allclose(res0, res)
    res = (manual_lind2 @ density.ravel('F')).reshape(np.prod(dims),
                                                      np.prod(dims)).T
    res0 = lind2 @ density
    assert np.allclose(res0, res)
    res = (manual_lind3 @ density.ravel('F')).reshape(np.prod(dims),
                                                      np.prod(dims)).T
    res0 = lind3 @ density
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit1():
    lind_test = LindbladObj([[mat0, mat2], [mat1]], dims, [[0, 2], [1]])
    manual_test = np.kron(mat0, np.eye(dims[1]))
    manual_test = np.kron(manual_test, mat2)
    manual_test_1 = np.kron(np.eye(dims[0]), mat1)
    manual_test_1 = np.kron(manual_test_1, np.eye(dims[2]))
    manual_lind_test = np.kron(manual_test, manual_test_1)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit2():
    lind_test = LindbladObj([[mat0], [mat0_1, mat1]], dims, [[0], [0, 1]])
    manual_test = np.kron(mat0, np.eye(dims[1]))
    manual_test = np.kron(manual_test, np.eye(dims[2]))
    manual_test_1 = np.kron(mat0_1, mat1)
    manual_test_1 = np.kron(manual_test_1, np.eye(dims[2]))
    manual_lind_test = np.kron(manual_test, manual_test_1)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density, 1)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit3():
    lind_test = LindbladObj([[mat0], [mat1]], dims, [[0], [1]])
    manual_test = np.kron(mat0, np.eye(dims[1]))
    manual_test = np.kron(manual_test, np.eye(dims[2]))
    manual_test_1 = np.kron(np.eye(dims[0]), mat1)
    manual_test_1 = np.kron(manual_test_1, np.eye(dims[2]))
    manual_lind_test = np.kron(manual_test, manual_test_1)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density, 1)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit4():
    lind_test = LindbladObj([[mat0_1], [mat0]], dims, [[0], [0]])
    manual_test = np.kron(mat0, np.eye(dims[1]))
    manual_test = np.kron(manual_test, np.eye(dims[2]))
    manual_test_1 = np.kron(mat0_1, np.eye(dims[1]))
    manual_test_1 = np.kron(manual_test_1, np.eye(dims[2]))
    manual_lind_test = np.kron(manual_test_1, manual_test)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density, 1)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit5():
    lind_test = LindbladObj([[], [mat0]], dims, [[], [0]])
    manual_test = np.kron(mat0, np.eye(dims[1]))
    manual_test = np.kron(manual_test, np.eye(dims[2]))
    manual_lind_test = np.kron(ident, manual_test)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density, 1)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit6():
    lind_test = LindbladObj([[mat0, mat1, mat2], []], dims, [[0, 1, 2], []])
    manual_test = np.kron(mat0, mat1)
    manual_test = np.kron(manual_test, mat2)
    manual_lind_test = np.kron(manual_test, ident)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density, 1)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit7():
    lind_test = LindbladObj([[mat0, mat1], [mat0, mat1]], dims,
                            [[0, 1], [0, 1]])
    manual_test = np.kron(mat0, mat1)
    manual_test = np.kron(manual_test, np.eye(dims[2]))
    manual_lind_test = np.kron(manual_test, manual_test)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density, 1)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_tensor_multiplication_unit8():
    lind_test = LindbladObj([[mat0, mat1], []], dims, [[0, 1], []])
    manual_test = np.kron(mat0, mat1)
    manual_test = np.kron(manual_test, np.eye(dims[2]))
    manual_lind_test = np.kron(manual_test, ident)
    res = (manual_lind_test @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0 = lind_test @ density
    res_exp = (sp.expm(manual_lind_test) @ density.ravel('F')).reshape(
        np.prod(dims), np.prod(dims)).T
    res0_exp = lind_test.expm(density, 1)
    assert np.allclose(res0_exp, res_exp)
    assert np.allclose(manual_lind_test,
                       lind_test.full().reshape(manual_lind_test.shape))
    assert np.allclose(res0, res)


def test_addition():
    assert np.allclose(manual_lind1 + manual_lind2,
                       (lind1 + lind2).full().reshape(manual_lind1.shape))
    assert np.allclose(manual_lind1 + manual_lind2,
                       (lind2 + lind1).full().reshape(manual_lind1.shape))
    assert np.allclose(manual_lind2 - manual_lind3,
                       (lind2 - lind3).full().reshape(manual_lind1.shape))
    assert np.allclose(manual_lind2 - manual_lind1,
                       (lind2 - lind1).full().reshape(manual_lind1.shape))


def test_scalar_multiplication():
    flt = rng.random()
    assert np.allclose(flt * (manual_lind1 + manual_lind2),
                       (flt * (lind1 + lind2)).full().reshape(
                           manual_lind1.shape))
    assert np.allclose(flt * (manual_lind1 + manual_lind2),
                       ((lind1 + lind2) * flt).full().reshape(
                           manual_lind1.shape))
