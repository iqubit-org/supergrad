"""Fidelity function for benchmark"""


def fidelity(U, V):
    """the fidelity function for unitary matrices."""
    inner_product = (U.conj() * V).sum()
    return (abs(inner_product) / U.shape[0])**2
