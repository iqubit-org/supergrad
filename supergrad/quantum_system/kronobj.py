"""The Kronecker Object (KronObj) class is designed to represent k-body Hamiltonian
in Kronecker product format. It's particularly useful for representing the sparse
total system Hamiltonian in a form convenient for computing Einstein summation.
"""
import numbers
from math import sqrt

import numpy as np
import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.scipy.linalg as jaxLA
import opt_einsum as oe

from supergrad.utils.utility import tensor, permute


@register_pytree_node_class
class KronObj(object):
    """The Kronecker Object (KronObj) class is designed to represent k-body Hamiltonian
    in Kronecker product format. It's particularly useful for representing the sparse
    total system Hamiltonian in a form convenient for computing Einstein summation.

    Additionally, the KronObj class supports mathematical operations such as addition(+)
    and multiplication(@) between KronObj instances.

    Args:
        inpt (list): list of array
            The local Hamiltonian that represents the Hamiltonian of each subsystem
            in a composite system. Each element of the `inpt` corresponds to
            the local Hamiltonian of the corresponding subsystem tagged in the `locs`.
        dims (list): list of int
            Dimensions of subsystems in the composited Hilbert Space.
        locs (list): list of int
            Location information of corresponding local Hamiltonian.
            For example, the first element of locs is the index of the first
            subsystem in `self.dims`, and so on.
        diag_unitary (array):
            Unitary of diagonal-unitary format.
        _nested_inpt (bool):
            For internal using only.
    """

    def __init__(self,
                 inpt=None,
                 dims=None,
                 locs=None,
                 _nested_inpt=False,
                 diag_unitary=None) -> None:

        if isinstance(inpt, (list, tuple, np.ndarray, jnp.ndarray)):
            # ndarray shape checker
            if _nested_inpt:
                # identify diagonal-unitary representation
                if diag_unitary is None:
                    self.diag_unitary = [None] * len(inpt)
                else:
                    self.diag_unitary = list(diag_unitary)
                # identify inpt location
                if locs is None:
                    self.locs = [None] * len(inpt)
                else:
                    self.locs = list(locs)
                if isinstance(inpt, (np.ndarray, jnp.ndarray)):
                    if inpt.ndim < 3:
                        raise ValueError('Unsupported data shape.')
                self._data = []
                if dims is None:
                    raise ValueError('No dimensions')
                else:
                    self.dims = dims
                for nest_list in inpt:
                    self._data.append([array for array in nest_list])
            else:
                # identify diag_unitary method
                if diag_unitary is None:
                    self.diag_unitary = [None]
                else:
                    self.diag_unitary = [diag_unitary]  # in the list form
                # identify localized method
                if locs is None:
                    self.locs = [None]
                else:
                    self.locs = [locs]
                self._data = [[array for array in inpt]]  # nested
                if locs is None and dims is None:
                    self.dims = [list(inpt[0].shape)[0]]
                else:
                    self.dims = dims
        else:
            raise ValueError('Unsupported data shape.')

    def tree_flatten(self):
        """Specifies a flattening recipe.

        Returns:
            a pair of an iterable with the children to be flattened recursively,
            and some opaque auxiliary data to pass back to the unflattening recipe.
            The auxiliary data is stored in the treedef for use during unflattening.
            The auxiliary data could be used, e.g., for dictionary keys.
        """
        return ((self.data, self.diag_unitary), (self.dims, self.locs))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe.

        Args:
            aux_data: the opaque data that was specified during flattening of the
            current treedef.
            children: the unflattened children

        Returns:
            a re-constructed object of the registered type, using the specified
            children and auxiliary data.
        """
        dims, locs = aux_data
        data, diag_unitary = children
        return cls(data,
                   dims,
                   locs,
                   _nested_inpt=True,
                   diag_unitary=diag_unitary)

    def get_data(self):
        return self._data

    def set_data(self, data):
        if not isinstance(data[0], list):
            raise TypeError('KronObj data must be in a nested list.')
        else:
            self._data = data

    data = property(get_data, set_data)

    def _build_sub_kronobj(self, idx):
        """Return a subset of `KronObj` at selected `idx`."""
        return KronObj(self.data[idx],
                       dims=self.dims,
                       locs=self.locs[idx],
                       diag_unitary=self.diag_unitary[idx])

    def _downcast_diagonal_unitary(self):
        """Convert the diagonal-unitary Hamiltonian format to dense matrix."""
        data = []
        for mat, eig_vec in zip(self.data, self.diag_unitary):
            if eig_vec is not None:  # downcasting
                if len(mat) == 1:
                    data.append([eig_vec * mat[0] @ jnp.conj(eig_vec).T])
                else:
                    raise TypeError('Unsupported diagonal-unitary format')
            else:
                data.append(mat)
        return KronObj(data, _nested_inpt=True, dims=self.dims, locs=self.locs)

    def _downcast_local_hamiltonian(self):
        """Cast the local Hamiltonian format to dense matrix."""
        # firstly downcast diag unitary
        if any(self.diag_status):
            self._downcast_diagonal_unitary()
        data = []
        for mats, local_list in zip(self.data, self.locs):
            if local_list is not None:  # downcasting
                # construct identity matrix list
                ilocs = [
                    idx for idx, _ in enumerate(self.dims)
                    if idx not in local_list
                ]
                idims = np.array(self.dims)[ilocs].prod()
                full_mat = tensor(*mats, jnp.eye(idims))
                tlocs = local_list + ilocs
                order = [tlocs.index(loc) for loc, _ in enumerate(self.dims)]
                # permute the full matrix in the correct order
                data.append([
                    permute(full_mat, [self.dims[loc] for loc in tlocs], order)
                ])
            else:
                data.append(mats)
        return KronObj(data, _nested_inpt=True, dims=self.dims)

    def __iter__(self):
        """iteration through nested list."""
        sublist = [
            self._build_sub_kronobj(ind) for ind, _ in enumerate(self.data)
        ]
        return iter(sublist)

    def __getitem__(self, key):

        if isinstance(key, int):
            return self._build_sub_kronobj(key)
        raise KeyError(f"Unrecognized key: {key}. Key must be an integer index")

    def __add__(self, other):
        """Addition with KronObj on left. The trivial addition is defined as
        concatenating the list.
        """
        if other == 0:  # identity of addition operator
            return self
        else:  # case for matching KronObj
            return KronObj(self.data + other.data,
                           self.dims,
                           _nested_inpt=True,
                           diag_unitary=self.diag_unitary + other.diag_unitary,
                           locs=self.locs + other.locs)

    def __radd__(self, other):
        """Addition with KronObj on right. Note the addition satisfies the
        commutative.
        """
        return self + other

    def __sub__(self, other):
        """Subtraction with KronObj on left."""
        return self + (-1.0 * other)

    def __rsub__(self, other):
        """Subtraction with KronObj on right."""
        return (-1.0 * self) + other

    def __mul__(self, other):
        """Multiplication with KronObj on left, the operator only acts on first
        term in the nested list. Note the scale multiplication satisfies the
        commutative.
        """
        if isinstance(other, (numbers.Number, jax.Array)):
            data = []
            for nest in self.data:
                # multiply scale number to first term
                array_0 = nest[0] * other
                data.append([array_0] + nest[1:])
            return KronObj(data,
                           self.dims,
                           _nested_inpt=True,
                           diag_unitary=self.diag_unitary,
                           locs=self.locs)
        else:
            raise NotImplementedError(f'Do not support instance {type(other)}.')

    def __rmul__(self, other):
        """Multiplication with KronObj on right"""
        return self * other

    def __matmul__(self, other):
        """Matrix multiplication with KronObj on left. The high efficient
        matrix-vector multiplication method working when `other` has matched shape.
        """
        new_self = self
        # downcast the diag unitary
        if any(self.diag_status):
            new_self = new_self._downcast_diagonal_unitary()
        if isinstance(other, KronObj):
            if all(self.loc_status) and all(other.loc_status) and not any(
                    self.diag_status) and not any(other.diag_status):
                # The product of two Kronecker products yields another Kronecker product
                res = []
                for kron_a in self:
                    for kron_b in other:
                        # concatenate loc info
                        dot_data = []
                        loc_info = sorted(set(kron_a.locs[0] + kron_b.locs[0]))
                        for loc in loc_info:
                            try:
                                id_a = kron_a.locs[0].index(loc)
                                mat_a = kron_a.data[0][id_a]
                            except ValueError:
                                mat_a = jnp.eye(kron_a.dims[loc])
                            try:
                                id_b = kron_b.locs[0].index(loc)
                                mat_b = kron_b.data[0][id_b]
                            except ValueError:
                                mat_b = jnp.eye(kron_b.dims[loc])
                            dot_data.append(mat_a @ mat_b)
                        # create new KronObj
                        res.append(KronObj(dot_data, self.dims, locs=loc_info))
                return sum(res)
            else:
                return self.full() @ other.full()
        elif isinstance(other, (np.ndarray, jnp.ndarray)):
            if other.shape[1] == 1:  # state vector
                if not all(new_self.loc_status):
                    return new_self.full() @ other
                res = []
                # implement Einstein summation
                for sub_kronobj in new_self:
                    # construct subscripts and operands
                    if len(sub_kronobj.data[0]) > 1:
                        unitary_op = tensor(*sub_kronobj.data[0])
                    else:
                        unitary_op = sub_kronobj.data[0][0]
                    locs = [loc + 1 for loc in sub_kronobj.locs[0]
                           ]  # index start from 1
                    aux_locs = [-loc for loc in locs]
                    unitary_dim = [
                        sub_kronobj.dims[idx] for idx in sub_kronobj.locs[0]
                    ]
                    unitary_op = unitary_op.reshape(unitary_dim * 2)
                    unitary_subscript = aux_locs + locs
                    psi_op = other.reshape(sub_kronobj.dims)
                    psi_subscript = list(np.arange(len(sub_kronobj.dims)) + 1)
                    res_subscript = psi_subscript.copy()
                    for loc in sub_kronobj.locs[0]:
                        res_subscript[loc] *= -1

                    new_psi = jnp.einsum(unitary_op, unitary_subscript, psi_op,
                                         psi_subscript,
                                         res_subscript).reshape(-1, 1)
                    res.append(new_psi)
                return sum(res)
            else:
                return new_self.full() @ other
        else:
            raise NotImplementedError(f'Do not support instance {type(other)}.')

    def __rmatmul__(self, other):
        """Matrix multiplication with KronObj on right"""
        if isinstance(other, (np.ndarray, jnp.ndarray)):
            return other @ self.full()
        else:
            raise NotImplementedError(f'Do not support instance {type(other)}.')

    def __str__(self):
        s = ""
        shape = self.shape
        s += ("Kronecker object: " + "dims = " + str(self.dims) + ", shape = " +
              str(shape) + ", diag_status = " + str(self.diag_status) +
              ", location_info = " + str(self.locs) + "\n")
        s += "KronObj data =\n"
        s += str(self.data)

        return s

    def __repr__(self):
        # give complete information on KronObj without print statement in
        # command-line we cant realistically serialize a KronObj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()

    @property
    def shape(self):
        """The shape of unitary"""
        res = [np.prod(self.dims), np.prod(self.dims)]
        return res

    @property
    def diag_status(self):
        """The diagonal-unitary format status of subsystems."""
        return [eig_vec is not None for eig_vec in self.diag_unitary]

    @property
    def loc_status(self):
        """The local Hamiltonian format status of subsystems."""
        return [loc_info is not None for loc_info in self.locs]

    def full(self):
        """Dense matrix from the result of Kronecker product."""
        new_self = self
        # downcast the diag unitary
        if any(self.diag_status):
            new_self = new_self._downcast_diagonal_unitary()
        # downcast the location representation
        if any(self.loc_status):
            new_self = new_self._downcast_local_hamiltonian()
        mat_list = [mats[0] for mats in new_self.data]
        return sum(mat_list)

    def dense(self):
        """Return new KronObj contain dense matrix"""
        data = [self.full()]
        return KronObj(data,
                       dims=self.dims,
                       locs=[i for i, _ in enumerate(self.dims)])

    def dag(self):
        """Conjugate transpose over tensor product"""
        data = []
        for nest_list in self.data:
            new_nest_list = []
            for mat in nest_list:
                new_nest_list.append(mat.conj().T)
            data.append(new_nest_list)
        return KronObj(data,
                       _nested_inpt=True,
                       dims=self.dims,
                       locs=self.locs,
                       diag_unitary=self.diag_unitary)

    def transpose(self):
        """Transpose over tensor product"""
        data = []
        diag_unitary = []
        for nest_list in self.data:
            new_nest_list = []
            for mat in nest_list:
                new_nest_list.append(mat.T)
            data.append(new_nest_list)
        for eig_vec in self.diag_unitary:
            if eig_vec is not None:
                diag_unitary.append(eig_vec.conj())
            else:
                diag_unitary.append(None)
        return KronObj(data,
                       _nested_inpt=True,
                       dims=self.dims,
                       locs=self.locs,
                       diag_unitary=diag_unitary)

    def conjugate(self):
        """Conjugate over tensor product"""
        data = []
        diag_unitary = []
        for nest_list in self.data:
            new_nest_list = []
            for mat in nest_list:
                new_nest_list.append(mat.conj())
            data.append(new_nest_list)
        for eig_vec in self.diag_unitary:
            if eig_vec is not None:
                diag_unitary.append(eig_vec.conj())
            else:
                diag_unitary.append(None)
        return KronObj(data,
                       _nested_inpt=True,
                       dims=self.dims,
                       locs=self.locs,
                       diag_unitary=diag_unitary)

    def diagonalize_operator(self):
        """Diagonalize Hermitian to diagonal-unitary representation."""
        data = []
        diag_unitary = []
        for kron_region in self.data:
            mat = tensor(*kron_region)
            # we cannot diagonalize a non-Hermite matrix
            # Temporarily comment it out to enhance jit support, you must ensure
            # the input is Hermitian to use diag evolution method.

            # if not np.allclose(np.conj(mat).T, mat):
            #     continue
            eig, eigv = jaxLA.eigh(mat)
            data.append([eig])
            diag_unitary.append(
                jax.lax.stop_gradient(eigv))  # Fix complex casting bug
        return KronObj(data,
                       _nested_inpt=True,
                       dims=self.dims,
                       locs=self.locs,
                       diag_unitary=diag_unitary)

    def compute_contraction_path(self, op_list=None, trotter_order=None):
        """Compute tensor network contraction path.
        Please provide either `op_list` or `trotter_order`. If `op_list` is None,
        it will be generated by `trotter_order`.

        Args:
            op_list: the list of local operator
            trotter_order (complex int): the order of suzuki-trotter decomposition.
                The following arguments are supported,
                a) `None`, calculating matrix exponentiation without trotter decomposition
                b) `1`, first order trotter decomposition
                c) `2`, second order trotter decomposition
                d) `4`, 4th order real decomposition
                e) `4j`, 4th order complex decomposition
        """
        assert op_list is not None or trotter_order is not None
        if op_list is None:
            op_list = self._trotter_decomposition(trotter_order)
        # initialize subscripts by state
        max_indices_per_layer = len(self.dims)
        num_subscript = np.zeros(len(self.dims), dtype=int)
        psi_subscript = ''.join([
            oe.get_symbol(i + max_indices_per_layer * num)
            for i, num in enumerate(num_subscript)
        ])
        oe_subscripts = []
        oe_shapes = []
        # construct tensor network
        oe_subscripts.append(psi_subscript)
        oe_shapes.append(self.dims)
        for sub_kronobj in op_list[::-1]:
            base_subscript = []
            new_subscript = []
            op_shape = []
            for loc in sub_kronobj.locs[0]:
                base_subscript.append(
                    oe.get_symbol(loc +
                                  max_indices_per_layer * num_subscript[loc]))
                num_subscript[loc] += 1
                new_subscript.append(
                    oe.get_symbol(loc +
                                  max_indices_per_layer * num_subscript[loc]))
                op_shape.append(self.dims[loc])
            oe_subscripts.append(''.join(new_subscript + base_subscript))
            oe_shapes.append(op_shape * 2)
        psi_output = ''.join([
            oe.get_symbol(i + max_indices_per_layer * num)
            for i, num in enumerate(num_subscript)
        ])
        eq = ','.join(oe_subscripts) + '->' + psi_output
        expr = oe.contract_expression(eq, *oe_shapes, optimize='auto-hq')
        return expr

    def _trotter_decomposition(self, trotter_order):
        """Generate Suzuki-Trotter decomposition sequence.

        Args:
            trotter_order (complex int): the order of suzuki-trotter decomposition.
                The following arguments are supported,
                a) `None`, calculating matrix exponentiation without trotter decomposition
                b) `1`, first order trotter decomposition
                c) `2`, second order trotter decomposition
                d) `4`, 4th order real decomposition
                e) `4j`, 4th order complex decomposition
        """

        def _loop(trotter_coeff, reverse=False):
            """Generate the matrix multiplication sequence.

            Args:
                trotter_coeff (float): the trotter coefficient
                reverse (bool): whether reverse the sequence or not.
            """
            op_list = []
            if reverse:
                scan_list = zip(reversed(self.data),
                                reversed(self.diag_unitary),
                                reversed(self.locs))
            else:
                scan_list = zip(self.data, self.diag_unitary, self.locs)
            for mat_list, eig_vec, local_info in scan_list:
                if eig_vec is not None:  # cast expm down to exp
                    lam = jnp.exp(trotter_coeff * mat_list[0])
                    op_list.append(
                        KronObj([eig_vec * lam @ jnp.conj(eig_vec).T],
                                self.dims, local_info))
                else:
                    # only calculate matrix exponentiation with the terms in `local_info`
                    obj = tensor(*mat_list) * trotter_coeff
                    op_list.append(
                        KronObj([jaxLA.expm(obj)], self.dims, local_info))
            return op_list

        def _second_order_expm(p=1.0):
            """Generate the matrix multiplication sequence for second order
            trotter decomposition.
            """
            # normal loop
            op_list = _loop(p / 2., reverse=False)
            # reversed loop
            op_list.extend(_loop(p / 2., reverse=True))
            return op_list

        if trotter_order == 1:
            op_list = _loop(1.)
        elif trotter_order == 2:
            op_list = _second_order_expm()
        elif trotter_order == 4:
            # the simplest real decomposition of 4th order STD
            p = 1 / (2 - 2**(1 / 3))
            op_list = _second_order_expm(p)
            op_list.extend(_second_order_expm(1 - 2 * p))
            op_list.extend(_second_order_expm(p))
        elif trotter_order == 4j:
            # the complex 4th STD
            p = (3 - 1j * sqrt(3)) / 6
            p_conj = (3 + 1j * sqrt(3)) / 6
            op_list = _second_order_expm(p)
            op_list.extend(_second_order_expm(p_conj))
        else:
            raise ValueError(f'Unsupported Trotter order {trotter_order}.')

        return op_list

    def expm(self, right_vec=None, trotter_order=None, tn_expr=None):
        """Calculate matrix exponentiation of `KronObj`. Combine matrix exponentiation
        with matrix-vector product if `right_vec` is not `None`.

        Args:
            right_vec (bool): matrix-vector product with the `right_vec` on the right
            trotter_order (complex int): the order of suzuki-trotter decomposition.
                The following arguments are supported,
                a) `None`, calculating matrix exponentiation without trotter decomposition
                b) `1`, first order trotter decomposition
                c) `2`, second order trotter decomposition
                d) `4`, 4th order real decomposition
                e) `4j`, 4th order complex decomposition
            tn_expr: Tensor network contract expression.
                if `tn_expr` is None, generate contract expression when `expm()`
                be called each time.

        """
        if right_vec is None or trotter_order not in [1, 2, 4, 4j]:
            # not implement trotter approximate
            res = jaxLA.expm(self.full())
            if right_vec is not None:
                return res @ right_vec
            else:
                return res
        else:
            op_list = self._trotter_decomposition(trotter_order)
            if tn_expr is None:
                tn_expr = self.compute_contraction_path(op_list)
            oe_arrays = []
            oe_arrays.append(right_vec.reshape(self.dims))
            for sub_kronobj in op_list[::-1]:
                op_dim = [sub_kronobj.dims[idx] for idx in sub_kronobj.locs[0]]
                op_array = sub_kronobj.data[0][0].reshape(op_dim * 2)
                oe_arrays.append(op_array)
            oe_res = tn_expr(*oe_arrays, backend='jax')

            return oe_res.reshape(-1, 1)
