"""The Lindblad Object (LindbladObj) class is designed to represent superoperator
in Kronecker product format. It's particularly useful for representing sparse total systems'
superoperator in a form convenient for computing Einstein summation.
"""
import numbers
from math import sqrt
import copy

import numpy as np
import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.scipy.linalg as jaxLA
import opt_einsum as oe

from supergrad.utils.utility import tensor
from supergrad.quantum_system.kronobj import KronObj


@register_pytree_node_class
class LindbladObj(object):
    """The Lindblad Object (LindbladObj) class is designed to represent
    superoperator in Kronecker product format. It's particularly useful for
    representing sparse total systems' superoperator in a form convenient for
    computing Einstein summation.

    We express the Lindblad equation for a tensor-4 superoperator, the Hamiltonian
    part and Lindblad generator part are expressed as the tensor product of two
    operators. The local Hamiltonian and corresponding location should be
    specified in the `inpt` and `locs` argument.

    Args:
        inpt (list): list of list of array
            `inpt` is a nested list represents a tensor product, while the first
            list corresponds to the left operand and the second list corresponds
            to right operand. Each element of `operand` corresponds to
            the local Hamiltonian of the corresponding subsystem tagged in the `locs`.
        dims (list): list of int
            Dimensions of subsystems in the composited Hilbert Space.
        locs (list): list of int
            Location information of corresponding local Hamiltonian. `locs` is a
            nested list represents a tensor product, while the first list corresponds
            to the left operand and the second list corresponds to right operand.
            For example, the first element of left operand is the index of the
            first subsystem in `self.dims`, and so on.
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
        if _nested_inpt:
            # identify diag_unitary method
            if diag_unitary is None:
                self.diag_unitary = [[None, None]] * len(inpt)
            else:
                self.diag_unitary = list(diag_unitary)
            # identify inpt location
            if locs is None:
                self.locs = [[None, None]] * len(inpt)
            else:
                self.locs = list(locs)
            self._data = []
            self.dims = dims
            # check the data formation
            for liou_region in inpt:
                self._data.append([ele for ele in liou_region])
        elif inpt is None:
            # idle constructor
            self.locs = []
            self.diag_unitary = []
            self._data = []
            self.dims = None
        else:
            # identify diagonal-unitary representation
            if diag_unitary is None or None in diag_unitary:
                self.diag_unitary = [[None, None]]
            else:
                # for ul_region in inpt:
                #     if ul_region and ul_region[0].ndim == 1:
                #         # convert eigenenergies to diagonal matrix
                #         ul_region = [jnp.diag(ul_region[0])]
                self.diag_unitary = [diag_unitary]  # nested list
            # identify localized method
            if locs is None:
                self.locs = [[None, None]]
            else:
                self.locs = [locs]
            self._data = [[ele for ele in inpt]]  # nested
            if dims is None:
                raise ValueError('No Hilbertspace dimensions. ')
            self.dims = dims

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
            raise TypeError('LindbladObj data must be in a nested list.')
        else:
            self._data = data

    data = property(get_data, set_data)

    def _build_sub_lindbladobj(self, idx):
        """Return a subset of `LindbladObj` at selected `idx`."""
        return LindbladObj(self.data[idx],
                           dims=self.dims,
                           locs=self.locs[idx],
                           diag_unitary=self.diag_unitary[idx])

    @property
    def shape(self):
        """The shape of unitary"""
        res = [np.prod(self.dims), np.prod(self.dims)] * 2
        return res

    @property
    def loc_status(self):
        """The local format status of subsystems."""
        return [[ul_loc_info is not None
                 for ul_loc_info in liou_loc_info]
                for liou_loc_info in self.locs]

    @property
    def diag_status(self):
        """The diagonal-unitary format status of subsystems."""
        return [[ul_eig_vec is not None
                 for ul_eig_vec in liou_diag_info]
                for liou_diag_info in self.diag_unitary]

    def __str__(self):
        s = ""
        shape = self.shape
        s += ("Lindblad object: " + "dims = " + str(self.dims) + ", shape = " +
              str(shape) + ", diag_status = " + str(self.diag_status) +
              ", location_info = " + str(self.locs) + "\n")
        s += "LindbladObj data =\n"
        s += str(self.data)

        return s

    def __repr__(self):
        # give complete information on LindbladObj without print statement in
        # command-line we cant realistically serialize a LindbladObj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()

    def _downcast_diagonal_unitary(self):
        """Convert the diagonal-unitary format to order-4 tensor."""
        data = []
        for liou_region, liou_diag_info in zip(self.data, self.diag_unitary):
            if None not in liou_diag_info:  # downcasting
                liou_list = []
                if all(liou_diag_info):
                    raise ValueError('Unsupported diagonal-unitary info.')
                for ul_region, ul_eig_vec in zip(liou_region, liou_diag_info):
                    if ul_region:
                        liou_list.append([
                            ul_eig_vec @ ul_region[0] @ jnp.conj(ul_eig_vec).T
                        ])
                    else:
                        liou_list.append([])
                data.append(liou_list)
            else:
                data.append(liou_region)
        return LindbladObj(data,
                           _nested_inpt=True,
                           dims=self.dims,
                           locs=self.locs)

    def _downcast_local_hamiltonian(self):
        """Convert the local format to the order-4 tensor."""
        # firstly downcast diag unitary
        data = []
        for liou_region, liou_local in zip(self.data, self.locs):
            if None not in liou_local:  # downcasting to order-4 tensor
                liou_list = []
                for ul_region, ul_local in zip(liou_region, liou_local):
                    if ul_local is None:
                        liou_list.append(ul_region[0])
                    elif ul_local:  # downcasting to matrix
                        # construct identity matrix list
                        mat_list = [jnp.eye(dim) for dim in self.dims]
                        for mat, local in zip(ul_region, ul_local):
                            mat_list[local] = mat
                        # calculate tensor product
                        liou_list.append(tensor(*mat_list))
                    else:  # []
                        liou_list.append(jnp.eye(np.prod(self.dims)))
                data.append([tensor(*liou_list).reshape(self.shape)])
            else:
                data.append(liou_local)

        return LindbladObj(data, _nested_inpt=True, dims=self.dims)

    def full(self):
        """Dense order-4 tensor from the result of Kronecker product."""
        new_self = self
        # downcast the diag unitary
        if any(self.diag_status):
            new_self = new_self._downcast_diagonal_unitary()
        # downcast the location representation
        if any(self.loc_status):
            new_self = new_self._downcast_local_hamiltonian()
        ts_list = [ts[0] for ts in new_self.data]
        return sum(ts_list)

    def __iter__(self):
        """iteration through nested list."""
        sublist = [
            self._build_sub_lindbladobj(ind) for ind, _ in enumerate(self.data)
        ]
        return iter(sublist)

    def __getitem__(self, key):

        if isinstance(key, int):
            return self._build_sub_lindbladobj(key)
        raise KeyError(f"Unrecognized key: {key}. Key must be an integer index")

    def add_liouvillian(self, ham: KronObj):
        """Convert Hamiltonian to Liouvillian.

        Args:
            ham (`KronObj`) : Hamiltonian
        """
        # Liouvillian is a 4-order tensor, but each index share the same dimension
        if self.dims is None:
            self.dims = ham.dims
        else:
            assert self.dims == ham.dims
        for aham in ham:
            if aham.diag_unitary[0] is None:
                part_diag_info = None
            else:
                part_diag_info = 0
            self.data += [[[], (-1.0j * aham).data[0]]]
            self.locs += [[[], aham.locs[0]]]
            self.diag_unitary += [[part_diag_info, aham.diag_unitary[0]]]

            self.data += [[(1.0j * aham).transpose().data[0], []]]
            self.locs += [[aham.locs[0], []]]
            self.diag_unitary += [[aham.diag_unitary[0], part_diag_info]]

    def add_lindblad_operator(self, a: KronObj, b: KronObj = None, chi=None):
        """Lindblad operator for a single pair of collapse operators
        (a, b), or for a single collapse operator (a) when b is not specified:

        .. math::

            \\mathcal{D}[a,b]\\rho = a \\rho b^\\dagger -
            \\frac{1}{2}a^\\dagger b\\rho - \\frac{1}{2}\\rho a^\\dagger b

        Args:
            a: KronObj
                Left part of collapse operator.
            b: KronObj
                Right part of collapse operator. If not specified, b defaults to a.
            chi: float
        """
        if self.dims is None:
            self.dims = a.dims
        else:
            assert self.dims == a.dims
        if b is None:
            temp_b = a
        else:
            assert b.dims == a.dims
            temp_b = b

        for aa in a:
            for bb in temp_b:
                if chi:
                    dissi0_data = [
                        bb.conjugate().data[0],
                        (jnp.exp(1.0j * chi) * aa).data[0]
                    ]
                else:
                    dissi0_data = [bb.conjugate().data[0], aa.data[0]]
                self.data += [dissi0_data]
                self.locs += [[bb.locs[0], aa.locs[0]]]
                self.diag_unitary += [[aa.diag_unitary[0], bb.diag_unitary[0]]]
                if b is None and len(a.data) == 1 and len(
                        a.data[0]) == 1 and all(a.diag_status):
                    # The product of two Kronecker products yields another Kronecker product
                    # output keeps the diag representation
                    # only support regular dissipator collapse_b = collapse_a
                    # note here kron_a = kron_b.dag()
                    # new_diag = -0.5 * jnp.diag(
                    #     a.data[0][0]).conjugate() * jnp.diag(a.data[0][0])
                    new_diag = -0.5 * jnp.conjugate(a.data[0][0]) * a.data[0][0]
                    ad_b = KronObj([new_diag],
                                   a.dims,
                                   locs=a.locs[0],
                                   diag_unitary=a.diag_unitary[0])
                else:
                    ad_b = -0.5 * aa.dag() @ bb
                if ad_b.diag_unitary[0] is None:
                    part_diag_info = None
                else:
                    part_diag_info = 0
                self.data += [[[], ad_b.data[0]]]
                self.locs += [[[], ad_b.locs[0]]]
                self.diag_unitary += [[part_diag_info, ad_b.diag_unitary[0]]]

                self.data += [[ad_b.transpose().data[0], []]]
                self.locs += [[ad_b.locs[0], []]]
                self.diag_unitary += [[ad_b.diag_unitary[0], part_diag_info]]

    def _diagonalize_super_operator(self):
        """Diagonalize super operator."""
        for idx, liou_region in enumerate(self.data):
            if all(liou_region):
                union_ul_local = sorted(
                    set(self.locs[idx][0] + self.locs[idx][1]))
                ul_list = []
                for ul_region, ul_local in zip(liou_region, self.locs[idx]):
                    mat_list = [
                        jnp.eye(self.dims[loc]) for loc in union_ul_local
                    ]
                    for mat, loc in zip(ul_region, ul_local):
                        loc_idx = union_ul_local.index(loc)
                        mat_list[loc_idx] = mat
                    ul_list.append(tensor(*mat_list))
                # only calculate matrix exponentiation with the terms in `local_info`
                obj = tensor(*ul_list[::-1])
                # we cannot diagonalize a non-Hermite matrix
                # TODO: add jit support
                if not jnp.allclose(jnp.conj(obj).T, obj):
                    continue
                eig, eigv = jaxLA.eigh(obj)
                # diagonalize the super operator
                tensor_dim = np.prod([self.dims[loc] for loc in union_ul_local])
                # self.data[idx] = [jnp.diag(eig).reshape(*[tensor_dim] * 4)]
                self.data[idx] = [eig.reshape(*[tensor_dim] * 2)]
                self.diag_unitary[idx] = eigv.reshape(*[tensor_dim] * 4)
                self.locs[idx] = [union_ul_local, union_ul_local]
            else:
                mat = tensor(*(liou_region[0] + liou_region[1]))
                # we cannot diagonalize a non-Hermite matrix
                # TODO: add jit support
                if not jnp.allclose(jnp.conj(mat).T, mat):
                    continue
                eig, eigv = jaxLA.eigh(mat)
                if liou_region[0]:
                    self.data[idx] = [[eig], []]
                    self.diag_unitary[idx] = [eigv, 0]
                else:
                    self.data[idx] = [[], [eig]]
                    self.diag_unitary[idx] = [0, eigv]

    def __add__(self, other):
        """Addition with LindbladObj on left. The trivial addition is defined as
        concatenating the list.
        """
        if other == 0:  # identity of addition operator
            return self
        else:  # case for matching KronObj
            return LindbladObj(self.data + other.data,
                               self.dims,
                               _nested_inpt=True,
                               locs=self.locs + other.locs,
                               diag_unitary=self.diag_unitary +
                               other.diag_unitary)

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
            for liou_region in self.data:
                if isinstance(liou_region[0], list):
                    if liou_region[0]:
                        # liou_region = [[3], [2]] or [[3,2], []]
                        # multiply scale number to first term
                        array_0 = liou_region[0][0] * other
                        temp_data = copy.deepcopy(liou_region)
                        temp_data[0][0] = array_0
                        data.append(temp_data)
                    elif liou_region[1]:
                        # liou_region = [[], [3,2]]
                        array_0 = liou_region[1][0] * other
                        temp_data = copy.deepcopy(liou_region)
                        temp_data[1][0] = array_0
                        data.append(temp_data)
                else:
                    # order-4 tensor be used as Lindblad
                    data.append([liou_region[0] * other])
            return LindbladObj(data,
                               self.dims,
                               _nested_inpt=True,
                               locs=self.locs,
                               diag_unitary=self.diag_unitary)
        else:
            raise NotImplementedError(f'Do not support instance {type(other)}.')

    def __rmul__(self, other):
        """Multiplication with KronObj on right"""
        return self * other

    def __matmul__(self, other):
        """Matrix multiplication with LindbladObj on left. The high efficient
        order-4 tensor <-> density matrix multiplication method working when `other` has matched shape.
        """
        if isinstance(other, (np.ndarray, jax.Array)):
            res = []
            # implement location Einstein summation
            for sub_lindbladobj in self:
                # construct subscripts and operands
                lindblad_op = []
                lindblad_dim = []
                lindblad_subscript = []
                if isinstance(sub_lindbladobj.data[0][0], list):
                    if all(sub_lindbladobj.locs[0]):
                        # both upper region and lower region have operator
                        union_loc_info = sorted(
                            set(sub_lindbladobj.locs[0][0] +
                                sub_lindbladobj.locs[0][1]))
                        for idx, ul_region, ul_loc in zip(
                                range(2), sub_lindbladobj.data[0],
                                sub_lindbladobj.locs[0]):
                            mat_list = [
                                jnp.eye(self.dims[loc])
                                for loc in union_loc_info
                            ]
                            for mat, loc in zip(ul_region, ul_loc):
                                mat_list[union_loc_info.index(loc)] = mat
                            lindblad_op.append(tensor(*mat_list))
                            lindblad_dim += [
                                sub_lindbladobj.dims[loc]
                                for loc in union_loc_info
                            ] * 2
                            # subscript start from 1
                            # for Lindblad L_{ijkl}
                            # | subscript_l | -len(dims) subscript_j | 0 | subscript_i len(dims) | subscript_k |
                            temp_subscript = [
                                len(sub_lindbladobj.dims) * idx + 1 + loc
                                for loc in union_loc_info
                            ]
                            lindblad_subscript += temp_subscript
                            lindblad_subscript += [
                                -subscript for subscript in temp_subscript
                            ]
                    else:
                        for idx, ul_loc in enumerate(sub_lindbladobj.locs[0]):
                            if ul_loc:
                                lindblad_op.append(
                                    tensor(*sub_lindbladobj.data[0][idx]))
                                lindblad_dim += [
                                    sub_lindbladobj.dims[loc] for loc in ul_loc
                                ] * 2
                                # subscript start from 1
                                # for Lindblad L_{ijkl}
                                # | subscript_l | -len(dims) subscript_j | 0 | subscript_i len(dims) | subscript_k |
                                temp_subscript = [
                                    len(sub_lindbladobj.dims) * idx + 1 + loc
                                    for loc in ul_loc
                                ]
                                lindblad_subscript += temp_subscript
                                lindblad_subscript += [
                                    -subscript for subscript in temp_subscript
                                ]
                            else:  # []
                                other_ul_loc = sub_lindbladobj.locs[0][1 - idx]
                                temp_lindblad_dim = [
                                    sub_lindbladobj.dims[loc]
                                    for loc in other_ul_loc
                                ]
                                lindblad_op.append(
                                    jnp.eye(np.prod(temp_lindblad_dim)))
                                lindblad_dim += temp_lindblad_dim * 2
                                temp_subscript = [
                                    len(sub_lindbladobj.dims) * idx + 1 + loc
                                    for loc in other_ul_loc
                                ]
                                lindblad_subscript += temp_subscript
                                lindblad_subscript += [
                                    -subscript for subscript in temp_subscript
                                ]
                    lindblad_op = tensor(
                        *lindblad_op[::-1]).reshape(lindblad_dim)
                else:
                    # use order-4 tensor as Lindblad
                    for idx, ul_loc in enumerate(sub_lindbladobj.locs[0]):
                        lindblad_dim += [
                            sub_lindbladobj.dims[loc] for loc in ul_loc
                        ] * 2
                        temp_subscript = [
                            len(sub_lindbladobj.dims) * idx + 1 + loc
                            for loc in ul_loc
                        ]
                        lindblad_subscript += temp_subscript
                        lindblad_subscript += [
                            -subscript for subscript in temp_subscript
                        ]
                    lindblad_op = sub_lindbladobj.data[0][0].reshape(
                        lindblad_dim)
                # density matrix in the dense format
                density_op = other.reshape(sub_lindbladobj.dims * 2)
                # for denisty matrix rho_{kl}
                # | subscript_l | -len(dims) None | 0 | None len(dims) | subscript_k |
                density_subscript = []
                temp_subscript = list(
                    range(
                        len(sub_lindbladobj.dims) + 1,
                        len(sub_lindbladobj.dims) * 2 + 1))
                density_subscript += temp_subscript
                density_subscript += [
                    -subscript for subscript in temp_subscript
                ]
                res_subscript = density_subscript.copy()
                for subscript in lindblad_subscript:
                    if abs(subscript) > len(sub_lindbladobj.dims):
                        idx = res_subscript.index(subscript)
                        res_subscript[idx] -= len(
                            sub_lindbladobj.dims) * np.sign(res_subscript[idx])

                new_density = jnp.einsum(lindblad_op, lindblad_subscript,
                                         density_op, density_subscript,
                                         res_subscript).reshape(other.shape)
                res.append(new_density)
            return sum(res)
        else:
            raise NotImplementedError(f'Do not support instance {type(other)}.')

    def compute_contraction_path(self, lindblad_list=None, trotter_order=None):
        """Compute tensor network contraction path.
        Please provide either `lindblad_list` or `trotter_order`. If `lindblad_list` is None,
        it will be generated by `trotter_order`.

        Args:
            lindblad_list: the list of local operator
            trotter_order (complex int): the order of suzuki-trotter decomposition.
                The following arguments are supported,
                a) `None`, calculating matrix exponentiation without trotter decomposition
                b) `1`, first order trotter decomposition
                c) `2`, second order trotter decomposition
                d) `4`, 4th order real decomposition
                e) `4j`, 4th order complex decomposition
        """
        assert lindblad_list is not None or trotter_order is not None
        if lindblad_list is None:
            lindblad_list = self._trotter_decomposition(trotter_order)
        # initialize subscripts by state
        num_indices = len(self.dims)
        max_indices_per_layer = len(self.dims)**2
        num_subscript = np.zeros((2, len(self.dims)), dtype=int)
        rho_subscript = ''.join([
            oe.get_symbol(i + max_indices_per_layer * num)
            for i, num in enumerate(num_subscript.flatten())
        ])
        oe_subscripts = []
        oe_shapes = []
        # construct tensor network
        oe_subscripts.append(rho_subscript)
        oe_shapes.append(list(self.dims) * 2)
        for sub_lindbladobj in lindblad_list[::-1]:
            # construct subscripts and operands
            base_subscript = []
            new_subscript = []
            lindblad_shape = []
            if isinstance(sub_lindbladobj.data[0][0], list):
                for idx, ul_loc in enumerate(sub_lindbladobj.locs[0]):
                    if ul_loc:
                        lindblad_shape += [
                            sub_lindbladobj.dims[loc] for loc in ul_loc
                        ] * 2
                        for loc in ul_loc:
                            base_subscript.append(
                                oe.get_symbol(loc + num_indices * idx +
                                              max_indices_per_layer *
                                              num_subscript[idx, loc]))
                            num_subscript[idx, loc] += 1
                            new_subscript.append(
                                oe.get_symbol(loc + num_indices * idx +
                                              max_indices_per_layer *
                                              num_subscript[idx, loc]))
                    else:  # []
                        other_ul_loc = sub_lindbladobj.locs[0][1 - idx]
                        temp_lindblad_dim = [
                            sub_lindbladobj.dims[loc] for loc in other_ul_loc
                        ]
                        lindblad_shape += temp_lindblad_dim * 2
                        for loc in other_ul_loc:
                            base_subscript.append(
                                oe.get_symbol(loc + num_indices * idx +
                                              max_indices_per_layer *
                                              num_subscript[idx, loc]))
                            num_subscript[idx, loc] += 1
                            new_subscript.append(
                                oe.get_symbol(loc + num_indices * idx +
                                              max_indices_per_layer *
                                              num_subscript[idx, loc]))
            else:
                # use order-4 tensor as Lindblad
                for idx, ul_loc in enumerate(sub_lindbladobj.locs[0]):
                    lindblad_shape += [
                        sub_lindbladobj.dims[loc] for loc in ul_loc
                    ] * 2
                    for loc in ul_loc:
                        base_subscript.append(
                            oe.get_symbol(loc + num_indices * idx +
                                          max_indices_per_layer *
                                          num_subscript[idx, loc]))
                        num_subscript[idx, loc] += 1
                        new_subscript.append(
                            oe.get_symbol(loc + num_indices * idx +
                                          max_indices_per_layer *
                                          num_subscript[idx, loc]))
            oe_subscripts.append(''.join(new_subscript + base_subscript))
            oe_shapes.append(lindblad_shape)
        rho_output = ''.join([
            oe.get_symbol(i + max_indices_per_layer * num)
            for i, num in enumerate(num_subscript.flatten())
        ])
        eq = ','.join(oe_subscripts) + '->' + rho_output
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
            lindblad_list = []
            if reverse:
                scan_list = zip(reversed(self.data),
                                reversed(self.diag_unitary),
                                reversed(self.locs))
            else:
                scan_list = zip(self.data, self.diag_unitary, self.locs)
            for liou_region, liou_diag, liou_local in scan_list:
                if all(liou_local):
                    # compute tensor product of upper_region and lower_region
                    union_ul_local = sorted(set(liou_local[0] + liou_local[1]))
                    if None not in liou_diag:  # cast expm down to exp
                        # diag info will be a order-4 tensor
                        eig_tensor = liou_diag
                        tensor_dim = np.prod(
                            [self.dims[loc] for loc in union_ul_local])
                        eig_vec = eig_tensor.reshape(*[tensor_dim**2] * 2)
                        # fix bug about zip function
                        # lam = jnp.exp(trotter_coeff * jnp.diag(
                        #     liou_region[0].reshape(*[tensor_dim**2] * 2)))
                        lam = jnp.exp(trotter_coeff * liou_region[0].flatten())
                        liou_local = [union_ul_local, union_ul_local]
                        tensor_expm = eig_vec * lam @ jnp.conj(eig_vec).T
                        inpt = [tensor_expm.reshape(*([
                            tensor_dim,
                        ] * 4))]
                    else:
                        ul_list = []
                        for ul_region, ul_local in zip(liou_region, liou_local):
                            mat_list = [
                                jnp.eye(self.dims[loc])
                                for loc in union_ul_local
                            ]
                            for mat, loc in zip(ul_region, ul_local):
                                loc_idx = union_ul_local.index(loc)
                                mat_list[loc_idx] = mat
                            ul_list.append(tensor(*mat_list))
                        # only calculate matrix exponentiation with the terms in `local_info`
                        obj = tensor(*ul_list[::-1]) * trotter_coeff
                        liou_local = [union_ul_local, union_ul_local]
                        tensor_dim = np.prod(
                            [self.dims[loc] for loc in union_ul_local])
                        inpt = [jaxLA.expm(obj).reshape(*([
                            tensor_dim,
                        ] * 4))]
                    lindblad_list.append(
                        LindbladObj(inpt, self.dims, liou_local))
                else:
                    # only one of region has non-trivial matrix
                    mat_list = liou_region[0] + liou_region[1]
                    if None not in liou_diag:  # cast expm down to exp
                        eig_vec = liou_diag[0] + liou_diag[1]
                        lam = jnp.exp(trotter_coeff * mat_list[0])
                        if liou_region[0]:
                            inpt = [[eig_vec * lam @ jnp.conj(eig_vec).T], []]
                        else:
                            inpt = [[], [eig_vec * lam @ jnp.conj(eig_vec).T]]
                    else:
                        obj = tensor(*mat_list) * trotter_coeff
                        if liou_region[0]:
                            inpt = [[jaxLA.expm(obj)], []]
                        else:
                            inpt = [[], [jaxLA.expm(obj)]]

                    lindblad_list.append(
                        LindbladObj(inpt, self.dims, liou_local))
                # note the `expm(obj)` cannot decomposition back to tensor product,
                # the `LindbladObj.full()` method does not support the form,
                # so `LindbladObj([expm(obj)])` only work in tensor-density multiplication
            return lindblad_list

        def _second_order_expm(p=1.0):
            """Generate the matrix multiplication sequence for second order
            trotter decomposition.
            """
            # normal loop
            lindblad_list = _loop(p / 2., reverse=False)
            # reversed loop
            lindblad_list.extend(_loop(p / 2., reverse=True))
            return lindblad_list

        if trotter_order == 1:
            lindblad_list = _loop(1.)
        elif trotter_order == 2:
            lindblad_list = _second_order_expm()
        elif trotter_order == 4:
            # the simplest real decomposition of 4th order STD
            p = 1 / (2 - 2**(1 / 3))
            lindblad_list = _second_order_expm(p)
            lindblad_list.extend(_second_order_expm(1 - 2 * p))
            lindblad_list.extend(_second_order_expm(p))
        elif trotter_order == 4j:
            # the complex 4th STD
            p = (3 - 1j * sqrt(3)) / 6
            p_conj = (3 + 1j * sqrt(3)) / 6
            lindblad_list = _second_order_expm(p)
            lindblad_list.extend(_second_order_expm(p_conj))
        else:
            raise ValueError(f'Unsupported trotter order {trotter_order}.')

        return lindblad_list

    def expm(self, right_density=None, trotter_order=None, tn_expr=None):
        """Calculate matrix exponentiation of `LindbladObj`. Combine matrix exponentiation
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
        if right_density is None or trotter_order not in [1, 2, 4, 4j]:
            # not implement trotter approximate
            res = jaxLA.expm(self.full().reshape(np.prod(self.shape[:2]),
                                                 np.prod(self.shape[2:])))
            if right_density is not None:
                return (res @ right_density.ravel('F')).reshape(
                    np.prod(self.dims), np.prod(self.dims)).T
            else:
                return res
        else:
            lindblad_list = self._trotter_decomposition(trotter_order)
            if tn_expr is None:
                tn_expr = self.compute_contraction_path(lindblad_list)
            oe_arrays = []
            oe_arrays.append(right_density.reshape(*self.dims * 2))
            for sub_lindbladobj in lindblad_list[::-1]:
                # construct subscripts and operands
                lindblad_shape = []
                lindblad_op = []
                if isinstance(sub_lindbladobj.data[0][0], list):
                    for idx, ul_loc in enumerate(sub_lindbladobj.locs[0]):
                        if ul_loc:
                            lindblad_op.append(
                                tensor(*sub_lindbladobj.data[0][idx]))
                            lindblad_shape += [
                                sub_lindbladobj.dims[loc] for loc in ul_loc
                            ] * 2

                        else:  # []
                            other_ul_loc = sub_lindbladobj.locs[0][1 - idx]
                            temp_lindblad_dim = [
                                sub_lindbladobj.dims[loc]
                                for loc in other_ul_loc
                            ]
                            lindblad_op.append(
                                jnp.eye(np.prod(temp_lindblad_dim)))
                            lindblad_shape += temp_lindblad_dim * 2
                    lindblad_op = tensor(
                        *lindblad_op[::-1]).reshape(lindblad_shape)
                else:
                    # use order-4 tensor as Lindblad
                    for idx, ul_loc in enumerate(sub_lindbladobj.locs[0]):
                        lindblad_shape += [
                            sub_lindbladobj.dims[loc] for loc in ul_loc
                        ] * 2
                    lindblad_op = sub_lindbladobj.data[0][0].reshape(
                        lindblad_shape)
                oe_arrays.append(lindblad_op)
            oe_res = tn_expr(*oe_arrays, backend='jax')

            return oe_res.reshape(np.prod(self.dims), np.prod(self.dims))
