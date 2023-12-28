import sympy as sp


###################
# sympy functions #
# matrix operations for arrays and a few other things
tp = lambda V, W: sp.tensorproduct(V, W)
tc = lambda V, ij: sp.tensorcontraction(V, ij)
tpc = lambda V, W, ij: tc(tp(V, W), ij)
tr = lambda V: tc(V, (0, 1))  # trace
mp = lambda A, B: tc(tp(A, B), (1, 2))  # matrix product
dot = lambda V, W: tc(tp(V, W), (0, 1))
norm = lambda u: sp.sqrt(dot(u, u))


def seinsum(tstr, tensor_list):
    """
    symbolic einsum

    tstr='abc,ab->c'
    tensor_list=[X,Y]
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # tensor_shapes = [sp.shape(tens) for tens in tensor_list]
    # product_shape = [*np.concatenate(tensor_shapes)]
    index_list = []

    tensor_indices = ""

    for n, char in enumerate(tstr):
        if char in alphabet:
            tensor_indices += char
        if char in ",-":
            index_list.append(tensor_indices)
            tensor_indices = ""
        if char == ">":
            pass
    free_indices = tensor_indices

    product_indices = ""
    for ind in index_list:
        product_indices += ind

    dummy_indices = ""
    for ind in product_indices:
        if ind in dummy_indices + free_indices:
            pass
        else:
            dummy_indices += ind

    dummy_slots = {}
    # contraction_axes = []
    for ind in dummy_indices:
        dummy_slots[ind] = []
        for ind_num, prod_ind in enumerate(product_indices):
            if ind == prod_ind:
                dummy_slots[ind].append(ind_num)

    contraction_axes = [axes for dummy, axes in dummy_slots.items()]

    # free_slots = {ind: product_indices.index(ind) for ind in free_indices}

    # out_shape = []
    # for ind in free_indices:
    #     product_index_number = product_indices.index(ind)
    #     index_range = product_shape[product_index_number]
    #     out_shape.append(index_range)

    # _T = 1
    # for tens in tensor_list:
    #     _T = tp(_T, tens)
    _T = sp.tensorproduct(*tensor_list)
    T = sp.tensorcontraction(_T, *contraction_axes)
    return T


def hat(v):
    """
    sympy hat map
    """
    vx, vy, vz = v
    vhat = sp.Array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
    return vhat


def unhat(vhat):
    """
    sympy inverse hat map
    """
    vx, vy, vz = -vhat[1, 2], vhat[0, 2], -vhat[0, 1]
    v = sp.Array([vx, vy, vz])
    return v


def cross(u, v):
    ux, uy, uz = u
    vx, vy, vz = v
    ucv = sp.Array([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx])
    return ucv


def floatify(expr):
    """
    converts sympy ints or int entries in sympy array to floats
    """
    if hasattr(expr, "__len__"):
        _expr = []
        for a in sp.flatten(expr):
            try:
                _expr.append(sp.Float(a))
            except Exception:
                _expr.append(a)
        _expr = sp.Array(_expr).reshape(*expr.shape)
    else:
        try:
            _expr = sp.Float(expr)
        except Exception:
            _expr = expr

    return _expr
