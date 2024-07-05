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


def einsum(tstr, tensor_list):
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


class SymTorus:

    def __init__(self, R=1, R2r=3):
        self.name = "torus"
        self.R = sp.symbols("R")  # R
        self.r = sp.symbols("r")  # sp.sympify(f"{R} / {R2r}")
        self.x, self.y, self.z = sp.symbols("x y z")
        self.theta, self.phi = sp.symbols("theta phi")
        self.num_subs = {self.R: R, self.r: sp.sympify(f"{R} / {R2r}")}
        self.R2r = R2r
        self.implicit_rep_xyz = (
            (sp.sqrt(self.x**2 + self.y**2) - self.R) ** 2 + self.z**2 - self.r**2
        )
        self.xyz_thetaphi = sp.Array(
            [
                sp.cos(self.phi) * (self.R + sp.cos(self.theta) * self.r),
                sp.sin(self.phi) * (self.R + sp.cos(self.theta) * self.r),
                sp.sin(self.theta) * self.r,
            ]
        )
        self.thetaphi_xyz = sp.Array(
            [sp.atan2(self.z, sp.sqrt(self.x**2 + self.y**2)), sp.atan2(self.y, self.x)]
        )
        self.surface_area = 4 * sp.pi**2 * self.R * self.r
        self.volume = 2 * sp.pi**2 * self.R * self.r**2
        ############################################################################
        self.jacobian = sp.derive_by_array(self.xyz_thetaphi, self.thetaphi).reshape(
            3, 2
        )
        self.hessian = sp.derive_by_array(self.jacobian, self.thetaphi).reshape(3, 2, 2)
        self.implicit_fun = sp.lambdify(
            self.xyz, self.implicit_rep_xyz.subs(self.num_subs)
        )
        self.parametric_fun = sp.lambdify(
            self.thetaphi, self.xyz_thetaphi.subs(self.num_subs)
        )
        self.unit_normal_fun = sp.lambdify(
            self.thetaphi, self.unit_normal.subs(self.num_subs)
        )
        self.mean_curvature_fun = sp.lambdify(
            self.thetaphi, self.mean_curvature.subs(self.num_subs)
        )
        self.gaussian_curvature_fun = sp.lambdify(
            self.thetaphi, self.gaussian_curvature.subs(self.num_subs)
        )
        self.orthonormal_frame_fun = sp.lambdify(self.thetaphi, self.orthonormal_frame)

    ############################################################################
    @property
    def xyz(self):
        return sp.Array([self.x, self.y, self.z])

    @property
    def thetaphi(self):
        return sp.Array([self.theta, self.phi])

    @property
    def unit_theta(self):
        u_theta = self.jacobian[:, 0]
        return u_theta / norm(u_theta)

    @property
    def unit_phi(self):
        u_phi = self.jacobian[:, 1]
        return u_phi / norm(u_phi)

    @property
    def unit_normal(self):
        n = cross(self.jacobian[:, 0], self.jacobian[:, 1])
        return n / norm(n)

    @property
    def orthonormal_frame(self):
        return sp.Array([self.unit_theta, self.unit_phi, self.unit_normal])

    @property
    def E(self):
        return dot(self.jacobian[:, 0], self.jacobian[:, 0])

    @property
    def F(self):
        return dot(self.jacobian[:, 0], self.jacobian[:, 1])

    @property
    def G(self):
        return dot(self.jacobian[:, 1], self.jacobian[:, 1])

    @property
    def L(self):
        return dot(self.unit_normal, self.hessian[:, 0, 0])

    @property
    def M(self):
        return dot(self.unit_normal, self.hessian[:, 0, 1])

    @property
    def N(self):
        return dot(self.unit_normal, self.hessian[:, 1, 1])

    @property
    def metric(self):
        return sp.Array([[self.E, self.F], [self.F, self.G]])

    @property
    def inverse_metric(self):
        return sp.Array([[self.G, -self.F], [-self.F, self.E]])

    @property
    def mean_curvature(self):
        return (self.L * self.N - self.M**2) / (self.E * self.G - self.F**2)

    @property
    def gaussian_curvature(self):
        return (self.L * self.G - 2 * self.M * self.F + self.N * self.E) / (
            2 * (self.E * self.G - self.F**2)
        )

    ############################################################################


class SymSphere:

    def __init__(self, R=1):
        self.name = "sphere"
        self.R = sp.symbols("R")
        self.x, self.y, self.z = sp.symbols("x y z")
        self.theta, self.phi = sp.symbols("theta phi")
        self.num_subs = {self.R: R}
        self.implicit_rep_xyz = self.x**2 + self.y**2 + self.z**2 - self.R**2
        self.xyz_thetaphi = sp.Array(
            [
                self.R * sp.sin(self.theta) * sp.cos(self.phi),
                self.R * sp.sin(self.theta) * sp.sin(self.phi),
                self.R * sp.cos(self.theta),
            ]
        )
        self.thetaphi_xyz = sp.Array(
            [sp.acos(self.z / self.R), sp.atan2(self.y, self.x)]
        )
        self.surface_area = 4 * sp.pi * self.R**2
        self.volume = 4 * sp.pi * self.R**3 / 3
        ############################################################################
        self.jacobian = sp.derive_by_array(self.xyz_thetaphi, self.thetaphi).reshape(
            3, 2
        )
        self.hessian = sp.derive_by_array(self.jacobian, self.thetaphi).reshape(3, 2, 2)
        self.implicit_fun = sp.lambdify(
            self.xyz, self.implicit_rep_xyz.subs(self.num_subs)
        )
        self.parametric_fun = sp.lambdify(
            self.thetaphi, self.xyz_thetaphi.subs(self.num_subs)
        )
        self.unit_normal_fun = sp.lambdify(
            self.thetaphi, self.unit_normal.subs(self.num_subs)
        )
        self.mean_curvature_fun = sp.lambdify(
            self.thetaphi, self.mean_curvature.subs(self.num_subs)
        )
        self.gaussian_curvature_fun = sp.lambdify(
            self.thetaphi, self.gaussian_curvature.subs(self.num_subs)
        )
        self.orthonormal_frame_fun = sp.lambdify(self.thetaphi, self.orthonormal_frame)

    ############################################################################
    @property
    def xyz(self):
        return sp.Array([self.x, self.y, self.z])

    @property
    def thetaphi(self):
        return sp.Array([self.theta, self.phi])

    @property
    def unit_theta(self):
        u_theta = self.jacobian[:, 0]
        return u_theta / norm(u_theta)

    @property
    def unit_phi(self):
        u_phi = self.jacobian[:, 1]
        return u_phi / norm(u_phi)

    @property
    def unit_normal(self):
        n = cross(self.jacobian[:, 0], self.jacobian[:, 1])
        return n / norm(n)

    @property
    def orthonormal_frame(self):
        return sp.Array([self.unit_theta, self.unit_phi, self.unit_normal])

    @property
    def E(self):
        return dot(self.jacobian[:, 0], self.jacobian[:, 0])

    @property
    def F(self):
        return dot(self.jacobian[:, 0], self.jacobian[:, 1])

    @property
    def G(self):
        return dot(self.jacobian[:, 1], self.jacobian[:, 1])

    @property
    def L(self):
        return dot(self.unit_normal, self.hessian[:, 0, 0])

    @property
    def M(self):
        return dot(self.unit_normal, self.hessian[:, 0, 1])

    @property
    def N(self):
        return dot(self.unit_normal, self.hessian[:, 1, 1])

    @property
    def metric(self):
        return sp.Array([[self.E, self.F], [self.F, self.G]])

    @property
    def inverse_metric(self):
        return sp.Array([[self.G, -self.F], [-self.F, self.E]])

    @property
    def mean_curvature(self):
        return (self.L * self.N - self.M**2) / (self.E * self.G - self.F**2)

    @property
    def gaussian_curvature(self):
        return (self.L * self.G - 2 * self.M * self.F + self.N * self.E) / (
            2 * (self.E * self.G - self.F**2)
        )

    ############################################################################
