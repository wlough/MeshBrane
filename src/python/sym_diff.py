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


class CoordinateCircle:
    """
    Attributes
    ----------
    lower_bound: float
        The lower bound of the interval.

    """
    def __init__(self, value_at_index=lambda k: .1*k, lower_bound=-1, upper_bound=1, include_lower=True, include_upper=True, num_samples=100):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_lower = include_lower
        self.include_upper = include_upper
        self.value_at_index = value_at_index
        self.num_samples = num_samples

    def __getitem__(self, key):
        return self.value_at_index(key)

    def generate_indices(self, kind="valid"):
        if kind == "valid":
            k_start = 0 if self.include_lower else 1
            k_end = self.num_samples if self.include_upper else self.num_samples - 1
        elif kind == "closure":
            k_start = 0
            k_end = self.num_samples
        elif kind == "interior":
            k_start = 1
            k_end = self.num_samples - 1
        return range(k_start, k_end)

    @property
    def samples(self):
        k_start = 0 if self.include_lower else 1
        k_end = self.num_samples if self.include_upper else self.num_samples -
        return [self.value_at_index(k) for k in range(self.num_samples)]


    @classmethod
    def from_key_value_strings(cls, key_str="k", value_str=".1*k", num_samples=100, include_lower=True, include_upper=True):
        key = sp.symbols(key_str)
        val = sp.sympify(value_str)
        value_at_index = sp.lambdify(key, val)

        return cls(value_at_index)

class CoordinateInterval:
    """
    used to generate coordinate samples for a given interval

    Attributes
    ----------
    lower_bound: float
        The lower bound of the interval.

    """
    def __init__(self, lower_bound=0, upper_bound=1, include_lower=False, include_upper=True):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_lower = include_lower
        self.include_upper = include_upper
        self.num_samples = sp.symbols("NumSamples")


    def __getitem__(self, key):
        return self.value_at_index(key)

    def generate_indices(self, kind="valid"):
        if kind == "valid":
            k_start = 0 if self.include_lower else 1
            k_end = self.num_samples if self.include_upper else self.num_samples - 1
        elif kind == "closure":
            k_start = 0
            k_end = self.num_samples
        elif kind == "interior":
            k_start = 1
            k_end = self.num_samples - 1
        return range(k_start, k_end)

    @property
    def samples(self):
        k_start = 0 if self.include_lower else 1
        k_end = self.num_samples if self.include_upper else self.num_samples -
        return [self.value_at_index(k) for k in range(self.num_samples)]


    @classmethod
    def from_key_value_strings(cls, key_str="k", value_str=".1*k", num_samples=100, include_lower=True, include_upper=True):
        key = sp.symbols(key_str)
        val = sp.sympify(value_str)
        value_at_index = sp.lambdify(key, val)

        return cls(value_at_index)
    # @property
    # def




class CoordinateSystem3D:
    def __init__(self, coords=None, coord_subs=None, coord_range=None):
        if coord_subs is None:
            coords = "r theta phi"
            coord_subs = {
                "x": "r*sin(theta)*cos(phi)",
                "y": "r*sin(theta)*sin(phi)",
                "z": "r*cos(theta)",
                "r": "sqrt(x**2 + y**2 + z**2)",
                "theta": "acos(z / r)",
                "phi": "atan2(y, x)",
            }
            coord_intervals = {"r": CoordinateInterval(0, 1), "theta": CoordinateCircle(), "phi": CoordinateCircle()}

        self.xyz = sp.Array([*sp.symbols("x y z")])
        self.uvw = sp.Array([*sp.symbols(coords)])
        self.xyz_uvw = sp.Array([sp.sympify(coord_subs[_]) for _ in coords.split()])
        self.uvw_xyz = sp.Array([sp.sympify(coord_subs[_]) for _ in coords.split()])

        self.jacobian = sp.Array(
            [[x_i.diff(phi_j) for phi_j in self.thetaphi] for x_i in self.xyz_thetaphi]
        )
        self.hessian = sp.Array(
            [
                [
                    [x_i.diff(phi_j).diff(phi_k) for phi_k in self.thetaphi]
                    for phi_j in self.thetaphi
                ]
                for x_i in self.xyz_thetaphi
            ]
        )


    def parse_range_string(self, range_str):
        """
        Parses a range string and returns a list of its components.

        Parameters:
        - range_str: A string representing a range, e.g., "[a, b)".

        Returns:
        - list: A list containing the opening bracket, start value, end value, and closing bracket.
        """
        # Trim the string to remove leading/trailing spaces
        trimmed_str = range_str.strip()

        # Extract the opening and closing brackets
        opening_bracket = trimmed_str[0]
        closing_bracket = trimmed_str[-1]

        # Extract the numbers, removing spaces
        numbers_part = trimmed_str[1:-1].replace(" ", "")

        # Split the numbers part by the comma
        start, end = numbers_part.split(',')

        return [opening_bracket, start, end, closing_bracket]

    # # Example usage
    # print(parse_range_string("(a,b)"))       # ["(", "a", "b", ")"]
    # print(parse_range_string("[a,    b]"))   # ["[", "a", "b", "]"]
    # print(parse_range_string("[a,          b)"))  # ["[", "a", "b", "]"]

class SymTorus:
    """ """

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
        # self.jacobian = sp.derive_by_array(self.xyz_thetaphi, self.thetaphi).reshape(
        #     3, 2
        # )
        self.jacobian = sp.Array(
            [[x_i.diff(phi_j) for phi_j in self.thetaphi] for x_i in self.xyz_thetaphi]
        )
        # self.hessian = sp.derive_by_array(self.jacobian, self.thetaphi).reshape(3, 2, 2)
        self.hessian = sp.Array(
            [
                [
                    [x_i.diff(phi_j).diff(phi_k) for phi_k in self.thetaphi]
                    for phi_j in self.thetaphi
                ]
                for x_i in self.xyz_thetaphi
            ]
        )
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
        self.surface_area = 4 * sp.pi * self.R**2
        self.volume = 4 * sp.pi * self.R**3 / 3
        ############################################################################
        # self.jacobian = sp.derive_by_array(self.xyz_thetaphi, self.thetaphi).reshape(
        #     3, 2
        # )
        self.jacobian = sp.Array(
            [[x_i.diff(phi_j) for phi_j in self.thetaphi] for x_i in self.xyz_thetaphi]
        )
        # self.hessian = sp.derive_by_array(self.jacobian, self.thetaphi).reshape(3, 2, 2)
        self.hessian = sp.Array(
            [
                [
                    [x_i.diff(phi_j).diff(phi_k) for phi_k in self.thetaphi]
                    for phi_j in self.thetaphi
                ]
                for x_i in self.xyz_thetaphi
            ]
        )
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
