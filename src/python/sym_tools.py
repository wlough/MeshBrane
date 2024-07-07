import sympy as sp
import numpy as np

###################
# Custom sympy functions, matrix operations for arrays and a few other things
tp = lambda V, W: sp.tensorproduct(V, W)
tc = lambda V, ij: sp.tensorcontraction(V, ij)
tpc = lambda V, W, ij: tc(tp(V, W), ij)
tr = lambda V: tc(V, (0, 1))  # trace
mp = lambda A, B: tc(tp(A, B), (1, 2))  # matrix product
dot = lambda V, W: tc(tp(V, W), (0, 1))
norm = lambda u: sp.sqrt(dot(u, u))


def dumbify(x):
    """
    Produces a sympy.Dummy from a string or sympy.Symbol
    """
    x_as_sympy = sp.sympify(x)

    return sp.Dummy(x_as_sympy.name, latex_name=sp.latex(x_as_sympy))


def sym_sorted(syms, skip=[]):
    """
    Sorts iterable of sympy symbols by string representation and length of string representation.
    """
    sorted_syms = sorted(syms, key=lambda sym: (str(sym), len(str(sym))))
    while skip:
        _ = skip.pop()
        try:
            sorted_syms.remove(_)
        except ValueError:
            pass
    return sorted_syms


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


############################################################
############################################################


class Samples1D:
    """
    Coordinate samples with a dict(symbolic: numeric) parameters. Can initialize with class methods linspace, expspace, logspace, geoseries to create instances.

    Args
    ----
    value_at_index: str or sympifiable expression
        The value of sample at dummy index.
    index: str or sympifiable expression
        Symbol for dummy index
    num_dict: dict
        A dictionary of symbolic parameters and their numerical values. Keys must be sympifiable.
    make_lamfun: bool
        Whether to assign lamfun method.
    make_lamfun_args: bool
        Whether to assign lamfun_args attribute.

    Attributes
    ----------
    index: sympy.Symbol
        Symbol for dummy index
    value_at_index: sympy.Expr
        The value of sample at dummy index.
    num_dict: dict
        A dictionary of symbolic parameters appearing in value_at_index and their numerical values.
    _lamfun_args: list
        A list of numerical values for symbolic parameters appearing in value_at_index.

    Methods
    -------
    subs(sym_dict)
        Substitutes sym_dict into self.value_at_index.
    update_num_dict(num_dict)
        Updates self.num_dict with values in num_dict
    simplify(kind="simplify", **kwargs)
        Simplifies value_at_index using sympy.simplify or sympy.ratsimp.
    refresh(lamfun=True, lamfun_args=True)
        Refreshes numerical functions so they use current index,value_at_index,num_dict.
    numerical_eval(expr)
        Returns numerical value of expr by substituting values in num_dict.
    sample_range(start, stop)
        Yields numerical value_at_index for index in range(start, stop).
    apply_index_transform(index_transform, refresh=True)
        Applies index_transform to index.
    apply_coord_transform(coord_transform, refresh=True)
        Applies coord_transform to value_at_index.
    __call__(k)
        Returns numerical value_at_index at index k.
    __getitem__(sym)
        Returns numerical value of sym in num_dict.
    __repr__()
        Returns a string representation of the object.
    __str__()
        Returns a string representation of the object.
    _repr_latex_()
        Returns a latex representation of the object.
    _lamfun(k, *lamfun_args)
        Returns numerical value_at_index at index k.
    _make_lamfun()
        Returns a lambdified value_at_index.
    _make_lamfun_args()
        Returns a list of numerical values for symbolic parameters appearing in value_at_index sorted by sym_sorted().
    """

    ###########################################
    # Class variables
    ###########################################
    # Initialization methods
    def __init__(
        self,
        value_at_index,
        index,
        num_dict=dict(),
        make_lamfun=True,
        make_lamfun_args=True,
    ):
        self.index = index
        self.value_at_index = value_at_index
        self.num_dict = num_dict
        if self.index in self.num_dict:
            raise ValueError("Index cannot be in num_dict")
        self.refresh(lamfun=make_lamfun, lamfun_args=make_lamfun_args)

    @classmethod
    def linspace(cls, start, stop, num=50):
        k, a, b, n = sp.symbols("k a b n")
        x = a + k * (b - a) / (n - 1)
        num_dict = {
            a: start,
            b: stop,
            n: num,
        }
        return cls(x, k, num_dict)

    @classmethod
    def expspace(cls, start, stop, num=50):
        k, a, b, n = sp.symbols("k a b n")
        x = a * (b / a) ** (k / (n - 1))
        num_dict = {
            a: start,
            b: stop,
            n: num,
        }
        return cls(x, k, num_dict)

    @classmethod
    def logspace(cls, start, stop, num=50):
        k, a, b, n = sp.symbols("k a b n")
        x = sp.exp(sp.log(a) + k * (sp.log(b) - sp.log(a)) / (n - 1))
        num_dict = {
            a: start,
            b: stop,
            n: num,
        }
        return cls(x, k, num_dict)

    @classmethod
    def geoseries(cls, seq_start, seq_ratio, num=50):
        # k, a, r, n = sp.symbols("k a r n")
        k, a, r, n = sp.symbols("k a r n")
        x = a * (1 - r**k) / (1 - r)
        num_dict = {
            a: seq_start,
            r: seq_ratio,
            n: num,
        }
        return cls(x, k, num_dict)

    ###########################################
    # Properties
    # ----------
    @property
    def index(self):
        """Index symbol"""
        return self._index

    @index.setter
    def index(self, index):
        if isinstance(index, sp.Symbol):
            self._index = index
        else:
            self._index = sp.sympify(index)

    @property
    def value_at_index(self):
        """Sample value in terms of index and symbolic parameters"""
        return self._value_at_index

    @value_at_index.setter
    def value_at_index(self, value_at_index):
        self._value_at_index = sp.sympify(value_at_index)

    @property
    def num_dict(self):
        return self._num_dict

    @num_dict.setter
    def num_dict(self, num_dict):
        _num_dict = dict()
        self._num_dict = {sp.sympify(sym): num for sym, num in num_dict.items()}

    ###########################################
    # Methods
    # -------
    def subs(self, sym_dict=None):
        self.value_at_index = self.value_at_index.subs(sym_dict)

    def update_num_dict(self, num_dict):
        self.num_dict.update(num_dict)
        self.refresh_lamfun()

    def simplify(self, kind="simplify", **kwargs):
        if kind == "simplify":
            self.value_at_index = sp.simplify(self.value_at_index, **kwargs)
        elif kind == "ratsimp":
            self.value_at_index = sp.ratsimp(self.value_at_index, **kwargs)

    def refresh(self, lamfun=True, lamfun_args=True):
        self._lamfun = self._make_lamfun()
        self._lamfun_args = self._make_lamfun_args()

    def numerical_eval(self, expr):
        if isinstance(expr, sp.Expr):
            return expr.subs(self.num_dict)
        else:
            return sp.sympify(expr).subs(self.num_dict)

    def sample_range(self, start, stop):
        for k in range(start, stop):
            yield self(k)

    def apply_index_transform(self, index_transform):
        self.subs({self.index: index_transform(self.index)})

    def apply_coord_transform(self, coordinate_transform):
        self.value_at_index = coordinate_transform(self.value_at_index)

    ###########################################
    # Private/special methods
    def __rep__(self):
        return f"Samples({self.value_at_index}, {self.index}, {self.num_dict})"

    def __str__(self):
        return f"Samples({self.value_at_index}, {self.index}, {self.num_dict})"

    def __call__(self, k):
        return self._lamfun(k, *self._lamfun_args)

    def __getitem__(self, sym):
        if isinstance(sym, sp.Symbol):
            return self.num_dict[sym]
        else:
            return self.num_dict[sp.sympify(sym)]

    def _repr_latex_(self):
        return f"${sp.latex(self.index)} \\mapsto {sp.latex(self.value_at_index)}$, ${sp.latex(self.num_dict)}$"

    def _make_lamfun(self):
        syms = [self.index] + sym_sorted(self.num_dict.keys())
        if not set(syms) >= set(self.value_at_index.free_symbols):
            raise ValueError("Mismatched symbols")
        return sp.lambdify(syms, self.value_at_index)

    def _make_lamfun_args(self):
        return [self[_] for _ in sym_sorted(self.num_dict.keys())]


############################################################
############################################################
class IndexTransform:
    def __init__(self, x, y_x):
        x_as_sympy = sp.sympify(x)
        self.x = sp.Dummy(x_as_sympy.name, latex_name=sp.latex(x_as_sympy))
        y_x_as_sympy = sp.sympify(y_x)
        self.y_x = y_x_as_sympy.subs({x_as_sympy: self.x})
        self.lamfun = sp.lambdify(self.x, self.y_x)

    def _repr_latex_(self):
        return f"${sp.latex(self.x)} \\mapsto {sp.latex(self.y_x)}$"

    def __rep__(self):
        return f"IndexTransform({self.x}, {self.y_x})"

    def __str__(self):
        return f"IndexTransform({self.x}, {self.y_x})"

    def __eq__(self, other):
        _x = sp.Dummy()
        return self.y_x.subs({self.x: _x}) == other.y_x.subs({self.x: _x})

    def __call__(self, x):
        return self.lamfun(x)

    def __mul__(self, other):
        return IndexTransform(other.x, self.y_x.subs({self.x: other.y_x}))

    def rinvert(self, x_inv):
        x_inv = dumbify(x_inv)
        subs_rinvert = sp.solve(self.y_x - x_inv, self.x, dict=True)
        if len(subs_rinvert) == 1:
            return IndexTransform(x_inv, subs_rinvert[0][self.x])
        else:
            return [IndexTransform(x_inv, sub[self.x]) for sub in subs_rinvert]

    def __invert__(self):
        rinv = self.rinvert("x")
        if hasattr(rinv, "__len__"):
            raise ValueError("Multivalued. Use rinvert() instead.")
        else:
            return rinv

    def __truediv__(self, other):
        return self * ~other

    def is_identity(self):
        return self.x == self.y_x

    ####################################################
    def right_action(self, x):
        return self.y_x.subs({self.x: x})


class Vertex:
    def __init__(self, coords=None, dart=None):
        self.coords = coords
        self.dart = dart


class Edge:
    def __init__(self, dart=None):
        self.dart = dart


class Triangle:
    def __init__(self, dart=None):
        self.dart = dart


class TriDart:
    def __init__(self):
        self.a = Vertex(coords=None, dart=self)
        self.ab = Edge(dart=self)
        self.abc = Triangle(dart=self)
        self.next = TriDart()
        self.twin = TriDart()


class PolyDisc:
    def __init__(self, vertex_number):
        self.vertex_number = sp.Symbol("n")
        self.vertex_index = sp.Symbol("k")


#######################################################
class UniformCoordinateInterval:
    """
    generates uniformly spaced coordinate samples for a given interval.

    Attributes
    ----------
    lower_bound: float
        The lower bound of the interval.

    start = 0
    stop = sp.symbols("upper_bound")
    num = sp.symbols("num_samples")
    Phi = UniformCoordinateInterval.linspace(start, stop, num_samples)

    """

    @classmethod
    def linspace(cls, start, stop, num_samples):
        return cls(
            lower_bound=start,
            upper_bound=stop,
            num_samples=num_samples,
        )

    def __init__(
        self,
        lower_bound="a",
        upper_bound="b",
        num_samples="2**p",
        dummy_index="k",
        subs={"a": 0, "b": 1, "p": 5},
        include_lower=False,
        include_upper=True,
    ):

        # super().__init__(
        #     dummy_index=dummy_index,
        #     value_at_index=f"ValMin+{dummy_index}*(ValMax-ValMin)/(NumSamples-1)",
        #     subs=subs,
        #     make_lamfun=False,
        # )
        self.lower_bound = sp.sympify(lower_bound)
        self.upper_bound = sp.sympify(upper_bound)
        self.num_samples = sp.sympify(num_samples)
        self.dummy_index = sp.sympify(dummy_index)
        self.include_lower = include_lower
        self.include_upper = include_upper
        self.subs = subs

        self.indexed_samples = IndexedSamples(
            dummy_index=dummy_index,
            value_at_index=f"{lower_bound}+({dummy_index}+StartShift)*({upper_bound}-{lower_bound})/({self.n_samples}+EndShift-1)",
            subs=subs | {"StartShift": self.start_shift, "EndShift": self.end_shift},
            make_lamfun=True,
        )

    def __getitem__(self, k):
        return self.indexed_samples[k]

    def __eval__(self):
        return self.numpy_arr

    @property
    def start_shift(self):
        if self.include_lower:
            return 0
        else:
            return 1

    @property
    def end_shift(self):
        if self.include_upper:
            return self.start_shift + 1
        else:
            return self.start_shift

    ###################################
    @property
    def val_min(self):
        if self.include_lower:
            return self.lower_bound
        else:
            return (
                (self.num_samples - 1) * self.lower_bound + self.upper_bound
            ) / self.num_samples

    @property
    def val_max(self):
        if self.include_upper:
            return self.upper_bound
        else:
            return (
                self.lower_bound + (self.num_samples - 1) * self.upper_bound
            ) / self.num_samples

    def generate_samples(self, kind="valid"):
        if kind == "valid":
            return self.sample_range(0, self.num_samples.subs(self.subs))
        if kind == "closed":
            return self.sample_range(
                -self.start_shift, self.num_samples.subs(self.subs) + self.end_shift
            )

    @property
    def numpy_arr(self, kind="valid"):
        return np.array([_ for _ in self.generate_samples(kind=kind)])


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
            coord_intervals = {
                "r": CoordinateInterval(0, 1),
                "theta": CoordinateCircle(),
                "phi": CoordinateCircle(),
            }

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
        start, end = numbers_part.split(",")

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
