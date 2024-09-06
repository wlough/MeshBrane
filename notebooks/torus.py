import sympy as sp
from sympy.physics.vector import ReferenceFrame  # , vlatex
from IPython.display import display, Latex

# https://docs.sympy.org/latest/modules/plotting.html#sympy.plotting.plot.plot3d_parametric_line
# from sympy.vector import CoordSys3D, BaseVector, Point, Vector
# from sympy.plotting import plot, plot_implicit, plot3d, plot3d_parametric_surface


def eq_tex_str(lhs, rhs, mode="inline"):
    if mode == "inline":
        tex_str = "$" + sp.latex(lhs) + " = " + sp.latex(rhs) + "$"
    elif mode == "plain":
        tex_str = sp.latex(lhs) + " = " + sp.latex(rhs)
    elif mode == "equation":
        tex_str = "\\begin{equation}" + sp.latex(lhs) + " = " + sp.latex(rhs) + "\\end{equation}"
    elif mode == "equation*":
        tex_str = "\\begin{equation*}" + sp.latex(lhs) + " = " + sp.latex(rhs) + "\\end{equation*}"

    return tex_str


# sp.init_printing()

#######################################
# Parameterization, orthonormal frame,...
########################################
OE = ReferenceFrame(
    "E",
    indices=["x", "y", "z"],
    latexs=[r"\bf{e}_x", r"\bf{e}_y", r"\bf{e}_z"],
    variables=["x", "y", "z"],
)
# major radius
a = sp.symbols("a", positive=True)
# minor radius
b = sp.symbols("b", positive=True)
x, y, z = OE[0], OE[1], OE[2]  # OE.varlist  # sp.symbols("x y z")
ex, ey, ez = OE["x"], OE["y"], OE["z"]

# surface coordinates
phi, psi = sp.symbols("phi psi")
# implicit surface function
implicit_fun = (sp.sqrt(x**2 + y**2) - a) ** 2 + z**2 - b**2
# surface parameterization
X = (a + sp.cos(psi) * b) * sp.cos(phi) * ex + (a + sp.cos(psi) * b) * sp.sin(phi) * ey + b * sp.sin(psi) * ez
# coordinate basis vectors
X_phi = X.diff(phi, frame=OE)
X_psi = X.diff(psi, frame=OE)
# moving frame
# e_phi = X_phi.normalize().simplify()
e_phi = -sp.sin(phi) * ex + sp.cos(phi) * ey
# e_psi = X_psi.normalize().simplify()
e_psi = -sp.sin(psi) * sp.cos(phi) * ex - sp.sin(psi) * sp.sin(phi) * ey + sp.cos(psi) * ez
# normal vector
n = (e_phi ^ e_psi).simplify()
# Metric and extrinsic curvature tensors
dphi = sp.symbols(r"d\phi")
dpsi = sp.symbols(r"d\psi")
dX = X_phi * dphi + X_psi * dpsi
dn = n.diff(phi, frame=OE) * dphi + n.diff(psi, frame=OE) * dpsi
metric = (dX & dX).trigsimp()
curvature = (-dX & dn).trigsimp()
# fundamental forms
E = metric.coeff(dphi**2).factor()
F = metric.coeff(dphi).coeff(dpsi) / 2
G = metric.coeff(dpsi**2)
L = curvature.coeff(dphi**2).factor()
M = curvature.coeff(dphi).coeff(dpsi) / 2
N = curvature.coeff(dpsi**2)
I = sp.Matrix([[E, F], [F, G]])
II = sp.Matrix([[L, M], [M, N]])
I_inv = sp.Matrix([[E, F], [F, G]]).inv().applyfunc(lambda _: _.factor())
# Shape operator and mean/Gaussian curvatures
shape = II @ I_inv
H = (shape.trace() / 2).factor()
K = shape.det().factor()

# Lame coefficients
h_phi = (e_phi & X_phi).simplify()
h_psi = (e_psi & X_psi).simplify()
# area element
J = h_phi * h_psi
# inverse coordinate transform
rho = sp.sqrt(x**2 + y**2)
# zp = sp.sqrt(b**2 - (a - rho) ** 2)
# zm = -sp.sqrt(b**2 - (a - rho) ** 2)
Phi = sp.atan2(y, x)
Psi = sp.atan2(z, rho - a)
# surface area
# A = sp.integrate(J, (phi, 0, 2 * sp.pi), (psi, 0, 2 * sp.pi))
A = 4 * sp.pi**2 * a * b
scale_unit_torus = 2 * sp.pi * sp.sqrt(a * b)
a_unit, b_unit = a / scale_unit_torus, b / scale_unit_torus
# %%
##############################
# print(some stuff)
implicit_tex_str = eq_tex_str(implicit_fun, 0)
parametric_tex_str = eq_tex_str(sp.Function(r"\bf{X}")(phi, psi), X)
X_phi_tex = eq_tex_str(sp.Symbol(r"\bf{X}_\phi"), X_phi)
X_psi_tex = eq_tex_str(sp.Symbol(r"\bf{X}_\psi"), X_psi)
e_phi_tex = eq_tex_str(sp.Symbol(r"\bf{e}_\phi"), e_phi)
e_psi_tex = eq_tex_str(sp.Symbol(r"\bf{e}_\psi"), e_psi)
n_tex = eq_tex_str(sp.Symbol(r"\bf{n}"), n)
I_str = eq_tex_str(sp.Matrix([[sp.Symbol("E"), sp.Symbol("F")], [sp.Symbol("F"), sp.Symbol("G")]]), I)
II_str = eq_tex_str(sp.Matrix([[sp.Symbol("L"), sp.Symbol("M")], [sp.Symbol("M"), sp.Symbol("N")]]), II)
H_str = eq_tex_str(sp.Symbol("H"), H)
K_str = eq_tex_str(sp.Symbol("K"), K)
h_phi_tex = eq_tex_str(sp.Symbol(r"h_\phi"), h_phi)
h_psi_tex = eq_tex_str(sp.Symbol(r"h_\psi"), h_psi)
# %%
print("-------\n|Torus|\n-------\n")
print("Implicit:")
display(Latex(implicit_tex_str))
print("Parametric:")
display(Latex(parametric_tex_str))
print("Coorinate basis:")
display(Latex(X_phi_tex))
display(Latex(X_psi_tex))
print("Frame:")
display(Latex(e_phi_tex))
display(Latex(e_psi_tex))
display(Latex(n_tex))
print("Lame coefficients:")
display(Latex(h_phi_tex))
display(Latex(h_psi_tex))
print("Fundamental forms:")
display(Latex(I_str))
display(Latex(II_str))
print("Mean and Gaussian curvatures")
display(Latex(H_str))
display(Latex(K_str))
# %%
# dphi,dpsi = sp.symbols(r"d\phi"), sp.symbols(r"d\psi")
# dA = J*dphi*dpsi
# A=sp.integrate(J, (phi, 0, 2*sp.pi), (psi, 0, 2*sp.pi))


# %%
v1 = X.diff(phi, OE).diff(phi, OE)
v2 = X.diff(psi, OE).diff(psi, OE)
v3 = X.diff(psi, OE)
c1 = 1 / h_phi**2
c2 = 1 / h_psi**2
c3 = -sp.sin(psi) / (h_phi * h_psi)
lboX = c1 * v1 + c2 * v2 + c3 * v3
(lboX / (2 * H)).simplify() - n
# %%
# %%
assumptions = (
    sp.Q.gt(a, b),
    sp.Q.gt(a, 0),
    sp.Q.gt(b, 0),
    sp.Q.ge(x, 0),
    sp.Q.ge(y, 0),
    sp.Q.ge(z, 0),
    sp.Q.ge(x, a - b),
    sp.Q.ge(y, a - b),
    sp.Q.ge(z, 0),
    # sp.Q.is_true(sp.And(sp.Q.ge(z, -b), sp.Q.le(z, b))),
    # sp.Q.is_true(sp.And(sp.Q.ge(x**2, (a-b)**2), sp.Q.le(x**2, -(a-b)**2))),
    # sp.Q.is_true(sp.And(sp.Q.ge(z, -b), sp.Q.le(z, b))),
    # sp.Q.is_true(sp.And(sp.Q.ge(x, -b), sp.Q.le(z, b))),
    # sp.Q.is_true(sp.And(sp.Q.gt(phi, -sp.pi), sp.Q.le(phi, sp.pi))),
    # sp.Q.is_true(sp.And(sp.Q.ge(psi, 0), sp.Q.le(psi, 2*sp.pi))),
)
# ((sp.sqrt(x**2 + y**2) - a) ** 2 + z**2 - b**2).simplify().expand().simplify()
with sp.assuming(*assumptions):
    simplified_expr = H.subs({phi: Phi, psi: Psi}).simplify()  # .expand().simplify()
# mcvec.applyfunc(lambda _: _.radsimp())
# %%
from sympy.diffgeom.rn import R3, R3_origin, R3_r

# from sympy.abc import a, b, x, y, z, phi, psi
import sympy as sp

# from sympy.diffgeom import *
from sympy.diffgeom.diffgeom import _find_coords

from sympy.diffgeom import (
    Manifold,
    Patch,
    CoordSystem,
    Differential,
    WedgeProduct,
    Commutator,
    metric_to_Christoffel_2nd,
    TensorProduct,
    BaseCovarDerivativeOp,
    LieDerivative,
    intcurve_series,
    metric_to_Riemann_components,
    metric_to_Ricci_components,
    twoform_to_matrix,
    metric_to_Christoffel_1st,
    covariant_order,
    contravariant_order,
)


def pdv_op_to_matrix(pdv_op, coord_sys):
    _pdv_op = pdv_op.expand()
    return sp.Matrix([_pdv_op.coeff(_) for _ in coord_sys.base_vectors()])


def matrix_to_pdv_op(matv, coord_sys):
    return matv.dot(coord_sys.base_vectors())


def eval_scalar_tensor(T, *u):
    T_terms = T.expand().as_ordered_terms()
    val = sp.sympify(0)
    for term in T_terms:
        term_factors = term.as_ordered_factors()
        _val = sp.sympify(1)
        for fac in term_factors:
            if isinstance(fac, TensorProduct) or isinstance(fac, Differential):
                _val *= fac(*u)
            else:
                _val *= fac
        val += _val
    return val


def eval_tensor(T, *u):
    # if isinstance(T, sp.Matrix):
    #     return sp.Matrix([eval_scalar_tensor(t, *u) for t in T])
    # elif isinstance(T, sp.Array):
    #     return sp.Array([eval_scalar_tensor(t, *u) for t in T])
    # else:
    #     return eval_scalar_tensor(T, *u)
    try:
        return T.rcall(*u)
    except AttributeError:
        return T.applyfunc(lambda _: _.rcall(*u))


def simplify_twoform(w):
    coordsys = _find_coords(w).pop()
    wmat = twoform_to_matrix(w)
    dim = coordsys.dim
    if dim == 2:
        dx0, dx1 = coordsys.base_oneforms()
        dx01 = WedgeProduct(dx0, dx1)
        return wmat[0, 1] * dx01
    elif dim == 3:
        dx0, dx1, dx2 = coordsys.base_oneforms()
        dx12 = WedgeProduct(dx1, dx2)
        dx20 = WedgeProduct(dx2, dx0)
        dx01 = WedgeProduct(dx0, dx1)
        return wmat[1, 2] * dx12 + wmat[2, 0] * dx20 + wmat[0, 1] * dx01
    else:
        raise ValueError("dim must be 2 or 3")


def twotensor_to_matrix(expr):
    if covariant_order(expr) != 2 or contravariant_order(expr):
        raise ValueError("The input expression is not a covariant rank two tensor.")
    coord_sys = _find_coords(expr)
    if len(coord_sys) != 1:
        raise ValueError(
            "The input expression concerns more than one "
            "coordinate systems, hence there is no unambiguous "
            "way to choose a coordinate system for the matrix."
        )
    coord_sys = coord_sys.pop()
    vectors = coord_sys.base_vectors()
    expr = expr.expand()
    matrix_content = [[expr.rcall(v1, v2) for v1 in vectors] for v2 in vectors]
    return sp.Matrix(matrix_content)


def WedgeCross(u, v):
    u1, u2, u3 = u
    v1, v2, v3 = v
    u_v = [
        WedgeProduct(u2, v3) - WedgeProduct(u3, v2),
        WedgeProduct(u3, v1) - WedgeProduct(u1, v3),
        WedgeProduct(u1, v2) - WedgeProduct(u2, v1),
    ]
    return sp.Matrix([simplify_twoform(_) for _ in u_v])


def WedgeDot(u, v):
    u1, u2, u3 = u
    v1, v2, v3 = v
    u_v = WedgeProduct(u1, v1) + WedgeProduct(u2, v2) + WedgeProduct(u3, v3)

    return simplify_twoform(u_v)


def TensorDot(u, v):
    u1, u2, u3 = u
    v1, v2, v3 = v
    u_v = TensorProduct(u1, v1) + TensorProduct(u2, v2) + TensorProduct(u3, v3)

    return u_v


def r3_coords():
    R3_toroid = Patch("toroid", R3)
    C = CoordSystem("cartesian", R3_toroid, sp.symbols("x y z"))
    # x,y,z=C.symbols
    x, y, z = C.base_scalars()
    dx, dy, dz = C.base_oneforms()
    d_dx, d_dy, d_dz = C.base_vectors()
    X = sp.Matrix([x, y, z])
    dX = sp.Matrix([dx, dy, dz])
    v = sp.Matrix([sp.cos(z) * dy, z**2 * dz, y * dx])
    # WedgeCross(dX, v)

    metric_C = TensorProduct(dx, dx) + TensorProduct(dy, dy) + TensorProduct(dz, dz)
    dvol_C = WedgeProduct(dx, dy, dz)
    darea_C = WedgeProduct(dx, dy) + WedgeProduct(dy, dz)


# %%
# major radius
a = sp.symbols("a")
a_num = 1
# minor radius
b = sp.symbols("b")
a2b = 3

M = Manifold("M", 2)
U = Patch("U", M)
# Uzp = Patch(r"$U^{z+}$", M)
# Uzm = Patch(r"$U^{z-}$", M)
# Mzm = Patch("Mzp", M)
_phi, _psi = sp.symbols("phi psi")
_x, _y = sp.symbols("x y")
rho = sp.sqrt(_x**2 + _y**2)
zp = sp.sqrt(b**2 - (a - rho) ** 2)
zm = -sp.sqrt(b**2 - (a - rho) ** 2)
relation_dict = {
    ("Monge+", "SimpleToroidal"): [(_x, _y), (sp.atan2(_y, _x), sp.atan2(zp, rho - a))],
    ("Monge-", "SimpleToroidal"): [(_x, _y), (sp.atan2(_y, _x), -sp.atan2(zm, rho - a))],
    ("SimpleToroidal", "Monge+"): [
        (_phi, _psi),
        (
            (a + sp.cos(_psi) * b) * sp.cos(_phi),
            (a + sp.cos(_psi) * b) * sp.sin(_phi),
        ),
    ],
}
Toroidal = CoordSystem("SimpleToroidal", U, (_phi, _psi), relation_dict)
Mongep = CoordSystem("Monge+", U, (_x, _y), relation_dict)
Mongem = CoordSystem("Monge-", U, (_x, _y), relation_dict)

phi, psi = Toroidal.base_scalars()
dphi, dpsi = Toroidal.base_oneforms()
d_dphi, d_dpsi = Toroidal.base_vectors()
x = (a + sp.cos(psi) * b) * sp.cos(phi)
y = (a + sp.cos(psi) * b) * sp.sin(phi)
z = b * sp.sin(phi)
ex, ey, ez = sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])
# X = x * ex + y * ey + z * ez
# dX = Differential(X)(d_dphi) * dphi + Differential(X)(d_dpsi) * dpsi
# Xphi = d_dphi(X)
# Xpsi = d_dpsi(X)
# Xphi.dot(Xphi).subs({phi: _phi, psi: _psi}).trigsimp()
# metric = x**2 * TensorProduct(dphi, dpsi) + TensorProduct(dphi, dphi)
# # sp.Matrix([metric, z * metric]).rcall(d_dphi, d_dpsi)
# e_phi = -sp.sin(phi) * ex + sp.cos(phi) * ey
# # e_psi = X_psi.normalize().simplify()
# e_psi = -sp.sin(psi) * sp.cos(phi) * ex - sp.sin(psi) * sp.sin(phi) * ey + sp.cos(psi) * ez

X = (a + sp.cos(psi) * b) * sp.cos(phi) * ex + (a + sp.cos(psi) * b) * sp.sin(phi) * ey + b * sp.sin(psi) * ez
# coordinate basis vectors
X_phi = d_dphi(X)
X_psi = d_dpsi(X)
# moving frame
# e_phi = X_phi.normalize().simplify()
e_phi = -sp.sin(phi) * ex + sp.cos(phi) * ey
# e_psi = X_psi.normalize().simplify()
e_psi = -sp.sin(psi) * sp.cos(phi) * ex - sp.sin(psi) * sp.sin(phi) * ey + sp.cos(psi) * ez
# normal vector
n = e_phi.cross(e_psi).applyfunc(lambda _: _.simplify())
# twoforms = sp.Matrix([TensorProduct(dx1,dx2) for dx2 in Toroidal.base_oneforms() for dx1 in Toroidal.base_oneforms()])
metric_components = sp.Matrix([[u.dot(v).simplify() for v in [X_phi, X_psi]] for u in [X_phi, X_psi]])
metric = X_phi.dot(X_phi).simplify() * TensorProduct(dphi, dphi) + X_psi.dot(X_psi).simplify() * TensorProduct(
    dpsi, dpsi
)


Riemann_components = metric_to_Riemann_components(metric)
Ricci_components = metric_to_Ricci_components(metric)
Christoffel_1st = metric_to_Christoffel_1st(metric).simplify()
