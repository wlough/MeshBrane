import sympy as sp
from sympy.physics.vector import ReferenceFrame  # , vlatex
from IPython.display import display, Latex

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
# %%
##################################
# Parameterization, orthonormal frame,...
##################################
OE = ReferenceFrame(
    "E", indices=["x", "y", "z"], latexs=[r"\bf{e}_x", r"\bf{e}_y", r"\bf{e}_z"], variables=["x", "y", "z"]
)
# major radius
a = sp.symbols("a")
# minor radius
b = sp.symbols("b")
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


# %%
# print(some stuff)
implicit_tex_str = eq_tex_str(implicit_fun, 0)
parametric_tex_str = eq_tex_str(sp.Function(r"\bf{X}")(phi, psi), X)
e_phi_tex = eq_tex_str(sp.Symbol(r"\bf{e}_\phi"), e_phi)
e_psi_tex = eq_tex_str(sp.Symbol(r"\bf{e}_\psi"), e_psi)
n_tex = eq_tex_str(sp.Symbol(r"\bf{n}"), n)
I_str = eq_tex_str(sp.Matrix([[sp.Symbol("E"), sp.Symbol("F")], [sp.Symbol("F"), sp.Symbol("G")]]), I)
II_str = eq_tex_str(sp.Matrix([[sp.Symbol("L"), sp.Symbol("M")], [sp.Symbol("M"), sp.Symbol("N")]]), II)
H_str = eq_tex_str(sp.Symbol("H"), H)
K_str = eq_tex_str(sp.Symbol("K"), K)

print("Torus\n-----")
print("Implicit:")
display(Latex(implicit_tex_str))
print("Parametric:")
display(Latex(parametric_tex_str))
print("Frame:")
display(Latex(e_phi_tex))
display(Latex(e_psi_tex))
display(Latex(n_tex))
print("Fundamental forms:")
display(Latex(I_str))
display(Latex(II_str))
print("Mean and Gaussian curvatures")
display(Latex(H_str))
display(Latex(K_str))
# %%
# from sympy.diffgeom.rn import R3, R3_origin, R3_r
from sympy.diffgeom import Manifold, Patch, CoordSystem
from sympy.physics.vector import ReferenceFrame

# R3_r.coord_functions()
Euc3 = Manifold(r"$\mathbb{E}^3$", 3)
OriginPatch = Patch("U", Euc3)
x, y, z = sp.symbols("x y z")
Cartesian3D = CoordSystem("Cartesian3D", OriginPatch, symbols=[x, y, z])
xp, yp, zp = sp.symbols("x' y' z'")
Cartesian3Dp = CoordSystem("Cartesian3D'", OriginPatch, symbols=[xp, yp, zp])
xpp, ypp, zpp = sp.symbols("x'' y'' z''")
Cartesian3Dpp = CoordSystem("Cartesian3D''", OriginPatch, symbols=[xpp, ypp, zpp])
q1, q2, q3 = sp.symbols("q^1 q^2 q^3")
Curvilinear3D = CoordSystem("Curvilinear3D", OriginPatch, symbols=[q1, q2, q3])

# major radius
a = sp.symbols("a")
a_num = 1
# minor radius
b = sp.symbols("b")
a2b = 3

M = Manifold("M", 2)
M0 = Patch("U", M)
phi, psi = sp.symbols("phi psi")
SimpleToroidal = CoordSystem("SimpleToroidal", M0, symbols=[phi, psi])
Monge = CoordSystem("Monge", M0, symbols=[x, y])
relation_dict = {
    ("Monge", "SimpleToroidal"): [(x, y), sp.atan2(y, x), 1],
    ("SimpleToroidal", "Monge"): [
        (phi, psi),
        (a + sp.cos(psi) * b) * sp.cos(phi),
        (a + sp.cos(psi) * b) * sp.sin(phi),
    ],
}

# from sympy import symbols, pi, sqrt, atan2, cos, sin
# from sympy.diffgeom import Manifold, Patch, CoordSystem
# (a + sp.cos(psi) * b) * sp.cos(phi) * ex + (a + sp.cos(psi) * b) * sp.sin(phi) * ey + b * sp.sin(psi) * ez
# m = Manifold("M", 2)
# p = Patch("P", m)
# x, y = symbols("x y", real=True)
# r, theta = symbols("r theta", nonnegative=True)
# relation_dict = {
#     ("Car2D", "Pol"): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
#     ("Pol", "Car2D"): [(r, theta), (r * cos(theta), r * sin(theta))],
# }
# Car2D = CoordSystem("Car2D", p, (x, y), relation_dict)
# Pol = CoordSystem("Pol", p, (r, theta), relation_dict)
