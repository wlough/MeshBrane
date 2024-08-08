import sympy as sp
from sympy.physics.vector import ReferenceFrame  # , vlatex
from src.python.pretty_pictures import eq_tex_str
from IPython.display import display, Latex

# https://docs.sympy.org/latest/modules/plotting.html#sympy.plotting.plot.plot3d_parametric_line
# from sympy.vector import CoordSys3D, BaseVector, Point, Vector
# from sympy.plotting import plot, plot_implicit, plot3d, plot3d_parametric_surface
# expr.rcall(*args)


# def eq_tex_str(lhs, rhs, mode="inline"):
#     if mode == "inline":
#         tex_str = "$" + latex(lhs) + " = " + latex(rhs) + "$"
#     elif mode == "plain":
#         tex_str = latex(lhs) + " = " + latex(rhs)
#     elif mode == "equation":
#         tex_str = (
#             "\\begin{equation}" + latex(lhs) + " = " + latex(rhs) + "\\end{equation}"
#         )
#     elif mode == "equation*":
#         tex_str = (
#             "\\begin{equation*}" + latex(lhs) + " = " + latex(rhs) + "\\end{equation*}"
#         )
#
#     return tex_str


# sp.init_printing()
#
##################################
# Parameterization, orthonormal frame,...
##################################
# ambient cartesian frame
OE = ReferenceFrame(
    "E",
    indices=["x", "y", "z"],
    latexs=[r"\bf{e}_x", r"\bf{e}_y", r"\bf{e}_z"],
    variables=["x", "y", "z"],
)
# surface adapted orthonormal frame
F = ReferenceFrame(
    "F",
    indices=["theta", "phi", "eta"],
    latexs=[r"\bf{e}_\theta", r"\bf{e}_\phi", r"\bf{n}"],
    variables=["theta", "phi", "eta"],
)
theta, phi, eta = F[0], F[1], F[2]  # OE.varlist  # sp.symbols("x y z")
e_theta, e_phi, e_eta = F["theta"], F["phi"], F["eta"]
# Lame coefficients
h_theta = sp.Function(r"\text{h}_{\theta}")(theta, phi)
h_phi = sp.Function(r"\text{h}_{\phi}")(theta, phi)
X_theta = h_theta * e_theta
X_phi = h_phi * e_phi

# Metric and extrinsic curvature tensors
dphi = sp.symbols(r"\text{d}\phi")
dtheta = sp.symbols(r"\text{d}\theta")
dX = X_theta * dtheta + X_phi * dphi
# %%
# radius
a = sp.symbols("a")
x, y, z = OE[0], OE[1], OE[2]  # OE.varlist  # sp.symbols("x y z")
ex, ey, ez = OE["x"], OE["y"], OE["z"]
# surface coordinates
theta, phi = sp.symbols("theta phi")
# implicit surface function
implicit_fun = x**2 + y**2 + z**2 - a**2
# surface parameterization
X = (
    a * sp.sin(theta) * sp.cos(phi) * ex
    + a * sp.sin(theta) * sp.sin(phi) * ey
    + a * sp.cos(theta) * ez
)

# coordinate basis vectors
X_theta = X.diff(theta, frame=OE)
X_phi = X.diff(phi, frame=OE)
# moving frame
# e_phi = X_phi.normalize().simplify()
e_phi = -sp.sin(phi) * ex + sp.cos(phi) * ey
# e_theta = X_theta.normalize().simplify()
e_theta = (
    sp.cos(theta) * sp.cos(phi) * ex
    + sp.cos(theta) * sp.sin(phi) * ey
    - sp.sin(theta) * ez
)
# normal vector
n = (e_theta ^ e_phi).simplify()

# Metric and extrinsic curvature tensors
dphi = sp.symbols(r"d\phi")
dtheta = sp.symbols(r"d\theta")
dX = X_theta * dtheta + X_phi * dphi
dn = n.diff(theta, frame=OE) * dtheta + n.diff(phi, frame=OE) * dphi
metric = (dX & dX).trigsimp().expand()
curvature = (-dX & dn).trigsimp().expand()
# fundamental forms
E = metric.coeff(dtheta**2).factor()
F = metric.coeff(dtheta).coeff(dphi) / 2
G = metric.coeff(dphi**2)
L = curvature.coeff(dtheta**2).factor()
M = curvature.coeff(dtheta).coeff(dphi) / 2
N = curvature.coeff(dphi**2)
I = sp.Matrix([[E, F], [F, G]])
II = sp.Matrix([[L, M], [M, N]])
I_inv = sp.Matrix([[E, F], [F, G]]).inv().applyfunc(lambda _: _.factor())
# Shape operator and mean/Gaussian curvatures
shape = II @ I_inv
H = (shape.trace() / 2).factor()
K = shape.det().factor()
#
# Lame coefficients
h_theta = (e_theta & X_theta).simplify()
h_phi = (e_phi & X_phi).simplify()
# area element
J = h_theta * h_phi
#
# print(some stuff)
implicit_tex_str = eq_tex_str(implicit_fun, 0)
parametric_tex_str = eq_tex_str(sp.Function(r"\bf{X}")(theta, phi), X)
X_theta_tex = eq_tex_str(sp.Symbol(r"\bf{X}_\theta"), X_theta)
X_phi_tex = eq_tex_str(sp.Symbol(r"\bf{X}_\phi"), X_phi)
e_theta_tex = eq_tex_str(sp.Symbol(r"\bf{e}_\theta"), e_theta)
e_phi_tex = eq_tex_str(sp.Symbol(r"\bf{e}_\phi"), e_phi)
n_tex = eq_tex_str(sp.Symbol(r"\bf{n}"), n)
I_str = eq_tex_str(
    sp.Matrix([[sp.Symbol("E"), sp.Symbol("F")], [sp.Symbol("F"), sp.Symbol("G")]]), I
)
II_str = eq_tex_str(
    sp.Matrix([[sp.Symbol("L"), sp.Symbol("M")], [sp.Symbol("M"), sp.Symbol("N")]]), II
)
H_str = eq_tex_str(sp.Symbol("H"), H)
K_str = eq_tex_str(sp.Symbol("K"), K)
h_theta_tex = eq_tex_str(sp.Symbol(r"h_\theta"), h_theta)
h_phi_tex = eq_tex_str(sp.Symbol(r"h_\phi"), h_phi)


print("Sphere\n-----")
print("Implicit:")
display(Latex(implicit_tex_str))
print("Parametric:")
display(Latex(parametric_tex_str))
print("Coorinate basis:")
display(Latex(X_theta_tex))
display(Latex(X_phi_tex))
print("Frame:")
display(Latex(e_theta_tex))
display(Latex(e_phi_tex))
display(Latex(n_tex))
print("Lame coefficients:")
display(Latex(h_theta_tex))
display(Latex(h_phi_tex))
print("Fundamental forms:")
display(Latex(I_str))
display(Latex(II_str))
print("Mean and Gaussian curvatures")
display(Latex(H_str))
display(Latex(K_str))
# %%
import sympy as sp
from sympy.functions.special.spherical_harmonics import Ynm

l, m = 2, 1
Y = Ynm(l, m, theta, phi)

laplacian0 = lambda Y: (
    (
        (1 / sp.sin(theta)) * sp.diff(sp.sin(theta) * sp.diff(Y, theta), theta)
        + (1 / (sp.sin(theta) ** 2)) * sp.diff(Y, phi, phi)
    )
    / a**2
)
laplacian = (
    lambda Y: Y.diff(theta, 2) / a**2
    + Y.diff(phi, 2) / (a**2 * sp.sin(theta) ** 2)
    + sp.cos(theta) * Y.diff(theta) / (a**2 * sp.sin(theta))
)
# Y_real, Y_imag = Y.expand(func=True)  # .as_real_imag()

# Compute the Laplacian in spherical coordinates
laplacian_Y = laplacian(Y).trigsimp()
laplacian_Y0 = laplacian0(Y).trigsimp()
#

# Display the Laplacian of the spherical harmonics
(laplacian_Y.expand(func=True).trigsimp() / Y.expand(func=True)).trigsimp().simplify()
# rcall
(1 / sp.sin(theta)) * sp.diff(sp.sin(theta) * sp.diff(Y, theta), theta) + (
    1 / (sp.sin(theta) ** 2)
) * sp.diff(Y, phi, phi)
