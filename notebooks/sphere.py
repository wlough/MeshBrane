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
        tex_str = (
            "\\begin{equation}"
            + sp.latex(lhs)
            + " = "
            + sp.latex(rhs)
            + "\\end{equation}"
        )
    elif mode == "equation*":
        tex_str = (
            "\\begin{equation*}"
            + sp.latex(lhs)
            + " = "
            + sp.latex(rhs)
            + "\\end{equation*}"
        )

    return tex_str


# sp.init_printing()
# %%
##################################
# Parameterization, orthonormal frame,...
##################################
OE = ReferenceFrame(
    "E",
    indices=["x", "y", "z"],
    latexs=[r"\bf{e}_x", r"\bf{e}_y", r"\bf{e}_z"],
    variables=["x", "y", "z"],
)
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
# %%
# Lame coefficients
h_theta = (e_theta & X_theta).simplify()
h_phi = (e_phi & X_phi).simplify()
# area element
J = h_theta * h_phi
# %%
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


print("Torus\n-----")
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
v1 = X.diff(theta, OE).diff(theta, OE)
v2 = X.diff(phi, OE).diff(phi, OE)
v3 = X.diff(theta, OE)
c1 = 1 / h_theta**2
c2 = 1 / h_phi**2
c3 = sp.cos(theta) / (a**2 * sp.sin(theta))
lboX = c1 * v1 + c2 * v2 + c3 * v3
(lboX / (2 * H)).simplify() - n
# %%
tau = sp.symbols(r"\tau")
num_subs = {tau: sp.GoldenRatio}
u = a * tau / sp.sqrt(1 + tau**2)
v = a / sp.sqrt(1 + tau**2)
V = sp.Matrix([])


V0 = [
    [0, v, -u],
    [v, u, 0],
    [-v, u, 0],
    [0, v, u],
    [0, -v, u],
    [-u, 0, v],
    [0, -v, -u],
    [u, 0, -v],
    [u, 0, v],
    [-u, 0, -v],
    [v, -u, 0],
    [-v, -u, 0],
]
F0 = [
    [2, 1, 0],
    [1, 2, 3],
    [5, 4, 3],
    [4, 8, 3],
    [7, 6, 0],
    [6, 9, 0],
    [11, 10, 4],
    [10, 11, 6],
    [9, 5, 2],
    [5, 9, 11],
    [8, 7, 1],
    [7, 8, 10],
    [2, 5, 3],
    [8, 1, 3],
    [9, 2, 0],
    [1, 7, 0],
    [11, 9, 6],
    [7, 10, 6],
    [5, 11, 4],
    [10, 8, 4],
]
#

#

#

#

#

#
