import sympy as sp
from sympy.physics.vector import ReferenceFrame  # , vlatex
from IPython.display import display, Latex

# https://docs.sympy.org/latest/modules/plotting.html#sympy.plotting.plot.plot3d_parametric_line
# from sympy.vector import CoordSys3D, BaseVector, Point, Vector
# from sympy.plotting import plot, plot_implicit, plot3d, plot3d_parametric_surface
# expr.rcall(*args)


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
#
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
a = sp.symbols("a", positive=True)
x, y, z = OE[0], OE[1], OE[2]  # OE.varlist  # sp.symbols("x y z")
ex, ey, ez = OE["x"], OE["y"], OE["z"]
# surface coordinates
theta, phi = sp.symbols("theta phi", real=True)
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

h = sp.symbols("h", real=True, positive=True)


def truncate(expr):
    try:
        return sp.series(expr, x=h, n=3).removeO()
    except Exception:
        a = truncate(expr & n)
        b = truncate(expr & e_theta)
        c = truncate(expr & e_phi)
        return a * n + b * e_theta + c * e_phi


u = sp.Function("u")(theta, phi)
u0, u10, u01, u20, u02, u11 = sp.symbols(
    r"u u_{\theta} u_{\phi} u_{\theta\theta} u_{\phi\phi} u_{\theta\phi}"
)


diff_subs = {
    u: u0,
    u.diff(theta): u10,
    u.diff(phi): u01,
    u.diff(theta, 2): u20,
    u.diff(phi, 2): u02,
    u.diff(theta).diff(phi): u11,
    u.diff(phi).diff(theta): u11,
}


Xh = X + h * u * n
Xh_theta = Xh.diff(theta, frame=OE)
Xh_phi = Xh.diff(phi, frame=OE)

eh_phi = Xh_phi.normalize()
eh_theta = Xh_theta.normalize()
nh = eh_theta ^ eh_phi


dXh = (Xh_theta * dtheta + Xh_phi * dphi).subs(diff_subs)
dnh = (nh.diff(theta, frame=OE) * dtheta + nh.diff(phi, frame=OE) * dphi).subs(
    diff_subs
)
truncate(dnh)
metrich = truncate(dXh & dXh).trigsimp().expand()
curvatureh = truncate(-dXh & dnh)  # .trigsimp().expand()

Eh = metrich.coeff(dtheta**2).factor()
Fh = metrich.coeff(dtheta).coeff(dphi) / 2
Gh = metrich.coeff(dphi**2)

Ih = sp.Matrix([[Eh, Fh], [Fh, Gh]])


truncate(sp.sqrt((Ih).det().trigsimp())) - truncate(sp.sqrt((I).det().trigsimp()))


# %%
import sympy as sp

R, r = sp.symbols("R r", positive=True)
u = sp.symbols("u", real=True)
h = sp.symbols("h", positive=True)
theta, phi = sp.symbols("theta phi", real=True)

r_theta, r_phi = sp.symbols(r"r_{\theta} r_{\phi}", real=True)
r_theta_theta, r_phi_phi, r_theta_phi = sp.symbols(
    r"r_{\theta\theta} r_{\phi\phi} r_{\theta\phi}", real=True
)

u_theta, u_phi = sp.symbols(r"u_{\theta} u_{\phi}", real=True)
u_theta_theta, u_phi_phi, u_theta_phi = sp.symbols(
    r"u_{\theta\theta} u_{\phi\phi} u_{\theta\phi}", real=True
)

u_subs = {
    r: R * (1 + h * u),
    r_theta: R * h * u_theta,
    r_phi: R * h * u_phi,
    r_theta_theta: R * h * u_theta_theta,
    r_phi_phi: R * h * u_phi_phi,
    r_theta_phi: R * h * u_theta_phi,
}

detg = (r_theta**2 + r**2) * (
    r_phi**2 + r**2 * sp.sin(theta) ** 2
) - r_theta**2 * r_phi**2

detg_actual = detg.subs(u_subs)
area_form_actual = sp.sqrt(detg).subs(u_subs)
area_form0 = R**2 * sp.Abs(sp.sin(theta))

a0 = area_form_actual.subs({h: 0}).simplify()
a1 = area_form_actual.diff(h).subs({h: 0}).simplify()
a2 = area_form_actual.diff(h, 2).subs({h: 0}).simplify()

area_form_ = a0 + a1 * h + a2 * h**2
area_form = area_form_actual.subs(u_subs).series(x=h, x0=0, n=3).removeO().subs({h: 1})
# %%
import sympy as sp
import numpy as np
from sympy.abc import l, m, n

l = sp.symbols("n", poitive=True, integer=True)
q = sp.symbols("q", poitive=True, integer=True)

theta = sp.pi / 2
phi = 0
g = sp.Ynm(l, q, theta, phi)

expr = g * g.conjugate()
X = sp.simplify(expr.expand(func=True))

l_max = 100

nums = [
    [X.subs({l: ll, q: qq}) for qq in range(0, ll + 1)] for ll in range(0, l_max + 1)
]

nums = [X.subs({l: ll, q: qq}) for ll in range(0, l_max + 1) for qq in range(0, ll + 1)]

[(ll, qq) for ll in range(0, l_max + 1) for qq in range(0, ll + 1)]

max(nums).evalf()
