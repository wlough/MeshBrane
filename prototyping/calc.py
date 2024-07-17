import sympy as sp
from src.python.sym_tools import dot, cross


# x, y, z, r, a, c = sp.symbols("x y z r a c")
# sigma, tau, phi, theta = sp.symbols("sigma tau phi theta")

# sphere
name = "torus"
a_num = 1
a2b = 3
a = sp.symbols("a")  # c
b = sp.symbols("b")  # sp.sympify(f"{c} / {c2r}")
x, y, z = sp.symbols("x y z")
psi, phi = sp.symbols("psi phi")
num_subs = {a: a_num, b: sp.sympify(f"{a_num} / {a2b}")}

implicit_rep_xyz = (sp.sqrt(x**2 + y**2) - a) ** 2 + z**2 - b**2
X = sp.Array(
    [
        (a + sp.cos(psi) * b) * sp.cos(phi),
        (a + sp.cos(psi) * b) * sp.sin(phi),
        b * sp.sin(psi),
    ]
)
X_psi = X.diff(psi)
X_phi = X.diff(phi)
E = dot(X_phi, X_phi).trigsimp()
F = dot(X_phi, X_psi).trigsimp()
G = dot(X_psi, X_psi).trigsimp()
I = sp.Array([[E, F], [F, G]])

e_phi = sp.Array([-sp.sin(phi), sp.cos(phi), 0])
e_psi = sp.Array([-sp.sin(psi) * sp.cos(phi), -sp.sin(psi) * sp.sin(phi), sp.cos(psi)])
n = cross(e_phi, e_psi).applyfunc(lambda _: _.trigsimp())
X.subs({phi: sp.pi / 2})
n.subs({phi: sp.pi / 2})

X_phi_phi = X.diff(phi).diff(phi)
X_phi_psi = X.diff(phi).diff(psi)
X_psi_psi = X.diff(psi).diff(psi)
L = dot(n, X_phi_phi).trigsimp()
M = dot(n, X_phi_psi).trigsimp()
N = dot(n, X_psi_psi).trigsimp()
II = sp.Array([[L, M], [M, N]])
II_Iinv = sp.Array([[L / E, 0], [0, N / G]])
# %%
# H=(G*L-2*F*M+E*N)/(2*(E*G-F**2))
H = (G * L + E * N) / (2 * (E * G))
a * b * H
K = L * N / (E * G)

# %%
thetaphi_xyz = sp.Array([sp.atan2(z, sp.sqrt(x**2 + y**2)), sp.atan2(y, x)])
surface_area = 4 * sp.pi**2 * R * r
volume = 2 * sp.pi**2 * R * r**2
# %%
import numpy as np
from mayavi import mlab


def plot_level_sets(func, x_range, y_range, z_range, levels):
    """
    Plots level sets of a given function using Mayavi mlab.

    Parameters:
    - func: The function to plot. It must take three numpy arrays (X, Y, Z) as input and return an array of values.
    - x_range, y_range, z_range: Tuples of (min, max) specifying the range to plot over for each dimension.
    - levels: Number of contour levels to plot.
    """
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    z = np.linspace(z_range[0], z_range[1], 100)
    X, Y, Z = np.meshgrid(x, y, z)

    # Evaluate the function on the grid
    values = func(X, Y, Z)

    # Plot the level sets
    mlab.contour3d(X, Y, Z, values, contours=levels)
    mlab.show()


# Example function: a simple scalar field
def my_func(X, Y, Z):
    return X**2 + Y**2 + Z**2


# Example usage
plot_level_sets(my_func, (-3, 3), (-3, 3), (-3, 3), levels=10)
