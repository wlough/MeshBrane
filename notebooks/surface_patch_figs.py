import sys
from pathlib import Path

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



##############
# Fig options
##############
OUTPUT = "../output/surface_patch_conormal.svg"

# Camera
ELEV = 24
AZIM = -58

# Surface geometry: z = SURF_SCALE * (u^2 + CROSS_TERM*u*v + V2_TERM*v^2)
SURF_SCALE = -0.18
CROSS_TERM = 0.0
V2_TERM = 1.0

# Outer displayed surface domain in parameter space
U_MIN, U_MAX = -2.2, 2.2
V_MIN, V_MAX = -1.8, 1.8
NU, NV = 3*180, 3*160

# Patch P: ellipse in parameter space
U0, V0 = 0.0, -0.7          # patch center
A_PATCH = .95             # semi-axis in u
B_PATCH = 0.70             # semi-axis in v
N_PATCH = 220

# Boundary point at which to draw n, u, t
THETA0 = -0.45 * np.pi

# Arrow lengths
N_LEN = 0.70
U_LEN = 0.70
T_LEN = 0.70


# Colors
# COLOR_SURFACE = "#d9d9d9"
# COLOR_PATCH = "#9ecae1"
# COLOR_BOUNDARY = "black"
# COLOR_N = "black"
# COLOR_U = "#d62728"
# COLOR_T = "#1f77b4"
# COLOR_TR = "#2ca02c"


# COLOR_SURFACE = "0.92"
# COLOR_PATCH = "C0"
# COLOR_BOUNDARY = "k"
# COLOR_N = "k"
# COLOR_U = "C3"
# COLOR_T = "C0"


# COLOR_SURFACE = "0.85"   # light gray
# COLOR_PATCH = "C0"       # default blue
# COLOR_BOUNDARY = "k"     # black
# COLOR_N = "k"            # black
# COLOR_U = "C1"           # default orange
# COLOR_T = "C2"           # default green

#
COLOR_SURFACE = "0.85"
COLOR_PATCH = "C0"
COLOR_BOUNDARY = "k"
COLOR_N = "k"
COLOR_U = "C3"       # red
COLOR_T = "C2"       # green


M_alpha = 0.9
P_alpha = 0.65

#
###########
# Geometry
###########
def surface(u, v):
    """Embedding r(u,v) = (u, v, z(u,v))."""
    z = SURF_SCALE * (u**2 + CROSS_TERM * u * v + V2_TERM * v**2)
    return np.stack([u, v, z], axis=-1)


def r_u(u, v):
    """Partial derivative dr_du."""
    dz_du = SURF_SCALE * (2.0 * u + CROSS_TERM * v)
    out = np.zeros(np.broadcast(u, v).shape + (3,))
    out[..., 0] = 1.0
    out[..., 2] = dz_du
    return out


def r_v(u, v):
    """Partial derivative dr_dv."""
    dz_dv = SURF_SCALE * (CROSS_TERM * u + 2.0 * V2_TERM * v)
    out = np.zeros(np.broadcast(u, v).shape + (3,))
    out[..., 1] = 1.0
    out[..., 2] = dz_dv
    return out


def normalize(vec):
    vec = np.asarray(vec, dtype=float)
    nrm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / nrm


def patch_param(theta):
    """Ellipse in parameter space."""
    u = U0 + A_PATCH * np.cos(theta)
    v = V0 + B_PATCH * np.sin(theta)
    return u, v


def patch_levelset_grad(theta):
    """
    unscaled outward normal to the ellipse in parameter space
    """
    gu = np.cos(theta) / A_PATCH
    gv = np.sin(theta) / B_PATCH
    g = np.array([gu, gv], dtype=float)
    return g / np.linalg.norm(g)


def boundary_tangent_param(theta):
    """Derivative of the ellipse with respect to theta."""
    du = -A_PATCH * np.sin(theta)
    dv =  B_PATCH * np.cos(theta)
    return np.array([du, dv], dtype=float)


#
###############
# M and P data
###############
u = np.linspace(U_MIN, U_MAX, NU)
v = np.linspace(V_MIN, V_MAX, NV)
U, V = np.meshgrid(u, v, indexing="xy")
R = surface(U, V)
X, Y, Z = R[..., 0], R[..., 1], R[..., 2]
# Mmask = (X/1.1)**2+(Y/.95)**2 <  1.8**2
Mmask = (X/1.)**2+(Y/1.)**2 <  1.8**2
X = np.where(Mmask, X, np.nan)
Y = np.where(Mmask, Y, np.nan)
Z = np.where(Mmask, Z, np.nan)

# Patch
up = np.linspace(U0 - A_PATCH, U0 + A_PATCH, N_PATCH)
vp = np.linspace(V0 - B_PATCH, V0 + B_PATCH, N_PATCH)
UP, VP = np.meshgrid(up, vp, indexing="xy")
mask = ((UP - U0) / A_PATCH) ** 2 + ((VP - V0) / B_PATCH) ** 2 <= 1.0
RP = surface(UP, VP)
XP, YP, ZP = RP[..., 0], RP[..., 1], RP[..., 2]
XPm = np.where(mask, XP, np.nan)
YPm = np.where(mask, YP, np.nan)
ZPm = np.where(mask, ZP, np.nan)

# Boundary curve
theta = np.linspace(0.0, 2.0 * np.pi, 600)
ub, vb = patch_param(theta)
Rb = surface(ub, vb)
Xb, Yb, Zb = Rb[:, 0], Rb[:, 1], Rb[:, 2]

# Boundary point
u1, v1 = patch_param(THETA0)
p = surface(u1, v1)

ru = r_u(u1, v1)
rv = r_v(u1, v1)
n_vec = normalize(np.cross(ru, rv))

# Outward conormal
g = patch_levelset_grad(THETA0)
u_guess = normalize(g[0] * ru + g[1] * rv)

# Boundary tangent
t_vec = normalize(np.cross(n_vec, u_guess))
u_vec = normalize(np.cross(t_vec, n_vec))

# A few points
pc = surface(U0, V0)
pm = surface(-1.8, -0.95)   # point to place M label
pb = surface(0.90 * A_PATCH, -0.55 * B_PATCH)  # point near boundary label


#
##########
# Plot
##########
fig = plt.figure(figsize=(7.6, 6.2))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)



##############
# Main surface
ax.plot_surface(
    X, Y, Z,
    # rstride=2, cstride=2,
    # rcount=50, ccount=50,
    color=COLOR_SURFACE,
    edgecolor="none",
    alpha=M_alpha,
    shade=True,
    zorder=-100
)
ax.text(pm[0], pm[1], pm[2] + 0.03, r"$M$", fontsize=18, zorder=20)

#
#
#

#########
# Interior P
ax.plot_surface(
    XPm, YPm, ZPm,
    rstride=1, cstride=1,
    color=COLOR_PATCH,
    edgecolor="none",
    alpha=P_alpha,
    shade=True,
    zorder=2
)
ax.text(pc[0] - 0.10, pc[1], pc[2] + 0.03, r"$P$", fontsize=18, zorder=20)

######################
# Boundary P
ax.plot(
    Xb, Yb, Zb,
    color=COLOR_BOUNDARY,
    linewidth=2.6,
    zorder=3
)
ax.text(pb[0] + 0.04, pb[1] - 0.02, pb[2] + 0.03, r"$\partial P$", fontsize=16, zorder=20)

#########################
# boundary point
ax.scatter([p[0]], [p[1]], [p[2]], color="black", s=20, zorder=10)

def draw_arrow(base, vec, length, color, lw=2.2):
    ax.quiver(
        base[0], base[1], base[2],
        vec[0], vec[1], vec[2],
        length=length,
        normalize=True,
        arrow_length_ratio=0.14,
        color=color,
        linewidth=lw,
        zorder=20
    )

def add_label(base, vec, length, text, offset=(0, 0, 0), color="black"):
    q = base + 1.10 * length * normalize(vec)
    q = q.reshape(3,) + np.array(offset)
    ax.text(q[0], q[1], q[2], text, color=color, fontsize=16, zorder=25)

# Normal n
draw_arrow(p, n_vec, N_LEN, COLOR_N)
add_label(p, n_vec, N_LEN, r"$\mathbf{n}$", offset=(0.02, 0.02, 0.02), color=COLOR_N)

# Conormal u
draw_arrow(p, u_vec, U_LEN, COLOR_U)
add_label(p, u_vec, U_LEN, r"$\mathbf{u}$", offset=(0.03, 0.00, 0.00), color=COLOR_U)

# Boundary tangent t
draw_arrow(p, t_vec, T_LEN, COLOR_T)
add_label(p, t_vec, T_LEN, r"$\mathbf{t}$", offset=(0.00, 0.00, 0.03), color=COLOR_T)



# Plot view and axis lims
ax.view_init(elev=ELEV, azim=AZIM)
ax.set_axis_off()

x_all = np.array([_ for _ in np.concatenate([X.ravel(), Xb.ravel()]) if not np.isnan(_)])
y_all = np.array([_ for _ in np.concatenate([Y.ravel(), Yb.ravel()]) if not np.isnan(_)])
z_all = np.array([_ for _ in np.concatenate([Z.ravel(), Zb.ravel()]) if not np.isnan(_)])

xmid = 0.5 * (x_all.min() + x_all.max())
ymid = 0.5 * (y_all.min() + y_all.max())
zmid = 0.5 * (z_all.min() + z_all.max())

xr = 0.5 * (x_all.max() - x_all.min())
yr = 0.5 * (y_all.max() - y_all.min())
zr = 0.5 * (z_all.max() - z_all.min())
Rbox = max(xr, yr, zr)

# ax.set_xlim(xmid - Rbox, xmid + Rbox)
# ax.set_ylim(ymid - Rbox, ymid + Rbox)
# ax.set_zlim(zmid - 0.72 * Rbox, zmid + 0.72 * Rbox)

ax.set_aspect("equal")
plt.tight_layout()



fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=-0.0)
plt.show()
plt.close(fig)
