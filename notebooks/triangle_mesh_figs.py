import sys
from pathlib import Path

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))

import matplotlib.pyplot as plt
import numpy as np

figsize_pts = np.array([235, 118])
figsize = figsize_pts / 72
fontzise = 8
dpi = 600

# 0: center center, 1...6: ccw from top
phi = np.array([2 * np.pi * _ / 6 + np.pi / 2 for _ in range(6)])
verts = np.array([[0.0, 0.0], *np.round([[np.cos(p), np.sin(p)] for p in phi], 1)])

# Initial faces
tri_before = [
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 5),
    (0, 5, 6),
    (0, 6, 1),
]

# Specify the diagonal to flip: (0,2) -> (1,3)
old_edge = (0, 2)
new_edge = (1, 3)

# Build tri_after by replacing triangles that include old_edge
tri_after = []
for tri in tri_before:
    if old_edge[0] in tri and old_edge[1] in tri:
        # skip these two triangles
        continue
    tri_after.append(tri)

# Add the two new triangles from the flipped diagonal
tri_after += [
    (1, 2, 3),
    (1, 3, 0),
]


# get edges from triangle list
def extract_edges(tris):
    e = set()
    for a, b, c in tris:
        for u, v in [(a, b), (b, c), (c, a)]:
            e.add(tuple(sorted((u, v))))
    return list(e)


edges_before = extract_edges(tri_before)
edges_after = extract_edges(tri_after)

# Plot before and after
fig, axes = plt.subplots(1, 2, figsize=figsize)
titles = ["Before Flip", "After Flip"]
edge_sets = [edges_before, edges_after]
highlight_edges = [old_edge, new_edge]

for ax, title, edges, highlight in zip(axes, titles, edge_sets, highlight_edges):
    ax.set_aspect("equal")
    # ax.set_title(title, fontsize=fontsize)
    ax.axis("off")

    # draw all edges
    for u, v in edges:
        x = [verts[u, 0], verts[v, 0]]
        y = [verts[u, 1], verts[v, 1]]
        lw = 2 if (u, v) == tuple(sorted(highlight)) else 1
        color = "red" if (u, v) == tuple(sorted(highlight)) else "black"
        ax.plot(x, y, color=color, lw=lw)

    # draw vertices
    ax.scatter(verts[:, 0], verts[:, 1], color="black", s=50, zorder=3)

plt.tight_layout()
plt.savefig("../output/edge_flip_plot.svg", dpi=dpi)
plt.show()
plt.close()

##################################################################
##################################################################
##################################################################
# %%
# Barycentric dual
import matplotlib.pyplot as plt
import numpy as np

figsize_pts = np.array([118, 118])
figsize = figsize_pts / 72
fontzise = 8
dpi = 600


# 0: center center, 1...6: ccw from top
# phi = np.array([2 * np.pi * _ / 6 for _ in range(6)])
phi = np.array([2 * np.pi * _ / 6 + (0.1 * np.pi * np.random.rand()) for _ in range(6)])
verts = np.array([[0.0, 0.0], *[[np.cos(p), np.sin(p)] for p in phi]])


tris = [
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 5),
    (0, 5, 6),
    (0, 6, 1),
]


def extract_dual_bdry(tris):
    bdry = []
    for a, b, c in tris:
        x0 = (verts[a] + verts[b]) / 2
        x1 = (verts[a] + verts[b] + verts[c]) / 3
        # x2 = (verts[a] + verts[c]) / 2
        bdry.append(x0)
        bdry.append(x1)
    return np.array(bdry)


# get edges from triangle list
def extract_edges(tris):
    e = set()
    for a, b, c in tris:
        for u, v in [(a, b), (b, c), (c, a)]:
            e.add(tuple(sorted((u, v))))
    return list(e)


edges = extract_edges(tris)
dual_bdry = extract_dual_bdry(tris)

# Plot before and after
fig, ax = plt.subplots(1, 1, figsize=figsize)

ax.set_aspect("equal")
# ax.set_title(title, fontsize=fontsize)
ax.axis("off")

# draw all edges
for u, v in edges:
    x = [verts[u, 0], verts[v, 0]]
    y = [verts[u, 1], verts[v, 1]]
    ax.plot(x, y, color="black", lw=1.5)

# draw vertices
ax.scatter(verts[:, 0], verts[:, 1], color="black", s=50, zorder=3)


# ax.plot(dual_bdry[:, 0], dual_bdry[:, 1], color="red", zorder=3)
ax.fill(dual_bdry[:, 0], dual_bdry[:, 1], facecolor="skyblue", edgecolor="blue", lw=1.5)


plt.tight_layout()
plt.savefig("../output/barycentric_dual_area_plot.svg", dpi=dpi)
plt.show()
plt.close()


##################################################################
##################################################################
##################################################################
# %%
# Circumcentric dual
import matplotlib.pyplot as plt
import numpy as np

# figsize_pts = np.array([118, 118])
# scale = 1
figsize_pts = np.array([235, 235])
scale = 2
figsize = figsize_pts / 72
fontzise = 8
dpi = 600


def get_circumcenter(x1, x2, x3):
    norm = np.linalg.norm
    dot = np.dot
    cross = np.cross

    xi, xj, xk = x1, x2, x3
    rij = xj - xi
    rik = xk - xi
    rjk = xk - xj
    s1 = 0.5 * norm(rjk) ** 2 * dot(rij, rik) / norm(cross(rij, rik)) ** 2

    xi, xj, xk = x2, x3, x1
    rij = xj - xi
    rik = xk - xi
    rjk = xk - xj
    s2 = 0.5 * norm(rjk) ** 2 * dot(rij, rik) / norm(cross(rij, rik)) ** 2

    xi, xj, xk = x3, x1, x2
    rij = xj - xi
    rik = xk - xi
    rjk = xk - xj
    s3 = 0.5 * norm(rjk) ** 2 * dot(rij, rik) / norm(cross(rij, rik)) ** 2

    return s1 * x1 + s2 * x2 + s3 * x3


rand = lambda m: m * (np.random.rand() - 0.5)


# 0: center center, 1...6: ccw from top
phi = np.array([2 * np.pi * _ / 6 for _ in range(6)])
verts = np.array(
    [[0.0, 0.0, 0.0], *[np.array([np.cos(p), np.sin(p), 0.0]) for p in phi]]
)
# phi = np.array([2 * np.pi * _ / 6 + rand(0.1 * np.pi) for _ in range(6)])
# verts = np.array([[0.0, 0.0], *[np.array([np.cos(p), np.sin(p)]) + rand(0.2) for p in phi]])


tris = [
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 5),
    (0, 5, 6),
    (0, 6, 1),
]


def extract_dual_bdry(tris):
    bdry = []
    for a, b, c in tris:
        x0 = (verts[a] + verts[b]) / 2
        x1 = get_circumcenter(verts[a], verts[b], verts[c])
        bdry.append(x0)
        bdry.append(x1)
    return np.array(bdry)


# get edges from triangle list
def extract_edges(tris):
    e = set()
    for a, b, c in tris:
        for u, v in [(a, b), (b, c), (c, a)]:
            e.add(tuple(sorted((u, v))))
    return list(e)


edges = extract_edges(tris)
dual_bdry = extract_dual_bdry(tris)

fig, ax = plt.subplots(1, 1, figsize=figsize)

ax.set_aspect("equal")
# ax.set_title(title, fontsize=fontsize)
ax.axis("off")

# draw all edges
for u, v in edges:
    x = [verts[u, 0], verts[v, 0]]
    y = [verts[u, 1], verts[v, 1]]
    ax.plot(x, y, color="black", lw=1.5)

# draw vertices
ax.scatter(verts[:, 0], verts[:, 1], color="black", s=50, zorder=3)


# ax.plot(dual_bdry[:, 0], dual_bdry[:, 1], color="red", zorder=3)
ax.fill(dual_bdry[:, 0], dual_bdry[:, 1], facecolor="skyblue", edgecolor="blue", lw=1.5)

plt.tight_layout()
plt.savefig("../output/circumcentric_dual_area_plot.svg", dpi=dpi)
plt.show()
plt.close()
