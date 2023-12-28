# from numdiff import diff
import numpy as np
from meshbrane import meshbrane, sparsebrane
import sympy as sp
from alphashape import alphashape as ashp
from scipy.sparse import csr_matrix, csc_matrix

x, y, z = sp.symbols("x y z")
# implicit_expr = (x - 0) ** 2 + (y - 0) ** 2 + ((z - 0) / 2.0) ** 2 - 1.0
rsq = x**2 + y**2
# zsqr = z**2
# r = 1.2-sp.cos(sp.pi*z)
# r=1.25-cos(pi*z/3)
# zzz0 =
implicit_expr1 = rsq - (1.25 - sp.cos(sp.pi * z / 4)) * (9 - z**2) / 4
implicit_expr2 = 1  # x**2 + y**2 + (z - 2) ** 2 - 1.0
implicit_expr3 = 1  # x**2 + y**2 + (z + 2) ** 2 - 1.0
implicit_expr = implicit_expr1 * implicit_expr2 * implicit_expr3
implicit_vars = (x, y, z)
nx, ny, nz = 10j, 10j, 20j
xyz_grid = np.mgrid[-3:3:nx, -3:3:ny, -3:3:nz]
random_args = {"Npts": 100, "alpha": 1.0}
implicit_args = {
    "implicit_expr": implicit_expr,
    "implicit_vars": implicit_vars,
    "xyz_grid": xyz_grid,
}
# brane2 = meshbrane2(random_args=random_args, implicit_args=implicit_args)
sbrane = sparsebrane(init_type="implicit", implicit_args=implicit_args)
# brane1 = meshbrane(random_args=random_args, implicit_args=implicit_args)
#
cloud_args = {
    "points": sbrane.vertices,
    "alpha": 1 / 3,
}


# brane = sparsebrane(init_type="cloud", cloud_args=cloud_args)
figure_kwargs = {
    "show": True,
    "save": False,
    "plot_vertices": True,
    "plot_edges": True,
    "plot_faces": True,
    "plot_face_normals": False,
    "plot_vertex_normals": False,
}

sbrane.plot_mesh(**figure_kwargs)


# %%
F = sbrane.faces
E = sbrane.edges
V = sbrane.vertices


Aef_data = sbrane.Aef.data
Aef_indices = sbrane.Aef.indices
Aef_indptr = sbrane.Aef.indptr

Afe_data = sbrane.Afe.data
Afe_indices = sbrane.Afe.indices
Afe_indptr = sbrane.Afe.indptr

Aev_data = sbrane.Aev.data
Aev_indices = sbrane.Aev.indices
Aev_indptr = sbrane.Aev.indptr

Ave_data = sbrane.Ave.data
Ave_indices = sbrane.Ave.indices
Ave_indptr = sbrane.Ave.indptr

Nv = len(sbrane.vertices)
Ne = len(sbrane.edges)
Nf = len(sbrane.faces)

data_v = np.ones(Nv)
indices_v = np.array([v for v in range(Nv)])
indptr_v = np.array([v for v in range(Nv + 1)])
_ketv = csc_matrix((data_v, indices_v, indptr_v), shape=(Nv, Nv))
_brav = csr_matrix((data_v, indices_v, indptr_v), shape=(Nv, Nv))

data_e = np.ones(Ne)
indices_e = np.array([e for e in range(Ne)])
indptr_e = np.array([e for e in range(Ne + 1)])
_kete = csc_matrix((data_e, indices_e, indptr_e), shape=(Ne, Ne))
_brae = csr_matrix((data_e, indices_e, indptr_e), shape=(Ne, Ne))

data_f = np.ones(Nf)
indices_f = np.array([f for f in range(Nf)])
indptr_f = np.array([f for f in range(Nf + 1)])
_ketf = csc_matrix((data_f, indices_f, indptr_f), shape=(Nf, Nf))
_braf = csr_matrix((data_f, indices_f, indptr_f), shape=(Nf, Nf))


def ketv(v):
    return _ketv[:, v]


def brav(v):
    return _brav[v, :]


def kete(e):
    return _kete[:, e]


def brae(e):
    return _brae[e, :]


def ketf(f):
    return _ketf[:, f]


def braf(f):
    return _braf[f, :]


Bdry_ve = sbrane.Aev.T
coBdry_ev = sbrane.Aev

Bdry_ef = sbrane.Afe.T
coBdry_fe = sbrane.Afe

e = 13
ket_e = kete(e)

ket_Bdry_e = Bdry_ve @ ket_e
v0 = Bdry_ket_e.indices[np.where(Bdry_ket_e.data == -1)]
ket_f2collapse = coBdry_fe @ ket_e


edge = E[e]
cobdry_e = sbrane.Aef[e].toarray()
_cobdry_e = sbrane.Afe
bdry_e = sbrane.Ave[:, e].toarray()


# edges with v2pop in boundary
e2pop = Ave_indices[Ave_indptr[v2pop] : Ave_indptr[v2pop + 1]]
# faces with edge in boundary
# f2pop = Aef_indices[Aef_indptr[e] : Aef_indptr[e + 1]]


def star_v(v):
    e_indices = Ave_indices[Ave_indptr[v] : Ave_indptr[v + 1]]
    f_indices = {
        f for f in Aef_indices[Aef_indptr[e] : Aef_indptr[e + 1]] for e in e_indices
    }
    return e_indices, f_indices


# %%
# faces with edge in boundary
f2pop = Aef_indices[Aef_indptr[e] : Aef_indptr[e + 1]]
# vertices of faces2pop
v2pop = {v for f in f2pop for v in F[f]}  # if v != v2keep}
# edges of faces2pop
e2pop = {e for f in f2pop for e in Afe_indices[Afe_indptr[f] : Afe_indptr[f + 1]]}

# f2relabel = {f for v in v2pop for f in Avf_indices[Avf_indptr[v] : Avf_indptr[v + 1]]}


part_vertices = np.array([V[v] for v in v2pop])
sbrane.plot_part(part_vertices=part_vertices)
# %%

#
#

#

#

#


# %%

simfig_kwargs = {
    "show": False,
    "save": True,
    "plot_vertices": True,
    "plot_edges": True,
    "plot_faces": True,
    "plot_face_normals": False,
    "plot_vertex_normals": False,
}

brane.wiggly_sim(T=0.02, dt=0.01, figure_kwargs=simfig_kwargs)


# %% #################################################################
vertex_positions = brane.vertices
vertices = np.array([_ for _ in range(len(vertex_positions))], dtype=np.int32)
faces = brane.faces
edges = brane.edges


Aev = brane.Aev.toarray()
Ave = brane.Ave.toarray()

Afe = brane.Afe.toarray()
Aef = brane.Aef.toarray()

Afv = brane.Afv.toarray()
Avf = brane.Avf.toarray()


vf_data = brane.Avf.data
vf_indices = brane.Avf.indices
vf_indptr = brane.Avf.indptr

ef_data = brane.Aef.data
ef_indices = brane.Aef.indices
ef_indptr = brane.Aef.indptr

ve_data = brane.Ave.data
ve_indices = brane.Ave.indices
ve_indptr = brane.Ave.indptr

# for v in vertices:
v = 17
faces_of_v = vf_indices[vf_indptr[v] : vf_indptr[v + 1]]
edges_of_v = ve_indices[ve_indptr[v] : ve_indptr[v + 1]]
cell_edges = []

bd_f = 0 * brane.Afe[0]
for f in faces_of_v:
    bd_f += brane.Afe[f]


bd_cell = 0 * brane.Aev[0]
for e in bd_f.indices:
    bd_cell += brane.Aev[e]

bd_cell.data
# %% #################################################################
from meshbrane.simulation_functions import *
from meshbrane.model import *
from mayavi import mlab

vertices = brane.vertices
edges = brane.edges
Avf = brane.Avf
Ave = brane.Ave
v = 17

faces_of_vertices = get_y_of_x_csr(Avf)
edges_of_vertices = get_y_of_x_csr(Ave)

faces_of_v = faces_of_vertices[v]
edges_of_v = edges_of_vertices[v]

# _e = 2
# e = edges_of_v[2]
# edge = edges[e]
#

Nev = len(edges_of_v)
cell_vertex_indices = np.zeros(Nev, dtype=np.int32)

for _e in range(Nev):
    e = edges_of_v[_e]
    edge = edges[e]
    edge = edges[e]
    ev_sgn = Ave[v, e]
    if ev_sgn == 1:
        # v_cell = sum(edge) - v
        v_cell = edge[0]
    elif ev_sgn == -1:
        # v_cell = sum(edge) - v
        v_cell = edge[1]
    cell_vertex_indices[_e] = v_cell
cell_vertices = np.array([vertices[v] for v in cell_vertex_indices])
vertex = vertices[v]
# %%
# def plot_mesh_cell(self, show=True, save=False, fig_path=None):
self = brane
show = True
save = False
fig_path = None
vertices = self.vertices
faces = self.faces
face_centroids = self.face_centroids
face_normals = self.face_normals
vertex_normals = self.vertex_normals
face_color = self.face_color
edge_color = self.edge_color
vertex_color = self.vertex_color
normal_color = self.normal_color
figsize = (2180, 2180)
##########################################################
if show:
    mlab.options.offscreen = False
else:
    mlab.options.offscreen = True

cell_X, cell_Y, cell_Z = cell_vertices.T

vertex_X, vertex_Y, vertex_Z = vertices.T
face_X, face_Y, face_Z = face_centroids.T
face_nX, face_nY, face_nZ = face_normals.T
vertex_nX, vertex_nY, vertex_nZ = vertex_normals.T
# bgcolor = (1.0, 1.0, 1.0)
# fgcolor = (0.0, 0.0, 0.0)
figsize = (2180, 2180)
title = f"Membrane mesh"
fig = mlab.figure(title, size=figsize)  # , bgcolor=bgcolor, fgcolor=fgcolor)
mem_edges = mlab.triangular_mesh(
    vertex_X,
    vertex_Y,
    vertex_Z,
    faces,
    # opacity=0.4,
    color=edge_color,
    # representation="wireframe"
    representation="mesh",
    # representation="surface"
    # representation="fancymesh",
    tube_radius=0.03,
)
mem_faces = mlab.triangular_mesh(
    vertex_X,
    vertex_Y,
    vertex_Z,
    faces,
    opacity=0.4,
    color=face_color,
    # representation="wireframe"
    # representation="mesh",
    representation="surface"
    # representation="fancymesh",
    # tube_radius=None
)
mem_vertices = mlab.points3d(
    *vertices.T, mode="sphere", scale_factor=0.1, color=vertex_color
)

cell_vertices = mlab.points3d(
    *cell_vertices.T, mode="sphere", scale_factor=0.1, color=normal_color
)

# f_normals = mlab.quiver3d(
#     face_X,
#     face_Y,
#     face_Z,
#     face_nX,
#     face_nY,
#     face_nZ,
# )
# v_normals = mlab.quiver3d(
#     vertex_X,
#     vertex_Y,
#     vertex_Z,
#     vertex_nX,
#     vertex_nY,
#     vertex_nZ,
#     color=normal_color,
# )

if show:
    mlab.show()
if save:
    mlab.savefig(fig_path, figure=fig, size=figsize)
mlab.close(all=True)

# mlab.close(all=True)
