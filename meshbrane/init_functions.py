import numpy as np
from numdiff import jitcross, fib_sphere
from scipy.sparse import csr_matrix
from alphashape import alphashape as ashp
import sympy as sp
from skimage.measure import marching_cubes
from numba import njit


######################################################
# Initialization functions
######################################################
def random_init(Npts=89, alpha=1.0):
    ###################################################
    big_sphere = fib_sphere(int(5 * Npts / 10))
    medium_sphere = fib_sphere(int(3 * Npts / 10))
    medium_sphere *= 0.513
    medium_sphere = np.array(
        [xyz - np.array([1.01, 0.0, 0.0]) for xyz in medium_sphere]
    )

    small_sphere = fib_sphere(int(2 * Npts / 10))
    small_sphere *= 0.2513
    small_sphere = np.array([xyz - np.array([1.51, 0.0, 0.0]) for xyz in small_sphere])

    point_cloud = np.array([*big_sphere, *medium_sphere, *small_sphere])

    alpha_shape = ashp(point_cloud, alpha)
    return alpha_shape


# @njit
def get_face_data(vertices, faces, surface_com):
    """
    computes what are hopefully outward pointing unit normal vectors
    and directed area vectors of the faces. Reorders vertices of each face to
    match unit normal direction.
    """
    # faces = faces_old
    Nfaces = len(faces)
    face_normals = np.zeros((Nfaces, 3))
    face_areas = np.zeros((Nfaces, 3))
    face_centroids = np.zeros((Nfaces, 3))
    for f in range(Nfaces):
        fv0, fv1, fv2 = faces[f]
        v0_xyz = vertices[fv0]
        v1_xyz = vertices[fv1]
        v2_xyz = vertices[fv2]

        # this is just (v1_xyz-v0_xyz) x (v2_xyz-v1_xyz)
        f_normal = (
            jitcross(v0_xyz, v1_xyz)
            + jitcross(v1_xyz, v2_xyz)
            + jitcross(v2_xyz, v0_xyz)
        )
        f_area = 0.5 * f_normal

        f_normal /= np.sqrt(f_normal @ f_normal)
        face_com = (v0_xyz + v1_xyz + v2_xyz) / 3.0
        face_centroids[f] = face_com
        n_dot_dr = f_normal @ (face_com - surface_com)
        if n_dot_dr > 0:
            face_normals[f, :] = f_normal
            face_areas[f, :] = f_area
        else:
            faces[f, :] = np.array([fv1, fv0, fv2])
            face_normals[f, :] = -f_normal
            face_areas[f, :] = -f_area
    return faces, face_centroids, face_normals, face_areas


# @njit
def get_edge_data(faces):
    """
    Computes vertex indices of edges, and edge indices of faces
    """
    edges_sets = []
    edges = []
    Nfaces = len(faces)
    edges_of_faces = np.empty((Nfaces, 3), dtype=np.int32)

    for nface in range(Nfaces):
        v0, v1, v2 = faces[nface]
        e01 = {v0, v1}
        e12 = {v1, v2}
        e20 = {v2, v0}

        for nedge, edge in enumerate((e01, e12, e20)):
            add_edge = not edge in edges_sets
            if add_edge:
                edges_sets.append(edge)
                edges.append(list(edge))
                edges_of_faces[nface, nedge] = len(edges) - 1
            else:
                edges_of_faces[nface, nedge] = edges_sets.index(edge)

    return np.array(edges), edges_of_faces


# @njit
def get_ev_adjacency_csr(vertices, edges):
    """
    A.data -- nonzero matrix elements
    A.indices -- column indices (vertex indices) for nonzero matrix elements
    A.indptr --- pointers for nonzero matrix elements and their column indices
    *** len(A.indptr) = #rows+1

    Nvertices = len(vertices)
    Nedges = len(edges)
    A0 = np.zeros((Nedges, Nvertices), dtype=np.int32)
    for e in range(Nedges):
        v0, v1 = edges[e]
        A0[e, v0] = -1
        A0[e, v1] = 1
    return A0
    """
    Nedges = len(edges)
    Nvertices = len(vertices)
    indices = edges.ravel()
    data = np.ravel(Nedges * [[-1, 1]])
    indptr = [2 * _ for _ in range(Nedges + 1)]
    A0 = csr_matrix((data, indices, indptr), shape=(Nedges, Nvertices))

    return A0


# @njit
def get_fe_adjacency_dense(edges, faces, edges_of_faces):
    # Nvertices = len(vertices)
    Nedges = len(edges)
    Nfaces = len(faces)
    A1 = np.zeros((Nfaces, Nedges), dtype=np.int32)  # most
    for f in range(Nfaces):
        e0, e1, e2 = edges_of_faces[f]
        v0, v1, v2 = faces[f]
        pos = [[v0, v1], [v1, v2], [v2, v0]]
        neg = [[v1, v0], [v2, v1], [v0, v2]]

        for e in edges_of_faces[f]:
            ev = list(edges[e])
            if ev in pos:
                # index = pos.index(ev)
                A1[f, e] = 1
            elif ev in neg:
                # index = neg.index(ev)
                A1[f, e] = -1
            else:
                raise ValueError
            # if ev in pos:
            #     index = pos.index(ev)
            #     A1[f, e] = 1
            # elif ev in neg:
            #     index = neg.index(ev)
            #     A1[f, e] = -1
            # else:
            #     raise ValueError
            # pos.pop(index)
            # neg.pop(index)
    return A1


# @njit
def get_fe_adjacency_csr_data(edges, faces, edges_of_faces):
    # Nvertices = len(vertices)
    Nfaces = len(faces)
    data = np.zeros(3 * Nfaces, dtype=np.int32)
    indices = edges_of_faces.ravel()  # np.zeros(3 * Nfaces, dtype=np.int32)
    indptr = np.zeros(Nfaces + 1, dtype=np.int32)
    # indptr = [3 * _ for _ in range(Nedges + 1)]
    for f in range(Nfaces):
        indptr[f + 1] = 3 * (f + 1)
        e0, e1, e2 = edges_of_faces[f]
        v0, v1, v2 = faces[f]
        pos = [[v0, v1], [v1, v2], [v2, v0]]
        neg = [[v1, v0], [v2, v1], [v0, v2]]

        # for e in edges_of_faces[f]:
        for n_e in range(3):
            e = edges_of_faces[f, n_e]
            ev = list(edges[e])
            if ev in pos:
                data[3 * f + n_e] = 1
            elif ev in neg:
                data[3 * f + n_e] = -1
            else:
                raise ValueError
    return data, indices, indptr


def get_fe_adjacency_csr(edges, faces, edges_of_faces):
    data, indices, indptr = get_fe_adjacency_csr_data(edges, faces, edges_of_faces)
    Nfaces, Nedges = len(faces), len(edges)
    A1 = csr_matrix((data, indices, indptr), shape=(Nfaces, Nedges))
    return A1


def get_fv_adjacency_csr(vertices, faces):
    Nfaces, Nvertices = len(faces), len(vertices)
    indices = faces.ravel()
    indptr = [3 * f for f in range(Nfaces + 1)]
    data = np.ones(3 * Nfaces)

    Afv = csr_matrix((data, indices, indptr), shape=(Nfaces, Nvertices))
    return Afv


def get_faces_of_vertices(vertices, faces):
    Nfaces = len(faces)
    Nvertices = len(vertices)
    faces_of_vertices = []
    for v in range(Nvertices):
        faces_of_vertices.append([])
        for f in range(Nfaces):
            if v in faces[f]:
                faces_of_vertices[-1].append(f)
    return faces_of_vertices


def get_area_weighted_vertex_normals(vertices, faces_of_vertices, face_areas):
    """
    computes unit normal vectors at vertices.
    """
    Nvertices = len(vertices)
    vertex_normals = np.zeros((Nvertices, 3))
    for v in range(Nvertices):
        # Nfaces_of_v = len(faces_of_vertices[v])
        # Nfaces_of_v = max([1,len(faces_of_vertices[v])])
        for f in faces_of_vertices[v]:
            vertex_normals[v] += face_areas[f]
        normal_norm = np.sqrt(vertex_normals[v] @ vertex_normals[v])
        if normal_norm > 0:
            vertex_normals[v] /= normal_norm
    return vertex_normals


######################################################
######################################################


def make_surf_mesh(mesh_fun, xyz_grid):
    """
    Makes triangle mesh from implicit function
    """
    x_grid, y_grid, z_grid = xyz_grid
    dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
    dy = y_grid[0, 1, 0] - y_grid[0, 0, 0]
    dz = z_grid[0, 0, 1] - z_grid[0, 0, 0]
    vol = mesh_fun(*xyz_grid)
    iso_val = 0.0
    vertex_positions, faces, vertex_normals, iso_vals = marching_cubes(
        vol, iso_val, spacing=(dx, dy, dz)
    )
    X, Y, Z = (
        vertex_positions[:, 0] + x_grid[0, 0, 0],
        vertex_positions[:, 1] + y_grid[0, 0, 0],
        vertex_positions[:, 2] + z_grid[0, 0, 0],
    )

    return X, Y, Z, vertex_positions, faces, vertex_normals, iso_vals


def implicit_init(implicit_expr, implicit_vars, xyz_grid=None):
    if xyz_grid is None:
        xyz_grid = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]
    xyz = sp.Array([*implicit_vars])
    mesh_fun = njit(sp.lambdify(xyz, implicit_expr))
    # mesh_fun = sp.lambdify(xyz, implicit_expr)
    X, Y, Z, vertex_positions, faces, vertex_normals, iso_vals = make_surf_mesh(
        mesh_fun, xyz_grid=xyz_grid
    )
    return vertex_positions, faces


# %%
