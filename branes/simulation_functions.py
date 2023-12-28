import numpy as np
from numba import njit
from numdiff import jitcross

# from alphashape import alphashape as alshp


######################################################
# Simulation functions
######################################################
@njit
def update_face_data(vertices, faces):
    """
    computes outward pointing unit normal vectors
    and directed area vectors of the faces.
    """
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
        face_normals[f, :] = f_normal
        face_areas[f, :] = f_area
    return face_centroids, face_normals, face_areas


@njit
def update_vertex_normals(vertices, Avf_indices, Avf_indptr, face_areas):
    """
    computes unit normal vectors at vertices.
    """
    Nvertices = len(vertices)

    vertex_normals = np.zeros((Nvertices, 3))
    for v in range(Nvertices):
        faces_of_v = Avf_indices[Avf_indptr[v] : Avf_indptr[v + 1]]
        for f in faces_of_v:
            vertex_normals[v] += face_areas[f]
        normal_norm = np.sqrt(vertex_normals[v] @ vertex_normals[v])
        if normal_norm > 0:
            vertex_normals[v] /= normal_norm
    return vertex_normals


def get_y_of_x_csr(Axy_csr):
    # Aef_csr = self.Aef_csr
    indices, indptr = Axy_csr.indices, Axy_csr.indptr
    Nx = len(indptr) - 1
    x_of_y = []
    for v in range(Nx):
        x_of_y.append(indices[indptr[v] : indptr[v + 1]])

    return x_of_y


######################################################
######################################################
