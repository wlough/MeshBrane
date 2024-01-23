import numpy as np
import sympy as sp
from skimage.measure import marching_cubes
from src.numdiff import jitcross
from plyfile import PlyData, PlyElement


def make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz):
    """
    xyz_minmax = [-3.0, 3.0, -3.0, 3.0, -3.0, 3.0]
    Nxyz = [60j, 60j, 60j]
    implicit_fun_str = (
        "1.0*(y**2 + z**2 + (x - 8)**2 - 1)*(y**2 + z**2 + (x + 8)**2 - 1) - 4200.0"
    )"""
    xyz = sp.Array(sp.symbols("x y z"))
    implicit_fun_sym = sp.sympify(implicit_fun_str)
    implicit_fun = sp.lambdify(xyz, implicit_fun_sym)

    x0, x1, y0, y1, z0, z1 = xyz_minmax
    Nx, Ny, Nz = Nxyz
    xyz_grid = np.mgrid[x0:x1:Nx, y0:y1:Ny, z0:z1:Nz]
    x, y, z = xyz_grid

    dx = x[1, 0, 0] - x[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    dz = z[0, 0, 1] - z[0, 0, 0]
    vol = implicit_fun(x, y, z)

    iso_val = 0.0
    verts, faces, normals, values = marching_cubes(vol, iso_val, spacing=(dx, dy, dz))

    verts[:, 0] += x[0, 0, 0]
    verts[:, 1] += y[0, 0, 0]
    verts[:, 2] += z[0, 0, 0]
    normals = -normals.astype(np.float64)
    return verts, faces, normals


def make_sample_mesh(surface_name, Nxyz=None, xyz_minmax=None):
    """ """
    surface_names = [
        "dumbbell",
        "dumbbell2",
        "torus",
        "double_torus",
        "triple_torus",
        "neovius",
        "sphere",
        "oblate",
        "prolate",
        "transverse_tori",
        "pyramid3",
        "pyramid4",
    ]
    if Nxyz is None:
        Nxyz = [30j, 30j, 30j]
    if xyz_minmax is None:
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    if surface_name not in surface_names:
        msg = f"{surface_name} undefined.\n"
        msg += "surface_name must be one of: "
        for _ in surface_names:
            msg += f"{_}, "
        raise ValueError(msg)
    elif surface_name == "dumbbell":
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [20j, 20j, 20j]
        implicit_fun_str = "9*x**2 + 9*y**2 - 9*(z**2 - 1)*(cos(3*pi*z/4) - 1.25)/4"
    elif surface_name == "dumbbell2":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [30j, 30j, 30j]
        implicit_fun_str = "(144*y**2 + 144*z**2 + (12*x - 8)**2 - 1)*(144*y**2 + 144*z**2 + (12*x + 8)**2 - 1) - 4200"
    elif surface_name == "torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [20j, 20j, 20j]
        R = 0.7  # big radius
        r = 0.7 / 3.0  # small radius
        implicit_fun_str = (
            f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (x**2 + y**2)"
        )
    elif surface_name == "double_torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [30j, 30j, 30j]
        # implicit_fun_str = "z**2 + (x**2*(x - 1)*(x + 1) + y**2)**2 - 0.01"
        implicit_fun_str = (
            "(z/0.2)**2 + (x**2*(x - 0.7)*(x + 0.7)/0.05 + y**2/0.05)**2 - 1"
        )
    elif surface_name == "triple_torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [40j, 40j, 40j]
        # implicit_fun_str = "0.16*z**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
        implicit_fun_str = "1.69*z**2*(1 - 0.769*cos(pi*Abs(y**2 + (x + 0.2)**2)/4))**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
    elif surface_name == "neovius":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        implicit_fun_str = "4*cos(3*x + 3)*cos(3*y + 3)*cos(3*z + 3) + 3*cos(3*x + 3) + 3*cos(3*y + 3) + 3*cos(3*z + 3)"
    elif surface_name == "sphere":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        R = 0.9
        implicit_fun_str = f"x**2+y**2+z**2-{R**2}"
    elif surface_name == "oblate":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        Rxy = 0.9
        Rz = 0.6
        implicit_fun_str = f"(x/{Rxy})**2+(y/{Rxy})**2+(z/{Rz})**2-1"
    elif surface_name == "prolate":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        Rxy = 0.6
        Rz = 0.9
        implicit_fun_str = f"(x/{Rxy})**2+(y/{Rxy})**2+(z/{Rz})**2-1"
    elif surface_name == "transverse_tori":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # # Nxyz = [60j, 60j, 60j]
        # Nxyz = [20j, 20j, 20j]
        # R = 0.7  # big radius
        # r = 0.7 / 3.0  # small radius
        # b = (0.5 * r) ** 2
        # xyz_minmax = [-1.3, 1.3, -1.3, 1.3, -1.3, 1.3]
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [30j, 30j, 30j]
        R = 1.0  # big radius
        r = 0.2  # small radius
        b = 0.01
        # F1 = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (x**2 + y**2)"
        # F2 = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (y**2 + z**2)"
        # F3 = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (z**2 + x**2)"
        F1 = f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((x/.7)**2 + (y/.7)**2)"
        F2 = f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((y/.7)**2 + (z/.7)**2)"
        F3 = f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((z/.7)**2 + (x/.7)**2)"
        implicit_fun_str = f"({F1})*({F2})*({F3})-{b}"
    elif surface_name == "pyramid3":
        implicit_fun_str = None
        verts = np.zeros((4, 3))
        faces = np.zeros((4, 3), dtype=np.int32)
        xyz_top = np.array([0.0, 0.0, 1.0])
        v_top = 0
        xyz_base1 = np.array([1.0, 0.0, -1.0])
        v_base1 = 1
        xyz_base2 = np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), -1.0])
        v_base2 = 2
        xyz_base3 = np.array([np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3), -1.0])
        v_base3 = 3
        f_base = np.array([v_base3, v_base2, v_base1])
        f_side1 = np.array([v_base1, v_base2, v_top])
        f_side2 = np.array([v_base2, v_base3, v_top])
        f_side3 = np.array([v_base3, v_base1, v_top])
        verts[v_top] = xyz_top
        verts[v_base1] = xyz_base1
        verts[v_base2] = xyz_base2
        verts[v_base3] = xyz_base3
        faces = np.array([f_base, f_side1, f_side2, f_side3], dtype=np.int32)
        verts *= 0.9
    elif surface_name == "pyramid4":
        implicit_fun_str = None
        verts = np.zeros((5, 3))
        faces = np.zeros((5, 3), dtype=np.int32)
        xyz_top = np.array([0.0, 0.0, 1.0])
        v_top = 0
        xyz_base1 = np.array([1.0, 0.0, -1.0])
        v_base1 = 1
        xyz_base2 = np.array([np.cos(2 * np.pi / 4), np.sin(2 * np.pi / 4), -1.0])
        v_base2 = 2
        xyz_base3 = np.array([np.cos(4 * np.pi / 4), np.sin(4 * np.pi / 4), -1.0])
        v_base3 = 3
        xyz_base4 = np.array([np.cos(6 * np.pi / 4), np.sin(6 * np.pi / 4), -1.0])
        v_base4 = 4

        f_base1 = np.array([v_base4, v_base3, v_base2])
        f_base2 = np.array([v_base2, v_base1, v_base4])
        f_side1 = np.array([v_base1, v_base2, v_top])
        f_side2 = np.array([v_base2, v_base3, v_top])
        f_side3 = np.array([v_base3, v_base4, v_top])
        f_side4 = np.array([v_base4, v_base1, v_top])
        verts[v_top] = xyz_top
        verts[v_base1] = xyz_base1
        verts[v_base2] = xyz_base2
        verts[v_base3] = xyz_base3
        verts[v_base4] = xyz_base4
        faces = np.array(
            [f_base1, f_base2, f_side1, f_side2, f_side3, f_side4], dtype=np.int32
        )
        verts *= 0.9
    if implicit_fun_str is None:
        pass
    else:
        verts, faces, normals = make_implicit_surface_mesh(
            implicit_fun_str, xyz_minmax, Nxyz
        )
    # normal_norms = np.linalg.norm(normals, axis=1)
    # normals = np.array([n / np.linalg.norm(n) for n in normals])

    # surf_dict = {"vertices": verts, "faces": faces, "normals": normals}
    # with open(f"./scratch/{surface_name}_dict.pickle", "wb") as _f:
    #     dill.dump(surf_dict, _f, recurse=True)

    return verts, faces


def save_sample_mesh(surface_name, file_path=None, Nxyz=None, xyz_minmax=None):
    if file_path is None:
        file_path = f"./data/ply_files/{surface_name}.ply"
    vertices, faces = make_sample_mesh(surface_name, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
    save_mesh_to_ply(vertices, faces, file_path)

    # surf_dict = {"vertices": verts, "faces": faces, "normals": normals}
    # with open(f"./scratch/{surface_name}_dict.pickle", "wb") as _f:
    #     dill.dump(surf_dict, _f, recurse=True)


def save_all_sample_meshes():
    surface_names = [
        "dumbbell",
        "dumbbell2",
        "torus",
        "double_torus",
        "triple_torus",
        "neovius",
        "sphere",
        "oblate",
        "prolate",
        "transverse_tori",
        "pyramid3",
        "pyramid4",
    ]
    print("saving\n")
    for _ in surface_names:
        surface_name = _
        Nxyz = [30j, 30j, 30j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        file_path = f"./data/ply_files/{surface_name}.ply"
        print(file_path + 10 * " ", end="\r")
        save_sample_mesh(
            surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax
        )
    for _ in surface_names:
        surface_name = _
        Nxyz = [60j, 60j, 60j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        file_path = f"./data/ply_files/{surface_name}_fine.ply"
        print(file_path + 10 * " ", end="\r")
        save_sample_mesh(
            surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax
        )
    for _ in surface_names:
        surface_name = _
        Nxyz = [20j, 20j, 20j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        print(file_path + 10 * " ", end="\r")
        file_path = f"./data/ply_files/{surface_name}_coarse.ply"
        save_sample_mesh(
            surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax
        )
    print("\ndone")


def make_trisurface_patch(Nfaces=5):
    N = 1 * Nfaces
    dr = 0.25
    dz = 0.1
    theta = np.array([2 * np.pi * _ / N for _ in range(N)])
    r = np.random.rand(N)
    r *= dr * np.max(r)
    r += 1 - dr
    z = dz * np.cos(theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    vertices = np.array([x, y, z]).T
    vertices = np.array([[0.0, 0.0, dz], *vertices])
    faces = [[0, i, i + 1] for i in range(1, N)]
    faces = np.array([*faces, [0, N, 1]], dtype=np.int32)
    return vertices, faces


def make_double_trisurface_patch(Nfaces=5):
    N = 1 * Nfaces
    dr = 0.25
    dz = 0.1
    theta = np.array([2 * np.pi * _ / N for _ in range(N)])
    r = np.random.rand(N)
    r *= dr * np.max(r)
    r += 1 - dr
    z = dz * np.cos(theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    vertices = np.array([x, y, z]).T
    vertices = np.array([[0.0, 0.0, dz], *vertices])
    faces = [[0, i, i + 1] for i in range(1, N)]
    faces = np.array([*faces, [0, N, 1]], dtype=np.int32)
    return vertices, faces


def make_quadsurface_patch(Nfaces=3):
    N = 2 * Nfaces
    dr = 0.25
    dz = 0.1
    theta = np.array([2 * np.pi * _ / N for _ in range(N)])
    r = np.random.rand(N)
    r *= dr * np.max(r)
    r += 1 - dr
    z = dz * np.cos(theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    vertices = np.array([x, y, z]).T
    vertices = np.array([[0.0, 0.0, dz], *vertices])
    faces = [[0, i, i + 1, i + 2] for i in range(1, N - 2, 2)]
    faces = np.array([*faces, [0, N - 1, N, 1]], dtype=np.int32)
    return vertices, faces


def save_mesh_to_ply(vertices, faces, file_path):
    # Create the vertex data
    vertex_data = np.array(
        [tuple(v) for v in vertices], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")]
    )

    # Create the face data
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces

    # Create the PlyElements
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    face_element = PlyElement.describe(face_data, "face")

    # Write to a .ply file
    PlyData([vertex_element, face_element], text=True).write(file_path)


def load_mesh_from_ply(file_path):
    # Read the ply file
    plydata = PlyData.read(file_path)

    # Extract the vertex and face data
    vertices = np.vstack(
        [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]
    ).T
    faces = np.vstack(plydata["face"]["vertex_indices"])

    if not isinstance(vertices[0], np.float64):
        # faces = [tuple(f) for f in faces]
        vertices = vertices.astype(np.float64)
    if not isinstance(faces[0], np.int32):
        # faces = [tuple(f) for f in faces]
        faces = faces.astype(np.int32)

    return vertices, faces


##########################
def check_face_orientation(vertices, faces):
    return vertices, faces


def get_face_data(vertices, faces, surface_com=None):
    """
    computes what are hopefully outward pointing unit normal vectors
    and directed area vectors of the faces. Reorders vertices of each face to
    match unit normal direction.
    """
    # faces = faces_old

    Nfaces = len(faces)

    if surface_com is None:
        Nvertices = len(vertices)
        surface_com = np.zeros(3)
        for v in vertices:
            surface_com += v
        surface_com /= Nvertices
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


# def get_boundary_ops_csr(vertices, faces):
#     Nvertices = len(vertices)
#     (
#         Afe_data,
#         Afe_indices,
#         Aev_indices,
#         edges,
#     ) = get_boundary_ops_csr_data(vertices, faces)
#     # edges = np.array(edges, dtype=np.int32)
#     # Afe ###############
#     Nfaces = len(faces)
#     # Afe_data = np.array(Afe_data_list, dtype=np.int32)  # [-1,1,...]
#     # Afe_indices = np.array(Afe_indices_list, dtype=np.int32)  #
#     Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)
#
#     # Aev ###############
#     Nedges = len(edges)
#     Aev_data = np.array(Nedges * [-1, 1], dtype=np.int32)  # [-1,1,...]
#     # Aev_indices = np.array(Aev_indices_list, dtype=np.int32)  # vertex indices
#     Aev_indptr = np.array(
#         [2 * _ for _ in range(Nedges + 1)], dtype=np.int32
#     )  # [0,2,4,...]
#
#     Afe = csr_matrix((Afe_data, Afe_indices, Afe_indptr), shape=(Nfaces, Nedges))
#     Aev = csr_matrix((Aev_data, Aev_indices, Aev_indptr), shape=(Nedges, Nvertices))
#
#     return Afe, Aev, edges


def get_boundary_ops_csr_data(vertices, faces):
    """
    Computes edges and boundary operators from vertices and faces
    """
    # Nvertices = len(vertices)
    Nfaces = len(faces)
    edges_list = []

    # Afe ###############
    Afe_data_list = []  # [-1,1,...]
    Afe_indices_list = []  #
    # Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

    # Aev ###############
    # Aev_data_list = []  # [-1,1,...]
    Aev_indices_list = []  # vertex indices
    # Aev_indptr = []  # [0,2,4,...]

    for f in range(Nfaces):
        face = faces[f]
        for _v in range(3):
            vm = face[_v]
            vp = face[np.mod(_v + 1, 3)]
            edge_p = [vm, vp]
            edge_m = [vp, vm]
            try:  # is negative edge already in edges?
                e = edges_list.index(edge_m)
                fe_orientation = -1
            except Exception:
                try:  # is positive edge already in edges?
                    e = edges_list.index(edge_p)
                    fe_orientation = 1
                except Exception:  # if neither, add positive edge to edges
                    edges_list.append(edge_p)
                    e = len(edges_list) - 1
                    fe_orientation = 1
                    Aev_indices_list.append(vm)
                    Aev_indices_list.append(vp)

            Afe_indices_list.append(e)
            Afe_data_list.append(fe_orientation)

    Afe_data = np.array(Afe_data_list, dtype=np.int32)
    Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
    Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
    edges = np.array(edges_list, dtype=np.int32)
    return Afe_data, Afe_indices, Aev_indices, edges
