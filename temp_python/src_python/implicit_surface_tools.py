import sympy as sp
import numpy as np
from skimage.measure import marching_cubes
from plyfile import PlyData, PlyElement
from scipy.spatial import Delaunay


class ImplicitSurfaceBase:
    def __init__(self, name, implicit_fun_str, xyz_minmaxN):
        # self.name = "unit_sphere"
        # self.implicit_fun_str = "x**2 + y**2 + z**2 - 1"
        # self.xyz_minmax = [-1.25, 1.25, -1.25, 1.25, -1.25, 1.25]
        # self.Nxyz = [30j, 30j, 30j]
        self.name = name
        self.implicit_fun_str = implicit_fun_str
        self.xyz_minmax = xyz_minmaxN

        self.xyz = sp.Array(sp.symbols("x y z"))
        self.implicit_fun_sym = sp.sympify(self.implicit_fun_str)
        self.implicit_fun = sp.lambdify(self.xyz, self.implicit_fun_sym)

    def marching_cubes(self):
        """
        uses marching cubes to get vertex/face list from implicit function
        """
        # xyz = sp.Array(sp.symbols("x y z"))
        # implicit_fun_sym = sp.sympify(implicit_fun_str)
        # implicit_fun = sp.lambdify(xyz, implicit_fun_sym)

        (x0, x1, Nx), (y0, y1, Ny), (z0, z1, Nz) = self.xyz_minmaxN
        xyz_grid = np.mgrid[x0:x1:Nx, y0:y1:Ny, z0:z1:Nz]
        x, y, z = xyz_grid

        dx = x[1, 0, 0] - x[0, 0, 0]
        dy = y[0, 1, 0] - y[0, 0, 0]
        dz = z[0, 0, 1] - z[0, 0, 0]
        vol = self.implicit_fun(x, y, z)

        iso_val = 0.0
        verts, faces, normals, values = marching_cubes(vol, iso_val, spacing=(dx, dy, dz))

        verts[:, 0] += x[0, 0, 0]
        verts[:, 1] += y[0, 0, 0]
        verts[:, 2] += z[0, 0, 0]
        normals = -normals.astype(np.float64)
        return verts, faces, normals

    def find_points(self):
        """
        finds points on the implicit surface
        """
        X, Y, Z = (
            np.linspace(*self.xyz_minmaxN[0]),
            np.linspace(*self.xyz_minmaxN[1]),
            np.linspace(*self.xyz_minmaxN[2]),
        )
        V = []
        return V


def implicit_surface_fun(surf):
    return 0


def f_surf(x, y, z):
    # return (x**2 + y**2 + z**2 + rad_big**2 - rad_small**2) ** 2 - 4 * rad_big**2 * (x**2 + y**2)
    return (np.sqrt(x**2 + y**2) - 1) ** 2 + z**2 - 1 / 3**2


def Df_surf(x, y, z):
    # xyz = np.array([x, y, z])
    # return 4 * (x**2 + y**2 + z**2 + rad_big**2 - rad_small**2) * xyz - 8 * rad_big**2 * np.array([x, y, 0.0])
    return 2 * (np.sqrt(x**2 + y**2) - 1) * np.array([x, y, 0]) / np.sqrt(x**2 + y**2) + np.array([0, 0, 2 * z])


def closest_point(xyz, iterations=18):
    """
    input
    -----
    r = [...,[x[s],y[s],z[s]],...]

    phi*Dphi/normDphi**2 ~ dist_S(r)
    """
    for _ in range(iterations):
        f = f_surf(*xyz)
        Df = Df_surf(*xyz)
        xyz -= f * Df / (Df @ Df)
    return xyz


################################################################
# implicit function --> vertex/face lists --> .ply files
def make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz):
    """
    uses marching cubes to get vertex/face list from implicit function

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


def make_trisurface_patch(Nfaces=5):
    """makes a triangle mesh patch around a vertex"""
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


def make_sample_mesh(surface_name, Nxyz=None, xyz_minmax=None):
    """makes meshes for some cool surfaces"""
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
        implicit_fun_str = (
            "(144*y**2 + 144*z**2 + (12*x - 8)**2 - 1)*(144*y**2 + 144*z**2 + (12*x + 8)**2 - 1) - 4200"
        )
    elif surface_name == "torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [20j, 20j, 20j]
        R = 0.7  # big radius
        r = 0.7 / 3.0  # small radius
        implicit_fun_str = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (x**2 + y**2)"
    elif surface_name == "double_torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [30j, 30j, 30j]
        # implicit_fun_str = "z**2 + (x**2*(x - 1)*(x + 1) + y**2)**2 - 0.01"
        implicit_fun_str = "(z/0.2)**2 + (x**2*(x - 0.7)*(x + 0.7)/0.05 + y**2/0.05)**2 - 1"
    elif surface_name == "triple_torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [40j, 40j, 40j]
        # implicit_fun_str = "0.16*z**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
        implicit_fun_str = "1.69*z**2*(1 - 0.769*cos(pi*Abs(y**2 + (x + 0.2)**2)/4))**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
    elif surface_name == "neovius":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        implicit_fun_str = (
            "4*cos(3*x + 3)*cos(3*y + 3)*cos(3*z + 3) + 3*cos(3*x + 3) + 3*cos(3*y + 3) + 3*cos(3*z + 3)"
        )
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
        F1 = (
            f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((x/.7)**2 + (y/.7)**2)"
        )
        F2 = (
            f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((y/.7)**2 + (z/.7)**2)"
        )
        F3 = (
            f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((z/.7)**2 + (x/.7)**2)"
        )
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
        faces = np.array([f_base1, f_base2, f_side1, f_side2, f_side3, f_side4], dtype=np.int32)
        verts *= 0.9
    if implicit_fun_str is None:
        pass
    else:
        verts, faces, normals = make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz)
    # normal_norms = np.linalg.norm(normals, axis=1)
    # normals = np.array([n / np.linalg.norm(n) for n in normals])

    # surf_dict = {"vertices": verts, "faces": faces, "normals": normals}
    # with open(f"./scratch/{surface_name}_dict.pickle", "wb") as _f:
    #     dill.dump(surf_dict, _f, recurse=True)

    return verts, faces


def save_mesh_to_ply(vertices, faces, file_path):
    """saves vertex+face list to a .ply file"""
    # Create the vertex data
    vertex_data = np.array([tuple(v) for v in vertices], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])

    # Create the face data
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces

    # Create the PlyElements
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    face_element = PlyElement.describe(face_data, "face")

    # Write to a .ply file
    PlyData([vertex_element, face_element], text=True).write(file_path)


def load_mesh_from_ply(file_path):
    """loads vertex+face list from a .ply file"""
    # Read the ply file
    plydata = PlyData.read(file_path)

    # Extract the vertex and face data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    faces = np.vstack(plydata["face"]["vertex_indices"])

    if not isinstance(vertices[0], np.float64):
        # faces = [tuple(f) for f in faces]
        vertices = vertices.astype(np.float64)
    if not isinstance(faces[0], np.int32):
        # faces = [tuple(f) for f in faces]
        faces = faces.astype(np.int32)

    return vertices, faces


def save_sample_mesh_to_ply(surface_name, file_path=None, Nxyz=None, xyz_minmax=None):
    """saves one of the meshes made by make_sample_mesh() to a .ply file"""
    if file_path is None:
        file_path = f"./data/ply_files/{surface_name}.ply"
    vertices, faces = make_sample_mesh(surface_name, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
    save_mesh_to_ply(vertices, faces, file_path)


def save_all_sample_meshes():
    """saves all sample meshes generated by make_sample_mesh() to .ply files"""
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
        print(file_path + 15 * " ", end="\r")
        save_sample_mesh_to_ply(surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
        print(file_path + " -done" + 9 * " ", end="\n")
    for _ in surface_names:
        surface_name = _
        Nxyz = [60j, 60j, 60j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        file_path = f"./data/ply_files/{surface_name}_fine.ply"
        print(file_path + 15 * " ", end="\r")
        save_sample_mesh_to_ply(surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
        print(file_path + " -done" + 9 * " ", end="\n")
    for _ in surface_names:
        surface_name = _
        Nxyz = [20j, 20j, 20j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        print(file_path + 15 * " ", end="\r")
        file_path = f"./data/ply_files/{surface_name}_coarse.ply"
        save_sample_mesh_to_ply(surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
        print(file_path + " -done" + 9 * " ", end="\n")
    print("\ndone")


def save_halfedge_mesh_from_ply(ply_path, output_directory):
    """loads vertex/face lists from .ply file, builds halfedge mesh data, and
    saves as numpy arrays"""
    os.system(f"rm -r {output_directory}")
    os.system(f"mkdir {output_directory}")
    plydata = PlyData.read(ply_path)

    # Extract the vertex and face data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    faces = np.vstack(plydata["face"]["vertex_indices"])

    if not isinstance(vertices[0], np.float64):
        # faces = [tuple(f) for f in faces]
        vertices = vertices.astype(np.float64)
    if not isinstance(faces[0], np.int32):
        # faces = [tuple(f) for f in faces]
        faces = faces.astype(np.int32)

    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = get_combinatorial_mesh_data(vertices, faces)
    np.save(f"{output_directory}/vertices.npy", vertices)
    np.save(f"{output_directory}/V_hedge.npy", V_hedge)
    np.save(f"{output_directory}/halfedges.npy", halfedges)
    np.save(f"{output_directory}/H_vertex.npy", H_vertex)
    np.save(f"{output_directory}/H_face.npy", H_face)
    np.save(f"{output_directory}/H_next.npy", H_next)
    np.save(f"{output_directory}/H_prev.npy", H_prev)
    np.save(f"{output_directory}/H_twin.npy", H_twin)
    np.save(f"{output_directory}/faces.npy", faces)
    np.save(f"{output_directory}/F_hedge.npy", F_hedge)


def load_halfedge_mesh_data(mesh_directory):
    """loads halfedge mesh numpy arrays"""
    mesh_data = {}
    mesh_data["vertices"] = np.load(f"{mesh_directory}/vertices.npy")
    mesh_data["V_hedge"] = np.load(f"{mesh_directory}/V_hedge.npy")
    mesh_data["halfedges"] = np.load(f"{mesh_directory}/halfedges.npy")
    mesh_data["H_vertex"] = np.load(f"{mesh_directory}/H_vertex.npy")
    mesh_data["H_face"] = np.load(f"{mesh_directory}/H_face.npy")
    mesh_data["H_next"] = np.load(f"{mesh_directory}/H_next.npy")
    mesh_data["H_prev"] = np.load(f"{mesh_directory}/H_prev.npy")
    mesh_data["H_twin"] = np.load(f"{mesh_directory}/H_twin.npy")
    mesh_data["faces"] = np.load(f"{mesh_directory}/faces.npy")
    mesh_data["F_hedge"] = np.load(f"{mesh_directory}/F_hedge.npy")
    return mesh_data


def save_all_halfedge_meshes():
    """applies save_halfedge_mesh_from_ply() to selected sample meshes"""
    surface_names = [
        "sphere",
        "oblate",
        "prolate",
        "dumbbell",
        "torus",
        "double_torus",
        "triple_torus",
        "transverse_tori",
    ]
    print("saving...\n")
    for surface_name in surface_names:
        ply_path = f"./data/ply_files/{surface_name}_coarse.ply"
        mesh_directory = f"./data/halfedge_meshes/{surface_name}_coarse"
        print(mesh_directory + 15 * " ", end="\r")
        save_halfedge_mesh_from_ply(ply_path, mesh_directory)
        print(mesh_directory + " -done" + 9 * " ", end="\n")
    for surface_name in surface_names:
        ply_path = f"./data/ply_files/{surface_name}.ply"
        mesh_directory = f"./data/halfedge_meshes/{surface_name}"
        print(mesh_directory + 15 * " ", end="\r")
        save_halfedge_mesh_from_ply(ply_path, mesh_directory)
        print(mesh_directory + " -done" + 9 * " ", end="\n")
    for surface_name in surface_names:
        ply_path = f"./data/ply_files/{surface_name}_fine.ply"
        mesh_directory = f"./data/halfedge_meshes/{surface_name}_fine"
        print(mesh_directory + 15 * " ", end="\r")
        save_halfedge_mesh_from_ply(ply_path, mesh_directory)
        print(mesh_directory + " -done" + 9 * " ", end="\n")
    print("\ndone")


def regularize_halfedge_mesh_data(
    vertices,
    V_hedge,
    halfedges,
    H_vertex,
    H_face,
    H_next,
    H_prev,
    H_twin,
    faces,
    F_hedge,
):
    iters = 20
    weight = 0.1
    Nflips = -1
    for iter in range(iters):
        print(
            f" regularizing halfedge mesh data iteration {iter} of {iters-1}    ",
            end="\r",
        )

        (
            V_hedge,
            halfedges,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            faces,
            F_hedge,
            Nflips,
        ) = regularize_by_flips(
            V_hedge,
            halfedges,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            faces,
            F_hedge,
        )

        vertices = regularize_by_shifts(
            weight,
            vertices,
            V_hedge,
            H_vertex,
            H_prev,
            H_twin,
        )
    return (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    )


def sphere_oblate_torus_dumbbell_mesh_inputs(reg=True, coarse=False, fine=False, ultrafine=False):
    """Gets list of inputs required to make .ply files and halfedge mesh
    data for sphere, oblate spheroid, torus, and dumbbell

    sphere={
        "implicit_fun_str": "x**2+y**2+z**2-0.9",
        "Nxyz": [30j, 30j, 30j],
        "xyz_minmax": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        "ply_path": "./data/ply_files/sphere.ply",
        "mesh_directory": "./data/halfedge_meshes/sphere",
    }"""
    Nxyz = [30j, 30j, 30j]
    Nxyz_coarse = [20j, 20j, 20j]
    Nxyz_fine = [60j, 60j, 60j]
    Nxyz_ultrafine = [200j, 200j, 200j]
    xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    surfs_dict = {
        "sphere": "x**2+y**2+z**2-0.9",
        "oblate": "(x/0.9)**2+(y/0.9)**2+(z/0.6)**2-1",
        "torus": "(x**2 + y**2 + z**2 + 0.7**2 - 0.24**2) ** 2 - 4 * 0.7**2 * (x**2 + y**2)",
        "dumbbell": "9*x**2 + 9*y**2 - 9*(z**2 - 1)*(cos(3*pi*z/4) - 1.25)/4",
    }
    reg_list = []
    coarse_list = []
    fine_list = []
    ultrafine_list = []
    for surface_name, implicit_fun_str in surfs_dict.items():
        reg_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}",
                "Nxyz": Nxyz,
                "xyz_minmax": xyz_minmax,
            }
        )
        coarse_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}_coarse.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}_coarse",
                "Nxyz": Nxyz_coarse,
                "xyz_minmax": xyz_minmax,
            }
        )
        fine_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}_fine.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}_fine",
                "Nxyz": Nxyz_fine,
                "xyz_minmax": xyz_minmax,
            }
        )
        ultrafine_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}_ultrafine.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}_ultrafine",
                "Nxyz": Nxyz_ultrafine,
                "xyz_minmax": xyz_minmax,
            }
        )

    surfaces = []
    if coarse:
        surfaces.extend(coarse_list)
    if reg:
        surfaces.extend(reg_list)
    if fine:
        surfaces.extend(fine_list)
    if ultrafine:
        surfaces.extend(ultrafine_list)
    return surfaces


def generate_regularize_save_mesh(implicit_fun_str, Nxyz, xyz_minmax, ply_path, mesh_directory):
    """Uses marching cubes to generate vertex/face list for the
    zero level set 'implicit_fun_str', builds halfedge mesh data, and regularizes
    mesh. Saves vertex/face list .ply file to 'ply_path' and saves
    halfedge mesh data as numpy arrays to 'mesh_directory'.
    """
    print(f"{implicit_fun_str}=0")
    print(f" ply_path: {ply_path}")
    print(f" mesh_directory: {mesh_directory}")
    os.system(f"rm -r {mesh_directory}")
    os.system(f"mkdir {mesh_directory}")
    print(" generating .ply file", end="\r")
    vertices_in, faces_in, normals = make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz)
    save_mesh_to_ply(vertices_in, faces_in, ply_path)
    print(" generating .ply file -done")
    print(" generating halfedge mesh data", end="\r")
    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = get_combinatorial_mesh_data(vertices_in, faces_in)
    print(" generating halfedge mesh data -done")
    print(" regularizing halfedge mesh data", end="\r")
    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = regularize_halfedge_mesh_data(
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    )
    print(" regularizing halfedge mesh data -done" + 15 * " ")
    print(" saving", end="\r")
    np.save(f"{mesh_directory}/vertices.npy", vertices)
    np.save(f"{mesh_directory}/V_hedge.npy", V_hedge)
    np.save(f"{mesh_directory}/halfedges.npy", halfedges)
    np.save(f"{mesh_directory}/H_vertex.npy", H_vertex)
    np.save(f"{mesh_directory}/H_face.npy", H_face)
    np.save(f"{mesh_directory}/H_next.npy", H_next)
    np.save(f"{mesh_directory}/H_prev.npy", H_prev)
    np.save(f"{mesh_directory}/H_twin.npy", H_twin)
    np.save(f"{mesh_directory}/faces.npy", faces)
    np.save(f"{mesh_directory}/F_hedge.npy", F_hedge)
    save_mesh_to_ply(vertices, faces, ply_path)

    print(" saving -done")


def generate_sphere_oblate_torus_dumbbell(reg=True, coarse=False, fine=False, ultrafine=False):
    surfaces = sphere_oblate_torus_dumbbell_mesh_inputs(reg=reg, coarse=coarse, fine=fine, ultrafine=ultrafine)
    for surf in surfaces:
        print(55 * "-")
        generate_regularize_save_mesh(**surf)

    print(55 * "-")
