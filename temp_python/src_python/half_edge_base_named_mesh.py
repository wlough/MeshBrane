from python.half_edge_mesh import HalfEdgeMeshBase
import numpy as np
from temp_python.src_python.global_vars import INT_TYPE, FLOAT_TYPE
import os
from temp_python.src_python.utilities.misc_utils import (
    make_output_dir,
    load_npz,
    save_npz,
    unchunk_file_with_cat,
)
from temp_python.src_python.half_edge_base_utils import find_h_right_B


def update_compressed_sphere_half_edge_arrays_with_h_right_B(
    input_dir="./data/half_edge_arrays_old",
    output_dir="./data/half_edge_arrays",
):
    _NUM_VERTS_ = [
        12,
        42,
        162,
        642,
        2562,
        10242,
        40962,
        163842,
        655362,
        2621442,
    ]
    make_output_dir(output_dir)
    paths_in = [f"{input_dir}/compressed_unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_]
    paths_out = [
        f"{output_dir}/compressed_unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_
    ]

    chunked_path_in = paths_in[-1]
    unchunked_path_in = paths_in[-1]
    unchunk_file_with_cat(chunked_path_in, unchunked_path_in)
    arrays_in = [load_npz(p) for p in paths_in]

    for arr, path_out in zip(arrays_in[:-1], paths_out[:-1]):
        print("saving " + path_out)
        arr["h_right_B"] = find_h_right_B(
            arr["xyz_coord_V"],
            arr["h_out_V"],
            arr["v_origin_H"],
            arr["h_next_H"],
            arr["h_twin_H"],
            arr["f_left_H"],
            arr["h_bound_F"],
        )

        save_npz(arr, path_out, compressed=True, chunk=False, remove_unchunked=False)

    arr, path_out = arrays_in[-1], paths_out[-1]
    print("saving " + path_out)
    arr["h_right_B"] = find_h_right_B(
        arr["xyz_coord_V"],
        arr["h_out_V"],
        arr["v_origin_H"],
        arr["h_next_H"],
        arr["h_twin_H"],
        arr["f_left_H"],
        arr["h_bound_F"],
    )
    save_npz(arr, path_out, compressed=True, chunk=True, remove_unchunked=True)
    print("done")
    return paths_out


def update_uncompressed_sphere_half_edge_arrays_with_h_right_B(
    input_dir="./output/half_edge_arrays_old",
    output_dir="./output/half_edge_arrays",
):
    _NUM_VERTS_ = [
        12,
        42,
        162,
        642,
        2562,
        10242,
        40962,
        163842,
        655362,
        2621442,
    ]
    make_output_dir(output_dir)
    paths_in = [f"{input_dir}/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_]
    paths_out = [f"{output_dir}/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_]
    arrays_in = [load_npz(p) for p in paths_in]

    for arr, path_out in zip(arrays_in, paths_out):
        print("saving " + path_out)
        arr["h_right_B"] = find_h_right_B(
            arr["xyz_coord_V"],
            arr["h_out_V"],
            arr["v_origin_H"],
            arr["h_next_H"],
            arr["h_twin_H"],
            arr["f_left_H"],
            arr["h_bound_F"],
        )

        save_npz(arr, path_out, compressed=False, chunk=False, remove_unchunked=False)
    print("done")
    return paths_out


def uncompress_sphere_half_edge_arrays(output_dir="./output/half_edge_arrays"):
    _NUM_VERTS_ = [
        12,
        42,
        162,
        642,
        2562,
        10242,
        40962,
        163842,
        655362,
        2621442,
    ]
    make_output_dir(output_dir)
    npz_paths = [f"{output_dir}/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_]
    compressed_npz_paths = [
        f"./data/half_edge_arrays/compressed_unit_sphere_{N:07d}.npz"
        for N in _NUM_VERTS_
    ]
    chunked_npz_path = compressed_npz_paths[-1]
    unchunked_npz_path = compressed_npz_paths[-1]
    unchunk_file_with_cat(chunked_npz_path, unchunked_npz_path)
    he_arrays = [load_npz(p) for p in compressed_npz_paths]
    os.system(f"rm {unchunked_npz_path}")
    for data, filename in zip(he_arrays, npz_paths):
        print("saving " + filename)
        save_npz(data, filename, compressed=False, chunk=False, remove_unchunked=False)
    print("done")
    return npz_paths


class HalfEdgeNamedMesh(HalfEdgeMeshBase):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
        *args,
        **kwargs,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
            *args,
            **kwargs,
        )
        # self.surface_params = [*surface_params]
        # self.surfcoord_array = self.compute_surfcoord_from_xyz()
        # self.mean_curvature = self.compute_mean_curvature()
        # self.gaussian_curvature = self.compute_gaussian_curvature()
        # self.unit_normal = self.compute_unit_normal()
        # self.mcvec_actual = np.einsum(
        #     "v,vi->vi", 2 * self.mean_curvature, self.unit_normal
        # )

    #######################################################
    # Initilization methods

    #######################################################
    def save(self, data_path=None):
        import pickle

        if data_path is not None:
            self.data_path = data_path
        if self.data_path is None:
            raise ValueError("No path to save data.")
        with open(self.data_path + ".pickle", "wb") as f:
            pickle.dump(self, f)

    def recompute_from_xyz(self, xyz_coord_V=None):
        if xyz_coord_V is None:
            xyz_coord_V = self.xyz_coord_V
        else:
            self.xyz_coord_V = xyz_coord_V
        self.surfcoord_array = self.compute_surfcoord_from_xyz()
        self.mean_curvature = self.compute_mean_curvature()
        self.unit_normal = self.compute_unit_normal()

    def recompute_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        else:
            self.surfcoord_array = surfcoord_array.copy()
        self.xyz_coord_V = self.compute_xyz_from_surfcoord()
        self.mean_curvature = self.compute_mean_curvature()
        self.unit_normal = self.compute_unit_normal()

    # Perturbations and noise
    ##########################################################
    # Overwrite these functions #
    #############################

    # Coordinate expressions/computations
    def compute_surfcoord_from_xyz(self, xyz_coord_V=None):
        if xyz_coord_V is None:
            xyz_coord_V = self.xyz_coord_V
        a, b = self.major_radius, self.minor_radius
        # xyz_coord_V = self.xyz_coord_V
        r = np.linalg.norm(xyz_coord_V, axis=-1)
        phi = np.arctan2(xyz_coord_V[:, 1], xyz_coord_V[:, 0])
        # rho = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        rho = np.linalg.norm(xyz_coord_V[:, :2], axis=-1)
        psi = np.arctan2(xyz_coord_V[:, 2], rho - a)
        return np.array([phi, psi]).T

    def compute_xyz_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        x = (a + b * np.cos(psi)) * np.cos(phi)
        y = (a + b * np.cos(psi)) * np.sin(phi)
        z = b * np.sin(psi)
        return np.array([x, y, z]).T

    def compute_mean_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        return -(a + 2 * b * np.cos(psi)) / (2 * b * (a + b * np.cos(psi)))

    def compute_gaussian_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        return np.cos(psi) / (b * (a + b * np.cos(psi)))

    def compute_unit_normal(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        nx = np.cos(phi) * np.cos(psi)
        ny = np.sin(phi) * np.cos(psi)
        nz = np.sin(psi)
        return np.array([nx, ny, nz]).T

    ######################################################
    ######################################################
    # to be deprecated
    @classmethod
    def _from_half_edge_ply(cls, ply_path, *surface_params):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(
            *VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples,
            *surface_params,
        )

    @classmethod
    def _from_data_arrays(cls, path, *surface_params):
        """Initialize a half-edge mesh from npz file containing data arrays."""
        data = np.load(path)
        return cls(
            data["xyz_coord_V"],
            data["h_out_V"],
            data["v_origin_H"],
            data["h_next_H"],
            data["h_twin_H"],
            data["f_left_H"],
            data["h_bound_F"],
            # data["h_comp_B"],
            *surface_params,
        )

    def perturbation_gaussian_xyz(self, loc=0, scale=0.01):
        """Adds a Gaussian perturbation to vertex cartesian coordinates"""
        self.xyz_coord_V = np.array(
            [
                self.xyz_coord_v(v) + np.random.normal(loc, scale, 3)
                for v in self.xyz_coord_V.keys()
            ]
        )

    def perturbation_gaussian_surfcoords(self, loc=0, scale=0.01):
        """
        Adds a Gaussian perturbation to vertex surface coordinates
        """
        self.surfcoord_array = np.array(
            [
                self.surfcoord_array[v] + np.random.normal(loc, scale, 2)
                for v in self.xyz_coord_V.keys()
            ]
        )
        # self.recompute_from_surfcoord()

    def perturbation_edge_flip(self, p=0.1):
        Nh = self.num_edges
        Hlist = list(self._v_origin_H.keys())
        for _ in range(Nh):
            h = np.random.choice(Hlist)
            if np.random.rand() < p:
                if self.h_is_flippable(h):
                    self.flip_edge(h)


class HalfEdgeSphere(HalfEdgeNamedMesh):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
        radius=1,
        rescale=False,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        self.radius = radius
        self.surfcoord_array = self.compute_surfcoord_from_xyz()
        self.mean_curvature = self.compute_mean_curvature()
        self.gaussian_curvature = self.compute_gaussian_curvature()
        self.unit_normal = self.compute_unit_normal()
        self.mcvec_actual = np.einsum(
            "v,vi->vi", 2 * self.mean_curvature, self.unit_normal
        )

    @classmethod
    def load_num(cls, p=5, input_dir="./output/half_edge_arrays"):
        _NUM_VERTS_ = [
            12,
            42,
            162,
            642,
            2562,
            10242,
            40962,
            163842,
            655362,
            2621442,
        ]
        N = _NUM_VERTS_[p]
        path_in = f"{input_dir}/unit_sphere_{N:07d}.npz"
        arr = load_npz(path_in)
        return cls(
            arr["xyz_coord_V"],
            arr["h_out_V"],
            arr["v_origin_H"],
            arr["h_next_H"],
            arr["h_twin_H"],
            arr["f_left_H"],
            arr["h_bound_F"],
            arr["h_right_B"],
        )

    # Coordinate expressions/computations
    def compute_surfcoord_from_xyz(self, xyz_coord_V=None):
        if xyz_coord_V is None:
            xyz_coord_V = self.xyz_coord_V
        a = self.radius
        # r = np.linalg.norm(xyz_coord_V, axis=-1)
        phi = np.arctan2(xyz_coord_V[:, 1], xyz_coord_V[:, 0])
        # rho = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        rho = np.linalg.norm(xyz_coord_V[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_coord_V[:, 2])
        return np.array([theta, phi]).T

    def compute_xyz_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a = self.radius
        theta, phi = surfcoord_array.T
        x = a * np.sin(theta) * np.cos(phi)
        y = a * np.sin(theta) * np.sin(phi)
        z = a * np.cos(theta)
        return np.array([x, y, z]).T

    def compute_mean_curvature(self, surfcoord_array=None):
        # if surfcoord_array is None:
        #     surfcoord_array = self.surfcoord_array
        a = self.radius
        return -np.ones(self.num_vertices) / a

    def compute_gaussian_curvature(self, surfcoord_array=None):
        # if surfcoord_array is None:
        #     surfcoord_array = self.surfcoord_array
        a = self.radius
        return np.ones(self.num_vertices) / a**2

    def compute_unit_normal(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        theta, phi = surfcoord_array.T
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        return np.array([nx, ny, nz]).T


class HalfEdgeTorus(HalfEdgeNamedMesh):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
        # major_radius=1,
        # minor_radius=0.5,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        self.major_radius = 1
        self.minor_radius = 1 / 3
        # self.recompute_from_xyz()
        # self.surface_params = [*surface_params]
        self.surfcoord_array = self.compute_surfcoord_from_xyz()
        self.mean_curvature = self.compute_mean_curvature()
        self.gaussian_curvature = self.compute_gaussian_curvature()
        self.unit_normal = self.compute_unit_normal()
        self.mcvec_actual = np.einsum(
            "v,vi->vi", 2 * self.mean_curvature, self.unit_normal
        )

    @classmethod
    def load_num(cls, p=5, input_dir="./output/half_edge_arrays"):
        _NUM_VERTS_ = [192, 768, 3072, 12288, 49152, 196608]
        N = _NUM_VERTS_[p]
        path_in = f"{input_dir}/torus_{N:07d}.npz"
        arr = load_npz(path_in)
        return cls(
            arr["xyz_coord_V"],
            arr["h_out_V"],
            arr["v_origin_H"],
            arr["h_next_H"],
            arr["h_twin_H"],
            arr["f_left_H"],
            arr["h_bound_F"],
            arr["h_right_B"],
        )

    # Coordinate expressions/computations
    def compute_surfcoord_from_xyz(self, xyz_coord_V=None):
        if xyz_coord_V is None:
            xyz_coord_V = self.xyz_coord_V
        a, b = self.major_radius, self.minor_radius
        xyz_coord_V = self.xyz_coord_V
        r = np.linalg.norm(xyz_coord_V, axis=-1)
        phi = np.arctan2(xyz_coord_V[:, 1], xyz_coord_V[:, 0])
        # rho = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        rho = np.linalg.norm(xyz_coord_V[:, :2], axis=-1)
        psi = np.arctan2(xyz_coord_V[:, 2], rho - a)
        return np.array([phi, psi]).T

    def compute_xyz_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        x = (a + b * np.cos(psi)) * np.cos(phi)
        y = (a + b * np.cos(psi)) * np.sin(phi)
        z = b * np.sin(psi)
        return np.array([x, y, z]).T

    def compute_mean_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        return -(a + 2 * b * np.cos(psi)) / (2 * b * (a + b * np.cos(psi)))

    def compute_gaussian_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        return np.cos(psi) / (b * (a + b * np.cos(psi)))

    def compute_unit_normal(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.major_radius, self.minor_radius
        phi, psi = surfcoord_array.T
        nx = np.cos(phi) * np.cos(psi)
        ny = np.sin(phi) * np.cos(psi)
        nz = np.sin(psi)
        return np.array([nx, ny, nz]).T

    @classmethod
    def refine(
        cls,
    ):
        r_b = self.rad_big
        r_s = self.rad_small
        Npsi_coarse = self.Npsi[-1]
        Nphi_coarse = self.Nphi[-1]
        pow_coarse = self.pow[-1]
        Npsi = 2 * Npsi_coarse
        Nphi = 2 * Nphi_coarse
        pow = pow_coarse + 1
        print(f"Refining {self.name}...")
        print(f"num_vertices: {Nphi_coarse*Npsi_coarse}-->{Nphi*Npsi}")
        self.Npsi.append(Npsi)
        self.Nphi.append(Nphi)
        self.pow.append(pow)
        F = []
        v_BS = []
        v_BS_coarse = self.v_BS[-1]

        for b_coarse in range(Nphi_coarse):
            ###################################################
            # add every other vertex to each ring in old mesh
            b = 2 * b_coarse
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s_coarse in range(Npsi_coarse):
                # every other vertex is the same as the coarse mesh
                s = 2 * s_coarse
                b_s_coarse = b_coarse * Npsi_coarse + s_coarse
                v_b_s = v_BS_coarse[b_s_coarse]  # v index
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])
                v_BS.append(v_b_s)
                # every other vertex is new
                s = 2 * s_coarse + 1
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                # bs_V[v_b_s] = b_s
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

            ###################################################
            # add every vertex to each new ring not in old mesh
            b = 2 * b_coarse + 1
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s in range(Npsi):
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

        self.F.append(F)
        self.v_BS.append(v_BS)
        if convert_to_half_edge:
            print("Converting to half-edge mesh...")
            self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))

    @classmethod
    def build_test_plys(cls, num_refine=5, p0=3, rad_big=1, ratio_big2small=3):
        pass


class TorusFactory:
    """
    Makes and refines meshes for the tori,
    (sqrt(x**2 + y**2) - rad_big)**2 + z**2 - rad_small**2 = 0.


    Refinements are computed by doubling the number of

    Attributes
    ----------
    name : str
        Name of the mesh (name="torus")
    rad_big : float
        Big radius of the torus. 1 by default.
    rad_small : float
        Small radius of the torus. 1/3 by default.
    p0 : int
        Resolution parameter for lowest level of refinement.
    Nphi : list of int
        Number of azimuthal angle samples in each mesh. Controls resolution along the big circumference. 3 * 2**p by default.
    Npsi : list of int
        Number of samples along the small circumference. 2**p by default.
    ---
    ---
    xyz_coord_V : list of numpy.array
        xyz coordinates of the vertices in the finest mesh.
    F : list of list
        Face list for each mesh level.
    implicit_fun_str : str
        Implicit function for the torus.
    """

    # @property
    # def name(self):
    #     return self._name

    # @property
    # def rad_small(self):
    #     return self.rad_big / self.ratio_big2small

    # def phi_of_xyz(self, x, y, z):
    #     return np.arctan2(y, x)

    # def psi_of_xyz(self, x, y, z):
    #     return np.arctan2(z, np.sqrt(x**2 + y**2))

    # def x_of_phi_psi(self, phi, psi):
    #     return np.cos(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    # def y_of_phi_psi(self, phi, psi):
    #     return np.sin(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    # def z_of_phi_psi(self, phi, psi):
    #     return np.sin(psi) * self.rad_small

    # def Nphi(self, p):
    #     return self.ratio_big2small * 2**p

    # def Npsi(self, p):
    #     return 2**p

    def __init__(self, p0=3, rad_big=1, ratio_big2small=3):
        self._name = "TESTtorus"
        self.p0 = p0
        self.rad_big = rad_big
        self.ratio_big2small = ratio_big2small
        rad_small = rad_big / ratio_big2small
        self.implicit_fun_str = (
            f"sqrt(x**2 + y**2) - {self.rad_big})**2 + z**2 - {rad_small}**2"
        )
        self.xyz_param_fun_str = [
            f"cos(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(psi) * {rad_small}",
        ]
        self.phipsi_param_fun_str = [
            f"cos(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(psi) * {rad_small}",
        ]

        pow = self.p0
        N_small = 2**pow
        N_big = 3 * N_small

        xyz_coord_V = []
        F = []
        v_BS = []
        for b in range(N_big):
            phi = 2 * np.pi * b / N_big
            bp1 = (b + 1) % N_big
            for s in range(N_small):
                sp1 = (s + 1) % N_small
                phi_small = 2 * np.pi * s / N_small
                x = np.cos(phi) * (rad_big + np.cos(phi_small) * rad_small)
                y = np.sin(phi) * (rad_big + np.cos(phi_small) * rad_small)
                z = np.sin(phi_small) * rad_small
                xyz_coord_V.append(np.array([x, y, z]))
                b_s = b * N_small + s
                b_sp1 = b * N_small + sp1
                bp1_s = bp1 * N_small + s
                bp1_sp1 = bp1 * N_small + sp1
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])
                v_b_s = b_s
                v_BS.append(v_b_s)

        self.xyz_coord_V = xyz_coord_V
        self.F = [F]
        # self.pow = [pow]
        self.Nphi = [N_big]
        self.Npsi = [N_small]
        self.v_BS = [v_BS]
        self.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*self.VF())]

    @classmethod
    def build_test_plys(cls, num_refine=5):
        b = cls()
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level)
        print("Done.")

    @property
    def pow(self):
        return [_ for _ in range(3, 3 + len(self.F))]

    def Vindices(self, level=-1):
        return self.v_BS[level]

    def num_vertices(self, level=-1):
        return len(self.Vindices(level))

    def num_faces(self, level=-1):
        return len(self.Vindices(level))

    def VF(self, level=-1):
        F = self.F[level]
        V = [self.xyz_coord_V[v] for v in self.Vindices(level)]
        return V, F

    def refine(self, convert_to_half_edge=True):
        r_b = self.rad_big
        r_s = self.rad_small
        Npsi_coarse = self.Npsi[-1]
        Nphi_coarse = self.Nphi[-1]
        pow_coarse = self.pow[-1]
        Npsi = 2 * Npsi_coarse
        Nphi = 2 * Nphi_coarse
        pow = pow_coarse + 1
        print(f"Refining {self.name}...")
        print(f"num_vertices: {Nphi_coarse*Npsi_coarse}-->{Nphi*Npsi}")
        self.Npsi.append(Npsi)
        self.Nphi.append(Nphi)
        self.pow.append(pow)
        F = []
        v_BS = []
        v_BS_coarse = self.v_BS[-1]

        for b_coarse in range(Nphi_coarse):
            ###################################################
            # add every other vertex to each ring in old mesh
            b = 2 * b_coarse
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s_coarse in range(Npsi_coarse):
                # every other vertex is the same as the coarse mesh
                s = 2 * s_coarse
                b_s_coarse = b_coarse * Npsi_coarse + s_coarse
                v_b_s = v_BS_coarse[b_s_coarse]  # v index
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])
                v_BS.append(v_b_s)
                # every other vertex is new
                s = 2 * s_coarse + 1
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                # bs_V[v_b_s] = b_s
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

            ###################################################
            # add every vertex to each new ring not in old mesh
            b = 2 * b_coarse + 1
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s in range(Npsi):
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

        self.F.append(F)
        self.v_BS.append(v_BS)
        if convert_to_half_edge:
            print("Converting to half-edge mesh...")
            self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))

    def write_plys(self, level=-1):
        if isinstance(level, int):
            vf_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_vf.ply"
            )
            he_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_he.ply"
            )
            print(f"Writing vertex-face ply to {vf_path}")
            self.v2h[level].write_source_ply(vf_path, use_ascii=False)
            print(f"Writing half-edge ply to {he_path}")
            self.v2h[level].write_target_ply(he_path, use_ascii=False)

        elif level == "all":
            for level in range(len(self.F)):
                self.write_plys(level=level)
            print(f"Done writing {self.name} plys.")

    def num_faces(self, p):
        """
        Nphi = 3 * 2**p
        Npsi = 2**p
        """
        return 6 * 4**p

    def num_vertices(self, p):
        return 3 * 4**p

    def num_edges(self, p):
        return 6 * 4**p + 3 * 4**p
