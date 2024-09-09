import numpy as np
from src.python.half_edge_base_ply_tools import (
    MeshConverterBase,
)  # VertTri2HalfEdgeConverter


class UnitDoughnutFactory:
    """
    Makes and refines meshes for the unit area torus. The torus is oriented such that the z-axis goes through the donut hole.

    Parameters
    --------------------
    scale_phi : int
        scale_phi ratio parameter
    scale_psi : int
        psi ratio parameter

    c = scale_phi / scale_psi
    Rphi = sqrt(c)/(2*pi)
    Rpsi = 1/(2*pi*sqrt(c))
    (sqrt(x**2 + y**2) - Rphi)**2 + z**2 - Rpsi**2 = 0

    Nphi_p = scale_phi * 2**p
    Npsi_p = scale_psi * 2**p

    Attributes
    ----------
    Rphi : float
        radius of the circle parameterized by phi (about the z-axis)
    Rpsi : float
        radius of the circle parameterized by psi (circle about the z-axis)

    Methods
    -------
    Nphi(self, p)
        Returns number of phi samples at resolution p.
    Npsi(self, p)
        Returns number of psi samples at resolution p.
    Phi(self, p)
        Returns ndarray of azimuthal angle samples at resolution p.
    Psi(self, p)
        Returns ndarray of poloidal angle samples at resolution p.
    V_of_F_at_resolution_p(self, p)
        Returns ndarray of face vertex indices at resolution p.
    surf_coords_V_at_resolution_p(self, p)
        Returns ndarray of surface coordinates of vertices at resolution p.
    XYZ_of_PhiPsi(self, Phi, Psi)
        Returns ndarray of xyz coordinates given Phi and Psi at each vertex.
    xyz_coord_V_at_resolution_p(self, p)
        Returns ndarray of xyz coordinates of vertices at resolution p.
    """

    def __init__(
        self,
        scale_phi=3,
        scale_psi=1,
        # resolution_min=3,
        # resolution_max=6,
    ):
        self.scale_phi = scale_phi
        self.scale_psi = scale_psi
        self.name = f"unit_torus_{scale_phi}_{scale_psi}"

        c = scale_phi / scale_psi

        self.Rphi = np.sqrt(c) / (2 * np.pi)
        self.Rpsi = 1 / (2 * np.pi * np.sqrt(c))

        ############################################################
        self.implicit_fun_str = (
            f"sqrt(x**2 + y**2) - {self.Rphi})**2 + z**2 - {self.Rpsi}**2"
        )
        ############################################################
        self.mesh_converter = dict()

    @property
    def scale_phi(self):
        return self._scale_phi

    @scale_phi.setter
    def scale_phi(self, value):
        if isinstance(value, int):
            self._scale_phi = value
        else:
            raise ValueError("scale_phi must be an integer.")

    @property
    def scale_psi(self):
        return self._scale_psi

    @scale_psi.setter
    def scale_psi(self, value):
        if isinstance(value, int):
            self._scale_psi = value
        else:
            raise ValueError("scale_psi must be an integer.")

    def Nphi(self, resolution):
        return self.scale_phi * 2**resolution

    def Npsi(self, resolution):
        return self.scale_psi * 2**resolution

    def Phi(self, resolution):
        # res_diff = self.resolution_max - resolution
        # i_skip = 2**res_diff
        # return self.Phi_rmax[::i_skip]
        Nphi = self.Nphi(resolution)
        Phi = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
        return Phi

    def Psi(self, resolution):
        # res_diff = self.resolution_max - resolution
        # i_skip = 2**res_diff
        # return self.Psi_rmax[::i_skip]
        Npsi = self.Npsi(resolution)
        Psi = np.linspace(0, 2 * np.pi, Npsi, endpoint=False)
        return Psi

    def V_of_F_at_resolution_p(self, p):
        Nphi = self.Nphi(p)
        Npsi = self.Npsi(p)
        Nf = self.num_faces(p)
        F = np.array(
            [
                [
                    # a face
                    b * Npsi + s,
                    ((b + 1) % Nphi) * Npsi + s,
                    ((b + 1) % Nphi) * Npsi + ((s + 1) % Npsi),
                    # another face
                    b * Npsi + s,
                    ((b + 1) % Nphi) * Npsi + ((s + 1) % Npsi),
                    b * Npsi + ((s + 1) % Npsi),
                ]
                for b in range(Nphi)
                for s in range(Npsi)
            ],
            dtype="int32",
        ).reshape((-1, 3))
        return F

    def surf_coords_V_at_resolution_p(self, p):

        # Nphi = self.Nphi(p)
        # Npsi = self.Npsi(p)
        # Phi = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
        # Psi = np.linspace(0, 2 * np.pi, Npsi, endpoint=False)
        Phi = self.Phi(p)
        Psi = self.Psi(p)
        surf_coords_V = np.array([[phi, psi] for phi in Phi for psi in Psi])
        return surf_coords_V

    def XYZ_of_PhiPsi(self, Phi, Psi):
        return np.array(
            [
                np.cos(Phi) * (self.Rphi + np.cos(Psi) * self.Rpsi),
                np.sin(Phi) * (self.Rphi + np.cos(Psi) * self.Rpsi),
                np.sin(Psi) * self.Rpsi,
            ]
        ).T

    def xyz_coord_V_at_resolution_p(self, p):
        surf_coords_V = self.surf_coords_V_at_resolution_p(p)
        xyz_coord_V = self.XYZ_of_PhiPsi(*surf_coords_V.T)
        return xyz_coord_V

    def init_mesh_converter_at_resolution_p(self, p):
        xyz_coord_V = self.xyz_coord_V_at_resolution_p(p)
        V_of_F = self.V_of_F_at_resolution_p(p)
        c = MeshConverterBase.from_vf_samples(
            xyz_coord_V, V_of_F, compute_he_stuff=True
        )
        self.mesh_converter[p] = c
        return c

    def write_he_ply_at_resolution_p(self, p, output_dir="./output"):

        if p in self.mesh_converter:
            c = self.mesh_converter[p]
            ply_path = f"{output_dir}/{self.name}_{self.num_faces(p):06d}_he.ply"
            c.write_he_ply(ply_path, use_binary=True)
        else:
            raise ValueError(f"Mesh converter for resolution {p} not found.")

    def write_he_samples_at_resolution_p(self, p, output_dir="./output"):
        if p in self.mesh_converter:
            c = self.mesh_converter[p]
            samples_path = f"{output_dir}/{self.name}_{self.num_faces(p):06d}_he.npz"
            c.write_he_samples(
                path=samples_path, compressed=False, chunk=False, remove_unchunked=False
            )
        else:
            raise ValueError(f"Mesh converter for resolution {p} not found.")

    ###################################
    def compute_surfcoord_from_xyz(self, x, y, z):
        a, b = self.Rphi, self.Rpsi
        phi = np.arctan2(y, x)
        cx, cy = a * np.cos(phi), a * np.sin(phi)
        rho = np.sqrt(x**2 + y**2)
        psi = np.arctan2(z, rho - a)
        return np.array([phi, psi]).T

    def phi_of_xyz(self, x, y, z):
        phi = np.arctan2(y, x)
        return phi

    def psi_of_xyz(self, x, y, z):
        rho = np.sqrt(x**2 + y**2)
        psi = np.arctan2(z, rho - a)
        return psi

    def x_of_phi_psi(self, phi, psi):
        return np.cos(phi) * (self.self.Rphi + np.cos(psi) * self.Rpsi)

    def y_of_phi_psi(self, phi, psi):
        return np.sin(phi) * (self.self.Rphi + np.cos(psi) * self.Rpsi)

    def z_of_phi_psi(self, phi, psi):
        return np.sin(psi) * self.Rpsi

    ###################################
    def num_faces(self, p):
        Nphi = self.Nphi(p)
        Npsi = self.Npsi(p)
        Nf = 2 * Nphi * Npsi
        return Nf

    def num_vertices(self, p):
        Nphi = self.Nphi(p)
        Npsi = self.Npsi(p)
        Nv = Nphi * Npsi
        return Nv

    def num_edges(self, p):
        Nv = self.num_vertices(p)
        Nf = self.num_faces(p)
        # g = 1  # genus
        Ne = Nv + Nf  # - 2 + 2 * g
        return Ne

    ###################################
    def project_to_torus(self, x, y, z):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        psi = np.arctan2(z, rho - self.Rphi)
        x = np.cos(phi) * (self.Rphi + np.cos(psi) * self.Rpsi)
        y = np.sin(phi) * (self.Rphi + np.cos(psi) * self.Rpsi)
        z = np.sin(psi) * self.Rpsi
        return np.array([x, y, z]).T

    ###################################
    # deprecated

    # def Phi_indices(self, resolution):
    #     res_diff = self.resolution_max - resolution
    #     i_skip = 2**res_diff
    #     return np.arange(0, self.Nphi(self.resolution_max), i_skip, dtype="int32")

    # def Psi_indices(self, resolution):
    #     res_diff = self.resolution_max - resolution
    #     i_skip = 2**res_diff
    #     return np.arange(0, self.Npsi(self.resolution_max), i_skip, dtype="int32")

    # def write_plys(self, level=-1, ply_dir="./output/torus_plys"):
    #     if isinstance(level, int):
    #         # p = self.pow[level]
    #         # Nv =
    #         he_path = f"{ply_dir}/{self.name}_{self.num_vertices(level):06d}_he.ply"
    #         print(f"Writing half-edge ply to {he_path}")
    #         self.v2h[level].write_target_ply(he_path, use_ascii=False)

    #     elif level == "all":
    #         for level in range(len(self.F)):
    #             self.write_plys(level=level)
    #         print(f"Done writing {self.name} plys.")

    # @property
    # def current_Nphi(self):
    #     return self.Nphi(self.current_num_refinemnts)

    # @property
    # def current_Npsi(self):
    #     return self.Npsi(self.current_num_refinemnts)

    # @classmethod
    # def build_test_plys(
    #     cls,
    #     num_refine=5,
    #     ply_dir="./output/torus_plys",
    #     p0=3,
    #     Rbig=1,
    #     ratio_Rbig2Rsmall=3,
    # ):
    #     b = cls(
    #         p0=3,
    #         Rbig=1,
    #         ratio_Rbig2Rsmall=3,
    #     )
    #     b.write_plys(level=0)
    #     for level in range(1, num_refine + 1):
    #         b.refine()
    #         b.write_plys(level=level, ply_dir=ply_dir)
    #     print("Done.")

    # # @property
    # # def name(self):
    # #     return self._name

    # @property
    # def pow(self):
    #     return [_ for _ in range(3, 3 + len(self.F))]

    # # def Vindices(self, level=-1):
    # #     return self.v_BS[level]

    # # def num_vertices(self, level=-1):
    # #     return len(self.Vindices(level))

    # # def num_faces(self, level=-1):
    # #     return len(self.Vindices(level))

    # def VF(self, level=-1):
    #     F = self.F[level]
    #     V = [self.xyz_coord_V[v] for v in self.Vindices(level)]
    #     return V, F

    # def refine(self, convert_to_half_edge=True):
    #     r_b = self.Rbig
    #     r_s = self.Rsmall
    #     Npsi_coarse = self.Npsi[-1]
    #     Nphi_coarse = self.Nphi[-1]
    #     pow_coarse = self.pow[-1]
    #     Npsi = 2 * Npsi_coarse
    #     Nphi = 2 * Nphi_coarse
    #     pow = pow_coarse + 1
    #     print(f"Refining {self.name}...")
    #     print(f"num_vertices: {Nphi_coarse*Npsi_coarse}-->{Nphi*Npsi}")
    #     self.Npsi.append(Npsi)
    #     self.Nphi.append(Nphi)
    #     self.pow.append(pow)
    #     F = []
    #     v_BS = []
    #     v_BS_coarse = self.v_BS[-1]

    #     for b_coarse in range(Nphi_coarse):
    #         ###################################################
    #         # add every other vertex to each ring in old mesh
    #         b = 2 * b_coarse
    #         bp1 = (b + 1) % Nphi
    #         phi = 2 * np.pi * b / Nphi
    #         for s_coarse in range(Npsi_coarse):
    #             # every other vertex is the same as the coarse mesh
    #             s = 2 * s_coarse
    #             b_s_coarse = b_coarse * Npsi_coarse + s_coarse
    #             v_b_s = v_BS_coarse[b_s_coarse]  # v index
    #             sp1 = (s + 1) % Npsi
    #             b_s = b * Npsi + s
    #             b_sp1 = b * Npsi + sp1
    #             bp1_s = bp1 * Npsi + s
    #             bp1_sp1 = bp1 * Npsi + sp1
    #             # F.append([b_s, bp1_sp1, bp1_s])
    #             # F.append([b_s, b_sp1, bp1_sp1])
    #             F.append([b_s, bp1_s, bp1_sp1])
    #             F.append([b_s, bp1_sp1, b_sp1])
    #             v_BS.append(v_b_s)
    #             # every other vertex is new
    #             s = 2 * s_coarse + 1
    #             v_b_s = len(self.xyz_coord_V)  # new v index
    #             psi = 2 * np.pi * s / Npsi
    #             x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
    #             y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
    #             z = np.sin(psi) * r_s
    #             self.xyz_coord_V.append(np.array([x, y, z]))
    #             sp1 = (s + 1) % Npsi
    #             b_s = b * Npsi + s
    #             b_sp1 = b * Npsi + sp1
    #             bp1_s = bp1 * Npsi + s
    #             bp1_sp1 = bp1 * Npsi + sp1
    #             # bs_V[v_b_s] = b_s
    #             v_BS.append(v_b_s)
    #             # F.append([b_s, bp1_sp1, bp1_s])
    #             # F.append([b_s, b_sp1, bp1_sp1])
    #             F.append([b_s, bp1_s, bp1_sp1])
    #             F.append([b_s, bp1_sp1, b_sp1])

    #         ###################################################
    #         # add every vertex to each new ring not in old mesh
    #         b = 2 * b_coarse + 1
    #         bp1 = (b + 1) % Nphi
    #         phi = 2 * np.pi * b / Nphi
    #         for s in range(Npsi):
    #             v_b_s = len(self.xyz_coord_V)  # new v index
    #             psi = 2 * np.pi * s / Npsi
    #             x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
    #             y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
    #             z = np.sin(psi) * r_s
    #             self.xyz_coord_V.append(np.array([x, y, z]))
    #             sp1 = (s + 1) % Npsi
    #             b_s = b * Npsi + s
    #             b_sp1 = b * Npsi + sp1
    #             bp1_s = bp1 * Npsi + s
    #             bp1_sp1 = bp1 * Npsi + sp1
    #             v_BS.append(v_b_s)
    #             # F.append([b_s, bp1_sp1, bp1_s])
    #             # F.append([b_s, b_sp1, bp1_sp1])
    #             F.append([b_s, bp1_s, bp1_sp1])
    #             F.append([b_s, bp1_sp1, b_sp1])

    #     self.F.append(F)
    #     self.v_BS.append(v_BS)
    #     if convert_to_half_edge:
    #         print("Converting to half-edge mesh...")
    #         self.v2h.append(MeshConverterBase.from_vf_samples(*self.VF()))
