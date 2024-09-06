import numpy as np
from src.python.half_edge_base_ply_tools import (
    MeshConverterBase,
)  # VertTri2HalfEdgeConverter


class UnitDoughnutFactory:
    """
    Makes and refines meshes for the unit area torus. The torus is oriented such that the z-axis goes through the donut hole.

    Geometric parameters
    --------------------
    scale_phi : int
        scale_phi ratio parameter
    scale_psi : int
        psi ratio parameter

    Discretization parameters
    -------------------------
    resolution_min : int
        Resolution parameter for lowest level of refinement.
    resolution_max : int
        Resolution parameter for highest level of refinement.

    c = scale_phi / scale_psi
    Rphi = sqrt(c)/(2*pi)
    Rpsi = 1/(2*pi*sqrt(c))
    (sqrt(x**2 + y**2) - Rphi)**2 + z**2 - Rpsi**2 = 0

    Nphi_p = c * 2**p = c * 2**(p0+q)
    Npsi_p = 2**p = 2**(p0+q)

    Attributes
    ----------
    Rphi : float
        radius of the circle parameterized by phi (about the z-axis)
    Rpsi : float
        radius of the circle parameterized by psi (circle about the z-axis)
    Phi_rmax : ndarray
        linspace(0, 2*pi, Nphi_rmax)
    Psi_rmax : ndarray
        linspace(-pi, pi, Npsi_rmax)

    Methods
    -------
    index_step(p)
        Returns the index step between resolution p samples in Phi_rmax and Psi_rmax.
    Phi(self, p)
        Returns ndarray of azimuthal angle samples for resolution p.
    Psi(self, p)
        Returns ndarray of poloidal angle samples for resolution p.



    """

    def __init__(
        self,
        scale_phi=3,
        scale_psi=1,
        resolution_min=3,
        resolution_max=6,
        name="torus_3_1",
    ):
        self.scale_phi = scale_phi
        self.scale_psi = scale_psi
        self.resolution_min = resolution_min
        self.resolution_max = resolution_max
        self.name = name

        c = scale_phi / scale_psi
        Nphi_rmax = scale_phi * 2**resolution_max
        Npsi_rmax = scale_psi * 2**resolution_max
        self.Rphi = np.sqrt(c) / (2 * np.pi)
        self.Rpsi = 1 / (2 * np.pi * np.sqrt(c))
        self.Phi_rmax = np.linspace(0, 2 * np.pi, Nphi_rmax, endpoint=False)
        self.Psi_rmax = np.linspace(0, 2 * np.pi, Npsi_rmax, endpoint=False)

        ################################
        self._name = "torus_3_1"
        self.p0 = p0
        self.max_num_refinements = max_num_refinements
        self.Rbig = Rbig

        self.Rsmall = Rbig / ratio_Rbig2Rsmall
        self.current_num_refinemnts = 0

        ######
        pow = self.p0
        Npsi = 2**pow
        Nphi = 3 * Npsi
        # Rbig = 1
        Rsmall = Rbig / ratio_Rbig2Rsmall
        self.Rbig = Rbig
        self.Rsmall = Rsmall
        self.implicit_fun_str = (
            f"sqrt(x**2 + y**2) - {self.Rbig})**2 + z**2 - {self.Rsmall}**2"
        )

        xyz_coord_V = []
        F = []
        v_BS = []
        for b in range(Nphi):
            phi = 2 * np.pi * b / Nphi
            bp1 = (b + 1) % Nphi
            for s in range(Npsi):
                sp1 = (s + 1) % Npsi
                phi_small = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (Rbig + np.cos(phi_small) * Rsmall)
                y = np.sin(phi) * (Rbig + np.cos(phi_small) * Rsmall)
                z = np.sin(phi_small) * Rsmall
                xyz_coord_V.append(np.array([x, y, z]))
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                # F.append([b_s, bp1_sp1, bp1_s])
                # F.append([b_s, b_sp1, bp1_sp1])
                F.append([b_s, bp1_s, bp1_sp1])
                F.append([b_s, bp1_sp1, b_sp1])
                v_b_s = b_s
                v_BS.append(v_b_s)

        self.xyz_coord_V = xyz_coord_V
        self.F = [F]
        # self.pow = [pow]
        self.Nphi = [Nphi]
        self.Npsi = [Npsi]
        self.v_BS = [v_BS]
        # self.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*self.VF())]
        self.v2h = [MeshConverterBase.from_vf_samples(*self.VF())]

    def phi_of_xyz(self, x, y, z):
        return np.arctan2(y, x)

    def psi_of_xyz(self, x, y, z):
        return np.arctan2(z, np.sqrt(x**2 + y**2))

    def x_of_phi_psi(self, phi, psi):
        return np.cos(phi) * (self.Rbig + np.cos(psi) * self.Rsmall)

    def y_of_phi_psi(self, phi, psi):
        return np.sin(phi) * (self.Rbig + np.cos(psi) * self.Rsmall)

    def z_of_phi_psi(self, phi, psi):
        return np.sin(psi) * self.Rsmall

    def Rphi_of_xyz(self, x, y, z):
        return np.sqrt(x**2 + y**2)

    ###################################
    # deprecated

    def write_plys(self, level=-1, ply_dir="./output/torus_plys"):
        if isinstance(level, int):
            # p = self.pow[level]
            # Nv =
            he_path = f"{ply_dir}/{self.name}_{self.num_vertices(level):06d}_he.ply"
            print(f"Writing half-edge ply to {he_path}")
            self.v2h[level].write_target_ply(he_path, use_ascii=False)

        elif level == "all":
            for level in range(len(self.F)):
                self.write_plys(level=level)
            print(f"Done writing {self.name} plys.")

    def Nphi(self, num_refinements):
        p = self.p0 + num_refinements
        M = self.ratio_Rbig2Rsmall
        N = M * 2**p
        return M * 2**p

    def Npsi(self, num_refinements):
        p = self.p0 + num_refinements
        M = self.ratio_Rbig2Rsmall
        N = M * 2**p
        return M * 2**p

    @property
    def current_Nphi(self):
        return self.Nphi(self.current_num_refinemnts)

    @property
    def current_Npsi(self):
        return self.Npsi(self.current_num_refinemnts)

    @classmethod
    def build_test_plys(
        cls,
        num_refine=5,
        ply_dir="./output/torus_plys",
        p0=3,
        Rbig=1,
        ratio_Rbig2Rsmall=3,
    ):
        b = cls(
            p0=3,
            Rbig=1,
            ratio_Rbig2Rsmall=3,
        )
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level, ply_dir=ply_dir)
        print("Done.")

    @property
    def name(self):
        return self._name

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
        r_b = self.Rbig
        r_s = self.Rsmall
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
                # F.append([b_s, bp1_sp1, bp1_s])
                # F.append([b_s, b_sp1, bp1_sp1])
                F.append([b_s, bp1_s, bp1_sp1])
                F.append([b_s, bp1_sp1, b_sp1])
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
                # F.append([b_s, bp1_sp1, bp1_s])
                # F.append([b_s, b_sp1, bp1_sp1])
                F.append([b_s, bp1_s, bp1_sp1])
                F.append([b_s, bp1_sp1, b_sp1])

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
                # F.append([b_s, bp1_sp1, bp1_s])
                # F.append([b_s, b_sp1, bp1_sp1])
                F.append([b_s, bp1_s, bp1_sp1])
                F.append([b_s, bp1_sp1, b_sp1])

        self.F.append(F)
        self.v_BS.append(v_BS)
        if convert_to_half_edge:
            print("Converting to half-edge mesh...")
            self.v2h.append(MeshConverterBase.from_vf_samples(*self.VF()))

    def _num_faces(self, p):
        """
        Nphi = 3 * 2**p
        Npsi = 2**p
        """
        return 6 * 4**p

    def _num_vertices(self, p):
        return 3 * 4**p

    def _num_edges(self, p):
        return 6 * 4**p + 3 * 4**p
