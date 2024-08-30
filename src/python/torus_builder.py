import numpy as np
from src.python.half_edge_mesh import VertTri2HalfEdgeConverter


class DoughnutFactory:
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

    def __init__(self, p0=3, rad_big=1, ratio_big2small=3):
        self._name = "torus_3_1"
        self.p0 = p0
        pow = self.p0
        N_small = 2**pow
        N_big = 3 * N_small
        # rad_big = 1
        rad_small = rad_big / ratio_big2small
        self.rad_big = rad_big
        self.rad_small = rad_small
        self.implicit_fun_str = (
            f"sqrt(x**2 + y**2) - {self.rad_big})**2 + z**2 - {self.rad_small}**2"
        )

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
                # F.append([b_s, bp1_sp1, bp1_s])
                # F.append([b_s, b_sp1, bp1_sp1])
                F.append([b_s, bp1_s, bp1_sp1])
                F.append([b_s, bp1_sp1, b_sp1])
                v_b_s = b_s
                v_BS.append(v_b_s)

        self.xyz_coord_V = xyz_coord_V
        self.F = [F]
        # self.pow = [pow]
        self.Nphi = [N_big]
        self.Npsi = [N_small]
        self.v_BS = [v_BS]
        self.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*self.VF())]

    def phi_of_xyz(self, x, y, z):
        return np.arctan2(y, x)

    def psi_of_xyz(self, x, y, z):
        return np.arctan2(z, np.sqrt(x**2 + y**2))

    def x_of_phi_psi(self, phi, psi):
        return np.cos(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    def y_of_phi_psi(self, phi, psi):
        return np.sin(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    def z_of_phi_psi(self, phi, psi):
        return np.sin(psi) * self.rad_small

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

    @classmethod
    def build_test_plys(
        cls,
        num_refine=5,
        ply_dir="./output/torus_plys",
        p0=3,
        rad_big=1,
        ratio_big2small=3,
    ):
        b = cls(
            p0=3,
            rad_big=1,
            ratio_big2small=3,
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
            self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))

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
