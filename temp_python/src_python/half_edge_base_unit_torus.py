import numpy as np
from temp_python.src_python.half_edge_base_ply_tools import (
    MeshConverterBase,
)  # VertTri2HalfEdgeConverter


class DoughnutFactory:
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
        name=None,
    ):
        self.scale_phi = scale_phi
        self.scale_psi = scale_psi
        if name is None:
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
        # Nf = self.num_faces(p)
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
        a = self.Rphi
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

    @classmethod
    def mesh_gen(
        cls,
        output_dir="./output/torus_mesh_gen_test",
        scale_phi=3,
        scale_psi=1,
        resolution_min=3,
        resolution_max=8,
    ):
        """
        Generates triangle meshes for unit area torii. Output files are labeled by number of faces in the mesh.

        Rphi:Rpsi=Nphi:Npsi=scale_phi:scale_psi
        Nphi = scale_phi * 2**resolution
        Npsi = scale_psi * 2**resolution
        num_vertices = Nphi * Npsi
        num_faces = 2 * Nphi * Npsi
                  = 2 * scale_phi * scale_psi * 4**resolution
        num_edges = 3 * Nphi * Npsi
        """
        from temp_python.src_python.utilities.misc_utils import make_output_dir

        make_output_dir(output_dir, overwrite=False)
        ##############################################################

        surfname = f"unit_torus_{scale_phi}_{scale_psi}"
        Nfaces = [
            2 * scale_phi * scale_psi * 4**p
            for p in range(resolution_min, resolution_max + 1)
        ]
        plys_raw = [f"{surfname}_raw_{num_faces:06d}_he.ply" for num_faces in Nfaces]
        samples_raw = [f"{surfname}_raw_{num_faces:06d}_he.npz" for num_faces in Nfaces]

        f = cls(scale_phi=scale_phi, scale_psi=scale_psi, name=surfname)

        for _, p in enumerate(range(resolution_min, resolution_max + 1)):
            print("------------------------------------------------")
            print(f"initializing mesh_converter at resolution {p=}")
            f.init_mesh_converter_at_resolution_p(p)
            c = f.mesh_converter[p]

            ply_path = f"{output_dir}/{plys_raw[_]}"
            samples_path = f"{output_dir}/{samples_raw[_]}"
            print(f"writing {ply_path=}")
            c.write_he_ply(ply_path, use_binary=True)
            print(f"writing {samples_path=}")
            c.write_he_samples(
                path=samples_path, compressed=False, chunk=False, remove_unchunked=False
            )

    ###################################
    # deprecated
