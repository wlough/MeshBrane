import numpy as np
from plyfile import PlyData, PlyElement
from temp_python.src_python.global_vars import INT_TYPE, FLOAT_TYPE
from temp_python.src_python.half_edge_base_utils import (
    vf_samples_to_he_samples,
    he_samples_to_vf_samples,
    find_h_right_B,
)
from temp_python.src_python.utilities.misc_utils import (
    # make_output_dir,
    # load_npz,
    save_npz,
    # unchunk_file_with_cat,
)


class MeshConverter:
    """
    Reading/writing ply files and converting between vertex-face (vf) and half-edge (he) data.

    Attributes
    ----------
    vf_ply_path : str
        Path to the vf ply file where vf_ply_data was saved to/loaded from.
    vf_ply_data : PlyData
        Data from the vf ply file.
    vf_samples : tuple of ndarray
        (xyz_coord_V, V_of_F)
    he_ply_path : str
        Path to the he ply file where he_ply_data was saved to/loaded from.
    he_ply_data : PlyData
        Data from the source ply file.
    he_samples : tuple of ndarray
        (xyz_coord_V,..., h_right_B)

    Initialization
    --------------
    from_vf_ply(ply_path, compute_he_stuff=True)
        Construct a MeshConverter object from a vf ply file and compute he samples.
    from_vf_samples(xyz_coord_V, V_of_F, compute_he_stuff=True)
        Construct a MeshConverter object from vf samples and compute he samples.
    from_he_ply(ply_path, compute_vf_stuff=True)
        Construct a MeshConverter object from a he ply file and compute vf samples.
    from_he_samples(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H, h_bound_F, h_right_B, compute_vf_stuff=True)
        Construct a MeshConverter object from he samples and compute vf samples.

    Methods
    -------
    vf_samples_to_he_samples()
        Convert vf samples to he samples.
    vf_ply_data_to_samples()
        Convert vf ply data to samples.
    vf_samples_to_ply_data(use_binary=True)
        Convert vf samples to ply data.
    he_samples_to_vf_samples()
        Convert he samples to vf samples.
    he_ply_data_to_samples()
        Convert he ply data to samples.
    he_samples_to_ply_data(use_binary=True)
        Convert he samples to ply data.
    write_vf_ply(ply_path, use_binary=True)
        Write vf ply data to a ply file.
    write_he_ply(ply_path, use_binary=True)
        Write he ply data to a ply file.
    write_he_samples(path=None, compressed=False, chunk=False, remove_unchunked=False)
        Save he samples to a npz file.
    """

    def __init__(self):
        self.vf_ply_path = None
        self.vf_ply_data = None
        self.vf_samples = None

        self.he_ply_path = None
        self.he_ply_data = None
        self.he_samples = None

    @classmethod
    def from_vf_ply(cls, ply_path, compute_he_stuff=True):
        c = cls()

        c.vf_ply_path = ply_path
        c.vf_ply_data = PlyData.read(ply_path)
        c.vf_samples = c.vf_ply_data_to_samples()

        if compute_he_stuff:
            c.he_samples = c.vf_samples_to_he_samples()
            c.he_ply_data = c.he_samples_to_ply_data()

        return c

    @classmethod
    def from_vf_samples(cls, xyz_coord_V, V_of_F, compute_he_stuff=True):
        c = cls()

        c.vf_samples = (xyz_coord_V, V_of_F)
        c.vf_ply_data = c.vf_samples_to_ply_data()

        if compute_he_stuff:
            c.he_samples = c.vf_samples_to_he_samples()
            c.he_ply_data = c.he_samples_to_ply_data()

        return c

    @classmethod
    def from_he_ply(cls, ply_path, compute_vf_stuff=True):
        c = cls()

        c.he_ply_path = ply_path
        c.he_ply_data = PlyData.read(ply_path)
        c.he_samples = c.he_ply_data_to_samples()

        if compute_vf_stuff:
            c.vf_samples = c.he_samples_to_vf_samples()
            c.vf_ply_data = c.vf_samples_to_ply_data()

        return c

    @classmethod
    def from_he_samples(
        cls,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
        compute_vf_stuff=True,
    ):
        c = cls()

        c.he_samples = (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        c.he_ply_data = c.he_samples_to_ply_data()

        if compute_vf_stuff:
            c.vf_samples = c.he_samples_to_vf_samples()
            c.vf_ply_data = c.vf_samples_to_ply_data()

        return c

    def vf_samples_to_he_samples(self):
        (xyz_coord_V, V_of_F) = self.vf_samples
        he_samples = vf_samples_to_he_samples(xyz_coord_V, V_of_F)
        return he_samples

    def vf_ply_data_to_samples(self):
        """Constructs a lists of data from a PlyData object using the schema"""
        plydata = self.vf_ply_data
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=FLOAT_TYPE,
        ).T
        V_of_F = np.array(
            [
                vertex_indices.tolist()
                for vertex_indices in plydata["face"]["vertex_indices"]
            ],
            dtype=INT_TYPE,
        )
        samples = (
            xyz_coord_V,
            V_of_F,
        )
        return samples

    def vf_samples_to_ply_data(self, use_binary=True):
        """Constructs a PlyData object using the schema"""
        (
            xyz_coord_V,
            V_of_F,
        ) = self.vf_samples
        V_data = np.array(
            [tuple(v) for v in xyz_coord_V],
            dtype=[
                ("x", FLOAT_TYPE),
                ("y", FLOAT_TYPE),
                ("z", FLOAT_TYPE),
            ],
        )
        F_data = np.empty(len(V_of_F), dtype=[("vertex_indices", INT_TYPE, (3,))])
        F_data["vertex_indices"] = V_of_F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        return PlyData([vertex_element, face_element], text=not use_binary)

    def he_samples_to_vf_samples(self):
        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = self.he_samples
        (xyz_coord_V, V_of_F) = he_samples_to_vf_samples(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        vf_samples = (xyz_coord_V, V_of_F)
        return vf_samples

    def he_ply_data_to_samples(self):
        """Constructs a lists of data from a PlyData object using the schema
        xyz_coord_V : list of numpy.array
            xyz_coord_V[i] = xyz coordinates of vertex i

        h_out_V : list of int
            h_out_V[i] = some outgoing half-edge incident on vertex i
        v_origin_H : list of int
            v_origin_H[j] = vertex at the origin of half-edge j
        h_next_H : list of int
            h_next_H[j] next half-edge after half-edge j in the face cycle
        h_twin_H : list of int
            h_twin_H[j] = half-edge antiparalel to half-edge j
        f_left_H : list of int
            f_left_H[j] = face to the left of half-edge j
            f_left_H[j] = -(b+1) if half-edge j is contained in boundary b and the complement of the mesh is left of j
        h_bound_F : list of int
            h_bound_F[k] = some half-edge on the boudary of face k
        h_right_B : list of int
            h_right_B[b] = some half-edge in the boundary b which is right of the complement of the mesh
        """
        plydata = self.he_ply_data
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=FLOAT_TYPE,
        ).T
        h_out_V = np.array(plydata["vertex"]["h"], dtype=INT_TYPE)
        v_origin_H = np.array(plydata["half_edge"]["v"], dtype=INT_TYPE)
        h_next_H = np.array(plydata["half_edge"]["n"], dtype=INT_TYPE)
        h_twin_H = np.array(plydata["half_edge"]["t"], dtype=INT_TYPE)
        f_left_H = np.array(plydata["half_edge"]["f"], dtype=INT_TYPE)
        h_bound_F = np.array(plydata["face"]["h"], dtype=INT_TYPE)
        h_right_B = np.array(plydata["boundary"]["h"], dtype=INT_TYPE)
        samples = (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        return samples

    def he_samples_to_ply_data(self, use_binary=True):
        """Constructs a PlyData object using the schema"""

        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = self.he_samples
        V_data = np.array(
            [(x, y, z, h) for (x, y, z), h in zip(xyz_coord_V, h_out_V)],
            dtype=[
                ("x", FLOAT_TYPE),
                ("y", FLOAT_TYPE),
                ("z", FLOAT_TYPE),
                ("h", INT_TYPE),
            ],
        )
        H_data = np.array(
            [
                (v, n, t, f)
                for v, n, t, f in zip(v_origin_H, h_next_H, h_twin_H, f_left_H)
            ],
            dtype=[
                ("v", INT_TYPE),
                ("n", INT_TYPE),
                ("t", INT_TYPE),
                ("f", INT_TYPE),
            ],
        )
        F_data = np.array(h_bound_F, dtype=[("h", INT_TYPE)])
        B_data = np.array(h_right_B, dtype=[("h", INT_TYPE)])

        vertex_element = PlyElement.describe(V_data, "vertex")
        half_edge_element = PlyElement.describe(H_data, "half_edge")
        face_element = PlyElement.describe(F_data, "face")
        boundary_element = PlyElement.describe(B_data, "boundary")
        return PlyData(
            [vertex_element, half_edge_element, face_element, boundary_element],
            text=not use_binary,
        )

    def write_vf_ply(self, ply_path, use_binary=True):
        self.vf_ply_data.text = not use_binary
        self.vf_ply_data.write(ply_path)

    def write_he_ply(self, ply_path, use_binary=True):
        self.he_ply_data.text = not use_binary
        self.he_ply_data.write(ply_path)

    def write_he_samples(
        self, path=None, compressed=False, chunk=False, remove_unchunked=False
    ):

        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = self.he_samples
        arr = {
            "xyz_coord_V": xyz_coord_V,
            "h_out_V": h_out_V,
            "v_origin_H": v_origin_H,
            "h_next_H": h_next_H,
            "h_twin_H": h_twin_H,
            "f_left_H": f_left_H,
            "h_bound_F": h_bound_F,
            "h_right_B": h_right_B,
        }
        save_npz(
            arr,
            path,
            compressed=compressed,
            chunk=chunk,
            remove_unchunked=remove_unchunked,
        )

    ###############################################################
    # for dealing with old data that doesn't include h_right_B
    def no_boundary_he_ply_data_to_samples(self):
        """Constructs a lists of data from a PlyData object using the schema
        xyz_coord_V : list of numpy.array
            xyz_coord_V[i] = xyz coordinates of vertex i

        h_out_V : list of int
            h_out_V[i] = some outgoing half-edge incident on vertex i
        v_origin_H : list of int
            v_origin_H[j] = vertex at the origin of half-edge j
        h_next_H : list of int
            h_next_H[j] next half-edge after half-edge j in the face cycle
        h_twin_H : list of int
            h_twin_H[j] = half-edge antiparalel to half-edge j
        f_left_H : list of int
            f_left_H[j] = face to the left of half-edge j
            f_left_H[j] = -(b+1) if half-edge j is contained in boundary b and the complement of the mesh is left of j
        h_bound_F : list of int
            h_bound_F[k] = some half-edge on the boudary of face k
        h_right_B : list of int
            h_right_B[b] = some half-edge in the boundary b which is right of the complement of the mesh
        """
        plydata = self.he_ply_data
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=FLOAT_TYPE,
        ).T
        h_out_V = np.array(plydata["vertex"]["h"], dtype=INT_TYPE)
        v_origin_H = np.array(plydata["half_edge"]["v"], dtype=INT_TYPE)
        h_next_H = np.array(plydata["half_edge"]["n"], dtype=INT_TYPE)
        h_twin_H = np.array(plydata["half_edge"]["t"], dtype=INT_TYPE)
        f_left_H = np.array(plydata["half_edge"]["f"], dtype=INT_TYPE)
        h_bound_F = np.array(plydata["face"]["h"], dtype=INT_TYPE)
        h_right_B = find_h_right_B(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
        samples = (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        return samples

    @classmethod
    def from_no_boundary_he_ply(cls, ply_path, compute_vf_stuff=True):
        c = cls()

        c.he_ply_path = ply_path
        c.he_ply_data = PlyData.read(ply_path)
        c.he_samples = c.no_boundary_he_ply_data_to_samples()
        c.he_ply_data = c.he_samples_to_ply_data()
        if compute_vf_stuff:
            c.vf_samples = c.he_samples_to_vf_samples()
            c.vf_ply_data = c.vf_samples_to_ply_data()

        return c

    @classmethod
    def update_no_boundary_he_plys(cls):
        old_ply_dir = "./data/ply/binary"
        new_ply_dir = "./data/half_edge_base/ply"
        misc = ["annulus.ply", "hex_patch.ply", "hex_sector.ply"]
        neovius = ["neovius.ply", "neovius_coarse.ply", "neovius_fine.ply"]
        dumbbell = ["dumbbell.ply", "dumbbell_coarse.ply", "dumbbell_fine.ply"]
        torus = []  # [f"torus_{N:06d}_he.ply" for N in [192, 768, 3072, 12288, 49152]]
        unit_sphere = [
            f"unit_sphere_{N:06d}_he.ply"
            for N in [12, 42, 162, 642, 2562, 10242, 40962]
        ]
        old_plys = misc + neovius + dumbbell + torus + unit_sphere
        he_plys = misc + neovius + dumbbell
        he_plys = [_[:-4] + "_he.ply" for _ in he_plys]
        he_plys += torus + unit_sphere
        vf_plys = [_[:-7] + "_vf.ply" for _ in he_plys]
        for old_ply, he_ply, vf_ply in zip(old_plys, he_plys, vf_plys):
            old_ply_path = f"{old_ply_dir}/{old_ply}"
            he_ply_path = f"{new_ply_dir}/{he_ply}"
            vf_ply_path = f"{new_ply_dir}/{vf_ply}"
            c = cls.from_no_boundary_he_ply(old_ply_path, compute_vf_stuff=True)
            c.write_he_ply(he_ply_path)
            c.write_vf_ply(vf_ply_path)

    ###############################################################
    # To be deprecated
    @classmethod
    def read_ply_header(cls, ply_path):
        print(f"Reading {ply_path}")
        print(len(f"Reading {ply_path}") * "=")
        with open(ply_path, "r") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
                print(line.strip())
                if line.strip() == "end_header":
                    break
        return lines
