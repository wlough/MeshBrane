# import h5py
import logging
import os
import numpy as np
import pickle
import yaml
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_patch import HalfEdgePatch
from src.python.half_edge_base_viewer import MeshViewer


def unit_bump(s):
    val = np.zeros_like(s)
    I = np.abs(s) < 1.0
    val[I] = np.exp(1 + -1 / (1 - s[I] ** 2))
    return val


def bump3(xyz, center, radius):
    s = np.linalg.norm(xyz - center, axis=-1) / radius
    return unit_bump(s)


class SpbForce:
    def __init__(self, envelope, magnitude, radius):
        self.magnitude = magnitude
        self.radius = radius
        self.envelope = envelope
        self.find_contact_forces()

    def find_contact_forces(self):
        envelope = self.envelope
        radius = self.radius
        x = envelope.xyz_coord_V[:, 0]
        xmin, xmax = np.min(x), np.max(x)
        seed_plus = np.where(x == xmax)[0][0]
        seed_minus = np.where(x == xmin)[0][0]
        patch_plus = HalfEdgePatch.from_seed_to_radius(seed_plus, envelope, radius)
        patch_minus = HalfEdgePatch.from_seed_to_radius(seed_minus, envelope, radius)

        V_plus = np.array(sorted(patch_plus.V))
        center_plus = envelope.xyz_coord_v(seed_plus)
        # f_plus = np.array(
        #     [bump3(envelope.xyz_coord_v(i), center_plus, radius) for i in V_plus]
        # )
        f_plus = bump3(envelope.xyz_coord_V[V_plus], center_plus, radius)

        f_plus *= self.magnitude / np.sum(f_plus)
        F_plus = np.einsum("i,j->ij", f_plus, [1, 0, 0])

        V_minus = np.array(sorted(patch_minus.V))
        center_minus = envelope.xyz_coord_v(seed_minus)
        f_minus = np.array(
            [bump3(envelope.xyz_coord_v(i), center_minus, radius) for i in V_minus]
        )
        f_minus *= self.magnitude / np.sum(f_minus)
        F_minus = np.einsum("i,j->ij", f_minus, [-1, 0, 0])

        self.seed_plus = seed_plus
        self.seed_minus = seed_minus
        self.patch_plus = patch_plus
        self.patch_minus = patch_minus
        self.V_plus = V_plus
        self.V_minus = V_minus
        self.F_plus = F_plus
        self.F_minus = F_minus

    def points_vecs(self):
        vecs = np.array([*self.F_plus, *self.F_minus])
        points = self.envelope.xyz_coord_V[np.array([*self.V_plus, *self.V_minus])]
        return points, vecs


class StretchSim:
    """
    Simulation of membrane being stretched by equal and opposite forces
    """

    def __init__(
        self, output_dir, run_name, T, dt, dt_record_data, dt_write_data, dt_checkpoint
    ):
        self.output_dir = output_dir  # "./output/stretch_sim"
        self.run_name = run_name
        self.T = T
        self.dt = dt
        self.dt_record_data = dt_record_data
        self.dt_write_data = dt_write_data
        self.dt_checkpoint = dt_checkpoint

        self.data_path = os.path.join(self.output_dir, "data", f"{self.run_name}.h5")
        self.checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"{self.run_name}.pkl"
        )
        self.log_path = os.path.join(self.output_dir, "logs", f"{self.run_name}.log")
        self.parameters_path = os.path.join(
            self.output_dir, "input", f"{self.run_name}.yaml"
        )

        self.input_dir = os.path.join(self.output_dir, "input")
        self.raw_dir = os.path.join(self.output_dir, "raw")
        self.processed_dir = os.path.join(self.output_dir, "processed")
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.temp_images_dir = os.path.join(self.output_dir, "temp_images")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_path), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized simulation with parameters: {self.__dict__}")

    @classmethod
    def from_parameters_file(cls, file_path, output_dir):

        with open(file_path, "r") as f:
            parameters = yaml.safe_load(f)
        envelope_params = parameters.pop("envelope")
        spb_force_params = parameters.pop("spb_force")
        mesh_viewer_params = parameters.pop("mesh_viewer")
        s = cls(output_dir, **parameters)
        os.system(f"cp {file_path} {s.parameters_path}")
        if "ply_path" in envelope_params:
            ply_path = envelope_params.pop("ply_path")
            s.logger.info(f"ply_path found: {ply_path}")
            m = Brane.from_he_ply(ply_path, **envelope_params)
        else:
            # raise ValueError("ply_path must be specified in envelope parameters")
            s.logger.warning("ply_path not found in envelope parameters")
        s.envelope = m
        mesh_viewer_params["figsize"] = (720, 720)
        mesh_viewer_params["image_dir"] = s.temp_images_dir
        s.mesh_viewer = MeshViewer(m, **mesh_viewer_params)
        s.logger.info(f"Initialized envelope with parameters: {envelope_params}")
        s.spb_force = SpbForce(m, **spb_force_params)
        s.logger.info(f"Initialized SPB force with parameters: {spb_force_params}")

        Fcolor = s.mesh_viewer.colors["purple50"]
        Findices = np.array(list(s.spb_force.patch_plus.F | s.spb_force.patch_minus.F))
        s.mesh_viewer.update_rgba_F(Fcolor, indices=Findices)
        # points, vectors = s.spb_force.points_vecs()
        # s.mesh_viewer.clear_vector_field_data()
        # s.mesh_viewer.add_vector_field(points, vectors)

        return s

    @staticmethod
    def make_output_dir(output_dir="./output/stretch_sim", overwrite=False):
        """
        Create sim directories
        """

        sub_dirs = [
            "input",
            "raw",
            "processed",
            "logs",
            "checkpoints",
            "temp_images",
            "visualizations",
        ]
        if os.path.exists(output_dir) and overwrite:
            os.system(f"rm -r {output_dir}")
        elif not os.path.exists(output_dir):
            pass
        else:
            raise ValueError(
                f"{output_dir} already exists. Choose a different output_dir, or set overwrite=True"
            )
        os.system(f"mkdir -p {output_dir}")
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(output_dir, sub_dir)
            os.system(f"mkdir -p {sub_dir_path}")

    def time_step(self):
        dt = self.dt
        m = self.envelope
        spb = self.spb_force
        num_flips = m.flip_non_delaunay()
        xyz_coord_V0 = m.xyz_coord_V.copy()
        Fa = m.Farea_harmonic()
        F = Fa
        Fv = m.Fvolume_harmonic()
        F += Fv
        # Ft = m.Ftether()
        # F += Ft
        Fb = m.Fbend_analytic()
        F += Fb
        F[spb.V_plus] += spb.F_plus
        F[spb.V_minus] += spb.F_minus
        Dxyz_coord_V = dt * F / m.linear_drag_coeff
        xyz_coord_V = xyz_coord_V0 + Dxyz_coord_V
        m.xyz_coord_V = xyz_coord_V

    def run(self):
        mv = self.mesh_viewer
        # mv.view = None
        # print(mv.view)
        # mv.view.pop("distance")
        m = self.envelope
        spb = self.spb_force
        spb_num_verts = len(spb.V_plus) + len(spb.V_minus)
        spb_scale = spb_num_verts * spb.radius / spb.magnitude
        t = 0
        dt = self.dt
        T = self.T
        points, vectors = spb.points_vecs()
        # mag = spb.magnitude
        # scale = 320 / mag

        mv.clear_vector_field_data()
        mv.add_vector_field(points, spb_scale * vectors)
        mv.plot(save=True, show=False, title=f"{t=}")
        while t <= T:
            self.time_step()
            points, vectors = spb.points_vecs()
            # mag = spb.magnitude
            # rad = spb.radius
            # spb_scale = spb.radius / spb.magnitude
            # Aspb = np.pi*rad**2
            # Fave = mag/Aspb
            # scale = 320 / mag
            # scale = 1

            mv.clear_vector_field_data()
            mv.add_vector_field(points, spb_scale * vectors)
            t += dt
            print(f"{t=}                ", end="\r")
            mv.plot(save=True, show=False, title=f"{t=}")
        mv.movie()

    @classmethod
    def load_checkpoint(cls, run_num):
        pass

    def save_checkpoint(self):
        chkpt = os.path.join(self.checkpoints_dir, run_name)
