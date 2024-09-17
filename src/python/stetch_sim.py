import h5py
import logging
import os
import numpy as np
import pickle
import yaml
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_patch import HalfEdgePatch
from src.python.half_edge_base_viewer import MeshViewer


class ParamManager:
    def __init__(self, **parameters):
        from copy import deepcopy

        self.parameters = deepcopy(parameters)

        # for key, val in parameters.items():
        #     setattr(self, key, val)
        self.envelope_parameters = self.parameters.get("envelope_parameters", {})
        self.spb_force_parameters = self.parameters.get("spb_force_parameters", {})
        self.mesh_viewer_parameters = self.parameters.get("mesh_viewer_parameters", {})
        self.sim_parameters = {
            k: v
            for k, v in self.parameters.items()
            if k not in ["envelope", "spb_force", "mesh_viewer"]
        }

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def __delitem__(self, key):
        del self.parameters[key]

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    def __str__(self):
        return str(self.parameters)

    def __repr__(self):
        return repr(self.parameters)

    def save_yaml(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self.parameters, f, sort_keys=False)

    @classmethod
    def default_sphere(cls, output_dir="./output/stretch_sim_unit_sphere"):
        length_unit = 1.0
        energy_unit = 6.25e-3  # .2/32
        time_unit = 1e5
        sim_params = {
            "output_dir": output_dir,
            "run_name": "run_0000",
            "T": 10.0,
            "dt": 0.01,
            "dt_record_data": 0.1,
            "dt_write_data": 0.5,
            "dt_checkpoint": 2.0,
            "dx_max": 0.1,
            "make_figs": True,
        }

        envelope_params = {
            "ply_path": "./data/half_edge_base/ply/unit_sphere_001280_he.ply",
            "spontaneous_curvature": 0.0 / length_unit,
            "bending_modulus": 20.0 * energy_unit,
            "splay_modulus": 0.0 * energy_unit,
            #
            "volume_reg_stiffness": 1.6e7 * energy_unit / length_unit**3,
            "area_reg_stiffness": 6.43e6 * energy_unit / length_unit**2,
            #
            "tether_stiffness": 80.0 * energy_unit,
            "tether_repulsive_onset": 0.8,
            "tether_repulsive_singularity": 0.6,
            "tether_attractive_onset": 1.2,
            "tether_attractive_singularity": 1.4,
            #
            "drag_coefficient": 0.4 * energy_unit * time_unit / length_unit**2,
            #
            "flipping_frequency": 1e6 / time_unit,
            "flipping_probability": 0.3,
        }

        spb_force_params = {
            "force_total": 320.0,
            "contact_radius": 0.3,
            "view_scale": 1.0,
            "force_profile": "bump",
            "contact_rgba": [0.5804, 0.0, 0.8275, 0.5],
            "force_rgba": [0.0, 0.0, 0.0, 0.9],
            "show_contact": True,
            "show_force": True,
        }

        mesh_viewer_params = {
            "figsize": [720, 720],
            "image_dir": f"{output_dir}/temp_images",
            "view": {
                "azimuth": 90.0,
                "elevation": 55.0,
                "distance": 6.75,
                "focalpoint": [0.0, 0.0, 0.0],
            },
        }
        sim_params["envelope_parameters"] = envelope_params
        sim_params["spb_force_parameters"] = spb_force_params
        sim_params["mesh_viewer_parameters"] = mesh_viewer_params
        return cls(**sim_params)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, "r") as f:
            parameters = yaml.safe_load(f)
        return cls(**parameters)


class SpbForce:
    def __init__(
        self,
        envelope,
        force_total,
        force_profile,
        contact_radius,
        show_contact=True,
        show_force=True,
        view_scale=0.3,
        contact_rgba=(0.5804, 0.0, 0.8275, 0.5),
        force_rgba=(0.0, 0.0, 0.0, 0.9),
    ):
        self.envelope = envelope
        self.force_total = force_total
        self.force_profile = force_profile
        self.contact_radius = contact_radius
        self.show_contact = show_contact
        self.show_force = show_force
        self.view_scale = view_scale
        self.contact_rgba = contact_rgba
        self.force_rgba = force_rgba

        self.find_contact_patches()
        self.find_contact_forces()

    def _bump3(self, xyz, center, contact_radius, force_total):
        s = np.linalg.norm(xyz - center, axis=-1) / contact_radius
        val = np.zeros_like(s)
        I = np.abs(s) < 1.0
        val[I] = np.exp(1 + -1 / (1 - s[I] ** 2))
        val *= force_total / np.sum(val)
        return val

    def _uniform3(self, xyz, center, contact_radius, force_total):
        s = np.linalg.norm(xyz - center, axis=-1) / contact_radius
        val = np.zeros_like(s)
        I = np.abs(s) < 1.0
        val[I] = 1.0
        val *= force_total / np.sum(val)
        return val

    def find_contact_patches(self):
        envelope = self.envelope
        contact_radius = self.contact_radius
        x_coord_V = envelope.xyz_coord_V[:, 0]
        xmin, xmax = np.min(x_coord_V), np.max(x_coord_V)
        seed_plus = np.where(x_coord_V == xmax)[0][0]
        seed_minus = np.where(x_coord_V == xmin)[0][0]
        patch_plus = HalfEdgePatch.from_seed_to_radius(
            seed_plus, envelope, contact_radius
        )
        patch_minus = HalfEdgePatch.from_seed_to_radius(
            seed_minus, envelope, contact_radius
        )
        V_plus = np.array(sorted(patch_plus.V))
        V_minus = np.array(sorted(patch_minus.V))
        V_contact = np.concatenate([V_plus, V_minus])
        F_plus = np.array(sorted(patch_plus.F))
        F_minus = np.array(sorted(patch_minus.F))
        F_contact = np.concatenate([F_plus, F_minus])

        self.seed_plus = seed_plus
        self.seed_minus = seed_minus
        self.patch_plus = patch_plus
        self.patch_minus = patch_minus
        self.V_plus = V_plus
        self.V_minus = V_minus
        self.V_contact = V_contact
        self.F_contact = F_contact

    def update_contact_patches(self):
        self.patch_plus.update_H_and_F_from_V()
        self.patch_minus.update_H_and_F_from_V()
        self.F_plus = np.array(sorted(self.patch_plus.F))
        self.F_minus = np.array(sorted(self.patch_minus.F))
        self.F_contact = np.concatenate([self.F_plus, self.F_minus])

    def find_contact_forces(self):

        force_profile = self.force_profile
        envelope = self.envelope
        contact_radius = self.contact_radius
        force_total = self.force_total
        view_scale = self.view_scale

        seed_plus = self.seed_plus
        seed_minus = self.seed_minus
        V_plus = self.V_plus
        V_minus = self.V_minus

        center_plus = envelope.xyz_coord_v(seed_plus)
        xyz_coord_V_plus = envelope.xyz_coord_V[V_plus]

        center_minus = envelope.xyz_coord_v(seed_minus)
        xyz_coord_V_minus = envelope.xyz_coord_V[V_minus]

        if force_profile == "bump":
            magnitude_plus = self._bump3(
                xyz_coord_V_plus, center_plus, contact_radius, force_total
            )
            magnitude_minus = self._bump3(
                xyz_coord_V_minus, center_minus, contact_radius, force_total
            )
        if force_profile == "uniform":
            magnitude_plus = self._uniform3(
                xyz_coord_V_plus, center_plus, contact_radius, force_total
            )
            magnitude_minus = self._uniform3(
                xyz_coord_V_minus, center_minus, contact_radius, force_total
            )

        max_magnitude = np.max([magnitude_plus, magnitude_minus])

        force_plus = np.einsum("i,j->ij", magnitude_plus, [1, 0, 0])
        force_minus = np.einsum("i,j->ij", magnitude_minus, [-1, 0, 0])
        force_contact = np.vstack([force_plus, force_minus])

        viewer_force_vectors = view_scale * force_contact / max_magnitude

        self.magnitude_plus = magnitude_plus
        self.magnitude_minus = magnitude_minus
        self.max_magnitude = max_magnitude

        self.force_plus = force_plus
        self.force_minus = force_minus
        self.force_contact = force_contact

        self.viewer_force_vectors = viewer_force_vectors

    def viewer_vector_field_kwargs(self):
        points = self.envelope.xyz_coord_V[self.V_contact]
        vectors = self.viewer_force_vectors
        rgba = self.force_rgba
        return {"points": points, "vectors": vectors, "rgba": rgba, "name": "spb_force"}

    def viewer_update_rgba_V_contact_kwargs(self):
        value = self.contact_rgba
        indices = self.V_contact
        return {"value": value, "indices": indices}

    def viewer_update_rgba_F_contact_kwargs(self):
        value = self.contact_rgba
        indices = self.F_contact
        return {"value": value, "indices": indices}


class SimData:
    def __init__(
        self,
        data_path,
        num_vertices,
        num_half_edges,
        num_faces,
        num_boundaries,
        samples_per_chunk,
    ):
        self.data_path = data_path
        self.num_vertices = num_vertices
        self.num_half_edges = num_half_edges
        self.num_faces = num_faces
        self.samples_per_chunk = samples_per_chunk
        self.sample_num = 0
        self.he_datasets = [
            ["vertex/xyz_coord_V", (num_vertices, 3), "int32"],
            ["vertex/h_out_V", (num_vertices,), "int32"],
            ["half_edge/v_origin_H", (num_half_edges,), "int32"],
            ["half_edge/h_next_H", (num_half_edges,), "int32"],
            ["half_edge/h_twin_H", (num_half_edges,), "int32"],
            ["half_edge/f_left_H", (num_half_edges,), "int32"],
            ["face/h_bound_F", (num_faces,), "int32"],
            ["boundary/h_right_B", (num_boundaries,), "int32"],
        ]
        self.force_datasets = [
            ["vertex/bending_force", (num_vertices, 3), "float64"],
            ["vertex/area_force", (num_vertices, 3), "float64"],
            ["vertex/volume_force", (num_vertices, 3), "float64"],
            ["vertex/tether_force", (num_vertices, 3), "float64"],
            ["vertex/spb_force", (num_vertices, 3), "float64"],
        ]

        with h5py.File(self.data_path, "w") as data_file:
            scalar_group = data_file.require_group("scalar")
            vertex_group = data_file.require_group("vertex")
            half_edge_group = data_file.require_group("half_edge")
            face_group = data_file.require_group("face")
            boundary_group = data_file.require_group("boundary")
            #
        self.scalar_chunk = dict()
        self.vertex_chunk = dict()
        self.half_edge_chunk = dict()
        self.face_chunk = dict()
        self.boundary_chunk = dict()

        self.key_dict = {
            group_key: dict()
            for group_key in ["scalar", "vertex", "half_edge", "face", "boundary"]
        }

    def define_he_datasets(self):

        ########################################################
        ########################################################
        scalar_stuff = [
            ["time", (), "float64"],
        ]
        for key, shape, dtype in scalar_stuff:
            self.define_scalar_data(key, dtype)

        ########################################################
        ########################################################
        # Half edge mesh samples
        vertex_stuff = [
            ["xyz_coord_V", (num_vertices, 3), "float64"],
            ["h_out_V", (num_vertices,), "int32"],
        ]
        for key, shape, dtype in vertex_stuff:
            self.define_vertex_data(key, shape, dtype)
        half_edge_stuff = [
            ["v_origin_H", (num_half_edges,), "int32"],
            ["h_next_H", (num_half_edges,), "int32"],
            ["h_twin_H", (num_half_edges,), "int32"],
            ["f_left_H", (num_half_edges,), "int32"],
        ]
        for key, shape, dtype in half_edge_stuff:
            self.define_half_edge_data(key, shape, dtype)

        face_stuff = [["h_bound_F", (num_faces,), "int32"]]
        for key, shape, dtype in face_stuff:
            self.define_face_data(key, shape, dtype)

        boundary_stuff = [["h_right_B", (num_boundaries,), "int32"]]
        if num_boundaries > 0:
            for key, shape, dtype in boundary_stuff:
                self.define_boundary_data(key, shape, dtype)

        ########################################################
        ########################################################

    @classmethod
    def load(cls, data_path):
        with h5py.File(data_path, "r") as f:
            num_vertices = f["vertex/xyz_coord_V"].shape[1]
            num_half_edges = f["half_edge/v_origin_H"].shape[1]
            num_faces = f["face/h_bound_F"].shape[1]
            num_boundaries = f["boundary/h_right_B"].shape[1]
            samples_per_chunk = f["vertex/xyz_coord_V"].chunks[0]
        self = cls(
            data_path,
            num_vertices,
            num_half_edges,
            num_faces,
            num_boundaries,
            samples_per_chunk,
        )

    def define_scalar_dataset(self, key, dtype):
        shape = ()
        with h5py.File(self.data_path, "a") as data_file:
            scalar_group = data_file["scalar"]
            scalar_group.require_dataset(
                key,
                (0,) + shape,
                maxshape=(None,) + shape,
                chunks=(self.samples_per_chunk,) + shape,
                dtype=dtype,
            )
        self.scalar_chunk[key] = np.zeros(
            (self.samples_per_chunk,) + shape, dtype=dtype
        )

    def define_vertex_dataset(self, key, shape, dtype):
        with h5py.File(self.data_path, "a") as data_file:
            vertex_group = data_file["vertex"]
            vertex_group.require_dataset(
                key,
                (0,) + shape,
                maxshape=(None,) + shape,
                chunks=(self.samples_per_chunk,) + shape,
                dtype=dtype,
            )
        self.vertex_chunk[key] = np.zeros(
            (self.samples_per_chunk,) + shape, dtype=dtype
        )

    def define_half_edge_dataset(self, key, shape, dtype):
        with h5py.File(self.data_path, "a") as data_file:
            half_edge_group = data_file["half_edge"]
            half_edge_group.require_dataset(
                key,
                (0,) + shape,
                maxshape=(None,) + shape,
                chunks=(self.samples_per_chunk,) + shape,
                dtype=dtype,
            )
        self.half_edge_chunk[key] = np.zeros(
            (self.samples_per_chunk,) + shape, dtype=dtype
        )

    def define_face_dataset(self, key, shape, dtype):
        with h5py.File(self.data_path, "a") as data_file:
            face_group = data_file["face"]
            face_group.require_dataset(
                key,
                (0,) + shape,
                maxshape=(None,) + shape,
                chunks=(self.samples_per_chunk,) + shape,
                dtype=dtype,
            )
        self.face_chunk[key] = np.zeros((self.samples_per_chunk,) + shape, dtype=dtype)

    def define_boundary_dataset(self, key, shape, dtype):
        with h5py.File(self.data_path, "a") as data_file:
            boundary_group = data_file["boundary"]
            boundary_group.require_dataset(
                key,
                (0,) + shape,
                maxshape=(None,) + shape,
                chunks=(self.samples_per_chunk,) + shape,
                dtype=dtype,
            )
        self.boundary_chunk[key] = np.zeros(
            (self.samples_per_chunk,) + shape, dtype=dtype
        )

    def define_dataset(group_key, dataset_key, shape, dtype):
        if group_key not in self.key_dict:
            self.key_dict[group_key] = dict()
            self.key_dict[group_key][dataset_key] = (shape, dtype)

        if group_key == "scalar":
            self.define_scalar_dataset(dataset_key, dtype)
        if group_key == "vertex":
            self.define_vertex_dataset(dataset_key, shape, dtype)
        if group_key == "half_edge":
            self.define_half_edge_dataset(dataset_key, shape, dtype)
        if group_key == "face":
            self.define_face_dataset(dataset_key, shape, dtype)
        if group_key == "boundary":
            self.define_boundary_dataset(dataset_key, shape, dtype)

    def record_scalar_samples(self, dataset_key, samples):
        self.scalar_chunk[dataset_key][self.sample_num] = samples

    def record_vertex_samples(self, dataset_key, samples):
        self.vertex_chunk[dataset_key][self.sample_num] = samples

    def record_half_edge_samples(self, dataset_key, samples):
        self.half_edge_chunk[dataset_key][self.sample_num] = samples

    def record_face_samples(self, dataset_key, samples):
        self.face_chunk[dataset_key][self.sample_num] = samples

    def record_boundary_samples(self, dataset_key, samples):
        self.boundary_chunk[dataset_key][self.sample_num] = samples

    def record_samples(self, group_key, dataset_key, samples):
        if group_key == "scalar":
            self.record_scalar_samples(dataset_key, samples)
        if group_key == "vertex":
            self.record_vertex_samples(dataset_key, samples)
        if group_key == "half_edge":
            self.record_half_edge_samples(dataset_key, samples)
        if group_key == "face":
            self.record_face_samples(dataset_key, samples)

    def save_chunks(self, group_key, dataset_keys):
        if group_key == "scalar":
            for key in dataset_keys:
                self.save_group(group_key, key=self.scalar_chunk[key])
        if group_key == "vertex":
            for key in dataset_keys:
                self.save_group(group_key, key=self.vertex_chunk[key])
        if group_key == "half_edge":
            for key in dataset_keys:
                self.save_group(group_key, key=self.half_edge_chunk[key])
        if group_key == "face":
            for key in dataset_keys:
                self.save_group(group_key, key=self.face_chunk[key])
        # if group_key == "boundary":
        #     for key in dataset_keys:
        #         self.save_group(group_key, key=self.boundary_chunk

    def save_group(self, group_key, **chunks):
        with h5py.File(self.data_path, "a") as f:
            for key, arr in chunks.items():
                self.append_to_dataset(f[f"{group}/{key}"], arr)

    def append_to_dataset(self, dataset, datachunk):
        # Get the current shape of the dataset
        shape_dataset_current = dataset.shape
        shape_datachunk = datachunk.shape
        num_samples_current = shape_dataset_current[0]
        samples_in_chunk = shape_datachunk[0]

        shape_dataset_new = (
            num_samples_current + samples_in_chunk,
        ) + shape_dataset_current[1:]

        # Resize the dataset to accommodate the new data
        dataset.resize(shape_dataset_new)

        # Append the new data
        dataset[num_samples_current:] = datachunk

    # def record_data(self):
    #     with h5py.File(self.data_path, "a") as f:
    #         t = self.t
    #         xyz_coord_V = self.envelope.xyz_coord_V
    #         F = self.envelope.F
    #         Fa = self.envelope.Farea_harmonic()
    #         Fv = self.envelope.Fvolume_harmonic()
    #         Ft, tether_success = self.envelope.Ftether()
    #         Fb = self.envelope.Fbend_analytic()
    #         F_contact = self.spb_force.force_contact
    #         force_plus = self.spb_force.force_plus
    #         force_minus = self.spb_force.force_minus
    #         viewer_force_vectors = self.spb_force.viewer_force_vectors
    #         magnitude_plus = self.spb_force.magnitude_plus
    #         magnitude_minus = self.spb_force.magnitude_minus
    #         max_magnitude = self.spb_force.max_magnitude


class StretchSim:
    """
    Simulation of membrane being stretched by equal and opposite forces

    Parameters
    ----------
    output_dir : str
        Path to output directory
    run_name : str
        Name of simulation run
    T : float
        Total simulation time to run simulation
    dt : float
        Time step for simulation
    dt_record_data : float
        Time step for recording data
    dt_write_data : float
        Time step for writing data
    dt_checkpoint : float
        Time step for saving checkpoint
    dx_max : float
        Maximum vertex displacement for a single time step as a fraction of the preferred edge length
    make_figs : bool
        Whether to make figures during simulation
    envelope_parameters : dict
        Parameters for envelope
    spb_force_parameters : dict
        Parameters for spb force
    mesh_viewer_parameters : dict
        Parameters for mesh viewer
    make_output_dir : bool
        Whether to create output directory
    overwrite_output_dir : bool
        Whether to overwrite existing output directory if it exists
    """

    def __init__(
        self,
        output_dir,
        run_name,
        T,
        dt,
        dt_record_data,
        dt_write_data,
        dt_checkpoint,
        dx_max,
        make_figs,
        envelope_parameters,
        spb_force_parameters,
        mesh_viewer_parameters,
        make_output_dir=True,
        overwrite_output_dir=False,
    ):
        if make_output_dir:
            self.make_output_dir(output_dir, overwrite_output_dir)
        self.output_dir = output_dir  # "./output/stretch_sim"
        self.run_name = run_name
        self.T = T
        self.dt = dt
        self.dt_record_data = dt_record_data
        self.dt_write_data = dt_write_data
        self.dt_checkpoint = dt_checkpoint
        self.dx_max = dx_max
        self.make_figs = make_figs

        self.envelope_parameters = envelope_parameters
        self.spb_force_parameters = spb_force_parameters
        self.mesh_viewer_parameters = mesh_viewer_parameters

        self.data_path = os.path.join(self.output_dir, "raw", f"{self.run_name}.hdf5")
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

        self.envelope = Brane.load(**envelope_parameters)
        self.spb_force = SpbForce(self.envelope, **spb_force_parameters)
        if "image_dir" not in mesh_viewer_parameters:
            mesh_viewer_parameters["image_dir"] = self.temp_images_dir
        self.mesh_viewer = MeshViewer(self.envelope, **mesh_viewer_parameters)
        ############################
        self.mesh_viewer.add_vector_field(**self.spb_force.viewer_vector_field_kwargs())
        self.mesh_viewer.update_rgba_V(
            **self.spb_force.viewer_update_rgba_V_contact_kwargs()
        )
        self.mesh_viewer.update_rgba_F(
            **self.spb_force.viewer_update_rgba_F_contact_kwargs()
        )
        ############################
        sim_data_params = dict()
        sim_data_params["data_path"] = self.data_path
        sim_data_params["num_vertices"] = self.envelope.num_vertices
        sim_data_params["num_half_edges"] = self.envelope.num_half_edges
        sim_data_params["num_faces"] = self.envelope.num_faces
        sim_data_params["num_boundaries"] = self.envelope.num_boundaries
        sim_data_params["samples_per_chunk"] = int(dt_write_data / dt_record_data)
        self.sim_data = SimData(**sim_data_params)
        he_keys_V = ["xyz_coord_V", "h_out_V"]
        he_keys_H = ["v_origin_H", "h_next_H", "h_twin_H", "f_left_H"]
        he_keys_F = ["h_bound_F"]
        he_keys_B = ["h_right_B"]
        scalar_samples = {"time": 0.0}
        vertex_samples = {
            "xyz_coord_V": self.envelope.xyz_coord_V,
            "h_out_V": self.envelope.h_out_V,
        }
        half_edge_samples = {
            "v_origin_H": self.envelope.v_origin_H,
            "h_next_H": self.envelope.h_next_H,
            "h_twin_H": self.envelope.h_twin_H,
            "f_left_H": self.envelope.f_left_H,
        }
        face_samples = {"h_bound_F": self.envelope.h_bound_F}

        ############################

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_path), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized simulation with parameters: {self.__dict__}")

    def update_viewer_vector_field(self):
        self.mesh_viewer.clear_vector_field_data()
        self.mesh_viewer.add_vector_field(**self.spb_force.viewer_vector_field_kwargs())

    @classmethod
    def from_parameters_file(cls, file_path, overwrite_output_dir=False):
        with open(file_path, "r") as f:
            parameters = yaml.safe_load(f)
        self = cls(**parameters, overwrite_output_dir=overwrite_output_dir)
        os.system(f"cp {file_path} {self.parameters_path}")

        return self

    @staticmethod
    def make_output_dir(output_dir="./output/stretch_sim", overwrite_output_dir=False):
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
        if os.path.exists(output_dir) and overwrite_output_dir:
            os.system(f"rm -r {output_dir}")
        elif not os.path.exists(output_dir):
            pass
        else:
            # raise ValueError(
            #     f"{output_dir} already exists. Choose a different output_dir, or set overwrite_output_dir=True"
            # )
            print(f"{output_dir} already exists.")
            return
        os.system(f"mkdir -p {output_dir}")
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(output_dir, sub_dir)
            os.system(f"mkdir -p {sub_dir_path}")

    def euler_step(self, dt, max_normDxyz):
        m = self.envelope
        spb = self.spb_force
        xyz_coord_V0 = m.xyz_coord_V.copy()
        Fa = m.Farea_harmonic()
        F = Fa
        Fv = m.Fvolume_harmonic()
        F += Fv
        Ft, tether_success = m.Ftether()

        F += Ft
        Fb = m.Fbend_analytic()
        F += Fb
        F[spb.V_plus] += spb.force_plus
        F[spb.V_minus] += spb.force_minus
        Dxyz = dt * F / m.drag_coefficient
        normDxyz = np.max(np.linalg.norm(Dxyz, axis=-1))
        Dxyz_good = normDxyz <= max_normDxyz
        # xyz_coord_V = xyz_coord_V0 + Dxyz
        # m.xyz_coord_V = xyz_coord_V
        success = tether_success and Dxyz_good
        return Dxyz, normDxyz, success, Dxyz_good, tether_success

    def bad_step(self, dt, max_normDxyz, Dxyz, normDxyz, Dxyz_good, tether_success):

        if tether_success:
            target_normDxyz = 0.25 * max_normDxyz
            dt_new = target_normDxyz / normDxyz * dt
            Dxyz *= dt_new / dt
            # normDxyz = np.max(np.linalg.norm(Dxyz, axis=-1))
            normDxyz = target_normDxyz
            Dxyz_good = normDxyz <= max_normDxyz

        success = tether_success and Dxyz_good
        return Dxyz, normDxyz, success, Dxyz_good, tether_success, dt_new

    def _run(self):
        m = self.envelope
        spb = self.spb_force
        mv = self.mesh_viewer
        max_normDxyz = self.dx_max * m.preferred_edge_length
        self.t = 0
        dt = self.dt
        T = self.T
        self.update_viewer_vector_field()
        mv.plot(save=True, show=False, title=f"t={self.t}")
        were_done_here = False
        while self.t <= T:
            if were_done_here:
                print("Failed at t={self.t}")
                print(f"{Dxyz_good=}")
                print(f"{tether_success=}")
                break
            num_flips = m.flip_non_delaunay()
            Dxyz, normDxyz, success, Dxyz_good, tether_success = self.euler_step(
                dt, max_normDxyz
            )
            if success:
                m.xyz_coord_V += Dxyz
                self.t += dt
                print(f"t={self.t}                ", end="\r")
                self.update_viewer_vector_field()
                mv.plot(save=True, show=False, title=f"t={self.t}")
            else:
                print("\ntrying smaller timestep...")

                Dxyz, normDxyz, success, Dxyz_good, tether_success, dt_new = (
                    self.bad_step(
                        dt, max_normDxyz, Dxyz, normDxyz, Dxyz_good, tether_success
                    )
                )
                print(f"dt={dt_new}")
                if success:
                    m.xyz_coord_V += Dxyz
                    self.t += dt_new
                    t_stop = self.t - dt_new + dt
                    while self.t <= t_stop:
                        Dxyz, normDxyz, success, Dxyz_good, tether_success = (
                            self.euler_step(dt_new, max_normDxyz)
                        )
                        if success:
                            m.xyz_coord_V += Dxyz
                            self.t += dt_new
                            print(f"t={self.t}                ", end="\r")
                        else:
                            were_done_here = True
                    print(f"t={self.t}                ", end="\r")
                    self.update_viewer_vector_field()
                    mv.plot(save=True, show=False, title=f"t={self.t}")
                else:
                    print("well, that didn't work...")
                    were_done_here = True
        mv.movie()

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
        F[spb.V_plus] += spb.force_plus
        F[spb.V_minus] += spb.force_minus
        Dxyz_coord_V = dt * F / m.linear_drag_coeff
        xyz_coord_V = xyz_coord_V0 + Dxyz_coord_V
        m.xyz_coord_V = xyz_coord_V

    @classmethod
    def load_checkpoint(cls, run_num):
        pass

    def save_checkpoint(self):
        chkpt = self.checkpoint_path
        with open(chkpt, "wb") as f:
            pickle.dump(self, f)

    def save_data(self):
        pass
