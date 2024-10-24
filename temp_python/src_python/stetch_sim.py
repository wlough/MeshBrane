# import h5py
import logging
import os
import numpy as np
import pickle
import yaml
from temp_python.src_python.half_edge_base_brane import Brane
from temp_python.src_python.half_edge_base_patch import HalfEdgePatch
from temp_python.src_python.half_edge_base_viewer import MeshViewer
from copy import deepcopy


class SPB:
    """
    Spindle pole body in contact with envelope.

    Parameters
    ----------
    envelope : Brane
        Envelope object
    contact_radius : float
        SPB radius
    axis_origin : np.ndarray
        3d coordinates of point on spindle axis
    axis_vec : np.ndarray
        3d vector directed along the spindle axis toward the SPB
    force_total : float
        Total force SPB exerts on the envelope
    force_profile : str
        Profile of force distribution. Options are "bump" and "uniform"
    visual_length : float
        Length of visual representation of SPB
    visual_max_force_magnitude : float
        Length of visual representation of max force vector
    contact_rgba : tuple
        RGBA color for contact patch
    force_rgba : tuple
        RGBA color for force vectors
    find_contact_data : bool
        Whether to find contact patches and forces during initialization

    Attributes
    ----------
    contact_patch : HalfEdgePatch
        Envelope patch in contact with SPB
    V_contact : np.ndarray
        Vertices in contact patch sorted by index
    contact_force_magnitude : np.ndarray
        Magnitude of force applied to each vertex in contact patch
    contact_force_vector : np.ndarray
        Force vector applied to each vertex in contact patch

    Methods
    -------
    find_contact_patch : HalfEdgePatch
        Find vertices in restrict_to_V with max and min distances along the spindle axis.
    sort_contact_vertices : np.ndarray
        Sort contact vertices by index.
    find_contact_force_magnitude : np.ndarray
        Find force magnitude for each vertex in contact patch.


    get_time_series_data:
        Return position and force data at current time.
    get_pretty_time_series_data:
        Return position and force data and MeshViewer params for patch and .
    viewer_add_vector_field_kwargs
        Return kwargs for adding contact force vector field to mesh viewer with MeshViewer.add_vector_field
    viewer_update_rgba_V_kwargs
        Return kwargs for updating contact patch colors in mesh viewer with MeshViewer.
    """

    def __init__(
        self,
        envelope,
        contact_radius,
        axis_origin,
        axis_vec,
        force_total,
        force_profile="bump",
        visual_length=0.1,
        # visual_max_force_magnitude=0.3,
        visual_force_scale=0.3,
        contact_rgba=(0.5804, 0.0, 0.8275, 0.5),
        force_rgba=(0.0, 0.0, 0.0, 0.9),
        name="spb",
        find_contact_data=True,
        show_contact=True,
        show_force=True,
    ):
        self.envelope = envelope
        self.contact_radius = contact_radius
        self.axis_origin = np.array(axis_origin)
        self.axis_vec = np.array(axis_vec) / np.linalg.norm(axis_vec)
        self.force_total = force_total
        self.force_profile = force_profile
        # self.max_search_vertices = max_search_vertices
        self.visual_length = visual_length
        # self.visual_max_force_magnitude = visual_max_force_magnitude
        self.visual_force_scale = visual_force_scale
        self.contact_rgba = np.array(contact_rgba)
        self.force_rgba = np.array(force_rgba)
        self.name = name
        self.show_contact = show_contact
        self.show_force = show_force

        if find_contact_data:
            self.contact_patch = self.find_contact_patch()
            self.V_contact = self.sort_contact_vertices()
            self.update(patch=False, pretty=True)
        else:
            self.contact_patch = None
            self.V_contact = None
            self.xyz_coord_V_contact = None
            self.contact_force_magnitude = None
            self.contact_force_vector = None
            self.contact_force_vector_pretty = None

    def get_xyz_seed_vertex(self):
        v_seed = self.contact_patch.seed_vertex
        return self.envelope.xyz_coord_v(v_seed)

    def get_xyz_contact_center(self):
        return self.envelope.xyz_coord_V[V_contact].mean(axis=0)

    def get_contact_force_vector(self):
        return self.axis_vec * self.contact_force_magnitude[:, np.newaxis]

    def get_contact_force_vector_pretty(self):
        # force_density = self.contact_force_magnitude / self.contact_radius**2
        scaled_axis_vec = (
            self.visual_force_scale
            * self.axis_vec
            # / (np.pi * self.contact_radius**2 * self.force_total)
            / self.force_total
        )
        return scaled_axis_vec * self.contact_force_magnitude[:, np.newaxis]

    def distance_from_axis(self, p):
        """
        Test if point p is contained in a cylinder or radius r_max with axis through p0 and direction ez.
        """
        dp = p - self.axis_origin
        dp_z = dp @ self.axis_vec
        r = np.sqrt(np.sum(dp**2, axis=-1) - dp_z**2)
        return r

    def find_contact_patch(self, restrict_to_V=None):
        """
        Find vertices in restrict_to_V with max and min distances along the spindle axis.
        If restrict_to_V is None, use all vertices in self.envelope.
        """
        if restrict_to_V is None:
            xyz_coord_V = self.envelope.xyz_coord_V - self.axis_origin
            z_coord_V = xyz_coord_V @ self.axis_vec
            seed_vertex = np.argmax(z_coord_V)
        else:
            xyz_coord_V = self.envelope.xyz_coord_V[restrict_to_V] - self.axis_origin
            z_coord_V = xyz_coord_V @ self.axis_vec
            seed_vertex = restrict_to_V[np.argmax(z_coord_V)]
        contact_patch = HalfEdgePatch.from_seed_to_cylinder(
            seed_vertex,
            self.envelope,
            self.axis_origin,
            self.contact_radius,
            self.axis_vec,
        )
        return contact_patch

    def sort_contact_vertices(self):
        return np.array(sorted(self.contact_patch.V), dtype="int32")

    def bump_contact_force_magnitude(self):
        V_contact = self.V_contact
        xyz_coord_V = self.envelope.xyz_coord_V[V_contact]
        s = self.distance_from_axis(xyz_coord_V) / self.contact_radius
        contact_force_magnitude = np.zeros_like(s)
        I = np.abs(s) < 1.0
        contact_force_magnitude[I] = np.exp(1 + -1 / (1 - s[I] ** 2))
        contact_force_magnitude *= self.force_total / np.sum(contact_force_magnitude)
        return contact_force_magnitude

    def uniform_contact_force_magnitude(self):
        # V_contact = self.V_contact
        num_vertices = len(self.contact_patch.V)
        contact_force_magnitude = np.zeros(num_vertices)
        contact_force_magnitude += self.force_total / num_vertices
        return contact_force_magnitude

    def find_contact_force_magnitude(self):
        if self.force_profile == "bump":
            return self.bump_contact_force_magnitude()
        if self.force_profile == "uniform":
            return self.uniform_contact_force_magnitude()

    def update_patch(self):
        dNf, dNh, dNv = 1, 1, 1
        while dNf != 0:
            dNf, dNh, dNv = self.contact_patch.move_towards_cylinder(
                self.axis_origin, self.contact_radius, self.axis_vec
            )
        self.V_contact = self.sort_contact_vertices()

    def update_force(self):
        self.contact_force_magnitude = self.find_contact_force_magnitude()
        self.contact_force_vector = self.get_contact_force_vector()

    def update_force_pretty(self):
        self.xyz_coord_V_contact = self.envelope.xyz_coord_V[self.V_contact]
        self.contact_force_vector_pretty = self.get_contact_force_vector_pretty()

    def update(self, patch=True, force=True, pretty=False):
        if patch:
            self.update_patch()
        if force:
            self.update_force()
        if pretty:
            self.update_force_pretty()

    def viewer_kwargs_add_vector_field(self):
        return {
            "points": self.xyz_coord_V_contact,
            "vectors": self.contact_force_vector_pretty,
            "rgba": self.force_rgba,
            "name": f"force_{self.name}",
        }

    def viewer_kwargs_update_rgba_V(self):
        return {"value": self.contact_rgba, "indices": self.V_contact}

    def export_time_series_data(self, pretty=False):
        data = {
            "V_contact": self.V_contact,
            "contact_force_magnitude": self.contact_force_magnitude,
            "contact_force_vector": self.contact_force_vector,
        }
        if pretty:
            data["contact_force_vector_pretty"] = self.contact_force_vector_pretty
            data["xyz_coord_V_contact"] = self.xyz_coord_V_contact
        return data

    def minimal_state(self):
        return {
            "V_contact": self.V_contact,
            "contact_force_magnitude": self.contact_force_magnitude,
        }


class Spindle:
    """ """

    def __init__(
        self,
        envelope,
        axis_origin,
        axis_vec,
        force_total,
        force_profile="bump",
        visual_spb_length=0.1,
        visual_force_scale=0.3,
        spb1=None,
        spb2=None,
        name="spindle",
        find_contact_data=True,
    ):
        self.envelope = envelope
        self.axis_origin = axis_origin
        self.axis_vec = axis_vec
        self.force_total = force_total
        self.force_profile = force_profile
        self.visual_spb_length = visual_spb_length
        self.visual_force_scale = visual_force_scale
        if spb1 is None:
            spb1 = {}
        if spb2 is None:
            spb2 = {}
        spb1.update(
            {
                "axis_origin": axis_origin,
                "axis_vec": axis_vec,
                "force_total": force_total,
                "force_profile": force_profile,
                "visual_length": visual_spb_length,
                "visual_force_scale": visual_force_scale,
                "find_contact_data": find_contact_data,
                "name": f"{name}_spb1",
            }
        )
        spb2.update(
            {
                "axis_origin": axis_origin,
                "axis_vec": -np.array(axis_vec),
                "force_total": force_total,
                "force_profile": force_profile,
                "visual_length": visual_spb_length,
                "visual_force_scale": visual_force_scale,
                "find_contact_data": find_contact_data,
                "name": f"{name}_spb1",
            }
        )
        self.spb1 = SPB(envelope, **spb1)
        self.spb2 = SPB(envelope, **spb2)
        if find_contact_data:
            self.length = self.get_spindle_length()

    def update(self, patch=True, force=True, pretty=False):
        self.spb1.update(patch, force, pretty)
        self.spb2.update(patch, force, pretty)
        if pretty:
            self.length = self.get_spindle_length()

    def get_spindle_length(self):
        v1 = self.spb1.contact_patch.seed_vertex
        v2 = self.spb2.contact_patch.seed_vertex
        x1 = self.envelope.xyz_coord_v(v1)
        x2 = self.envelope.xyz_coord_v(v2)
        return np.linalg.norm(x1 - x2)


class Envelope(Brane):
    def __init__(
        self,
        *args,
        rgba_surface=(0.0, 0.6745, 0.2784, 0.5),
        rgba_edge=(0.0, 0.4471, 0.698, 0.8),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rgba_surface = np.array(rgba_surface)
        self.rgba_edge = np.array(rgba_edge)

    def physical_parameters(self):
        return {
            "preferred_area": self.preferred_area,
            "preferred_volume": self.preferred_volume,
            "spontaneous_curvature": self.spontaneous_curvature,
            "bending_modulus": self.bending_modulus,
            "splay_modulus": self.splay_modulus,
            "volume_reg_stiffness": self.volume_reg_stiffness,
            "area_reg_stiffness": self.area_reg_stiffness,
            "tether_stiffness": self.tether_stiffness,
            "tether_repulsive_onset": self.tether_repulsive_onset,
            "tether_repulsive_singularity": self.tether_repulsive_singularity,
            "tether_attractive_onset": self.tether_attractive_onset,
            "tether_attractive_singularity": self.tether_attractive_singularity,
            "drag_coefficient": self.drag_coefficient,
            "flipping_frequency": self.flipping_frequency,
            "flipping_probability": self.flipping_probability,
        }

    def mesh_state(self):
        data = {
            "xyz_coord_V": self.xyz_coord_V,
            "h_out_V": self.h_out_V,
            "v_origin_H": self.v_origin_H,
            "h_next_H": self.h_next_H,
            "h_twin_H": self.h_twin_H,
            "f_left_H": self.f_left_H,
            "h_bound_F": self.h_bound_F,
            "h_right_B": self.h_right_B,
        }

    def viewer_kwargs(self):
        return {
            "rgba_surface": self.rgba_surface,
            "rgba_edge": self.rgba_edge,
        }

    def update_force(self):
        self.force_surface_tension = self.Farea_harmonic()
        self.force_pressure = self.Fvolume_harmonic()
        self.force_tether, self.tether_success = self.Ftether()
        self.force_bending = self.Fbend_analytic()


class StretchSim:
    """

    time_samples T = [t0, t1, t2,...]=[t0, t0+dt, t0+2*dt,...,]
    Trecord
    state samples S = [S(), ]
    Trecord = [t0, t1, t2, t3, ]
    """

    def __init__(
        self,
        output_dir,
        run_name,
        T,
        envelope_parameters,
        spindle_parameters,
        mesh_viewer_parameters,
        dt=1e-3,
        dt_record_data=1e-2,
        dt_write_data=1.0,
        dt_checkpoint=5.0,
        dx_max=0.05,
        make_movie_frames=True,
        make_output_dir=True,
        overwrite_output_dir=False,
    ):
        if make_output_dir:
            self.make_output_dir(output_dir, overwrite_output_dir)
        self.output_dir = output_dir
        self.run_name = run_name
        self.T = T
        self.dt = dt
        self.dt_record_data = dt_record_data
        self.dt_write_data = dt_write_data
        self.dt_checkpoint = dt_checkpoint
        self.dx_max = dx_max
        self.make_movie_frames = make_movie_frames

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

        self.envelope_parameters = envelope_parameters
        self.spindle_parameters = spindle_parameters
        mesh_viewer_parameters = mesh_viewer_parameters
        mesh_viewer_parameters.update(
            {
                "image_dir": self.temp_images_dir,
                "show_face_colored_surface": False,
                "show_vertex_colored_surface": True,
                "show_wireframe_surface": True,
            }
        )
        mesh_viewer_parameters["rgba_vertex"] = envelope_parameters.get("rgba_surface")
        mesh_viewer_parameters["rgba_edge"] = envelope_parameters.get("rgba_edge")
        self.mesh_viewer_parameters = mesh_viewer_parameters
        self.envelope = Envelope.load(**envelope_parameters)
        self.spindle = Spindle(self.envelope, **spindle_parameters)
        self.mesh_viewer = MeshViewer(self.envelope, **mesh_viewer_parameters)
        self.t = 0.0

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

    @classmethod
    def from_parameters_file(cls, file_path, overwrite_output_dir=False):
        with open(file_path, "r") as f:
            parameters = yaml.safe_load(f)
        parameters["envelope_parameters"] = parameters.pop("envelope")
        parameters["spindle_parameters"] = parameters.pop("spindle")
        parameters["mesh_viewer_parameters"] = parameters.pop("mesh_viewer")
        self = cls(**parameters, overwrite_output_dir=overwrite_output_dir)
        os.system(f"cp {file_path} {self.parameters_path}")

        return self

    def get_force_total(self):
        m = self.envelope
        spindle = self.spindle
        spb1 = spindle.spb1
        spb2 = spindle.spb2
        force_spb1 = spb1.contact_force_vector
        force_spb2 = spb2.contact_force_vector
        force_surface_tension = m.force_surface_tension
        force_pressure = m.force_pressure
        force_tether = m.force_tether
        force_bending = m.force_bending
        force_total = (
            force_bending + force_tether + force_pressure + force_surface_tension
        )
        force_total[spb1.V_contact] += force_spb1
        force_total[spb2.V_contact] += force_spb2

        # return {
        #     "force_spb1": force_spb1,
        #     "force_spb2": force_spb2,
        #     "force_surface_tension": force_surface_tension,
        #     "force_pressure": force_pressure,
        #     "force_tether": force_tether,
        #     "force_bending": force_bending,
        #     "force_total": force_total,
        # }
        return force_total

    def update_mesh_viewer(self):
        self.mesh_viewer.update_rgba_V(self.envelope.rgba_surface)
        self.mesh_viewer.clear_vector_field_data()
        for spb in [self.spindle.spb1, self.spindle.spb2]:
            self.mesh_viewer.add_vector_field(**spb.viewer_kwargs_add_vector_field())
            self.mesh_viewer.update_rgba_V(**spb.viewer_kwargs_update_rgba_V())

    def update(self, patch=True, force=True, pretty=False):
        self.spindle.update(patch, force, pretty)
        if force:
            self.envelope.update_force()
            self.force_total = self.get_force_total()
            self.max_force_actual = np.max(np.linalg.norm(self.force_total, axis=-1))
        if pretty:
            self.update_mesh_viewer()

    def plot(self, save=True, show=False, title=""):
        self.mesh_viewer.plot(save=save, show=show, title=title)

    def dt_max(self):
        dx_max = self.dx_max * self.envelope.preferred_edge_length
        dt_max = self.envelope.drag_coefficient * dx_max / self.max_force_actual
        return dt_max

    def euler_step(self, dt):
        m = self.envelope
        force_total = self.force_total
        Dxyz = dt * force_total / m.drag_coefficient
        return Dxyz

    def step_was_good(self):
        success = True
        success = success and self.envelope.tether_success
        return success

    def try_smaller_timestep(self, dt0):
        dt_max = self.dt_max()
        dt = np.min([dt_max, 0.2 * dt0])
        Dxyz = self.euler_step(dt)
        return Dxyz, dt

    def get_smaller_timestep(self, dt0):
        dt_max = self.dt_max()
        dt = np.min([dt_max, 0.2 * dt0])
        return dt

    def evolve_for_DT(self, DT, dt0, patch=True):
        dt_min = 1e-6
        iters_to_reset_dt = 10
        dt = dt0
        t_stop = self.t + DT
        # bad_step = False
        num_fails = 0
        while self.t < t_stop:
            if dt < dt_min:
                print(f"dt={dt} is too small. Exiting...")
                return False
            if iters_to_reset_dt <= 0:
                dt = dt0
                iters_to_reset_dt = 10
            if num_fails > 10:
                print(f"Failed too many times. Exiting...")
                return False
            Dxyz = self.euler_step(dt)
            if self.step_was_good():
                self.envelope.xyz_coord_V += Dxyz
                # self.num_flips = self.envelope.flip_non_delaunay()
                self.update(patch=patch, force=True, pretty=False)
                self.t += dt
                num_fails = 0
            else:
                print(" ")
                dt_max = self.dt_max()
                dt = np.min([dt_max, 0.2 * dt])
                num_fails += 1
                print(f"trying smaller timestep dt={dt}")
            iters_to_reset_dt -= 1
        return True

    def run(self):
        self.update(patch=True, force=True, pretty=True)
        dt0 = self.dt
        T = self.T
        dt_record_data = self.dt_record_data
        print(f"T={self.T}         ", end="\n")
        while self.t < T:
            success = self.evolve_for_DT(dt_record_data, dt0, patch=False)
            if success:
                self.num_flips = self.envelope.flip_non_delaunay()
                self.update(patch=True, force=True, pretty=True)
                self.plot(save=True, show=False, title=f"t={self.t}")
                V = self.envelope.total_volume()
                A = self.envelope.total_area_of_faces()
                V0 = self.envelope.preferred_volume
                A0 = self.envelope.preferred_area
                self.envelope.preferred_edge_length = np.mean(self.envelope.length_H())

                t_print = np.round(self.t, 3)
                A_percent_error = np.round(100 * (A - A0) / A0, 3)
                V_percent_error = np.round(100 * (V - V0) / V0, 3)
                L_spindle = np.round(self.spindle.length, 3)
                # A_percent_error = 100 * (A - A0) / A0
                # V_percent_error = 100 * (V - V0) / V0

                print(
                    f"t={t_print}, L_spindle={L_spindle}, A_error={A_percent_error}, V_error={V_percent_error}         ",
                    end="\r",
                )

            else:
                print(f"Failed at t={self.t}")
                break
        self.mesh_viewer.movie()

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


#####################################################
class RawData:
    def __init__(
        self,
        dt_sim,
        dt_record,
        dt_write,
        envelope,
        spb_force,
        *args,
        **kwargs,
    ):
        self.dt_sim = dt_sim
        self.dt_record = dt_record
        self.dt_write = dt_write
        self.envelope = envelope
        self.spb_force = spb_force

        num_sim_samples = 5
        num_vertices = envelope.num_vertices
        num_half_edges = envelope.num_half_edges
        num_faces = envelope.num_faces

        self.t = np.zeros(num_sim_samples)
        self.envelope_data = {
            "xyz_coord_V": np.zeros((num_sim_samples, num_vertices, 3)),
            "h_out_V": np.zeros((num_sim_samples, num_vertices), dtype="int32"),
            "v_origin_H": np.zeros((num_sim_samples, num_half_edges), dtype="int32"),
            "h_next_H": np.zeros((num_sim_samples, num_half_edges), dtype="int32"),
            "h_twin_H": np.zeros((num_sim_samples, num_half_edges), dtype="int32"),
            "f_left_H": np.zeros((num_sim_samples, num_half_edges), dtype="int32"),
            "h_bound_F": np.zeros((num_sim_samples, num_faces), dtype="int32"),
            "h_right_B": np.zeros((num_sim_samples, num_boundaries), dtype="int32"),
            #
            "pressure": np.zeros(num_time_samples),
            "surface_tension_V": np.zeros((num_time_samples, num_faces)),
            "area_V": np.zeros((num_time_samples, num_vertices)),
            "laplacian_weights_VV_csr_data": np.zeros((num_time_samples, num_vertices)),
            "unit_normal_V": np.zeros((num_time_samples, num_vertices, 3)),
            "force_bend_V": np.zeros((num_time_samples, num_vertices, 3)),
            "force_area_V": np.zeros((num_time_samples, num_vertices, 3)),
            "force_volume_V": np.zeros((num_time_samples, num_vertices, 3)),
        }

        num_record_samples = int(dt_write // dt_record) + 1
        self.t_record = np.zeros(dt_write_dt_record_ratio)

        self.envelope_data = {
            "xyz_coord_V": envelope.xyz_coord_V,
        }

        self.t = t
        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.f_left_H = f_left_H
        self.h_bound_F = h_bound_F
        self.h_right_B = h_right_B
        self.force_bending_V = kwargs.get("bending_force", np.zeros_like(xyz_coord_V))
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.args = args

    def extrapolate(self, t):
        dt = t - self.t
        xyz_coord_V = self.xyz_coord_V + dt * self.dN_dt
        return RawData(t, xyz_coord_V, *self.args, **self.__dict__)


class TimeStepper:
    """ """

    def __init__(self, envelope, spb_force, sim_parameters, t=0.0, dt=0.01, **data):
        self.t = t
        self.dt = dt
        self.data = data

    def __str__(self):
        return f"TimeStepper(t={self.t}, dt={self.dt})"

    def __repr__(self):
        return f"TimeStepper(t={self.t}, dt={self.dt})"

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def save_yaml(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self.data, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class DataManager:
    def __init__(self, **data):
        self.data = deepcopy(data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def save_yaml(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self.data, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ParamManager:
    def __init__(self, **parameters):

        self.parameters = deepcopy(parameters)

        # for key, val in parameters.items():
        #     setattr(self, key, val)
        # self.envelope_parameters = self.parameters.get("envelope_parameters", {})
        # self.spb_force_parameters = self.parameters.get("spb_force_parameters", {})
        # self.mesh_viewer_parameters = self.parameters.get("mesh_viewer_parameters", {})
        # self.sim_parameters = {
        #     k: v
        #     for k, v in self.parameters.items()
        #     if k not in ["envelope", "spb_force", "mesh_viewer"]
        # }

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
            "run_name": "default_sphere",
            "T": 10.0,
            "dt": 0.01,
            "dt_record_data": 0.05,
            "dt_write_data": 1.0,
            "dt_checkpoint": 2.0,
            "dx_max": 0.05,
            "make_movie_frames": True,
            "show_contact_patches": True,
            "show_contact_forces": True,
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
            "force_rgba": [0.5804, 0.0, 0.8275, 1.0],
        }

        mesh_viewer_params = {
            "figsize": [720, 720],
            "image_dir": f"{output_dir}/temp_images",
            "show_face_colored_surface": False,
            "show_vertex_colored_surface": True,
            "rgba_vertex": [0.0, 0.63335, 0.05295, 0.65],
            "view": {
                "azimuth": 90.0,
                "elevation": 55.0,
                # "distance": 6.75,
                # "focalpoint": [0.0, 0.0, 0.0],
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
    """
    Spindle pole body force

    Parameters
    ----------
    envelope : Brane
        Envelope object
    contact_radius : float
        Radius of contact patches
    force_total : float
        Total force on each spindle pole body
    force_profile : str
        Profile of force distribution
    view_scale : float
        Scale for viewer force vectors
    contact_rgba : tuple
        RGBA color for contact patches
    force_rgba : tuple
        RGBA color for force vectors
    find_contact_data : bool
        Whether to find contact patches and forces during initialization

    Attributes
    ----------
    V_plus : np.ndarray
        Vertices in contact patch plus
    V_minus : np.ndarray
        Vertices in contact patch minus
    force_plus : np.ndarray
        Force vector for each vertex in contact patch plus
    force_minus : np.ndarray
        Force vector for each vertex in contact patch minus

    Methods
    -------
    viewer_add_vector_field_kwargs
        Return kwargs for adding contact force vector field to mesh viewer with MeshViewer.add_vector_field
    viewer_update_rgba_V_kwargs
        Return kwargs for updating contact patch colors in mesh viewer with MeshViewer.update_rgba_V

    Other Attributes
    ----------------
    V_contact : np.ndarray
        Vertices in contact patches
    force_contact : np.ndarray
        Force vectors for contact patches
    scaled_force : np.ndarray
        Scaled force vectors for viewer
    envelope : Brane
        Nuclear envelope
    patch_plus : HalfEdgePatch
        Contact patch for spindle pole body plus
    patch_minus : HalfEdgePatch
        Contact patch for spindle pole body minus
    """

    def __init__(
        self,
        envelope,
        contact_radius,
        force_total,
        max_search_vertices=10**3,
        force_profile="bump",
        view_scale=0.3,
        contact_rgba=(0.5804, 0.0, 0.8275, 0.5),
        force_rgba=(0.0, 0.0, 0.0, 0.9),
        find_contact_data=True,
        axis=0,
    ):
        self.envelope = envelope
        self.contact_radius = contact_radius
        self.force_total = force_total
        self.force_profile = force_profile
        self.max_search_vertices = max_search_vertices
        self.view_scale = view_scale
        self.contact_rgba = contact_rgba
        self.force_rgba = force_rgba
        self.axis = axis

        if find_contact_data:
            self.patch_plus, self.patch_minus = self.find_contact_patches()
            self.V_plus = np.array(sorted(self.patch_plus.V), dtype="int32")  # ***
            self.V_minus = np.array(sorted(self.patch_minus.V), dtype="int32")  # ***
            self.magnitude_plus, self.magnitude_minus = (
                self.find_force_magnitudes()
            )  # ***
            #############################################################################

            # self.V_contact = np.concatenate([self.V_plus, self.V_minus], dtype="int32")
            self.force_contact = np.vstack([self.force_plus, self.force_minus])
            max_magnitude = np.max(
                [
                    np.linalg.norm(self.force_plus, axis=-1),
                    np.linalg.norm(self.force_minus, axis=-1),
                ]
            )
            self.scaled_force = self.view_scale * self.force_contact / max_magnitude
        else:
            self.patch_plus = None
            self.patch_minus = None
            self.V_plus = np.zeros(0, dtype="int32")
            self.V_minus = np.zeros(0, dtype="int32")
            self.V_contact = np.zeros(0, dtype="int32")
            self.force_plus = np.zeros((0, 3))
            self.force_minus = np.zeros((0, 3))
            self.scaled_force = np.zeros((0, 3))

    @property
    def V_contact(self):
        return np.concatenate([self.V_plus, self.V_minus], dtype="int32")

    @property
    def max_magnitude(self):
        v_plus = self.patch_plus.seed_vertex
        v_minus = self.patch_minus.seed_vertex
        # return max([self.magnitude_plus[self.]])

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

    def find_seeds(self, restrict_to_V=None):
        """
        Find vertices in self.envelope with maximum and minimum (self.axis)-coordinates.
        """
        x_coord_V = self.envelope.xyz_coord_V[:, self.axis]
        seed_plus = np.argmax(x_coord_V)
        seed_minus = np.argmax(x_coord_V)
        return seed_plus, seed_minus

    def find_contact_patches(self):
        patch_plus = HalfEdgePatch.from_seed_to_radius(
            self.seed_plus, self.envelope, self.contact_radius
        )
        patch_minus = HalfEdgePatch.from_seed_to_radius(
            self.seed_minus, self.envelope, self.contact_radius
        )

        return patch_plus, patch_minus

    def find_force_magnitudes(self):

        force_profile = self.force_profile
        envelope = self.envelope
        contact_radius = self.contact_radius
        force_total = self.force_total
        view_scale = self.view_scale

        seed_plus = self.patch_plus.seed_vertex
        seed_minus = self.patch_minus.seed_vertex
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
        return force_plus, force_minus

    def viewer_add_vector_field_kwargs(self):
        points = self.envelope.xyz_coord_V[self.V_contact]
        vectors = self.scaled_force
        rgba = self.force_rgba
        return {"points": points, "vectors": vectors, "rgba": rgba, "name": "spb_force"}

    def viewer_update_rgba_V_kwargs(self):
        value = self.contact_rgba
        indices = self.V_contact
        return {"value": value, "indices": indices}

    # def dataset_info(self):
    #     num_vertices_plus = len(self.V_plus)
    #     num_vertices_minus = len(self.V_minus)
    #     info = [
    #         ["spb_force/V_plus", (num_vertices_plus,), "int32"],
    #         ["spb_force/V_minus", (num_vertices_minus,), "int32"],
    #         ["spb_force/force_plus", (num_vertices_plus, 3), "float64"],
    #         ["spb_force/force_minus", (num_vertices_minus, 3), "float64"],
    #         ["spb_force/contact_radius", (), "float64"],
    #         ["spb_force/force_total", (), "float64"],
    #         ["spb_force/force_profile", (), "str"],
    #         ["spb_force/view_scale", (), "float64"],
    #         ["spb_force/contact_rgba", (4,), "float64"],
    #         ["spb_force/force_rgba", (4,), "float64"],
    #     ]
    #     return info

    # def export_timeseries_data(self):
    #     data = {"spb_force/V_plus":self.V_plus,
    #         "spb_force/V_minus":,
    #         "spb_force/force_plus":,
    #         "spb_force/force_minus":,
    #         "spb_force/contact_radius":,
    #         "spb_force/force_total":,
    #         "spb_force/force_profile":,
    #         "spb_force/view_scale":,
    #         "spb_force/contact_rgba":,
    #         "spb_force/force_rgba":,}

    #     return data
    # def get_dataset(self):

    #     dataset = {
    #         "spb_force/V_plus": self.V_plus,
    #         "spb_force/V_minus": self.V_minus,
    #         "spb_force/force_plus": self.force_plus,
    #         "spb_force/force_minus": self.force_minus,
    #         "spb_force/contact_radius": self.contact_radius,
    #         "spb_force/force_total": self.force_total,
    #         "spb_force/force_profile": self.force_profile,
    #         "spb_force/view_scale": self.view_scale,
    #         "spb_force/contact_rgba": self.contact_rgba,
    #         "spb_force/force_rgba": self.force_rgba,
    #     }
    #     return dataset

    # @classmethod
    # def from_dataset(cls, envelope, datasets):
    #     contact_radius = datasets["spb_force/contact_radius"]
    #     force_total = datasets["spb_force/force_total"]
    #     force_profile = datasets["spb_force/force_profile"]
    #     view_scale = datasets["spb_force/view_scale"]
    #     contact_rgba = datasets["spb_force/contact_rgba"]
    #     force_rgba = datasets["spb_force/force_rgba"]
    #     V_plus = datasets["spb_force/V_plus"]
    #     V_minus = datasets["spb_force/V_minus"]
    #     force_plus = datasets["spb_force/force_plus"]
    #     force_minus = datasets["spb_force/force_minus"]

    #     self = cls(
    #         envelope,
    #         contact_radius,
    #         force_total,
    #         force_profile,
    #         view_scale,
    #         contact_rgba,
    #         force_rgba,
    #         find_contact_data=False,
    #     )
    #     self.patch_plus = HalfEdgePatch.from_vertex_set(set(self.V_plus), self.envelope)
    #     self.patch_minus = HalfEdgePatch.from_vertex_set(
    #         set(self.V_minus), self.envelope
    #     )
    #     self.V_plus = V_plus
    #     self.V_minus = V_minus
    #     self.force_plus = force_plus
    #     self.force_minus = force_minus

    #     self.V_contact = np.concatenate([self.V_plus, self.V_minus], dtype="int32")
    #     self.force_contact = np.vstack([self.force_plus, self.force_minus])
    #     max_magnitude = np.max(
    #         [
    #             np.linalg.norm(self.force_plus, axis=-1),
    #             np.linalg.norm(self.force_minus, axis=-1),
    #         ]
    #     )
    #     self.scaled_force = self.view_scale * self.force_contact / max_magnitude
    #     return self


class __Envelope(Brane):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parameters(self):
        return {
            "preferred_area": self.preferred_area,
            "preferred_volume": self.preferred_volume,
            "spontaneous_curvature": self.spontaneous_curvature,
            "bending_modulus": self.bending_modulus,
            "splay_modulus": self.splay_modulus,
            "volume_reg_stiffness": self.volume_reg_stiffness,
            "area_reg_stiffness": self.area_reg_stiffness,
            "tether_stiffness": self.tether_stiffness,
            "tether_repulsive_onset": self.tether_repulsive_onset,
            "tether_repulsive_singularity": self.tether_repulsive_singularity,
            "tether_attractive_onset": self.tether_attractive_onset,
            "tether_attractive_singularity": self.tether_attractive_singularity,
            "drag_coefficient": self.drag_coefficient,
            "flipping_frequency": self.flipping_frequency,
            "flipping_probability": self.flipping_probability,
        }

    def mesh_state(self):
        data = {
            "xyz_coord_V": self.xyz_coord_V,
            "h_out_V": self.h_out_V,
            "v_origin_H": self.v_origin_H,
            "h_next_H": self.h_next_H,
            "h_twin_H": self.h_twin_H,
            "f_left_H": self.f_left_H,
            "h_bound_F": self.h_bound_F,
            "h_right_B": self.h_right_B,
        }

    def _parameters_dataset_info(self):
        info = [
            ["envelope/spontaneous_curvature", (), "float64"],
            ["envelope/bending_modulus", (), "float64"],
            ["envelope/splay_modulus", (), "float64"],
            ["envelope/volume_reg_stiffness", (), "float64"],
            ["envelope/area_reg_stiffness", (), "float64"],
            ["envelope/tether_stiffness", (), "float64"],
            ["envelope/tether_repulsive_onset", (), "float64"],
            ["envelope/tether_repulsive_singularity", (), "float64"],
            ["envelope/tether_attractive_onset", (), "float64"],
            ["envelope/tether_attractive_singularity", (), "float64"],
            ["envelope/drag_coefficient", (), "float64"],
            ["envelope/flipping_frequency", (), "float64"],
            ["envelope/flipping_probability", (), "float64"],
        ]
        return info

    def _mesh_dataset_info(self):
        num_vertices = self.num_vertices
        num_half_edges = self.num_half_edges
        num_faces = self.num_faces
        info = [
            ["envelope/xyz_coord_V", (num_vertices, 3), "float64"],
            ["envelope/h_out_V", (num_vertices,), "int32"],
            ["envelope/v_origin_H", (num_half_edges,), "int32"],
            ["envelope/h_next_H", (num_half_edges,), "int32"],
            ["envelope/h_twin_H", (num_half_edges,), "int32"],
            ["envelope/f_left_H", (num_half_edges,), "int32"],
            ["envelope/h_bound_F", (num_faces,), "int32"],
            ["envelope/h_right_B", (num_boundaries,), "int32"],
        ]
        return info

    def _dynamics_dataset_info(self):
        num_vertices = self.num_vertices
        num_half_edges = self.num_half_edges
        num_faces = self.num_faces
        info = [
            ["envelope/bending_force", (num_vertices, 3), "float64"],
            ["envelope/area_force", (num_vertices, 3), "float64"],
            ["envelope/volume_force", (num_vertices, 3), "float64"],
            ["envelope/tether_force", (num_vertices, 3), "float64"],
            ["envelope/num_flips", (), "int32"],
        ]
        return info

    def get_parameters_dataset(self):
        dataset = {
            "envelope/spontaneous_curvature": self.spontaneous_curvature,
            "envelope/bending_modulus": self.bending_modulus,
            "envelope/splay_modulus": self.splay_modulus,
            "envelope/volume_reg_stiffness": self.volume_reg_stiffness,
            "envelope/area_reg_stiffness": self.area_reg_stiffness,
            "envelope/tether_stiffness": self.tether_stiffness,
            "envelope/tether_repulsive_onset": self.tether_repulsive_onset,
            "envelope/tether_repulsive_singularity": self.tether_repulsive_singularity,
            "envelope/tether_attractive_onset": self.tether_attractive_onset,
            "envelope/tether_attractive_singularity": self.tether_attractive_singularity,
            "envelope/drag_coefficient": self.drag_coefficient,
            "envelope/flipping_frequency": self.flipping_frequency,
            "envelope/flipping_probability": self.flipping_probability,
        }
        return dataset


class StretchSimData:
    def __init__(
        self,
        data_path,
        envelope,
        spb_force,
        samples_per_chunk=100,
    ):
        self.data_path = data_path
        self.samples_per_chunk = samples_per_chunk
        self.sample_num = 0

        envelope_parameters = {
            "envelope/spontaneous_curvature": envelope.spontaneous_curvature,
            "envelope/bending_modulus": envelope.bending_modulus,
            "envelope/splay_modulus": envelope.splay_modulus,
            "envelope/volume_reg_stiffness": envelope.volume_reg_stiffness,
            "envelope/area_reg_stiffness": envelope.area_reg_stiffness,
            "envelope/tether_stiffness": envelope.tether_stiffness,
            "envelope/tether_repulsive_onset": envelope.tether_repulsive_onset,
            "envelope/tether_repulsive_singularity": envelope.tether_repulsive_singularity,
            "envelope/tether_attractive_onset": envelope.tether_attractive_onset,
            "envelope/tether_attractive_singularity": envelope.tether_attractive_singularity,
            "envelope/drag_coefficient": envelope.drag_coefficient,
            "envelope/flipping_frequency": envelope.flipping_frequency,
            "envelope/flipping_probability": envelope.flipping_probability,
        }

        spb_force_parameters = {
            "spb_force/V_plus": spb_force.V_plus,
            "spb_force/V_minus": spb_force.V_minus,
            "spb_force/force_plus": spb_force.force_plus,
            "spb_force/force_minus": spb_force.force_minus,
            "spb_force/contact_radius": spb_force.contact_radius,
            "spb_force/force_total": spb_force.force_total,
            "spb_force/force_profile": spb_force.force_profile,
            "spb_force/view_scale": spb_force.view_scale,
            "spb_force/contact_rgba": spb_force.contact_rgba,
            "spb_force/force_rgba": spb_force.force_rgba,
        }

        num_vertices = envelope.num_vertices
        num_half_edges = envelope.num_half_edges
        num_faces = envelope.num_faces

        self.mesh_time_series_info = [
            ["mesh/xyz_coord_V", (num_vertices, 3), "float64"],
            ["mesh/h_out_V", (num_vertices,), "int32"],
            ["mesh/v_origin_H", (num_half_edges,), "int32"],
            ["mesh/h_next_H", (num_half_edges,), "int32"],
            ["mesh/h_twin_H", (num_half_edges,), "int32"],
            ["mesh/f_left_H", (num_half_edges,), "int32"],
            ["mesh/h_bound_F", (num_faces,), "int32"],
            ["mesh/h_right_B", (num_boundaries,), "int32"],
        ]
        self.mesh_time_series_data = {
            key: np.zeros((self.samples_per_chunk,) + shape, dtype=dtype)
            for key, shape, dtype in self.mesh_time_series_info
        }
        num_vertices_plus = spb_force.V_plus.shape[0]
        num_vertices_minus = spb_force.V_minus.shape[0]
        self.spb_force_info = [
            ["spb_force/V_plus", (num_vertices_plus,), "int32"],
            ["spb_force/V_minus", (num_vertices_minus,), "int32"],
            ["spb_force/force_plus", (num_vertices_plus, 3), "float64"],
            ["spb_force/force_minus", (num_vertices_minus, 3), "float64"],
            ["spb_force/contact_radius", (), "float64"],
            ["spb_force/force_total", (), "float64"],
            ["spb_force/force_profile", (), "str"],
            ["spb_force/view_scale", (), "float64"],
            ["spb_force/contact_rgba", (4,), "float64"],
            ["spb_force/force_rgba", (4,), "float64"],
        ]
        self.spb_force_data = {
            "spb_force/V_plus": spb_force.V_plus,
            "spb_force/V_minus": spb_force.V_minus,
            "spb_force/force_plus": spb_force.force_plus,
            "spb_force/force_minus": spb_force.force_minus,
            "spb_force/contact_radius": spb_force.contact_radius,
            "spb_force/force_total": spb_force.force_total,
            "spb_force/force_profile": spb_force.force_profile,
            "spb_force/view_scale": spb_force.view_scale,
            "spb_force/contact_rgba": spb_force.contact_rgba,
            "spb_force/force_rgba": spb_force.force_rgba,
        }

        with h5py.File(self.data_path, "w") as f:
            envelope_group = f.require_group("envelope")
            spb_force_group = f.require_group("spb_force")
            half_edge_group = f.require_group("half_edge")
            face_group = f.require_group("face")
            boundary_group = f.require_group("boundary")
        # mesh_data_sets = [
        #     ["mesh/xyz_coord_V", (num_vertices, 3), "float64"],
        #     ["mesh/h_out_V", (num_vertices,), "int32"],
        #     ["mesh/v_origin_H", (num_half_edges,), "int32"],
        #     ["mesh/h_next_H", (num_half_edges,), "int32"],
        #     ["mesh/h_twin_H", (num_half_edges,), "int32"],
        #     ["mesh/f_left_H", (num_half_edges,), "int32"],
        #     ["mesh/h_bound_F", (num_faces,), "int32"],
        #     ["mesh/h_right_B", (num_boundaries,), "int32"],
        # ]

        # self.he_datasets = [
        #     ["vertex/xyz_coord_V", (num_vertices, 3), "float64"],
        #     ["vertex/h_out_V", (num_vertices,), "int32"],
        #     ["half_edge/v_origin_H", (num_half_edges,), "int32"],
        #     ["half_edge/h_next_H", (num_half_edges,), "int32"],
        #     ["half_edge/h_twin_H", (num_half_edges,), "int32"],
        #     ["half_edge/f_left_H", (num_half_edges,), "int32"],
        #     ["face/h_bound_F", (num_faces,), "int32"],
        #     ["boundary/h_right_B", (num_boundaries,), "int32"],
        # ]
        # self.force_datasets = [
        #     ["vertex/bending_force", (num_vertices, 3), "float64"],
        #     ["vertex/area_force", (num_vertices, 3), "float64"],
        #     ["vertex/volume_force", (num_vertices, 3), "float64"],
        #     ["vertex/tether_force", (num_vertices, 3), "float64"],
        #     ["vertex/spb_force", (num_vertices, 3), "float64"],
        # ]

        # with h5py.File(self.data_path, "w") as data_file:
        #     scalar_group = data_file.require_group("scalar")
        #     vertex_group = data_file.require_group("vertex")
        #     half_edge_group = data_file.require_group("half_edge")
        #     face_group = data_file.require_group("face")
        #     boundary_group = data_file.require_group("boundary")
        #     #
        # self.scalar_chunk = dict()
        # self.vertex_chunk = dict()
        # self.half_edge_chunk = dict()
        # self.face_chunk = dict()
        # self.boundary_chunk = dict()

        # self.key_dict = {
        #     group_key: dict()
        #     for group_key in ["scalar", "vertex", "half_edge", "face", "boundary"]
        # }

    ########################################################
    ########################################################
    ########################################################

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


class _StretchSim:
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
    make_movie_frames : bool
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
        make_movie_frames,
        envelope_parameters,
        spb_force_parameters,
        mesh_viewer_parameters,
        make_output_dir=True,
        overwrite_output_dir=False,
    ):
        if make_output_dir:
            self.make_output_dir(output_dir, overwrite_output_dir)
        self.output_dir = output_dir
        self.run_name = run_name
        self.T = T
        self.dt = dt
        self.dt_record_data = dt_record_data
        self.dt_write_data = dt_write_data
        self.dt_checkpoint = dt_checkpoint
        self.dx_max = dx_max
        self.make_movie_frames = make_movie_frames

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
        self.mesh_viewer.add_vector_field(
            **self.spb_force.viewer_add_vector_field_kwargs()
        )
        self.mesh_viewer.update_rgba_V(
            **self.spb_force.viewer_update_rgba_V_contact_kwargs()
        )
        # self.mesh_viewer.update_rgba_F(
        #     **self.spb_force.viewer_update_rgba_F_contact_kwargs()
        # )
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

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        self = cls(**checkpoint)
        return self

    def update_viewer_vector_field(self):
        self.mesh_viewer.clear_vector_field_data()
        self.mesh_viewer.add_vector_field(
            **self.spb_force.viewer_add_vector_field_kwargs()
        )

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


class SimAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        pass

    def analyze_data(self):
        pass

    def save_results(self):
        pass
