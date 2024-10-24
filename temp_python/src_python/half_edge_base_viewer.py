from mayavi import mlab
import numpy as np
import os
from temp_python.src_python.pretty_pictures import movie
from temp_python.src_python.pyvista_utils import decimate_VF
from temp_python.src_python.global_vars import INT_TYPE, FLOAT_TYPE
import pyvista as pv

# from mayavi.modules.vectors import Vectors
# mlab.pipeline.vector_field


def get_half_edge_vector_field(self):
    """halfedge vector shifted toward face centroid for visualization"""
    shift_to_center = 0.15
    num_half_edges = len(self.v_origin_H)
    points_vecs = np.array(
        [
            (
                #########################################################
                # if h is in mesh interior or positively oriented boundary
                [
                    # shift origin toward face centroid
                    shift_to_center
                    * (
                        self.xyz_coord_V[self.v_origin_H[h]]
                        + self.xyz_coord_V[self.v_origin_H[self.h_next_H[h]]]
                        + self.xyz_coord_V[
                            self.v_origin_H[self.h_next_H[self.h_next_H[h]]]
                        ]
                    )
                    / 3
                    + (1 - shift_to_center) * self.xyz_coord_V[self.v_origin_H[h]],
                    # rescale half-edge vector
                    (1 - shift_to_center)
                    * (
                        self.xyz_coord_V[self.v_origin_H[self.h_next_H[h]]]
                        - self.xyz_coord_V[self.v_origin_H[h]]
                    ),
                ]
                if self.f_left_H[h] >= 0
                #########################################################
                # if h is in negatively oriented boundary
                else [
                    # shift origin AWAY from twin face centroid
                    shift_to_center
                    * (
                        (
                            self.xyz_coord_V[self.v_origin_H[h]]
                            + self.xyz_coord_V[self.v_origin_H[self.h_twin_H[h]]]
                        )
                        - (
                            self.xyz_coord_V[self.v_origin_H[self.h_twin_H[h]]]
                            + self.xyz_coord_V[
                                self.v_origin_H[self.h_next_H[self.h_twin_H[h]]]
                            ]
                            + self.xyz_coord_V[
                                self.v_origin_H[
                                    self.h_next_H[self.h_next_H[self.h_twin_H[h]]]
                                ]
                            ]
                        )
                        / 3
                    )
                    + (1 - shift_to_center) * self.xyz_coord_V[self.v_origin_H[h]],
                    ####################################################
                    (1 - shift_to_center)
                    * (
                        self.xyz_coord_V[self.v_origin_H[self.h_next_H[h]]]
                        - self.xyz_coord_V[self.v_origin_H[h]]
                    ),
                ]
            )
            for h in range(num_half_edges)
        ]
    )
    return (points_vecs[:, 0], points_vecs[:, 1])


class MeshViewer:
    def __init__(
        self,
        # HalfEdgeMesh
        mesh,
        # image/movie output options
        image_dir="./output/temp_images",
        image_count=0,
        image_format="png",
        image_prefix="frame",
        image_index_length=5,
        movie_dir=None,
        movie_name="movie",
        movie_format="mp4",
        # Vertex/Edge/Face visual parameters
        radius_vertex=0.0125,
        rgba_vertex=(0.7057, 0.0156, 0.1502, 0.75),
        rgba_half_edge=(1.0, 0.498, 0.0, 1.0),
        rgba_face=(0.0, 0.63335, 0.05295, 0.65),
        rgba_boundary_half_edge=(0.0, 0.4471, 0.6980, 1.0),
        rgba_edge=(0.0, 0.4471, 0.698, 0.8),
        # rgba_surface=(0.0, 0.6745, 0.2784, 0.5),
        # mlab data that depends on mesh size
        radius_V=None,
        rgba_V=None,
        rgba_H=None,
        rgba_F=None,
        # Additional vector fields to plot
        vector_field_data=None,
        # mlab data that does NOT depend on mesh size
        show_wireframe_surface=True,
        show_face_colored_surface=True,
        show_vertex_colored_surface=False,
        show_vertices=False,
        show_half_edges=False,
        show_vector_fields=True,
        show_plot_axes=False,
        view=None,
        figsize=(2180, 2180),
        target_faces=None,
    ):
        self.M = mesh
        if target_faces is not None:
            self.Msimp, self.indicesVsimp = self.get_decimated_mesh(
                target_faces=target_faces
            )
        else:
            self.Msimp = None
            self.indicesVsimp = None

        ###############################################################
        # image/movie output options
        ###############################################################
        self.image_dir = image_dir
        self.image_count = image_count
        self.image_format = image_format
        self.image_prefix = image_prefix
        self.image_index_length = image_index_length
        if movie_dir is None:
            self.movie_dir = image_dir
        else:
            self.movie_dir = movie_dir
        self.movie_name = movie_name
        self.movie_format = movie_format
        ###############################################################
        # Size-independent params and defaults
        ###############################################################
        self.colors = {
            "black": [0.0, 0.0, 0.0, 1.0],
            "white": [1.0, 1.0, 1.0, 1.0],
            "transparent": [0.0, 0.0, 0.0, 0.0],
            "red": [0.8392, 0.1529, 0.1569, 1.0],
            "red10": [0.8392, 0.1529, 0.1569, 0.1],
            "red20": [0.8392, 0.1529, 0.1569, 0.2],
            "red50": [0.8392, 0.1529, 0.1569, 0.5],
            "red80": [0.8392, 0.1529, 0.1569, 0.8],
            "green": [0.0, 0.6745, 0.2784, 1.0],
            "green10": [0.0, 0.6745, 0.2784, 0.1],
            "green20": [0.0, 0.6745, 0.2784, 0.2],
            "green50": [0.0, 0.6745, 0.2784, 0.5],
            "green80": [0.0, 0.6745, 0.2784, 0.8],
            "blue": [0.0, 0.4471, 0.6980, 1.0],
            "blue10": [0.0, 0.4471, 0.6980, 0.1],
            "blue20": [0.0, 0.4471, 0.6980, 0.2],
            "blue50": [0.0, 0.4471, 0.6980, 0.5],
            "blue80": [0.0, 0.4471, 0.6980, 0.8],
            "yellow": [1.0, 0.8431, 0.0, 1.0],
            "yellow10": [1.0, 0.8431, 0.0, 0.1],
            "yellow20": [1.0, 0.8431, 0.0, 0.2],
            "yellow50": [1.0, 0.8431, 0.0, 0.5],
            "yellow80": [1.0, 0.8431, 0.0, 0.8],
            "cyan": [0.0, 0.8431, 0.8431, 1.0],
            "cyan10": [0.0, 0.8431, 0.8431, 0.1],
            "cyan20": [0.0, 0.8431, 0.8431, 0.2],
            "cyan50": [0.0, 0.8431, 0.8431, 0.5],
            "cyan80": [0.0, 0.8431, 0.8431, 0.8],
            "magenta": [0.8784, 0.0, 0.8784, 1.0],
            "magenta10": [0.8784, 0.0, 0.8784, 0.1],
            "magenta20": [0.8784, 0.0, 0.8784, 0.2],
            "magenta50": [0.8784, 0.0, 0.8784, 0.5],
            "magenta80": [0.8784, 0.0, 0.8784, 0.8],
            "orange": [1.0, 0.5490, 0.0, 1.0],
            "orange10": [1.0, 0.5490, 0.0, 0.1],
            "orange20": [1.0, 0.5490, 0.0, 0.2],
            "orange50": [1.0, 0.5490, 0.0, 0.5],
            "orange80": [1.0, 0.5490, 0.0, 0.8],
            "purple": [0.5804, 0.0, 0.8275, 1.0],
            "purple10": [0.5804, 0.0, 0.8275, 0.1],
            "purple20": [0.5804, 0.0, 0.8275, 0.2],
            "purple50": [0.5804, 0.0, 0.8275, 0.5],
            "purple80": [0.5804, 0.0, 0.8275, 0.8],
            "vertex_default": [0.7057, 0.0156, 0.1502, 1.0],
            "half_edge_default": [1.0, 0.498, 0.0, 1.0],
            "face_default": [0.0, 0.63335, 0.05295, 0.65],
            "boundary_half_edge_default": [0.0, 0.4471, 0.6980, 1.0],
        }
        ################
        self.show_wireframe_surface = show_wireframe_surface
        self.show_face_colored_surface = show_face_colored_surface
        self.show_vertex_colored_surface = show_vertex_colored_surface
        self.show_vertices = show_vertices
        self.show_half_edges = show_half_edges
        self.show_vector_fields = show_vector_fields
        self.show_plot_axes = show_plot_axes
        self.view = view
        self.figsize = figsize

        self.radius_vertex = radius_vertex
        self.rgba_vertex = rgba_vertex
        self.rgba_half_edge = rgba_half_edge
        self.rgba_face = rgba_face
        self.rgba_edge = rgba_edge

        ################################
        # Size-dependent params
        ################################
        if radius_V is not None:
            self.radius_V = radius_V
        else:
            self.radius_V = np.array(self.M.num_vertices * [self.radius_vertex])
        if rgba_V is not None:
            self.rgba_V = rgba_V
        else:
            self.rgba_V = np.array(self.M.num_vertices * [self.rgba_vertex])
        if rgba_H is not None:
            self.rgba_H = rgba_H
        else:
            self.rgba_H = np.array(self.M.num_half_edges * [self.rgba_half_edge])
        if rgba_F is not None:
            self.rgba_F = rgba_F
        else:
            self.rgba_F = np.array(self.M.num_faces * [self.rgba_face])

        ################################
        # Additional vector field data
        ################################
        if vector_field_data is None:
            self.vector_field_data = []
        else:
            self.vector_field_data = vector_field_data
        # self.fancy_mayavi_vector_fields = [
        #     FancyMayaviVectorField(*data) for data in vector_field_data
        # ]

    @property
    def rgba_vertex(self):
        return self._rgba_vertex

    @rgba_vertex.setter
    def rgba_vertex(self, value):
        self._rgba_vertex = np.array(value, dtype=FLOAT_TYPE)

    @property
    def radius_vertex(self):
        return self._radius_vertex

    @radius_vertex.setter
    def radius_vertex(self, value):
        self._radius_vertex = value

    @property
    def rgba_half_edge(self):
        return self._rgba_half_edge

    @rgba_half_edge.setter
    def rgba_half_edge(self, value):
        self._rgba_half_edge = np.array(value, dtype=FLOAT_TYPE)

    @property
    def rgba_face(self):
        return self._rgba_face

    @rgba_face.setter
    def rgba_face(self, value):
        self._rgba_face = np.array(value, dtype=FLOAT_TYPE)

    @property
    def radius_V(self):
        return self._radius_V

    @radius_V.setter
    def radius_V(self, value):
        self._radius_V = np.array(value, dtype=FLOAT_TYPE)

    @property
    def rgba_V(self):
        return self._rgba_V

    @rgba_V.setter
    def rgba_V(self, value):
        self._rgba_V = np.array(value, dtype=FLOAT_TYPE)

    @property
    def rgba_H(self):
        return self._rgba_H

    @rgba_H.setter
    def rgba_H(self, value):
        self._rgba_H = np.array(value, dtype=FLOAT_TYPE)

    @property
    def rgba_F(self):
        return self._rgba_F

    @rgba_F.setter
    def rgba_F(self, value):
        self._rgba_F = np.array(value, dtype=FLOAT_TYPE)

    def update_radius_V(self, value, indices=None):
        if value is not None:
            if indices is None:
                self._radius_V[:] = value
            else:
                self._radius_V[indices] = value

    def update_rgba_V(self, value, indices=None):
        if value is not None:
            if indices is None:
                self._rgba_V[:] = value
            else:
                self._rgba_V[indices] = value

    def update_rgba_H(self, value, indices=None):
        if value is not None:
            if indices is None:
                self._rgba_H[:] = value
            else:
                self._rgba_H[indices] = value

    def update_rgba_F(self, value, indices=None):
        if value is not None:
            if indices is None:
                self._rgba_F[:] = value
            else:
                self._rgba_F[indices] = value

    def update_rgba_h_right_B(self, value, b):
        h_right_B = np.array(list(self.M.generate_h_right_B(b)))
        self.update_rgba_H(value, indices=h_right_B)

    def update_rgba_h_right_B(self, value):
        """."""
        for b in range(self.M.num_boundaries):
            self.update_rgba_h_right_B(value, b)

    def update_rgba_F_incident_b(self, value, b):
        F_incident_b = self.M.F_incident_b(b)
        self.update_rgba_F(value, indices=F_incident_b)

    def update_rgba_F_incident_B(self, value):
        for b in range(self.M.num_boundaries):
            self.update_rgba_F_incident_b(value, b)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)

        if "radius_vertex" in kwargs:
            self.update_radius_V(kwargs["radius_vertex"])
        if "rgba_vertex" in kwargs:
            self.update_rgba_V(kwargs["rgba_vertex"])
        if "rgba_half_edge" in kwargs:
            self.update_rgba_H(kwargs["rgba_half_edge"])
        if "rgba_face" in kwargs:
            self.update_rgba_F(kwargs["rgba_face"])
        if "rgba_boundary_half_edge" in kwargs:
            self.update_rgba_h_right_B(kwargs["rgba_boundary_half_edge"])
        if "rgba_boundary_face" in kwargs:
            self.update_rgba_F_incident_B(kwargs["rgba_boundary_face"])

    def set_visual_defaults(self):
        self.show_surface = True
        self.show_halfedges = True
        self.show_edges = False
        self.show_vertices = False
        self.show_plot_axes = False
        self.color_by_rgba_V = False
        self.view = None
        self.radius_vertex = 0.0125
        self.rgba_vertex = [0.7057, 0.0156, 0.1502, 1.0]
        self.rgba_edge = [1.0, 0.498, 0.0, 1.0]
        self.rgba_half_edge = [1.0, 0.498, 0.0, 1.0]
        self.rgba_face = [0.0, 0.63335, 0.05295, 0.65]
        self.radius_V = self.update_radius_V(self.radius_vertex)
        self.rgba_V = self.update_rgba_V(self.rgba_vertex)
        self.rgba_H = self.update_rgba_H(self.rgba_half_edge)
        self.rgba_F = self.update_rgba_F(self.rgba_face)

    def add_vector_field(self, points, vectors, rgba=None, name="", mask_points=1):
        vec_field = {
            "points": points,
            "vectors": vectors,
            "rgba": rgba,
            "name": name,
            "mask_points": mask_points,
        }
        self.vector_field_data.append(vec_field)

    def get_decimated_mesh(self, target_faces=1000):
        Vsimp, Fsimp, indicesVsimp = decimate_VF(
            self.M.xyz_coord_V, self.M.V_of_F, target_faces=target_faces
        )
        Msimp = self.M.from_vf_data(Vsimp, Fsimp)
        previous_flips = 0
        while True:
            num_flips = Msimp.flip_non_delaunay()
            if num_flips >= previous_flips:
                break
            else:
                previous_flips = num_flips

        return Msimp, indicesVsimp

    def get_fig_path(self):
        # image_name = f"{image_prefix}_%0{index_length}d.{image_format}"
        # image_name = f"{self.image_prefix}_{self.image_count:0>self.image_index_length}.{self.image_format}"
        image_name = f"{self.image_prefix}_{self.image_count:0{self.image_index_length}d}.{self.image_format}"
        image_path = os.path.join(self.image_dir, image_name)
        return image_path

    def add_vertices_to_fig(self, downsampled=False):
        if downsampled:
            V = self.Msimp.xyz_coord_V
            V_indices = self.indicesVsimp
            radius_V = self.radius_V[V_indices]
            rgba_V = self.rgba_V[V_indices]
        else:
            V = self.M.xyz_coord_V
            radius_V = self.radius_V
            rgba_V = self.rgba_V
        ####################################################
        vert_cloud_kwargs = {
            "name": "vert_cloud",
            "scale_mode": "vector",
            "scale_factor": 1.0,
        }
        rad_vecs = np.zeros_like(V)
        rad_vecs[:, 0] = radius_V
        rgba = rgba_V
        rgba_int = (rgba * 255).round().astype(int)
        color_scalars = np.linspace(0, 1, rgba_int.shape[0])

        vert_cloud = mlab.points3d(*V.T, **vert_cloud_kwargs)
        vert_cloud.glyph.glyph.clamping = False

        vert_cloud.mlab_source.dataset.point_data.vectors = rad_vecs
        vert_cloud.mlab_source.dataset.point_data.vectors.name = "radius_vectors"

        vert_cloud.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            rgba_int
        )
        vert_cloud.module_manager.scalar_lut_manager.lut.table = rgba_int
        vert_cloud.module_manager.lut_data_mode = "point data"
        vert_cloud.mlab_source.dataset.point_data.scalars = color_scalars
        vert_cloud.mlab_source.dataset.point_data.scalars.name = "vertex_colors"
        vert_cloud.mlab_source.update()
        # vert_cloud2 = mlab.pipeline.set_active_attribute(
        #     vert_cloud, point_scalars="vertex_colors", point_vectors="radius_vectors"
        # )
        mlab.pipeline.set_active_attribute(
            vert_cloud, point_scalars="vertex_colors", point_vectors="radius_vectors"
        )
        return vert_cloud

    def add_next_cycle_to_fig(self, h_start=None, h_end=None, max_num=1000):
        if h_end is None:
            h_end = h_start
        if h_start is None:
            h_start = 0
        h = h_start
        H = []
        while True:
            H.append(h)
            h = self.M.h_next_h(h)
            if h == h_end:
                break
            if h == h_start:
                break
            if len(H) > max_num:
                break
        H = np.array(H, dtype=INT_TYPE)
        V = self.M.xyz_coord_v(self.M.v_origin_h(H))
        edge_radius = 0.3 * self.radius_vertex
        rgb_edge = self.rgba_edge[:3]
        ####################################################
        curve_kwargs = {
            "name": "edge_curve",
            "color": rgb_edge,
            "tube_radius": edge_radius,
        }
        edge_tube = mlab.plot3d(*V.T, **curve_kwargs)
        return edge_tube

    def add_edge_curve_to_fig(self, ordered_V=True):

        if ordered_V:
            h_start = self.M.h_out_v(0)
            max_num = self.M.num_vertices
            return self.add_next_cycle_to_fig(h_start=h_start, max_num=max_num)
        else:
            raise NotImplementedError("Unordered edge curves not implemented")

    def add_wireframe_surface_to_fig(self, fig, downsampled=False):
        if downsampled:
            V = self.Msimp.xyz_coord_V
            F = self.Msimp.V_of_F
        else:
            V = self.M.xyz_coord_V
            F = self.M.V_of_F
        ####################################################
        edge_mesh_kwargs = {
            "name": "edge_mesh",
            "color": tuple(self.rgba_edge[:3]),
            "representation": "wireframe",
        }
        mesh = mlab.triangular_mesh(*V.T, F, **edge_mesh_kwargs, figure=fig)
        return mesh

    def add_face_colored_surface_to_fig(self, downsampled=False):
        if downsampled:
            V = self.Msimp.xyz_coord_V
            F = self.Msimp.V_of_F
            rgba = np.array(self.Msimp.num_faces * [self.rgba_face])
        else:
            V = self.M.xyz_coord_V
            F = self.M.V_of_F
            rgba = self.rgba_F

        ####################################################
        edge_mesh_kwargs = {
            "name": "edge_mesh",
            "representation": "surface",
        }
        mesh = mlab.triangular_mesh(*V.T, F, **edge_mesh_kwargs)
        rgba_int = (rgba * 255).round().astype(int)
        color_scalars = np.linspace(0, 1, rgba.shape[0])
        mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(rgba)
        mesh.module_manager.scalar_lut_manager.lut.table = rgba_int

        mesh.module_manager.lut_data_mode = "cell data"
        mesh.mlab_source.dataset.cell_data.scalars = color_scalars
        mesh.mlab_source.dataset.cell_data.scalars.name = "surface_colors"
        mesh.mlab_source.update()
        # mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars="face colors")
        mlab.pipeline.set_active_attribute(mesh, cell_scalars="surface_colors")
        return mesh

    def add_vertex_colored_surface_to_fig(self, downsampled=False):
        if downsampled:
            V = self.Msimp.xyz_coord_V
            F = self.Msimp.V_of_F
            rgba = self.rgba_V[self.indicesVsimp]
        else:
            V = self.M.xyz_coord_V
            F = self.M.V_of_F
            rgba = self.rgba_V

        ####################################################
        edge_mesh_kwargs = {
            "name": "edge_mesh",
            "representation": "surface",
        }
        mesh = mlab.triangular_mesh(*V.T, F, **edge_mesh_kwargs)

        rgba_int = (rgba * 255).round().astype(int)
        color_scalars = np.linspace(0, 1, rgba.shape[0])
        mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(rgba)
        mesh.module_manager.scalar_lut_manager.lut.table = rgba_int

        mesh.module_manager.lut_data_mode = "point data"
        mesh.mlab_source.dataset.point_data.scalars = color_scalars
        mesh.mlab_source.dataset.point_data.scalars.name = "surface_colors"
        mesh.mlab_source.update()
        mlab.pipeline.set_active_attribute(mesh, point_scalars="surface_colors")

        return mesh

    def add_vector_field_to_fig(
        self,
        points,
        vectors,
        rgba=None,
        name="",
        tip_length=0.25,
        tip_radius=0.03,
        shaft_radius=0.01,
        mask_points=1,
    ):
        clamping = False
        color_mode = "color_by_scalar"
        quiver3d_kwargs = {
            "name": name,
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": 1.0,
            "mask_points": mask_points,
        }
        if rgba is None:
            rgba = np.array(len(points) * [self.colors["black"]])
        elif len(np.shape(rgba)) == 1:
            rgba = np.array(len(points) * [rgba])
        elif len(rgba) != len(points):
            raise ValueError("len(rgba) != len(points)")
        rgba_int = (rgba * 255).round().astype(int)
        color_scalars = np.linspace(0, 1, rgba_int.shape[0])
        vfield = mlab.quiver3d(*points.T, *vectors.T, **quiver3d_kwargs)
        ###
        # vfield_src = mlab.pipeline.vector_field(*points.T, *vectors.T)
        # vfield = mlab.pipeline.vectors(vfield_src, mask_points=20, **quiver3d_kwargs)
        ####
        vfield.glyph.glyph.clamping = clamping
        vfield.glyph.glyph_source.glyph_source.tip_length = tip_length
        vfield.glyph.glyph_source.glyph_source.tip_radius = tip_radius
        vfield.glyph.glyph_source.glyph_source.shaft_radius = shaft_radius
        vfield.glyph.color_mode = color_mode

        vfield.module_manager.scalar_lut_manager.lut.number_of_colors = len(rgba)
        vfield.module_manager.scalar_lut_manager.lut.table = rgba_int
        vfield.mlab_source.dataset.point_data.scalars = color_scalars
        vfield.mlab_source.dataset.point_data.scalars.name = f"{name}_colors"
        vfield.mlab_source.update()
        return vfield

    def add_half_edges_fig(self, downsampled=False):
        if downsampled:
            points, vectors = get_half_edge_vector_field(self.Msimp)
            rgba = np.array(self.Msimp.num_half_edges * [self.rgba_half_edge])
        else:
            points, vectors = get_half_edge_vector_field(self.M)
            rgba = self.rgba_H
        name = "half_edges"
        tip_length = 0.25
        tip_radius = 0.03
        shaft_radius = 0.01
        return self.add_vector_field_to_fig(
            points,
            vectors,
            rgba=rgba,
            name=name,
            tip_length=tip_length,
            tip_radius=tip_radius,
            shaft_radius=shaft_radius,
        )

    def options_plot(
        self,
        show=True,
        save=False,
        title="",
        show_wireframe_surface=True,
        show_face_colored_surface=False,
        show_vertex_colored_surface=False,
        show_vertices=True,
        show_half_edges=False,
        show_vector_fields=True,
        show_plot_axes=False,
        view=None,
        figsize=(2180, 2180),
        fig_path=None,
        downsampled=False,
    ):
        """
        fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
        """
        mlab.options.offscreen = not show
        fig = mlab.figure(title, size=figsize)
        ################################
        if show_vector_fields:
            for data in self.vector_field_data:
                self.add_vector_field_to_fig(**data)

        if show_wireframe_surface:
            wireframe_surface = self.add_wireframe_surface_to_fig(
                fig, downsampled=downsampled
            )
        if show_face_colored_surface:
            face_colored_surface = self.add_face_colored_surface_to_fig(
                downsampled=downsampled
            )
        if show_vertex_colored_surface:
            vertex_colored_surface = self.add_vertex_colored_surface_to_fig(
                downsampled=downsampled
            )
        if show_vertices:
            vert_cloud = self.add_vertices_to_fig(downsampled=downsampled)
        if show_half_edges:
            half_edge_vector_field = self.add_half_edges_fig(downsampled=downsampled)

        if show_plot_axes:
            mlab.axes()
            mlab.orientation_axes()
        if view is not None:
            mlab.view(**view)

        # mview = mlab.view()
        # print(mview)
        if save:
            if fig_path is None:
                print("fig_path is None")
            else:
                mlab.savefig(fig_path, figure=fig, size=figsize)
        if show:
            mlab.show()

        mlab.close(all=True)

    def curve_plot(
        self,
        show=True,
        save=False,
        title="",
        show_vertices=True,
        show_edges=True,
        show_plot_axes=False,
        view=None,
        figsize=(2180, 2180),
        fig_path=None,
    ):
        """
        fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
        """
        mlab.options.offscreen = not show
        fig = mlab.figure(title, size=figsize)
        ################################

        if show_vertices:
            vert_cloud = self.add_vertices_to_fig()
        if show_edges:
            edge_tube = self.add_edge_curve_to_fig()

        if show_plot_axes:
            mlab.axes()
            mlab.orientation_axes()
        if view is not None:
            mlab.view(**view)

        # mview = mlab.view()
        # print(mview)
        if save:
            if fig_path is None:
                print("fig_path is None")
            else:
                mlab.savefig(fig_path, figure=fig, size=figsize)
        if show:
            mlab.show()

        mlab.close(all=True)

    def plot(self, show=True, save=False, title="", downsampled=False):
        """
        Default plot with options set in __init__
        """
        if save:
            fig_path = self.get_fig_path()
            self.image_count += 1
        else:
            fig_path = None
        self.options_plot(
            show=show,
            save=save,
            title=title,
            show_wireframe_surface=self.show_wireframe_surface,
            show_face_colored_surface=self.show_face_colored_surface,
            show_vertex_colored_surface=self.show_vertex_colored_surface,
            show_vertices=self.show_vertices,
            show_half_edges=self.show_half_edges,
            show_vector_fields=self.show_vector_fields,
            show_plot_axes=self.show_plot_axes,
            view=self.view,
            figsize=self.figsize,
            fig_path=fig_path,
            downsampled=downsampled,
        )

    def fast_plot(self, show=True, save=False, title="", downsampled=False):
        """
        Default plot with options set in __init__
        """
        if save:
            fig_path = self.get_fig_path()
            self.image_count += 1
        else:
            fig_path = None
        self.options_plot(
            show=show,
            save=save,
            title=title,
            show_wireframe_surface=self.show_wireframe_surface,
            show_face_colored_surface=self.show_face_colored_surface,
            show_vertex_colored_surface=self.show_vertex_colored_surface,
            show_vertices=self.show_vertices,
            show_half_edges=self.show_half_edges,
            show_vector_fields=self.show_vector_fields,
            show_plot_axes=self.show_plot_axes,
            view=self.view,
            figsize=self.figsize,
            fig_path=fig_path,
            downsampled=downsampled,
        )

    def clear_mlab_figures(self):
        mlab.close(all=True)

    def clear_vector_field_data(self):
        self.vector_field_data = []

    def movie(self):
        movie(
            self.image_dir,
            image_format=self.image_format,
            image_prefix=self.image_prefix,
            index_length=self.image_index_length,
            movie_name=self.movie_name,
            movie_dir=self.movie_dir,
            movie_format=self.movie_format,
        )

    def apply_fun_iter(self, fun, num_iters=1):
        m = self.M
        self.plot(save=True, show=False, title=f"iter_{0}")
        for i in range(num_iters):
            print(f"Applying fun to mesh {i+1} of {num_iters}")
            fun(m)
            self.plot(save=True, show=False, title=f"iter_{i+1}")
        self.movie()

    def vec_field_apply_fun_iter(self, fun, num_iters=1):
        m = self.M
        self.plot(save=True, show=False, title=f"iter_{0}")
        for i in range(num_iters):
            print(f"Applying fun to mesh {i+1} of {num_iters}")
            points, vectors = fun(m)
            self.clear_vector_field_data()
            self.add_vector_field(points, vectors)
            com = np.sum(m.xyz_coord_V, axis=0) / m.num_vertices
            self.view["focalpoint"] = com
            self.plot(save=True, show=False, title=f"iter_{i+1}")
        self.movie()

    ###############################################################
    # to be deprecated
    def _color_boundary_H(self, b, rgba=None):
        if rgba is None:
            rgba = self.colors["boundary_half_edge_default"]
        h = self.M.h_right_B(b)
        Hindices = list(self.M.generate_H_next_h(h))
        self._rgba_H[indices] = rgba


class MultiMeshViewer:
    def __init__(
        self,
        meshes,
        image_dir="./output/temp_images",
        image_count=0,
        image_format="png",
        image_prefix="frame",
        image_index_length=5,
        movie_name="movie",
        movie_format="mp4",
        show_plot_axes=False,
        view=None,
        figsize=(720, 720),
        mesh_params=None,
    ):
        self.meshes = meshes
        self.params = {
            "image_dir": image_dir,
            "image_count": image_count,
            "image_format": image_format,
            "image_prefix": image_prefix,
            "image_index_length": image_index_length,
            "movie_name": movie_name,
            "movie_format": movie_format,
            "show_plot_axes": show_plot_axes,
            "view": view,
            "figsize": figsize,
        }
        if mesh_params is None:
            self.mesh_params = [self.params.copy() for _ in range(len(meshes))]

        self.mesh_viewers = [
            MeshViewer(mesh, **params) for mesh, params in zip(meshes, self.mesh_params)
        ]
        ###############################################################
        # image/movie output options
        ###############################################################
        self.image_dir = image_dir
        self.image_count = image_count
        self.image_format = image_format
        self.image_prefix = image_prefix
        self.image_index_length = image_index_length
        self.movie_name = movie_name
        self.movie_format = movie_format
        ###############################################################
        # Size-independent params and defaults
        ###############################################################
        self.colors = {
            "black": [0.0, 0.0, 0.0, 1.0],
            "white": [1.0, 1.0, 1.0, 1.0],
            "transparent": [0.0, 0.0, 0.0, 0.0],
            "red": [0.8392, 0.1529, 0.1569, 1.0],
            "red10": [0.8392, 0.1529, 0.1569, 0.1],
            "red20": [0.8392, 0.1529, 0.1569, 0.2],
            "red50": [0.8392, 0.1529, 0.1569, 0.5],
            "red80": [0.8392, 0.1529, 0.1569, 0.8],
            "green": [0.0, 0.6745, 0.2784, 1.0],
            "green10": [0.0, 0.6745, 0.2784, 0.1],
            "green20": [0.0, 0.6745, 0.2784, 0.2],
            "green50": [0.0, 0.6745, 0.2784, 0.5],
            "green80": [0.0, 0.6745, 0.2784, 0.8],
            "blue": [0.0, 0.4471, 0.6980, 1.0],
            "blue10": [0.0, 0.4471, 0.6980, 0.1],
            "blue20": [0.0, 0.4471, 0.6980, 0.2],
            "blue50": [0.0, 0.4471, 0.6980, 0.5],
            "blue80": [0.0, 0.4471, 0.6980, 0.8],
            "yellow": [1.0, 0.8431, 0.0, 1.0],
            "yellow10": [1.0, 0.8431, 0.0, 0.1],
            "yellow20": [1.0, 0.8431, 0.0, 0.2],
            "yellow50": [1.0, 0.8431, 0.0, 0.5],
            "yellow80": [1.0, 0.8431, 0.0, 0.8],
            "cyan": [0.0, 0.8431, 0.8431, 1.0],
            "cyan10": [0.0, 0.8431, 0.8431, 0.1],
            "cyan20": [0.0, 0.8431, 0.8431, 0.2],
            "cyan50": [0.0, 0.8431, 0.8431, 0.5],
            "cyan80": [0.0, 0.8431, 0.8431, 0.8],
            "magenta": [0.8784, 0.0, 0.8784, 1.0],
            "magenta10": [0.8784, 0.0, 0.8784, 0.1],
            "magenta20": [0.8784, 0.0, 0.8784, 0.2],
            "magenta50": [0.8784, 0.0, 0.8784, 0.5],
            "magenta80": [0.8784, 0.0, 0.8784, 0.8],
            "orange": [1.0, 0.5490, 0.0, 1.0],
            "orange10": [1.0, 0.5490, 0.0, 0.1],
            "orange20": [1.0, 0.5490, 0.0, 0.2],
            "orange50": [1.0, 0.5490, 0.0, 0.5],
            "orange80": [1.0, 0.5490, 0.0, 0.8],
            "purple": [0.5804, 0.0, 0.8275, 1.0],
            "purple10": [0.5804, 0.0, 0.8275, 0.1],
            "purple20": [0.5804, 0.0, 0.8275, 0.2],
            "purple50": [0.5804, 0.0, 0.8275, 0.5],
            "purple80": [0.5804, 0.0, 0.8275, 0.8],
            "vertex_default": [0.7057, 0.0156, 0.1502, 1.0],
            "half_edge_default": [1.0, 0.498, 0.0, 1.0],
            "face_default": [0.0, 0.63335, 0.05295, 0.65],
            "boundary_half_edge_default": [0.0, 0.4471, 0.6980, 1.0],
        }
        ################
        self.show_plot_axes = show_plot_axes
        self.view = view
        self.figsize = figsize

    @property
    def num_meshes(self):
        return len(self.meshes)

    def get_fig_path(self):
        image_name = f"{self.image_prefix}_{self.image_count:0{self.image_index_length}d}.{self.image_format}"
        image_path = os.path.join(self.image_dir, image_name)
        return image_path

    def update_mesh_params(self, mesh_index, **kwargs):
        self.mesh_params[mesh_index].update(**kwargs)
        self.mesh_viewers[mesh_index].update(**kwargs)

    def options_plot(
        self,
        show=True,
        save=False,
        title="",
        show_plot_axes=False,
        view=None,
        figsize=(720, 720),
        fig_path=None,
    ):
        mlab.options.offscreen = not show
        fig = mlab.figure(title, size=figsize)
        for viewer in self.mesh_viewers:
            if viewer.show_wireframe_surface:
                viewer.add_wireframe_surface_to_fig(fig)
            if viewer.show_face_colored_surface:
                viewer.add_face_colored_surface_to_fig()
            if viewer.show_vertex_colored_surface:
                viewer.add_vertex_colored_surface_to_fig()
            if viewer.show_vertices:
                viewer.add_vertices_to_fig()
            if viewer.show_half_edges:
                viewer.add_half_edges_fig()
            if viewer.show_vector_fields:
                for data in viewer.vector_field_data:
                    viewer.add_vector_field_to_fig(**data)
        if show_plot_axes:
            mlab.axes()
            mlab.orientation_axes()
        if view is not None:
            mlab.view(**view)
        mview = mlab.view()
        print(mview)
        if save:
            if fig_path is None:
                print("fig_path is None")
            else:
                mlab.savefig(fig_path, figure=fig, size=figsize)
        if show:
            mlab.show()

        mlab.close(all=True)

    def plot(self, show=True, save=False, title=""):
        """
        Default plot with options set in __init__
        """
        if save:
            fig_path = self.get_fig_path()
            self.image_count += 1
        else:
            fig_path = None
        self.options_plot(
            show=show,
            save=save,
            title=title,
            show_plot_axes=self.show_plot_axes,
            view=self.view,
            figsize=self.figsize,
            fig_path=fig_path,
        )

    def movie(self):
        movie(
            self.image_dir,
            image_format=self.image_format,
            image_prefix=self.image_prefix,
            index_length=self.image_index_length,
            movie_name=self.movie_name,
            movie_dir=self.image_dir,
            movie_format=self.movie_format,
        )

    def spread_meshes(self, pad=0.05, axis=-1):
        diameters = np.zeros(self.num_meshes)
        for _, m in enumerate(self.meshes):
            m.xyz_coord_V -= np.mean(m.xyz_coord_V, axis=0)
            diameters[_] = np.ptp(m.xyz_coord_V[:, axis])
        diameters += pad * np.mean(diameters)
        translations = np.cumsum(diameters)
        total_diameter = translations[-1]
        translations[1:] = translations[:-1]
        translations[0] = 0.0
        translations -= total_diameter / 2
        for _, m in enumerate(self.meshes):
            m.xyz_coord_V[:, axis] += translations[_]
