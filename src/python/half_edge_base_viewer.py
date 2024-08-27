from mayavi import mlab
import numpy as np
import os
import subprocess
import pyvista as pv
from src.python.half_edge_base_mesh import HalfEdgeMeshBase
from src.python.global_vars import _INT_TYPE_, _FLOAT_TYPE_

# def downsample(V, F, target_faces=1000):
#     vertices = V
#     faces = F
#     num_faces = faces.shape[0]
#     if num_faces < target_faces:
#         original_indices = np.arange(vertices.shape[0])
#         return vertices, faces, original_indices
#     target_reduction = target_faces / num_faces

#     faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()

#     # Create a PyVista mesh
#     mesh = pv.PolyData(vertices, faces_pv)

#     # Add original point IDs to the mesh
#     mesh.point_data["orig_ids"] = np.arange(mesh.n_points)

#     # Simplify the mesh (reduce the number of faces) and preserve original point IDs
#     simplified_mesh = mesh.decimate_pro(target_reduction, preserve_topology=True)

#     # Extract simplified vertices and faces
#     simplified_vertices = simplified_mesh.points
#     simplified_faces = simplified_mesh.faces.reshape(-1, 4)[:, 1:]

#     # Extract the indices of the original vertices used in the simplified mesh
#     original_indices = np.unique(simplified_mesh.point_data["orig_ids"])

#     return np.array(simplified_vertices), simplified_faces, np.array(original_indices)


class MeshViewer(HalfEdgeMeshBase):
    def __init__(
        self,
        # HalfEdgeMesh array data
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
        # image/movie output options
        image_dir="./output/temp_images",
        image_count=0,
        image_format="png",
        image_prefix="frame",
        image_index_length=5,
        movie_name="movie",
        movie_format="mp4",
        # Vertex/Edge/Face visual parameters
        v_radius=0.0125,
        v_rgba=[0.7057, 0.0156, 0.1502, 1.0],
        e_rgba=[1.0, 0.498, 0.0, 1.0],
        h_rgba=[1.0, 0.498, 0.0, 1.0],
        f_rgba=[0.0, 0.63335, 0.05295, 0.65],
        # mlab data that depends on mesh size
        radius_V=None,
        rgba_V=None,
        rgba_H=None,
        rgba_F=None,
        # Additional vector fields to plot
        vector_field_data=[],
        # mlab data that does NOT depend on mesh size
        show_surface=True,
        show_halfedges=True,
        show_edges=False,
        show_vertices=False,
        show_plot_axes=False,
        # color_by_rgba_V=False,
        view=None,
        figsize=(2180, 2180),
        target_faces=None,
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
        if target_faces is not None:
            self.V, self.F, self.V_indices = self.downsample(target_faces=target_faces)
        else:
            self.V = xyz_coord_V
            self.F = self.V_of_F
            self.V_indices = np.arange(len(xyz_coord_V))

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
            "rgba_V": [0.7057, 0.0156, 0.1502, 1.0],
            "E_rgba": [1.0, 0.498, 0.0, 1.0],
            "F_rgba": [0.0, 0.63335, 0.05295, 0.65],
        }
        ################
        self.show_surface = show_surface
        self.show_halfedges = show_halfedges
        self.show_edges = show_edges
        self.show_vertices = show_vertices
        self.show_plot_axes = show_plot_axes
        # self.color_by_rgba_V = color_by_rgba_V
        self.view = view
        self.figsize = figsize
        self.v_radius = v_radius
        self.v_rgba = v_rgba
        self.h_rgba = h_rgba
        self.f_rgba = f_rgba

        self.e_rgba = e_rgba

        ################################
        # Size-dependent params
        ################################
        if radius_V is not None:
            self.radius_V = radius_V
        else:
            self.radius_V = np.array(self.num_vertices * [v_radius])
        if rgba_V is not None:
            self.rgba_V = rgba_V
        else:
            self.rgba_V = np.array(self.num_vertices * [v_rgba])
        if rgba_H is not None:
            self.rgba_H = rgba_H
        else:
            self.rgba_H = np.array(self.num_half_edges * [h_rgba])
        if rgba_F is not None:
            self.rgba_F = rgba_F
        else:
            self.rgba_F = np.array(self.num_faces * [f_rgba])
        # self.set_V_radius(V_radius=V_radius, v_radius=v_radius)
        # self.set_rgba_V(rgba_V=rgba_V)
        # self.set_E_rgba(E_rgba=E_rgba, update_fancy_E_field=False)
        # self.set_F_rgba(F_rgba=F_rgba)
        # self.set_fancy_E_field()

        ################################
        # Additional vector field data
        ################################
        # self.fancy_mayavi_vector_fields = [
        #     FancyMayaviVectorField(*data) for data in vector_field_data
        # ]

    @property
    def radius_V(self):
        return self._radius_V

    @radius_V.setter
    def radius_V(self, value):
        self._radius_V = np.array(value, dtype=_FLOAT_TYPE_)

    @property
    def rgba_V(self):
        return self._rgba_V

    @rgba_V.setter
    def rgba_V(self, value):
        self._rgba_V = np.array(value, dtype=_FLOAT_TYPE_)

    @property
    def rgba_H(self):
        return self._rgba_H

    @rgba_H.setter
    def rgba_H(self, value):
        self._rgba_H = np.array(value, dtype=_FLOAT_TYPE_)

    @property
    def rgba_F(self):
        return self._rgba_F

    @rgba_F.setter
    def rgba_F(self, value):
        self._rgba_F = np.array(value, dtype=_FLOAT_TYPE_)

    def update_radius_V(self, value, indices=None):
        if indices is None:
            self.radius_V[:] = value
        else:
            self.radius_V[indices] = value

    def update_rgba_V(self, value, indices=None):
        if indices is None:
            self.rgba_V[:] = value
        else:
            self.rgba_V[indices] = value

    def update_rgba_H(self, value, indices=None):
        if indices is None:
            self.rgba_H[:] = value
        else:
            self.rgba_H[indices] = value

    def update_rgba_F(self, value, indices=None):
        if indices is None:
            self.rgba_F[:] = value
        else:
            self.rgba_F[indices] = value

    def set_visual_defaults(self):
        self.show_surface = True
        self.show_halfedges = True
        self.show_edges = False
        self.show_vertices = False
        self.show_plot_axes = False
        self.color_by_rgba_V = False
        self.view = None
        self.v_radius = 0.0125
        self.v_rgba = [0.7057, 0.0156, 0.1502, 1.0]
        self.e_rgba = [1.0, 0.498, 0.0, 1.0]
        self.h_rgba = [1.0, 0.498, 0.0, 1.0]
        self.f_rgba = [0.0, 0.63335, 0.05295, 0.65]
        self.radius_V = self.update_radius_V(self.v_radius)
        self.rgba_V = self.update_rgba_V(self.v_rgba)
        self.rgba_H = self.update_rgba_H(self.h_rgba)
        self.rgba_F = self.update_rgba_F(self.f_rgba)

    def downsample(self, target_faces=1000, boundary_vertex_deletion=True):
        F = self.V_of_F
        V = self.xyz_coord_V
        num_faces = F.shape[0]
        num_vertices = V.shape[0]
        #
        if num_faces <= target_faces:
            print("num_faces <= target_faces")
            V_indices = np.arange(V.shape[0])
            return V, F, V_indices
        target_reduction = 1 - target_faces / num_faces

        F_pv = np.zeros((num_faces, 4), dtype="int32")
        F_pv[:, 0] = 3
        F_pv[:, 1:] = F
        F_pv = F_pv.ravel()

        # Create a PyVista mesh
        M = pv.PolyData(V, F_pv)

        # Add original point IDs to the mesh
        M.point_data["V_indices"] = np.arange(num_vertices)

        # Simplify the mesh (reduce the number of faces) and preserve original point IDs
        M_simp = M.decimate_pro(
            target_reduction,
            preserve_topology=True,
            boundary_vertex_deletion=boundary_vertex_deletion,
        )

        # Extract simplified vertices and faces
        V_simp = np.array(M_simp.points)
        F_simp = M_simp.faces.reshape(-1, 4)[:, 1:]

        # Extract the indices of the original vertices used in the simplified mesh
        V_indices = np.unique(M_simp.point_data["V_indices"])
        return V_simp, F_simp, V_indices

    def simple_plot(
        self,
        show=True,
        save=False,
        title="",
        representation="wireframe",
    ):
        """
        fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
        """
        mlab.options.offscreen = not show

        fig = mlab.figure(title, size=self.figsize)

        ################################
        V, F, V_indices = self.V, self.F, self.V_indices

        # brane_mesh

        brane_mesh_kwargs = {
            "name": "brane_mesh",
            "representation": representation,
        }

        brane_mesh = mlab.triangular_mesh(*V.T, F, **brane_mesh_kwargs)

        F_rgba_int = (self.rgba_V[V_indices] * 255).round().astype(int)
        F_color_scalars = np.linspace(0, 1, F_rgba_int.shape[0])
        brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            F_rgba_int
        )
        brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba_int
        brane_mesh.module_manager.lut_data_mode = "point data"
        brane_mesh.mlab_source.dataset.point_data.scalars = F_color_scalars
        brane_mesh.mlab_source.dataset.point_data.scalars.name = "face colors"
        brane_mesh.mlab_source.update()
        brane_mesh2 = mlab.pipeline.set_active_attribute(
            brane_mesh, point_scalars="face colors"
        )

        if self.show_plot_axes:
            mlab.axes()
            mlab.orientation_axes()
        # mview = mlab.view()
        # print(mview)
        if self.view is not None:
            mlab.view(**self.view)
        if save:
            fig_path = self.get_fig_path()
            self.image_count += 1
            mlab.savefig(fig_path, figure=fig, size=self.figsize)
        if show:
            mlab.show()

        mlab.close(all=True)


# h_out_V,
# v_origin_H,
# h_next_H,
# h_twin_H,
# f_left_H,
# h_bound_F,
# h_right_B,
class MeshViewerBase:
    def __init__(
        self,
        # HalfEdgeMesh array data
        V,
        h_out_V,
        v_origin_H,
        E_next,
        E_twin,
        E_face,
        F_edge,
        h_right_B=None,
        # Computes from that^ if None
        F=None,
        # Additional vector fields to plot
        vector_field_data=[],
        # mlab data that does NOT depend on mesh size
        show_surface=True,
        show_halfedges=True,
        show_edges=False,
        show_vertices=False,
        show_plot_axes=False,
        color_by_V_rgba=False,
        view=None,
        figsize=(2180, 2180),
        v_radius=0.0125,
        v_rgba=[0.7057, 0.0156, 0.1502, 1.0],
        e_rgba=[1.0, 0.498, 0.0, 1.0],
        f_rgba=[0.0, 0.63335, 0.05295, 0.65],
        # mlab data that depends on mesh size
        V_radius=None,
        V_rgba=None,
        E_rgba=None,
        F_rgba=None,
        # image and movie output options
        image_dir="./output/temp_images",
        image_count=0,
        image_format="png",
        image_prefix="frame",
        image_index_length=5,
        movie_name="movie",
        movie_format="mp4",
    ):
        ###############################################################
        # HalfEdgeMesh array data
        ###############################################################
        self.V = np.array(V)
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.E_face = E_face
        self.E_next = E_next
        self.E_twin = E_twin
        self.F_edge = F_edge
        #
        self.Nvertices = len(V)
        self.Nedges = len(v_origin_H)
        self.Nfaces = len(F_edge)
        ###################
        # if F=None then self.set_faces(F=F) computes
        # from HalfEdgeMesh data
        self.set_faces(F=F)
        ###############################################################
        # output params
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
        self.defaults = {}
        self.defaults["show_surface"] = True
        self.defaults["show_halfedges"] = True
        self.defaults["show_edges"] = False
        self.defaults["show_vertices"] = False
        self.defaults["show_plot_axes"] = False
        self.defaults["color_by_V_rgba"] = False
        self.defaults["view"] = None
        self.defaults["v_radius"] = 0.0125
        self.defaults["v_rgba"] = [0.7057, 0.0156, 0.1502, 1.0]
        self.defaults["e_rgba"] = [1.0, 0.498, 0.0, 1.0]
        self.defaults["f_rgba"] = [0.0, 0.63335, 0.05295, 0.65]
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
            "V_rgba": [0.7057, 0.0156, 0.1502, 1.0],
            "E_rgba": [1.0, 0.498, 0.0, 1.0],
            "F_rgba": [0.0, 0.63335, 0.05295, 0.65],
        }
        ################
        self.show_surface = show_surface
        self.show_halfedges = show_halfedges
        self.show_edges = show_edges
        self.show_vertices = show_vertices
        self.show_plot_axes = show_plot_axes
        self.color_by_V_rgba = color_by_V_rgba
        self.view = view
        self.figsize = figsize
        self.v_radius = v_radius
        self.v_rgba = v_rgba
        self.e_rgba = e_rgba
        self.f_rgba = f_rgba

        ################################
        # Size-dependent params
        ################################

        self.set_V_radius(V_radius=V_radius, v_radius=v_radius)
        self.set_V_rgba(V_rgba=V_rgba)
        self.set_E_rgba(E_rgba=E_rgba, update_fancy_E_field=False)
        self.set_F_rgba(F_rgba=F_rgba)
        self.set_fancy_E_field()

        ################################
        # Additional vector field data
        ################################
        self.fancy_mayavi_vector_fields = [
            FancyMayaviVectorField(*data) for data in vector_field_data
        ]

    def update_positions(self, V, update_E_field=True):
        self.V = np.array(V)
        self.update_fancy_E_field(update_vec_field=update_E_field)

    def update_mesh_topology(
        self,
        V,
        h_out_V,
        v_origin_H,
        E_next,
        E_twin,
        E_face,
        F_edge,
        F=None,
        V_radius=None,
        V_rgba=None,
        E_rgba=None,
        F_rgba=None,
    ):
        ################################
        # HalfEdgeMesh array data
        ################################
        self.V = np.array(V)
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.E_face = E_face
        self.E_next = E_next
        self.E_twin = E_twin
        self.F_edge = F_edge
        #####
        self.Nvertices = len(V)
        self.Nedges = len(self.v_origin_H)
        self.Nfaces = len(self.F_edge)
        if F is None:
            self.set_faces(F=F)

        ################################
        # Size-dependent params
        ################################
        self.set_V_radius(V_radius=V_radius)
        self.set_V_rgba(V_rgba=V_rgba)
        self.set_E_rgba(E_rgba=E_rgba)
        self.set_F_rgba(F_rgba=F_rgba)
        self.update_fancy_E_field(update_vec_field=True, update_rgba=True)

    def update_from_HalfEdgeMesh(self, halfEdgeMesh, update_connect_data=False):
        # mv = MeshViewer(*m.get_data_lists(), **viewer_kwargs)
        self.V = np.array(halfEdgeMesh.xyz_coord_V)
        self.update_fancy_E_field(update_vec_field=update_connect_data)

    ###################################################
    def set_fancy_E_field(self):
        shifted_E_field = self.get_shifted_E_field()
        self.fancy_E_field = FancyMayaviVectorField(
            *shifted_E_field, self.E_rgba, name="shifted_E_field"
        )

    def update_fancy_E_field(self, update_vec_field=True, update_rgba=True):
        if update_vec_field:
            points, vecs = self.get_shifted_E_field()
        else:
            points, vecs = None, None
        if update_rgba:
            rgba = self.E_rgba
        else:
            rgba = None
        self.fancy_E_field.update(points=points, vectors=vecs, rgba=rgba)

    def set_V_rgba(self, V_rgba=None, v_rgba=None):
        if V_rgba is not None:
            self.V_rgba = V_rgba
        if V_rgba is None and v_rgba is None:
            self.V_rgba = np.zeros((len(self.V), 4))
            self.v_rgba = self.defaults["v_rgba"].copy()
            self.V_rgba[:] = self.v_rgba
        if V_rgba is None and v_rgba is not None:
            self.V_rgba = np.zeros((len(self.V), 4))
            self.v_rgba = v_rgba
            self.V_rgba[:] = self.v_rgba

    def set_subset_V_rgba(self, rgba, indices):
        self.V_rgba[indices] = rgba

    def set_V_radius(self, V_radius=None, v_radius=None):
        if V_radius is not None:
            self.V_radius = V_radius
        if V_radius is None and v_radius is None:
            self.V_radius = np.zeros(len(self.V))
            self.v_radius = self.defaults["v_radius"]
            self.V_radius[:] = self.v_radius
        if V_radius is None and v_radius is not None:
            self.V_radius = np.zeros(len(self.V))
            self.v_radius = v_radius
            self.V_radius[:] = self.v_radius

    def set_subset_V_radius(self, radius, indices):
        # set V_radius for specific vertices
        self.V_radius[indices] = radius

    def set_E_rgba(self, E_rgba=None, e_rgba=None, update_fancy_E_field=True):
        if E_rgba is not None:
            self.E_rgba = E_rgba
        if E_rgba is None and e_rgba is None:
            self.E_rgba = np.zeros((len(self.v_origin_H), 4))
            self.e_rgba = self.defaults["e_rgba"].copy()
            self.E_rgba[:] = self.e_rgba
        if E_rgba is None and e_rgba is not None:
            self.E_rgba = np.zeros((len(self.v_origin_H), 4))
            self.e_rgba = e_rgba
            self.E_rgba[:] = self.e_rgba
        if update_fancy_E_field:
            self.update_fancy_E_field(update_vec_field=False, update_rgba=True)

    def set_subset_E_rgba(self, rgba, indices):
        self.E_rgba[indices] = rgba
        self.update_fancy_E_field(update_vec_field=False, update_rgba=True)

    def set_F_rgba(self, F_rgba=None, f_rgba=None):
        if F_rgba is not None:
            self.F_rgba = F_rgba
        if F_rgba is None and f_rgba is None:
            self.F_rgba = np.zeros((len(self.F), 4))
            self.f_rgba = self.defaults["f_rgba"].copy()
            self.F_rgba[:] = self.f_rgba
        if F_rgba is None and f_rgba is not None:
            self.F_rgba = np.zeros((len(self.F), 4))
            self.f_rgba = f_rgba
            self.F_rgba[:] = self.f_rgba

    def set_subset_F_rgba(self, rgba, indices):
        self.F_rgba[indices] = rgba

    def set_faces(self, F=None):
        if F is None:
            self.F = np.array(
                [
                    [
                        self.v_origin_H[e],
                        self.v_origin_H[self.E_next[e]],
                        self.v_origin_H[self.E_next[self.E_next[e]]],
                    ]
                    for e in self.F_edge
                ]
            )
        else:
            self.F = F

    def get_shifted_E_field_no_bdry_twin(self):
        """halfedge vector shifted toward face centroid for visualization"""
        shift_to_center = 0.15
        Ne = len(self.v_origin_H)
        # vecs = np.zeros((Ne, 3))
        # points = np.zeros((Ne, 3))
        # for e in range(Ne):
        #     v0 = self.v_origin_H[e]
        #     v1 = self.v_origin_H[self.E_next[e]]
        #     V2 = self.v_origin_H[self.E_next[self.E_next[e]]]
        #     com = (self.V[v0] + self.V[v1] + self.V[v2]) / 3
        #     points[e, :] = shift_to_center * com + (
        #         1 - shift_to_center
        #     ) * self.V[v0]
        #     vecs[e, :] = (1 - shift_to_center) * (self.V[v1] - self.V[v0])
        points_vecs = np.array(
            [
                [
                    shift_to_center
                    * (
                        self.V[self.v_origin_H[e]]
                        + self.V[self.v_origin_H[self.E_next[e]]]
                        + self.V[self.v_origin_H[self.E_next[self.E_next[e]]]]
                    )
                    / 3
                    + (1 - shift_to_center) * self.V[self.v_origin_H[e]],
                    (1 - shift_to_center)
                    * (
                        self.V[self.v_origin_H[self.E_next[e]]]
                        - self.V[self.v_origin_H[e]]
                    ),
                ]
                for e in range(Ne)
            ]
        )
        return (points_vecs[:, 0], points_vecs[:, 1])

    def get_shifted_E_field(self):
        """halfedge vector shifted toward face centroid for visualization"""
        shift_to_center = 0.15
        Ne = len(self.v_origin_H)
        # vecs = np.zeros((Ne, 3))
        # points = np.zeros((Ne, 3))
        # for e in range(Ne):
        #     v0 = self.v_origin_H[e]
        #     v1 = self.v_origin_H[self.E_next[e]]
        #     V2 = self.v_origin_H[self.E_next[self.E_next[e]]]
        #     com = (self.V[v0] + self.V[v1] + self.V[v2]) / 3
        #     points[e, :] = shift_to_center * com + (
        #         1 - shift_to_center
        #     ) * self.V[v0]
        #     vecs[e, :] = (1 - shift_to_center) * (self.V[v1] - self.V[v0])
        points_vecs = np.array(
            [
                (
                    [
                        shift_to_center
                        * (
                            self.V[self.v_origin_H[e]]
                            + self.V[self.v_origin_H[self.E_next[e]]]
                            + self.V[self.v_origin_H[self.E_next[self.E_next[e]]]]
                        )
                        / 3
                        + (1 - shift_to_center) * self.V[self.v_origin_H[e]],
                        ####################################################
                        (1 - shift_to_center)
                        * (
                            self.V[self.v_origin_H[self.E_next[e]]]
                            - self.V[self.v_origin_H[e]]
                        ),
                    ]
                    if self.E_face[e] != -1  # if on the boundary
                    else [
                        shift_to_center  # shift AWAY from twin face centroid
                        * (
                            (
                                self.V[self.v_origin_H[e]]
                                + self.V[self.v_origin_H[self.E_twin[e]]]
                            )
                            - (
                                self.V[self.v_origin_H[self.E_twin[e]]]
                                + self.V[self.v_origin_H[self.E_next[self.E_twin[e]]]]
                                + self.V[
                                    self.v_origin_H[
                                        self.E_next[self.E_next[self.E_twin[e]]]
                                    ]
                                ]
                            )
                            / 3
                        )
                        + (1 - shift_to_center) * self.V[self.v_origin_H[e]],
                        ####################################################
                        (1 - shift_to_center)
                        * (
                            self.V[self.v_origin_H[self.E_next[e]]]
                            - self.V[self.v_origin_H[e]]
                        ),
                    ]
                )
                for e in range(Ne)
            ]
        )
        return (points_vecs[:, 0], points_vecs[:, 1])

    def get_fig_path(self):
        # image_name = f"{image_prefix}_%0{index_length}d.{image_format}"
        # image_name = f"{self.image_prefix}_{self.image_count:0>self.image_index_length}.{self.image_format}"
        image_name = f"{self.image_prefix}_{self.image_count:0{self.image_index_length}d}.{self.image_format}"
        image_path = os.path.join(self.image_dir, image_name)
        return image_path

    def plot(self, show=True, save=False, title=""):
        """
        fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
        """
        mlab.options.offscreen = not show

        fig = mlab.figure(title, size=self.figsize)

        ################################
        # vert_cloud
        if self.show_vertices:
            vert_cloud_kwargs = {
                "name": "vert_cloud",
                "scale_mode": "vector",
                "scale_factor": 1.0,
            }

            vert_cloud = mlab.points3d(*self.V.T, **vert_cloud_kwargs)
            vert_cloud.glyph.glyph.clamping = False
            V_rad_vecs = np.zeros_like(self.V)
            V_rad_vecs[:, 0] = self.V_radius
            vert_cloud.mlab_source.dataset.point_data.vectors = V_rad_vecs
            vert_cloud.mlab_source.dataset.point_data.vectors.name = "vertex rads"

            V_rgba_int = (self.V_rgba * 255).round().astype(int)
            V_color_scalars = np.linspace(0, 1, V_rgba_int.shape[0])
            vert_cloud.module_manager.scalar_lut_manager.lut.number_of_colors = len(
                V_rgba_int
            )

            vert_cloud.module_manager.scalar_lut_manager.lut.table = V_rgba_int
            vert_cloud.module_manager.lut_data_mode = "point data"
            vert_cloud.mlab_source.dataset.point_data.scalars = V_color_scalars
            vert_cloud.mlab_source.dataset.point_data.scalars.name = "vertex colors"
            vert_cloud.mlab_source.update()
            vert_cloud2 = mlab.pipeline.set_active_attribute(
                vert_cloud, point_scalars="vertex colors", point_vectors="vertex rads"
            )
        ################################
        # brane_mesh
        if self.show_surface:
            brane_mesh_kwargs = {
                "name": "brane_mesh",
                "representation": "surface",
            }

            brane_mesh = mlab.triangular_mesh(*self.V.T, self.F, **brane_mesh_kwargs)
            if self.color_by_V_rgba:
                F_rgba_int = (self.V_rgba * 255).round().astype(int)
                F_color_scalars = np.linspace(0, 1, F_rgba_int.shape[0])
                brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(
                    F_rgba_int
                )
                brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba_int
                brane_mesh.module_manager.lut_data_mode = "point data"
                brane_mesh.mlab_source.dataset.point_data.scalars = F_color_scalars
                brane_mesh.mlab_source.dataset.point_data.scalars.name = "face colors"
                brane_mesh.mlab_source.update()
                brane_mesh2 = mlab.pipeline.set_active_attribute(
                    brane_mesh, point_scalars="face colors"
                )
            else:
                F_rgba_int = (self.F_rgba * 255).round().astype(int)
                F_color_scalars = np.linspace(0, 1, F_rgba_int.shape[0])

                brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(
                    F_rgba_int
                )
                brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba_int
                brane_mesh.module_manager.lut_data_mode = "cell data"
                brane_mesh.mlab_source.dataset.cell_data.scalars = F_color_scalars
                brane_mesh.mlab_source.dataset.cell_data.scalars.name = "face colors"
                brane_mesh.mlab_source.update()
                brane_mesh2 = mlab.pipeline.set_active_attribute(
                    brane_mesh, cell_scalars="face colors"
                )
            # # # #  #surf = mlab.pipeline.surface(brane_mesh)

        ################################
        # edge_mesh
        if self.show_edges:
            edge_mesh_kwargs = {
                "name": "edge_mesh",
                # "color": (1.0, 0.498, 0.0),
                "color": self.colors["brane_orange"],
                "representation": "wireframe",
            }

            edge_mesh = mlab.triangular_mesh(*self.V.T, self.F, **edge_mesh_kwargs)
        ###############################
        # hedge vecs
        hedge_vfield = self.fancy_E_field.plot()
        ###############################
        for fvf in self.fancy_mayavi_vector_fields:
            fvf.plot()

        if self.show_plot_axes:
            mlab.axes()
            mlab.orientation_axes()
        # mview = mlab.view()
        # print(mview)
        if self.view is not None:
            mlab.view(**self.view)
        if save:
            fig_path = self.get_fig_path()
            self.image_count += 1
            mlab.savefig(fig_path, figure=fig, size=self.figsize)
        if show:
            mlab.show()

        mlab.close(all=True)

    def simple_plot(
        self,
        show=True,
        save=False,
        title="",
        target_faces=1000,
        representation="wireframe",
    ):
        """
        fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
        """
        mlab.options.offscreen = not show

        fig = mlab.figure(title, size=self.figsize)

        ################################
        V, F, Iv = downsample(self.V, self.F, target_faces=target_faces)

        # brane_mesh
        if self.show_surface:
            brane_mesh_kwargs = {
                "name": "brane_mesh",
                "representation": representation,
            }

            brane_mesh = mlab.triangular_mesh(*V.T, F, **brane_mesh_kwargs)

            F_rgba_int = (self.V_rgba[Iv] * 255).round().astype(int)
            F_color_scalars = np.linspace(0, 1, F_rgba_int.shape[0])
            brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(
                F_rgba_int
            )
            brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba_int
            brane_mesh.module_manager.lut_data_mode = "point data"
            brane_mesh.mlab_source.dataset.point_data.scalars = F_color_scalars
            brane_mesh.mlab_source.dataset.point_data.scalars.name = "face colors"
            brane_mesh.mlab_source.update()
            brane_mesh2 = mlab.pipeline.set_active_attribute(
                brane_mesh, point_scalars="face colors"
            )

        if self.show_plot_axes:
            mlab.axes()
            mlab.orientation_axes()
        # mview = mlab.view()
        # print(mview)
        if self.view is not None:
            mlab.view(**self.view)
        if save:
            fig_path = self.get_fig_path()
            self.image_count += 1
            mlab.savefig(fig_path, figure=fig, size=self.figsize)
        if show:
            mlab.show()

        mlab.close(all=True)

    def save_plot(self, title=""):
        self.plot(show=False, save=True, title=title)

    def show_plot(self, title=""):
        self.plot(show=True, save=False, title=title)

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
