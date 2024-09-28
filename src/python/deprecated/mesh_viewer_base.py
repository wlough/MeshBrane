from mayavi import mlab
import numpy as np
import os
import subprocess


class MeshViewerBase:
    """
    Base class for MeshViewer and FancyMeshViewer
    """

    def __init__(
        self,
        # HalfEdgeMesh array data
        V,
        V_edge,
        E_vertex,
        E_next,
        E_twin,
        E_face,
        F_edge,
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
        self.V_edge = V_edge
        self.E_vertex = E_vertex
        self.E_face = E_face
        self.E_next = E_next
        self.E_twin = E_twin
        self.F_edge = F_edge
        #
        self.Nvertices = len(V)
        self.Nedges = len(E_vertex)
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
        V_edge,
        E_vertex,
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
        self.V_edge = V_edge
        self.E_vertex = E_vertex
        self.E_face = E_face
        self.E_next = E_next
        self.E_twin = E_twin
        self.F_edge = F_edge
        #####
        self.Nvertices = len(V)
        self.Nedges = len(self.E_vertex)
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
            self.E_rgba = np.zeros((len(self.E_vertex), 4))
            self.e_rgba = self.defaults["e_rgba"].copy()
            self.E_rgba[:] = self.e_rgba
        if E_rgba is None and e_rgba is not None:
            self.E_rgba = np.zeros((len(self.E_vertex), 4))
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
                        self.E_vertex[e],
                        self.E_vertex[self.E_next[e]],
                        self.E_vertex[self.E_next[self.E_next[e]]],
                    ]
                    for e in self.F_edge
                ]
            )
        else:
            self.F = F

    def get_shifted_E_field_no_bdry_twin(self):
        """halfedge vector shifted toward face centroid for visualization"""
        shift_to_center = 0.15
        Ne = len(self.E_vertex)
        # vecs = np.zeros((Ne, 3))
        # points = np.zeros((Ne, 3))
        # for e in range(Ne):
        #     v0 = self.E_vertex[e]
        #     v1 = self.E_vertex[self.E_next[e]]
        #     V2 = self.E_vertex[self.E_next[self.E_next[e]]]
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
                        self.V[self.E_vertex[e]]
                        + self.V[self.E_vertex[self.E_next[e]]]
                        + self.V[self.E_vertex[self.E_next[self.E_next[e]]]]
                    )
                    / 3
                    + (1 - shift_to_center) * self.V[self.E_vertex[e]],
                    (1 - shift_to_center)
                    * (
                        self.V[self.E_vertex[self.E_next[e]]] - self.V[self.E_vertex[e]]
                    ),
                ]
                for e in range(Ne)
            ]
        )
        return (points_vecs[:, 0], points_vecs[:, 1])

    def get_shifted_E_field(self):
        """halfedge vector shifted toward face centroid for visualization"""
        shift_to_center = 0.15
        Ne = len(self.E_vertex)
        # vecs = np.zeros((Ne, 3))
        # points = np.zeros((Ne, 3))
        # for e in range(Ne):
        #     v0 = self.E_vertex[e]
        #     v1 = self.E_vertex[self.E_next[e]]
        #     V2 = self.E_vertex[self.E_next[self.E_next[e]]]
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
                            self.V[self.E_vertex[e]]
                            + self.V[self.E_vertex[self.E_next[e]]]
                            + self.V[self.E_vertex[self.E_next[self.E_next[e]]]]
                        )
                        / 3
                        + (1 - shift_to_center) * self.V[self.E_vertex[e]],
                        ####################################################
                        (1 - shift_to_center)
                        * (
                            self.V[self.E_vertex[self.E_next[e]]]
                            - self.V[self.E_vertex[e]]
                        ),
                    ]
                    if self.E_face[e] != -1  # if on the boundary
                    else [
                        shift_to_center  # shift AWAY from twin face centroid
                        * (
                            (
                                self.V[self.E_vertex[e]]
                                + self.V[self.E_vertex[self.E_twin[e]]]
                            )
                            - (
                                self.V[self.E_vertex[self.E_twin[e]]]
                                + self.V[self.E_vertex[self.E_next[self.E_twin[e]]]]
                                + self.V[
                                    self.E_vertex[
                                        self.E_next[self.E_next[self.E_twin[e]]]
                                    ]
                                ]
                            )
                            / 3
                        )
                        + (1 - shift_to_center) * self.V[self.E_vertex[e]],
                        ####################################################
                        (1 - shift_to_center)
                        * (
                            self.V[self.E_vertex[self.E_next[e]]]
                            - self.V[self.E_vertex[e]]
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


class MeshViewer:
    def __init__(
        self,
        # HalfEdgeMesh array data
        V,
        V_edge,
        E_vertex,
        E_next,
        E_twin,
        E_face,
        F_edge,
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
        self.V_edge = V_edge
        self.E_vertex = E_vertex
        self.E_face = E_face
        self.E_next = E_next
        self.E_twin = E_twin
        self.F_edge = F_edge
        #
        self.Nvertices = len(V)
        self.Nedges = len(E_vertex)
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
        V_edge,
        E_vertex,
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
        self.V_edge = V_edge
        self.E_vertex = E_vertex
        self.E_face = E_face
        self.E_next = E_next
        self.E_twin = E_twin
        self.F_edge = F_edge
        #####
        self.Nvertices = len(V)
        self.Nedges = len(self.E_vertex)
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
            self.E_rgba = np.zeros((len(self.E_vertex), 4))
            self.e_rgba = self.defaults["e_rgba"].copy()
            self.E_rgba[:] = self.e_rgba
        if E_rgba is None and e_rgba is not None:
            self.E_rgba = np.zeros((len(self.E_vertex), 4))
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
                        self.E_vertex[e],
                        self.E_vertex[self.E_next[e]],
                        self.E_vertex[self.E_next[self.E_next[e]]],
                    ]
                    for e in self.F_edge
                ]
            )
        else:
            self.F = F

    def get_shifted_E_field_no_bdry_twin(self):
        """halfedge vector shifted toward face centroid for visualization"""
        shift_to_center = 0.15
        Ne = len(self.E_vertex)
        # vecs = np.zeros((Ne, 3))
        # points = np.zeros((Ne, 3))
        # for e in range(Ne):
        #     v0 = self.E_vertex[e]
        #     v1 = self.E_vertex[self.E_next[e]]
        #     V2 = self.E_vertex[self.E_next[self.E_next[e]]]
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
                        self.V[self.E_vertex[e]]
                        + self.V[self.E_vertex[self.E_next[e]]]
                        + self.V[self.E_vertex[self.E_next[self.E_next[e]]]]
                    )
                    / 3
                    + (1 - shift_to_center) * self.V[self.E_vertex[e]],
                    (1 - shift_to_center)
                    * (
                        self.V[self.E_vertex[self.E_next[e]]] - self.V[self.E_vertex[e]]
                    ),
                ]
                for e in range(Ne)
            ]
        )
        return (points_vecs[:, 0], points_vecs[:, 1])

    def get_shifted_E_field(self):
        """halfedge vector shifted toward face centroid for visualization"""
        shift_to_center = 0.15
        Ne = len(self.E_vertex)
        # vecs = np.zeros((Ne, 3))
        # points = np.zeros((Ne, 3))
        # for e in range(Ne):
        #     v0 = self.E_vertex[e]
        #     v1 = self.E_vertex[self.E_next[e]]
        #     V2 = self.E_vertex[self.E_next[self.E_next[e]]]
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
                            self.V[self.E_vertex[e]]
                            + self.V[self.E_vertex[self.E_next[e]]]
                            + self.V[self.E_vertex[self.E_next[self.E_next[e]]]]
                        )
                        / 3
                        + (1 - shift_to_center) * self.V[self.E_vertex[e]],
                        ####################################################
                        (1 - shift_to_center)
                        * (
                            self.V[self.E_vertex[self.E_next[e]]]
                            - self.V[self.E_vertex[e]]
                        ),
                    ]
                    if self.E_face[e] != -1  # if on the boundary
                    else [
                        shift_to_center  # shift AWAY from twin face centroid
                        * (
                            (
                                self.V[self.E_vertex[e]]
                                + self.V[self.E_vertex[self.E_twin[e]]]
                            )
                            - (
                                self.V[self.E_vertex[self.E_twin[e]]]
                                + self.V[self.E_vertex[self.E_next[self.E_twin[e]]]]
                                + self.V[
                                    self.E_vertex[
                                        self.E_next[self.E_next[self.E_twin[e]]]
                                    ]
                                ]
                            )
                            / 3
                        )
                        + (1 - shift_to_center) * self.V[self.E_vertex[e]],
                        ####################################################
                        (1 - shift_to_center)
                        * (
                            self.V[self.E_vertex[self.E_next[e]]]
                            - self.V[self.E_vertex[e]]
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


def movie(
    image_dir,
    image_format="png",
    image_prefix="frame",
    index_length=5,
    movie_name="movie",
    movie_dir=None,
    movie_format="mp4",
):
    # print(os.getcwd())
    # os.chdir('/desired/path')
    image_name = f"{image_prefix}_%0{index_length}d.{image_format}"
    ###############################################################
    image_path = os.path.join(image_dir, image_name)
    if movie_dir is None:
        movie_dir = image_dir
    movie_path = os.path.join(image_dir, f"{movie_name}.{movie_format}")
    ###############################################################
    wkdir = image_dir
    relative_movie_path = os.path.relpath(movie_path, wkdir)
    relative_image_path = os.path.relpath(image_path, wkdir)
    ###############################################################
    run_command = [
        "ffmpeg",
        # overwrite output file without asking
        "-y",
        # frame rate (Hz)
        "-r",
        "20",
        # frame size (width x height)
        "-s",
        "1080x720",
        # input files
        "-i",
        # image_path,
        relative_image_path,
        # video codec
        "-vcodec",
        "libx264",
        # video quality, lower means better
        "-crf",
        "25",
        # pixel format
        "-pix_fmt",
        "yuv420p",
        # output file
        # movie_path,
        relative_movie_path,
    ]

    # Start the process
    process = subprocess.Popen(
        run_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=image_dir,
    )
    for line in iter(process.stdout.readline, b""):
        print(line.decode(), end="")
    # relative_path = os.path.relpath(target_dir, start_dir)
    # os.path.join,os.path.basename,os.makedirs,os.system,os.path.exists,os.makedirs,shutil.rmtree


ColorDict = {
    "black": [0.0, 0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
    "transparent": [0.0, 0.0, 0.0, 0.0],
    "red": [0.8392, 0.1529, 0.1569, 1.0],
    "red10": [0.8392, 0.1529, 0.1569, 0.1],
    "red20": [0.8392, 0.1529, 0.1569, 0.2],
    "red50": [0.8392, 0.1529, 0.1569, 0.5],
    "green": [0.0, 0.6745, 0.2784, 1.0],
    "green10": [0.0, 0.6745, 0.2784, 0.1],
    "green20": [0.0, 0.6745, 0.2784, 0.2],
    "green50": [0.0, 0.6745, 0.2784, 0.5],
    "blue": [0.0, 0.4471, 0.6980, 1.0],
    "blue10": [0.0, 0.4471, 0.6980, 0.1],
    "blue20": [0.0, 0.4471, 0.6980, 0.2],
    "blue50": [0.0, 0.4471, 0.6980, 0.5],
    "yellow": [1.0, 0.8431, 0.0, 1.0],
    "yellow10": [1.0, 0.8431, 0.0, 0.1],
    "yellow20": [1.0, 0.8431, 0.0, 0.2],
    "yellow50": [1.0, 0.8431, 0.0, 0.5],
    "cyan": [0.0, 0.8431, 0.8431, 1.0],
    "cyan10": [0.0, 0.8431, 0.8431, 0.1],
    "cyan20": [0.0, 0.8431, 0.8431, 0.2],
    "cyan50": [0.0, 0.8431, 0.8431, 0.5],
    "magenta": [0.8784, 0.0, 0.8784, 1.0],
    "magenta10": [0.8784, 0.0, 0.8784, 0.1],
    "magenta20": [0.8784, 0.0, 0.8784, 0.2],
    "magenta50": [0.8784, 0.0, 0.8784, 0.5],
    "orange": [1.0, 0.5490, 0.0, 1.0],
    "orange10": [1.0, 0.5490, 0.0, 0.1],
    "orange20": [1.0, 0.5490, 0.0, 0.2],
    "orange50": [1.0, 0.5490, 0.0, 0.5],
    "purple": [0.5804, 0.0, 0.8275, 1.0],
    "purple10": [0.5804, 0.0, 0.8275, 0.1],
    "purple20": [0.5804, 0.0, 0.8275, 0.2],
    "purple50": [0.5804, 0.0, 0.8275, 0.5],
    "brane_red": [0.7057, 0.0156, 0.1502, 1.0],
    "brane_orange": [1.0, 0.498, 0.0, 1.0],
    "brane_green": [0.0, 0.63335, 0.05295, 0.65],
}


class FancyMayaviVectorField:
    def __init__(self, points, vectors, rgba=None, name=None, show=True):
        self.points = points
        self.vectors = vectors
        self.Nvecs = len(points)
        if rgba is None:
            _rgba = np.array([0.8392, 0.1529, 0.1569, 1.0])
            self.rgba = np.zeros((self.Nvecs, 4))
            self.rgba[:] = _rgba
        elif len(np.shape(rgba)) == 1:
            self.rgba = np.zeros((self.Nvecs, 4))
            self.rgba[:] = rgba
        elif np.shape(rgba)[0] == self.Nvecs:
            self.rgba = rgba
        else:
            raise ValueError(
                f"rgba must be of shape (Nvecs, 4) or (4,). Got {np.shape(rgba)}"
            )
        if name is None:
            name = "fancy_vector_field"
        self.show = show
        self.name = name
        self.rgba_int = (self.rgba * 255).round().astype(int)
        self.color_scalars = np.linspace(0, 1, self.rgba_int.shape[0])
        self.clamping = False
        self.tip_length = 0.25
        self.tip_radius = 0.03
        self.shaft_radius = 0.01
        self.color_mode = "color_by_scalar"

    def update(self, points=None, vectors=None, rgba=None, name=None, show=None):
        if points is not None:
            self.points = points
            self.Nvecs = len(points)
        if vectors is not None:
            self.vectors = vectors
        if rgba is not None:
            self.rgba = rgba
            self.rgba_int = (self.rgba * 255).round().astype(int)
            self.color_scalars = np.linspace(0, 1, self.rgba_int.shape[0])
        elif self.Nvecs != len(
            self.rgba
        ):  # only update if number of points has changed
            _rgba = np.array([0.8392, 0.1529, 0.1569, 1.0])
            self.rgba = np.zeros((self.Nvecs, 4))
            self.rgba[:] = _rgba
            self.rgba_int = (self.rgba * 255).round().astype(int)
            self.color_scalars = np.linspace(0, 1, self.rgba_int.shape[0])
        if name is not None:
            self.name = name
        if show is not None:
            self.show = show

    def plot(self):
        if not self.show:
            return None
        quiver3d_kwargs = {
            "name": self.name,
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": 1.0,
        }
        vfield = mlab.quiver3d(*self.points.T, *self.vectors.T, **quiver3d_kwargs)

        vfield.glyph.glyph.clamping = self.clamping
        vfield.glyph.glyph_source.glyph_source.tip_length = self.tip_length
        vfield.glyph.glyph_source.glyph_source.tip_radius = self.tip_radius
        vfield.glyph.glyph_source.glyph_source.shaft_radius = self.shaft_radius
        vfield.glyph.color_mode = self.color_mode

        vfield.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            self.rgba_int
        )
        vfield.module_manager.scalar_lut_manager.lut.table = self.rgba_int
        vfield.mlab_source.dataset.point_data.scalars = self.color_scalars
        vfield.mlab_source.dataset.point_data.scalars.name = f"{self.name}_colors"
        vfield.mlab_source.update()
        return vfield
