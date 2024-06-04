from mayavi import mlab
import numpy as np


def vf_plot(
    V,
    F,
    show=True,
    save=False,
    fig_path=None,
    figsize=(2180, 2180),
    show_plot_axes=False,
    view=None,
):
    """
    fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    """
    ################################

    ################################
    if show:
        mlab.options.offscreen = False
    else:
        mlab.options.offscreen = True
    # figsize = (2180, 2180)
    title = "Membrane mesh"
    fig = mlab.figure(title, size=figsize)

    ################################

    brane_mesh_kwargs = {
        "name": "brane_mesh",
        "representation": "surface",
    }

    brane_mesh = mlab.triangular_mesh(*V.T, F, **brane_mesh_kwargs)

    if show_plot_axes:
        mlab.axes()
        mlab.orientation_axes()
    # mview = mlab.view()
    # print(mview)
    if view is not None:
        mlab.view(**view)
    if show:
        mlab.options.offscreen = False
        mlab.show()
    if save:
        mlab.options.offscreen = True
        mlab.savefig(fig_path, figure=fig, size=figsize)

    mlab.close(all=True)


def brane_plot(
    b,
    show=True,
    save=False,
    fig_path=None,
    figsize=(2180, 2180),
    show_surface=True,
    show_halfedges=False,
    show_edges=False,
    show_vertices=False,
    show_normals=False,
    show_plot_axes=False,
    color_by_V_rgba=False,
    V_vector_data=None,
    V_vector_data_rgba=None,
    show_V_vector_data=False,
    frame_scale=0.07,
    view=None,
):
    """
    fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    """
    ################################
    V_rgba = b.V_rgba
    V_normal_rgba = b.V_normal_rgba
    V_radius = b.V_radius
    E_rgba = b.E_rgba
    F_rgba = b.F_rgba
    V = b.V
    F = b.get_faces()
    ################################
    if show:
        mlab.options.offscreen = False
    else:
        mlab.options.offscreen = True
    # figsize = (2180, 2180)
    title = "Membrane mesh"
    fig = mlab.figure(title, size=figsize)

    ################################
    # vert_cloud
    if show_vertices:
        vert_cloud_kwargs = {
            "name": "vert_cloud",
            "scale_mode": "vector",
            "scale_factor": 1.0,
        }

        vert_cloud = mlab.points3d(*V.T, **vert_cloud_kwargs)
        vert_cloud.glyph.glyph.clamping = False
        V_rad_vecs = np.array([[_, 0, 0] for _ in V_radius])
        vert_cloud.mlab_source.dataset.point_data.vectors = V_rad_vecs
        vert_cloud.mlab_source.dataset.point_data.vectors.name = "vertex rads"

        V_rgba_int = np.round(np.array([_ for _ in V_rgba]) * 255).astype(int)
        V_color_scalars = np.linspace(0, 1, V_rgba.shape[0])
        vert_cloud.module_manager.scalar_lut_manager.lut.number_of_colors = len(V_color_scalars)

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
    if show_surface:
        brane_mesh_kwargs = {
            "name": "brane_mesh",
            "representation": "surface",
        }

        brane_mesh = mlab.triangular_mesh(*V.T, F, **brane_mesh_kwargs)
        if color_by_V_rgba:
            F_rgba_int = np.round(np.array([_ for _ in V_rgba]) * 255).astype(int)
            F_color_scalars = np.linspace(0, 1, V_rgba.shape[0])
            brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(F_color_scalars)
            brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba_int
            brane_mesh.module_manager.lut_data_mode = "point data"
            brane_mesh.mlab_source.dataset.point_data.scalars = F_color_scalars
            brane_mesh.mlab_source.dataset.point_data.scalars.name = "face colors"
            brane_mesh.mlab_source.update()
            brane_mesh2 = mlab.pipeline.set_active_attribute(brane_mesh, point_scalars="face colors")
        else:
            F_rgba_int = np.round(np.array([_ for _ in F_rgba]) * 255).astype(int)
            F_color_scalars = np.linspace(0, 1, F_rgba.shape[0])

            brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(F_color_scalars)
            brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba_int
            brane_mesh.module_manager.lut_data_mode = "cell data"
            brane_mesh.mlab_source.dataset.cell_data.scalars = F_color_scalars
            brane_mesh.mlab_source.dataset.cell_data.scalars.name = "face colors"
            brane_mesh.mlab_source.update()
            brane_mesh2 = mlab.pipeline.set_active_attribute(brane_mesh, cell_scalars="face colors")
        # # # #  #surf = mlab.pipeline.surface(brane_mesh)

    ################################
    # edge_mesh
    if show_edges:
        edge_mesh_kwargs = {
            "name": "edge_mesh",
            "color": (1.0, 0.498, 0.0),
            "representation": "wireframe",
        }

        edge_mesh = mlab.triangular_mesh(*V.T, F, **edge_mesh_kwargs)
    ###############################
    # hedge vecs
    if show_halfedges:
        E_points, E_vecs = b.shifted_hedge_vectors()
        hedge_vec_kwargs = {
            "name": "halfedges",
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": 1.0,
        }
        hedge_vfield = mlab.quiver3d(*E_points.T, *E_vecs.T, **hedge_vec_kwargs)

        E_rgba_int = np.round(np.array([_ for _ in E_rgba]) * 255).astype(int)
        E_color_scalars = np.linspace(0, 1, E_rgba.shape[0])

        hedge_vfield.glyph.glyph.clamping = False
        hedge_vfield.glyph.glyph_source.glyph_source.tip_length = 0.25
        hedge_vfield.glyph.glyph_source.glyph_source.tip_radius = 0.03
        hedge_vfield.glyph.glyph_source.glyph_source.shaft_radius = 0.01
        # hedge_vfield.glyph.glyph_source.shaft_resolution = 3
        # hedge_vfield.glyph.glyph_source.tip_resolution = 4
        # hedge_vfield.glyph.glyph.scale_factor = 1
        # hedge_vfield.glyph.scale_mode = 'scale_by_vector'
        hedge_vfield.glyph.color_mode = "color_by_scalar"

        hedge_vfield.module_manager.scalar_lut_manager.lut.number_of_colors = len(E_color_scalars)
        hedge_vfield.module_manager.scalar_lut_manager.lut.table = E_rgba_int
        # hedge_vfield.module_manager.lut_data_mode = "point data"
        hedge_vfield.mlab_source.dataset.point_data.scalars = E_color_scalars
        hedge_vfield.mlab_source.dataset.point_data.scalars.name = "halfedge colors"
        # hedge_vfield.mlab_source.dataset.point_data.vectors = E_vecs
        # hedge_vfield.mlab_source.dataset.point_data.vectors.name = "halfedge vectors"
        hedge_vfield.mlab_source.update()
        # hedge_vfield2 = mlab.pipeline.set_active_attribute(
        #     hedge_vfield, point_scalars="halfedge colors", point_vectors="halfedge vectors"
        # )
    ###############################
    if show_V_vector_data:
        V_vector_data = b.V_vector_data

        V_vector_data_kwargs = {
            "name": "vector_data",
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": 1.0,
        }
        V_vector_data_field = mlab.quiver3d(*V.T, *V_vector_data.T, **V_vector_data_kwargs)
        # V_normal_rgba, V_normal_color_scalars = set_rgba_colors(V_normal_rgb, 1.0)
        V_vector_data_field.glyph.glyph.clamping = False
        # V_normal_field.glyph.glyph_source.glyph_source.tip_length = 0.25
        # V_normal_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
        # V_normal_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
        V_vector_data_field.glyph.glyph_source.glyph_source.tip_length = 0.25
        V_vector_data_field.glyph.glyph_source.glyph_source.tip_radius = 0.03 * 2
        V_vector_data_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01 * 2
        # V_vector_data_field.glyph.color_mode = "color_by_scalar"

        # V_vector_data_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(
        #     V_normal_color_scalars
        # )
        # V_vector_data_field.module_manager.scalar_lut_manager.lut.table = V_normal_rgba
        # V_vector_data_field.mlab_source.dataset.point_data.scalars = V_normal_color_scalars
        # V_vector_data_field.mlab_source.dataset.point_data.scalars.name = "normal colors"
        # V_vector_data_field.mlab_source.update()

    # if show_normals:
    #     V_normal_rgb = b.V_normal_rgb
    #     try:
    #         V_normal = V_frames[:, :, 2]
    #     except NameError:
    #         V_frames = quaternion_to_matrix_vectorized(b.V_pq[:, 3:])
    #         V_normal = V_frames[:, :, 2]
    #
    #     V_normal_kwargs = {
    #         "name": "normals",
    #         "mode": "arrow",
    #         "scale_mode": "vector",
    #         "scale_factor": frame_scale,
    #     }
    #     V_normal_field = mlab.quiver3d(*vertices.T, *V_normal.T, **V_normal_kwargs)
    #     V_normal_rgba, V_normal_color_scalars = set_rgba_colors(V_normal_rgb, 1.0)
    #     V_normal_field.glyph.glyph.clamping = False
    #     # V_normal_field.glyph.glyph_source.glyph_source.tip_length = 0.25
    #     # V_normal_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
    #     # V_normal_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
    #     V_normal_field.glyph.glyph_source.glyph_source.tip_length = 0.25
    #     V_normal_field.glyph.glyph_source.glyph_source.tip_radius = 0.03 * 2
    #     V_normal_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01 * 2
    #     V_normal_field.glyph.color_mode = "color_by_scalar"
    #
    #     V_normal_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(V_normal_color_scalars)
    #     V_normal_field.module_manager.scalar_lut_manager.lut.table = V_normal_rgba
    #     V_normal_field.mlab_source.dataset.point_data.scalars = V_normal_color_scalars
    #     V_normal_field.mlab_source.dataset.point_data.scalars.name = "normal colors"
    #     V_normal_field.mlab_source.update()

    if show_plot_axes:
        mlab.axes()
        mlab.orientation_axes()
    # mview = mlab.view()
    # print(mview)
    if view is not None:
        mlab.view(**view)
    if show:
        mlab.options.offscreen = False
        mlab.show()
    if save:
        mlab.options.offscreen = True
        mlab.savefig(fig_path, figure=fig, size=figsize)

    mlab.close(all=True)
