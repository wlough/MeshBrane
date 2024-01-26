from mayavi import mlab
import numpy as np
from matplotlib import colormaps as plt_cmap
import os

# import tvtk
from src.numdiff import quaternion_to_matrix_vectorized

# red = plt_cmap["Set1"](0)
# blue = plt_cmap["Set1"](1)
# green = plt_cmap["Greens_r"](0)
# purple = plt_cmap["Set1"](3)
# orange = plt_cmap["Set1"](4)
# yellow = plt_cmap["Set1"](5)
# brown = plt_cmap["Set1"](6)
# pink = plt_cmap["Set1"](7)
# grey = plt_cmap["Set1"](8)
# white = plt_cmap["Greys"](0)
# black = plt_cmap["Greys_r"](0)
#
# membrane_sureface_color = (
#     0.0,
#     0.26666666666666666,
#     0.10588235294117647,
#     0.6,
# )  # plt_cmap["Greens_r"](0), plt_cmap["Set1"](2)
# membrane_wireframe_color = (1.0, 0.4980392156862745, 0.0, 0.6)  # plt_cmap["Set1"](4)

default_color_dict = {
    "face_color": (0.0, 0.2667, 0.1059),
    "face_alpha": 0.8,
    "edge_color": (1.0, 0.498, 0.0),
    "vertex_color": (1.0, 0.498, 0.0),  # (0.7057, 0.0156, 0.1502)
    "normal_color": (0.7057, 0.0156, 0.1502),  # (1.0, 0.0, 0.0)
    "tangent_color": (0.2298, 0.2987, 0.7537),
}


def rgb_float_to_int(rgb_float):
    """converts normalized rgb 0<r,g,b<1 to 0<r,g,b<255
    rgb_float=[r,g,b]
    rgb_float=[r,g,b,alpha]
    rgb_float=[...,[r,g,b],...]
    rgb_float=[...,[r,g,b,alpha],...]"""
    rgb_int = np.round(np.array([_ for _ in rgb_float]) * 255).astype(int)
    return rgb_int


def rgb_int_to_float(rgb_int):
    """converts normalized rgb 0<r,g,b<255 to 0<r,g,b<1"""
    rgb_float = np.array([_ for _ in rgb_int], dtype=np.float64) / 255
    return rgb_float


def get_cmap(cmin=0.0, cmax=1.0, name="hsv"):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
    'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
    'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
    'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
    'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
    'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
    'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
    'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
    'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
    'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
    'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
    'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
    'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
    'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot',
    'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma',
    'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',
    'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
    'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
    'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
    'winter_r'

    """
    cnum = lambda x: (x - cmin) / (cmax - cmin)
    cmap01 = plt_cmap[name]
    my_cmap = lambda x: cmap01(cnum(float(x)))
    return my_cmap


def movie(image_dir):
    image_type = "png"
    # image_directory = "./output/temp_images/"
    # movie_directory = "./output/temp_images/movie.mp4"
    image_directory = f"{image_dir}/"
    movie_directory = f"{image_dir}/movie.mp4"
    os.system(
        "ffmpeg "
        # frame rate (Hz)
        + "-r 20 "
        # frame size (width x height)
        + "-s 1080x720 "
        # input files
        + "-i "
        + image_directory
        + f"/fig_%04d.{image_type} "
        # video codec
        + "-vcodec libx264 "
        # video quality, lower means better
        + "-crf 25 "
        # pixel format
        + "-pix_fmt yuv420p "
        # output file
        + movie_directory
    )


#######################################################
def mayavi_mesh_plot(
    vertices,
    faces,
    edges=None,
    frames=None,
    vector_field_data=None,
    submesh_data=None,
    show=True,
    save=False,
    fig_path=None,
    plot_vertices=True,
    plot_edges=True,
    plot_faces=True,
):
    face_color = (0.0, 0.2667, 0.1059)
    edge_color = (1.0, 0.498, 0.0)
    vertex_color = (0.7057, 0.0156, 0.1502)
    frame_color = (0.2298, 0.2987, 0.7537)

    # figsize = (2180, 2180)
    figsize = (720, 720)
    ##########################################################
    if show:
        mlab.options.offscreen = False
    else:
        mlab.options.offscreen = True

    # bgcolor = (1.0, 1.0, 1.0)
    # fgcolor = (0.0, 0.0, 0.0)
    figsize = (2180, 2180)
    title = "Membrane mesh"
    # , bgcolor=bgcolor, fgcolor=fgcolor)
    fig = mlab.figure(title, size=figsize)

    if plot_edges:
        # mem_edges = #
        mlab.triangular_mesh(
            *vertices.T,
            faces,
            # opacity=0.4,
            color=edge_color,
            # representation="wireframe"
            representation="mesh",
            # representation="surface"
            # representation="fancymesh",
            tube_radius=0.002,
            tube_sides=3,
        )
    if plot_faces:
        # mem_faces = #
        mlab.triangular_mesh(
            *vertices.T,
            faces,
            opacity=0.4,
            color=face_color,
            # representation="wireframe"
            # representation="mesh",
            representation="surface"
            # representation="fancymesh",
            # tube_radius=None
        )
    if plot_vertices:
        # mem_vertices = #
        mlab.points3d(
            *vertices.T, mode="sphere", scale_factor=0.015, color=vertex_color
        )
    if frames is not None:
        # tangents1 = #
        mlab.quiver3d(
            *vertices.T,
            *frames[:, :, 0].T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=0.075,
            color=frame_color,
        )
        # tangents2 =
        # mlab.quiver3d(
        #     *vertices.T,
        #     *frames[:, :, 1].T,
        #     # line_width=5,
        #     mode="arrow",
        #     # scale_mode="vector",
        #     scale_factor=0.075,
        #     color=frame_color,
        # )
        # normals =
        mlab.quiver3d(
            *vertices.T,
            *frames[:, :, 2].T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=0.075,
            color=frame_color,
        )

    if vector_field_data is not None:
        vectors = vector_field_data["vectors"]
        vector_positions = vector_field_data["positions"]
        vector_color = vector_field_data["color"]
        # vecs =
        mlab.quiver3d(
            *vector_positions.T,
            *vectors.T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=1,
            color=vector_color,
        )
    if submesh_data is not None:
        vectors = vector_field_data["vectors"]
        vector_positions = vector_field_data["positions"]
        vector_color = vector_field_data["color"]
        # vecs =
        mlab.quiver3d(
            *vector_positions.T,
            *vectors.T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=1,
            color=vector_color,
        )

    if show:
        # mlab.axes()
        mlab.orientation_axes()
        mlab.show()
    if save:
        mlab.savefig(fig_path, figure=fig, size=figsize)
    mlab.close(all=True)


def mayavi_mesh_minimesh_plot(
    vertices,
    faces,
    mini_vertices,
    mini_faces,
    edges=None,
    frames=None,
    vector_field_data=None,
    submesh_data=None,
    show=True,
    save=False,
    fig_path=None,
    plot_vertices=True,
    plot_edges=True,
    plot_faces=True,
):
    face_color = (0.0, 0.2667, 0.1059)
    edge_color = (1.0, 0.498, 0.0)
    vertex_color = (0.7057, 0.0156, 0.1502)
    frame_color = (0.2298, 0.2987, 0.7537)

    # figsize = (2180, 2180)
    figsize = (720, 720)
    ##########################################################
    if show:
        mlab.options.offscreen = False
    else:
        mlab.options.offscreen = True

    # bgcolor = (1.0, 1.0, 1.0)
    # fgcolor = (0.0, 0.0, 0.0)
    figsize = (2180, 2180)
    title = "Membrane mesh"
    # , bgcolor=bgcolor, fgcolor=fgcolor)
    fig = mlab.figure(title, size=figsize)

    if plot_edges:
        # mem_edges = #
        mlab.triangular_mesh(
            *vertices.T,
            faces,
            # opacity=0.4,
            color=edge_color,
            # representation="wireframe"
            representation="mesh",
            # representation="surface"
            # representation="fancymesh",
            tube_radius=0.002,
            tube_sides=3,
        )
    if plot_faces:
        # mem_faces = #
        mlab.triangular_mesh(
            *vertices.T,
            faces,
            opacity=0.4,
            color=face_color,
            # representation="wireframe"
            # representation="mesh",
            representation="surface"
            # representation="fancymesh",
            # tube_radius=None
        )
    if plot_vertices:
        # mem_vertices = #
        mlab.points3d(
            *vertices.T, mode="sphere", scale_factor=0.015, color=vertex_color
        )
    if frames is not None:
        # tangents1 = #
        mlab.quiver3d(
            *vertices.T,
            *frames[:, :, 0].T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=0.075,
            color=frame_color,
        )
        # tangents2 =
        # mlab.quiver3d(
        #     *vertices.T,
        #     *frames[:, :, 1].T,
        #     # line_width=5,
        #     mode="arrow",
        #     # scale_mode="vector",
        #     scale_factor=0.075,
        #     color=frame_color,
        # )
        # normals =
        mlab.quiver3d(
            *vertices.T,
            *frames[:, :, 2].T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=0.075,
            color=frame_color,
        )

    if vector_field_data is not None:
        vectors = vector_field_data["vectors"]
        vector_positions = vector_field_data["positions"]
        vector_color = vector_field_data["color"]
        # vecs =
        mlab.quiver3d(
            *vector_positions.T,
            *vectors.T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=1,
            color=vector_color,
        )
    if submesh_data is not None:
        vectors = vector_field_data["vectors"]
        vector_positions = vector_field_data["positions"]
        vector_color = vector_field_data["color"]
        # vecs =
        mlab.quiver3d(
            *vector_positions.T,
            *vectors.T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=1,
            color=vector_color,
        )

    if show:
        # mlab.axes()
        mlab.orientation_axes()
        mlab.show()
    if save:
        mlab.savefig(fig_path, figure=fig, size=figsize)
    mlab.close(all=True)


#
# def curve_plot():
#     # Generate data for a 3D curve
#     t = np.linspace(0, 4 * np.pi, 100)
#     ex, ey, ez = np.eye(3)
#     a = 0.1 * np.pi
#     u1 = (ex * np.cos(a) + ez * np.sin(a)) / np.sqrt(2)
#     u2 = 0.5 * (ex * np.cos(a) + ez * np.sin(a)) / np.sqrt(2)
#     w = ez
#     psi1 = np.array([*u1, *w])
#     psi2 = np.array([*u2, *w])
#     pq1 = np.array([exp_se3_quaternion(_ * psi1) for _ in t])
#     pq2 = np.array([exp_se3_quaternion(_ * psi2) for _ in t])
#     x1, y1, z1 = pq1[:, 0], pq1[:, 1], pq1[:, 2]
#     x2, y2, z2 = pq2[:, 0], pq2[:, 1], pq2[:, 2]
#
#     # Plot the curve
#     mlab.plot3d(x1, y1, z1, color=(1, 0, 0), tube_radius=None)
#     mlab.plot3d(x2, y2, z2, color=(0, 0, 1), tube_radius=None)
#
#     # Show the plot
#     mlab.show()


def set_rgba_colors(rgb, a):
    rgba = np.array([[_[0], _[1], _[2], a] for _ in rgb])
    rgba = rgb_float_to_int(rgba)
    scalars = np.arange(rgba.shape[0])
    return rgba, scalars


def brane_plot(
    brane,
    show=True,
    save=False,
    fig_path=None,
    figsize=(2180, 2180),
    show_surface=True,
    show_halfedges=False,
    show_edges=False,
    show_vertices=False,
    show_normals=False,
    show_tangent1=False,
    show_tangent2=False,
    show_plot_axes=False,
    color_by_verts=False,
):
    """
    fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    """
    ################################
    V_rgb = brane.V_rgb
    V_radius = brane.V_radius
    H_rgb = brane.H_rgb
    # H_radius = brane.H_radius
    F_rgb = brane.F_rgb
    F_opacity = brane.F_opacity
    H_opacity = brane.H_opacity
    V_opacity = brane.V_opacity

    vertices = brane.vertex_positions()
    faces = brane.faces
    # hedges = brane.halfedges
    frame_scale = 0.15
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

        vert_cloud = mlab.points3d(*vertices.T, **vert_cloud_kwargs)
        vert_cloud.glyph.glyph.clamping = False
        V_rad_vecs = np.array([[_, 0, 0] for _ in V_radius])
        vert_cloud.mlab_source.dataset.point_data.vectors = V_rad_vecs
        vert_cloud.mlab_source.dataset.point_data.vectors.name = "vertex rads"

        V_rgba, V_color_scalars = set_rgba_colors(V_rgb, V_opacity)
        vert_cloud.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            V_color_scalars
        )

        vert_cloud.module_manager.scalar_lut_manager.lut.table = V_rgba
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

        brane_mesh = mlab.triangular_mesh(*vertices.T, faces, **brane_mesh_kwargs)
        if color_by_verts:
            F_rgba, F_scalars = set_rgba_colors(V_rgb, F_opacity)
            brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(
                F_scalars
            )
            brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba
            brane_mesh.module_manager.lut_data_mode = "point data"
            brane_mesh.mlab_source.dataset.point_data.scalars = F_scalars
            brane_mesh.mlab_source.dataset.point_data.scalars.name = "face colors"
            brane_mesh.mlab_source.update()
            brane_mesh2 = mlab.pipeline.set_active_attribute(
                brane_mesh, point_scalars="face colors"
            )
        else:
            F_rgba, F_scalars = set_rgba_colors(F_rgb, F_opacity)
            brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(
                F_scalars
            )
            brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba
            brane_mesh.module_manager.lut_data_mode = "cell data"
            brane_mesh.mlab_source.dataset.cell_data.scalars = F_scalars
            brane_mesh.mlab_source.dataset.cell_data.scalars.name = "face colors"
            brane_mesh.mlab_source.update()
            brane_mesh2 = mlab.pipeline.set_active_attribute(
                brane_mesh, cell_scalars="face colors"
            )
        # surf = mlab.pipeline.surface(brane_mesh)
    ################################
    # edge_mesh
    if show_edges:
        edge_mesh_kwargs = {
            "name": "edge_mesh",
            "color": (1.0, 0.498, 0.0),
            "representation": "wireframe",
        }

        edge_mesh = mlab.triangular_mesh(*vertices.T, faces, **edge_mesh_kwargs)
    ###############################
    # hedge vecs
    if show_halfedges:
        H_points, H_vecs = brane.shifted_hedge_vectors()
        hedge_vec_kwargs = {
            "name": "halfedges",
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": 1.0,
        }
        hedge_vfield = mlab.quiver3d(*H_points.T, *H_vecs.T, **hedge_vec_kwargs)
        H_rgba, H_color_scalars = set_rgba_colors(H_rgb, H_opacity)
        hedge_vfield.glyph.glyph.clamping = False
        hedge_vfield.glyph.glyph_source.glyph_source.tip_length = 0.25
        hedge_vfield.glyph.glyph_source.glyph_source.tip_radius = 0.03
        hedge_vfield.glyph.glyph_source.glyph_source.shaft_radius = 0.01
        # hedge_vfield.glyph.glyph_source.shaft_resolution = 3
        # hedge_vfield.glyph.glyph_source.tip_resolution = 4
        # hedge_vfield.glyph.glyph.scale_factor = 1
        # hedge_vfield.glyph.scale_mode = 'scale_by_vector'
        hedge_vfield.glyph.color_mode = "color_by_scalar"

        hedge_vfield.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            H_color_scalars
        )
        hedge_vfield.module_manager.scalar_lut_manager.lut.table = H_rgba
        # hedge_vfield.module_manager.lut_data_mode = "point data"
        hedge_vfield.mlab_source.dataset.point_data.scalars = H_color_scalars
        hedge_vfield.mlab_source.dataset.point_data.scalars.name = "halfedge colors"
        # hedge_vfield.mlab_source.dataset.point_data.vectors = H_vecs
        # hedge_vfield.mlab_source.dataset.point_data.vectors.name = "halfedge vectors"
        hedge_vfield.mlab_source.update()
        # hedge_vfield2 = mlab.pipeline.set_active_attribute(
        #     hedge_vfield, point_scalars="halfedge colors", point_vectors="halfedge vectors"
        # )
    ###############################
    if show_normals:
        V_normal_rgb = brane.V_normal_rgb
        try:
            V_normal = V_frames[:, :, 2]
        except NameError:
            V_frames = brane.orthogonal_matrices()
            V_normal = V_frames[:, :, 2]

        V_normal_kwargs = {
            "name": "normals",
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": frame_scale,
        }
        V_normal_field = mlab.quiver3d(*vertices.T, *V_normal.T, **V_normal_kwargs)
        V_normal_rgba, V_normal_color_scalars = set_rgba_colors(V_normal_rgb, 1.0)
        V_normal_field.glyph.glyph.clamping = False
        V_normal_field.glyph.glyph_source.glyph_source.tip_length = 0.25
        V_normal_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
        V_normal_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
        V_normal_field.glyph.color_mode = "color_by_scalar"

        V_normal_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            V_normal_color_scalars
        )
        V_normal_field.module_manager.scalar_lut_manager.lut.table = V_normal_rgba
        V_normal_field.mlab_source.dataset.point_data.scalars = V_normal_color_scalars
        V_normal_field.mlab_source.dataset.point_data.scalars.name = "normal colors"
        V_normal_field.mlab_source.update()

    if show_tangent1:
        V_tangent1_rgb = brane.V_tangent1_rgb
        try:
            V_tangent1 = V_frames[:, :, 0]
        except NameError:
            V_frames = brane.orthogonal_matrices()
            V_tangent1 = V_frames[:, :, 0]

        V_tangent1_kwargs = {
            "name": "tangent1",
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": frame_scale,
        }
        V_tangent1_field = mlab.quiver3d(
            *vertices.T, *V_tangent1.T, **V_tangent1_kwargs
        )
        V_tangent1_rgba, V_tangent1_color_scalars = set_rgba_colors(V_tangent1_rgb, 1.0)
        V_tangent1_field.glyph.glyph.clamping = False
        V_tangent1_field.glyph.glyph_source.glyph_source.tip_length = 0.25
        V_tangent1_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
        V_tangent1_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
        V_tangent1_field.glyph.color_mode = "color_by_scalar"

        V_tangent1_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            V_tangent1_color_scalars
        )
        V_tangent1_field.module_manager.scalar_lut_manager.lut.table = V_tangent1_rgba
        V_tangent1_field.mlab_source.dataset.point_data.scalars = (
            V_tangent1_color_scalars
        )
        V_tangent1_field.mlab_source.dataset.point_data.scalars.name = "tangent1 colors"
        V_tangent1_field.mlab_source.update()

    if show_tangent2:
        V_tangent2_rgb = brane.V_tangent2_rgb
        try:
            V_tangent2 = V_frames[:, :, 1]
        except NameError:
            V_frames = brane.orthogonal_matrices()
            V_tangent2 = V_frames[:, :, 1]

        V_tangent2_kwargs = {
            "name": "tangent2",
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": frame_scale,
        }
        V_tangent2_field = mlab.quiver3d(
            *vertices.T, *V_tangent2.T, **V_tangent2_kwargs
        )
        V_tangent2_rgba, V_tangent2_color_scalars = set_rgba_colors(V_tangent2_rgb, 1.0)
        V_tangent2_field.glyph.glyph.clamping = False
        V_tangent2_field.glyph.glyph_source.glyph_source.tip_length = 0.25
        V_tangent2_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
        V_tangent2_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
        V_tangent2_field.glyph.color_mode = "color_by_scalar"

        V_tangent2_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            V_tangent2_color_scalars
        )
        V_tangent2_field.module_manager.scalar_lut_manager.lut.table = V_tangent2_rgba
        V_tangent2_field.mlab_source.dataset.point_data.scalars = (
            V_tangent2_color_scalars
        )
        V_tangent2_field.mlab_source.dataset.point_data.scalars.name = "tangent2 colors"
        V_tangent2_field.mlab_source.update()

    if show_plot_axes:
        mlab.axes()
        mlab.orientation_axes()
    if show:
        mlab.options.offscreen = False
        mlab.show()
    if save:
        mlab.options.offscreen = True
        mlab.savefig(fig_path, figure=fig, size=figsize)

    mlab.close(all=True)


def plot_from_data(
    V_pq,
    faces,
    V_rgb,
    V_radius,
    H_rgb,
    F_rgb,
    F_opacity,
    H_opacity,
    V_opacity,
    V_normal_rgb,
    show=True,
    save=False,
    fig_path=None,
    figsize=(2180, 2180),
    show_surface=True,
    show_halfedges=False,
    show_edges=False,
    show_vertices=False,
    show_normals=False,
    show_tangent1=False,
    show_tangent2=False,
    show_plot_axes=False,
):
    """
    fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    """
    ################################

    vertices = V_pq[:, :3]  # brane.vertex_positions()
    frame_scale = 0.15
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

        vert_cloud = mlab.points3d(*vertices.T, **vert_cloud_kwargs)
        vert_cloud.glyph.glyph.clamping = False
        V_rad_vecs = np.array([[_, 0, 0] for _ in V_radius])
        vert_cloud.mlab_source.dataset.point_data.vectors = V_rad_vecs
        vert_cloud.mlab_source.dataset.point_data.vectors.name = "vertex rads"

        V_rgba, V_color_scalars = set_rgba_colors(V_rgb, V_opacity)
        vert_cloud.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            V_color_scalars
        )

        vert_cloud.module_manager.scalar_lut_manager.lut.table = V_rgba
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
            # "mask": mask,
            # "opacity": F_opacity,
            # "representation": "wireframe",
            # "representation": "mesh",
            "representation": "surface",
            # "representation": "fancymesh",,
        }

        brane_mesh = mlab.triangular_mesh(*vertices.T, faces, **brane_mesh_kwargs)

        F_rgba, F_scalars = set_rgba_colors(F_rgb, F_opacity)
        brane_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            F_scalars
        )
        brane_mesh.module_manager.scalar_lut_manager.lut.table = F_rgba
        brane_mesh.module_manager.lut_data_mode = "cell data"

        brane_mesh.mlab_source.dataset.cell_data.scalars = F_scalars
        brane_mesh.mlab_source.dataset.cell_data.scalars.name = "face colors"
        brane_mesh.mlab_source.update()
        brane_mesh2 = mlab.pipeline.set_active_attribute(
            brane_mesh, cell_scalars="face colors"
        )
        # surf = mlab.pipeline.surface(brane_mesh)
    ################################
    # edge_mesh
    if show_edges:
        edge_mesh_kwargs = {
            "name": "edge_mesh",
            "color": (1.0, 0.498, 0.0),
            "representation": "wireframe",
        }

        edge_mesh = mlab.triangular_mesh(*vertices.T, faces, **edge_mesh_kwargs)

    ###############################
    if show_normals:
        V_frames = quaternion_to_matrix_vectorized(b.V_pq[:, 3:])
        V_normal = V_frames[:, :, 2]

        V_normal_kwargs = {
            "name": "normals",
            "mode": "arrow",
            "scale_mode": "vector",
            "scale_factor": frame_scale,
        }
        V_normal_field = mlab.quiver3d(*vertices.T, *V_normal.T, **V_normal_kwargs)
        V_normal_rgba, V_normal_color_scalars = set_rgba_colors(V_normal_rgb, 1.0)
        V_normal_field.glyph.glyph.clamping = False
        V_normal_field.glyph.glyph_source.glyph_source.tip_length = 0.25
        V_normal_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
        V_normal_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
        V_normal_field.glyph.color_mode = "color_by_scalar"

        V_normal_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(
            V_normal_color_scalars
        )
        V_normal_field.module_manager.scalar_lut_manager.lut.table = V_normal_rgba
        V_normal_field.mlab_source.dataset.point_data.scalars = V_normal_color_scalars
        V_normal_field.mlab_source.dataset.point_data.scalars.name = "normal colors"
        V_normal_field.mlab_source.update()

    # if show_tangent1:
    #     V_tangent1_rgb = brane.V_tangent1_rgb
    #     try:
    #         V_tangent1 = V_frames[:, :, 0]
    #     except NameError:
    #         V_frames = brane.orthogonal_matrices()
    #         V_tangent1 = V_frames[:, :, 0]
    #
    #     V_tangent1_kwargs = {
    #         "name": "tangent1",
    #         "mode": "arrow",
    #         "scale_mode": "vector",
    #         "scale_factor": frame_scale,
    #     }
    #     V_tangent1_field = mlab.quiver3d(
    #         *vertices.T, *V_tangent1.T, **V_tangent1_kwargs
    #     )
    #     V_tangent1_rgba, V_tangent1_color_scalars = set_rgba_colors(V_tangent1_rgb, 1.0)
    #     V_tangent1_field.glyph.glyph.clamping = False
    #     V_tangent1_field.glyph.glyph_source.glyph_source.tip_length = 0.25
    #     V_tangent1_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
    #     V_tangent1_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
    #     V_tangent1_field.glyph.color_mode = "color_by_scalar"
    #
    #     V_tangent1_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(
    #         V_tangent1_color_scalars
    #     )
    #     V_tangent1_field.module_manager.scalar_lut_manager.lut.table = V_tangent1_rgba
    #     V_tangent1_field.mlab_source.dataset.point_data.scalars = (
    #         V_tangent1_color_scalars
    #     )
    #     V_tangent1_field.mlab_source.dataset.point_data.scalars.name = "tangent1 colors"
    #     V_tangent1_field.mlab_source.update()

    # if show_tangent2:
    #     V_tangent2_rgb = brane.V_tangent2_rgb
    #     try:
    #         V_tangent2 = V_frames[:, :, 1]
    #     except NameError:
    #         V_frames = brane.orthogonal_matrices()
    #         V_tangent2 = V_frames[:, :, 1]
    #
    #     V_tangent2_kwargs = {
    #         "name": "tangent2",
    #         "mode": "arrow",
    #         "scale_mode": "vector",
    #         "scale_factor": frame_scale,
    #     }
    #     V_tangent2_field = mlab.quiver3d(
    #         *vertices.T, *V_tangent2.T, **V_tangent2_kwargs
    #     )
    #     V_tangent2_rgba, V_tangent2_color_scalars = set_rgba_colors(V_tangent2_rgb, 1.0)
    #     V_tangent2_field.glyph.glyph.clamping = False
    #     V_tangent2_field.glyph.glyph_source.glyph_source.tip_length = 0.25
    #     V_tangent2_field.glyph.glyph_source.glyph_source.tip_radius = 0.03
    #     V_tangent2_field.glyph.glyph_source.glyph_source.shaft_radius = 0.01
    #     V_tangent2_field.glyph.color_mode = "color_by_scalar"
    #
    #     V_tangent2_field.module_manager.scalar_lut_manager.lut.number_of_colors = len(
    #         V_tangent2_color_scalars
    #     )
    #     V_tangent2_field.module_manager.scalar_lut_manager.lut.table = V_tangent2_rgba
    #     V_tangent2_field.mlab_source.dataset.point_data.scalars = (
    #         V_tangent2_color_scalars
    #     )
    #     V_tangent2_field.mlab_source.dataset.point_data.scalars.name = "tangent2 colors"
    #     V_tangent2_field.mlab_source.update()

    if show_plot_axes:
        mlab.axes()
        mlab.orientation_axes()
    if show:
        mlab.options.offscreen = False
        mlab.show()
    if save:
        mlab.options.offscreen = True
        mlab.savefig(fig_path, figure=fig, size=figsize)

    mlab.close(all=True)


def simple_plot(
    vertices,
    faces,
    show=True,
    save=False,
    fig_path=None,
    figsize=(2180, 2180),
    show_surface=True,
    show_edges=False,
    show_vertices=False,
):
    """
    fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    """
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

        vert_cloud = mlab.points3d(*vertices.T, **vert_cloud_kwargs)

    ################################
    # brane_mesh
    if show_surface:
        brane_mesh_kwargs = {
            "name": "brane_mesh",
            # "mask": mask,
            # "opacity": F_opacity,
            # "representation": "wireframe",
            # "representation": "mesh",
            "representation": "surface",
            # "representation": "fancymesh",,
        }

        brane_mesh = mlab.triangular_mesh(*vertices.T, faces, **brane_mesh_kwargs)

    ################################
    # edge_mesh
    if show_edges:
        edge_mesh_kwargs = {
            "name": "edge_mesh",
            "color": (1.0, 0.498, 0.0),
            "representation": "wireframe",
        }

        edge_mesh = mlab.triangular_mesh(*vertices.T, faces, **edge_mesh_kwargs)

    if show:
        mlab.options.offscreen = False
        mlab.show()
    if save:
        mlab.options.offscreen = True
        mlab.savefig(fig_path, figure=fig, size=figsize)

    mlab.close(all=True)
