from mayavi import mlab
import numpy as np

# from matplotlib import colormaps as plt_cmap
import os
from plyfile import PlyData

# default_color_dict = {
#     "face_color": (0.0, 0.2667, 0.1059),
#     "face_alpha": 0.8,
#     "edge_color": (1.0, 0.498, 0.0),
#     "vertex_color": (1.0, 0.498, 0.0),  # (0.7057, 0.0156, 0.1502)
#     "normal_color": (0.7057, 0.0156, 0.1502),  # (1.0, 0.0, 0.0)
#     "tangent_color": (0.2298, 0.2987, 0.7537),
# }


# def rgb_float_to_int(rgb_float):
#     """converts normalized rgb 0<r,g,b<1 to 0<r,g,b<255
#     rgb_float=[r,g,b]
#     rgb_float=[r,g,b,alpha]
#     rgb_float=[...,[r,g,b],...]
#     rgb_float=[...,[r,g,b,alpha],...]"""
#     rgb_int = np.round(np.array([_ for _ in rgb_float]) * 255).astype(int)
#     return rgb_int
#
#
# def rgb_int_to_float(rgb_int):
#     """converts normalized rgb 0<r,g,b<255 to 0<r,g,b<1"""
#     rgb_float = np.array([_ for _ in rgb_int], dtype=np.float64) / 255
#     return rgb_float
#
#
# def get_cmap(cmin=0.0, cmax=1.0, name="hsv"):
#     """
#     Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#     RGB color; the keyword argument name must be a standard mpl colormap name.
#
#     'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
#     'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
#     'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
#     'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
#     'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
#     'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
#     'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
#     'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
#     'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
#     'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
#     'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
#     'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
#     'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
#     'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
#     'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
#     'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
#     'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
#     'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot',
#     'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma',
#     'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',
#     'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
#     'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
#     'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
#     'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
#     'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
#     'winter_r'
#
#     """
#     cnum = lambda x: (x - cmin) / (cmax - cmin)
#     cmap01 = plt_cmap[name]
#     my_cmap = lambda x: cmap01(cnum(float(x)))
#     return my_cmap
#
#
# def movie(image_dir):
#     image_type = "png"
#     # image_directory = "./output/temp_images/"
#     # movie_directory = "./output/temp_images/movie.mp4"
#     image_directory = f"{image_dir}/"
#     movie_directory = f"{image_dir}/movie.mp4"
#     os.system(
#         "ffmpeg "
#         # frame rate (Hz)
#         + "-r 20 "
#         # frame size (width x height)
#         + "-s 1080x720 "
#         # input files
#         + "-i "
#         + image_directory
#         + f"/fig_%04d.{image_type} "
#         # video codec
#         + "-vcodec libx264 "
#         # video quality, lower means better
#         + "-crf 25 "
#         # pixel format
#         + "-pix_fmt yuv420p "
#         # output file
#         + movie_directory
#     )
#
#
# def set_rgba_colors(rgb, a):
#     if len(np.atleast_1d(a).shape) == 1:
#         A = a * np.ones(len(rgb))
#     else:
#         A = a
#     rgba = np.array([[_rgb[0], _rgb[1], _rgb[2], _a] for _rgb, _a in zip(rgb, A)])
#     rgba = rgb_float_to_int(rgba)
#     # scalars = np.arange(rgba.shape[0])
#     scalars = np.linspace(0, 1, rgba.shape[0])
#     return rgba, scalars
#


#######################################################
def load_mesh_from_ply(file_path):
    """loads vertex+face list from a .ply file"""
    # Read the ply file
    plydata = PlyData.read(file_path)

    # Extract the vertex and face data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    faces = np.vstack(plydata["face"]["vertex_indices"])

    if not isinstance(vertices[0], np.float64):
        # faces = [tuple(f) for f in faces]
        vertices = vertices.astype(np.float64)
    if not isinstance(faces[0], np.int32):
        # faces = [tuple(f) for f in faces]
        faces = faces.astype(np.int32)

    return vertices, faces


def ply_plot(
    file_path,
    show=True,
    save=False,
    fig_path=None,
    plot_vertices=True,
    plot_edges=True,
    plot_faces=True,
):
    """
    fig_path=f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    """
    ################################
    vertices, faces = load_mesh_from_ply(file_path)
    face_color = (0.0, 0.2667, 0.1059)
    edge_color = (1.0, 0.498, 0.0)
    vertex_color = (0.7057, 0.0156, 0.1502)

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
            tube_radius=0.003,
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
            representation="surface",
            # representation="fancymesh",
            # tube_radius=None
        )
    if plot_vertices:
        # mem_vertices = #
        mlab.points3d(*vertices.T, mode="sphere", scale_factor=0.015, color=vertex_color)

    if show:
        # mlab.axes()
        mlab.orientation_axes()
        mlab.show()
    if save:
        mlab.savefig(fig_path, figure=fig, size=figsize)
    mlab.close(all=True)


# %%


file_path = "./src/cbrane/example_cube-ascii.ply"
file_path = "./src/cbrane/data/ply_files/dumbbell_ultracoarse.ply"
vertices, faces = load_mesh_from_ply(file_path)

ply_plot(
    file_path,
    show=True,
    save=False,
    fig_path=None,
    plot_vertices=True,
    plot_edges=True,
    plot_faces=True,
)
