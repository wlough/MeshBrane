# %%
from mayavi import mlab
import numpy as np
import os
from src.utils import save_mesh_to_ply, load_mesh_from_ply, make_trisurface_patch
from plyfile import PlyData, PlyElement


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


def make_output_directory(output_directory):
    os.system(f"rm -r {output_directory}")
    os.system(f"mkdir {output_directory}")
    os.system(f"mkdir {output_directory}/ply_files")
    os.system(f"mkdir {output_directory}/temp_images")
    os.system(f"mkdir {output_directory}/logs")
    os.system(f"mkdir {output_directory}/config")
    os.system(f"mkdir {output_directory}/checkpoints")


def load_he_mesh_from_ply(file_path):
    """loads vertex+face list from a .ply file"""
    # Read the ply file
    plydata = PlyData.read(file_path)

    # Extract the vertex and face data
    V = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    V_edge = plydata["vertex"]["e"]

    F_edge = plydata["face"]["e"]

    E_vertex = plydata["edge"]["v"]
    E_face = plydata["edge"]["f"]
    E_next = plydata["edge"]["n"]
    E_twin = plydata["edge"]["t"]  # np.vstack(plydata["edge"]["t"])

    if not isinstance(V[0], np.float64):
        V = V.astype(np.float64)
    if not isinstance(V_edge[0], np.int32):
        V_edge = V_edge.astype(np.int32)

    if not isinstance(F_edge[0], np.int32):
        F_edge = F_edge.astype(np.int32)

    if not isinstance(E_vertex[0], np.int32):
        E_vertex = E_vertex.astype(np.int32)
    if not isinstance(E_face[0], np.int32):
        E_face = E_face.astype(np.int32)
    if not isinstance(E_next[0], np.int32):
        E_next = E_next.astype(np.int32)
    if not isinstance(E_twin[0], np.int32):
        E_twin = E_twin.astype(np.int32)

    return V, V_edge, F_edge, E_vertex, E_face, E_next, E_twin


he_path = "./data/ply_files/hex_patch_he.ply"
V, V_edge, F_edge, E_vertex, E_face, E_next, E_twin = load_he_mesh_from_ply(he_path)
# %%
output_directory = "../output/test_output"
make_output_directory(output_directory)

# %%
file_path = "./data/ply_files/hex_patch.ply"
file_path = "/home/wlough/git/MeshBrane/data/ply_files/dumbbell_nobinary.ply"
ply_plot(
    file_path,
    show=True,
    save=False,
    fig_path=None,
    plot_vertices=True,
    plot_edges=True,
    plot_faces=True,
)

# %%
from src.utils import save_mesh_to_ply, load_mesh_from_ply, make_trisurface_patch

# theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)


# def make_trisurface_patch(Nfaces=5):
#     """makes a triangle mesh patch around a vertex"""
#     N = 1 * Nfaces
#     dr = 0.25
#     dz = 0.1
#     theta = np.array([2 * np.pi * _ / N for _ in range(N)])
#     r = np.random.rand(N)
#     r *= dr * np.max(r)
#     r += 1 - dr
#     z = dz * np.cos(theta)
#     x, y = r * np.cos(theta), r * np.sin(theta)
#     vertices = np.array([x, y, z]).T
#     vertices = np.array([[0.0, 0.0, dz], *vertices])
#     faces = [[0, i, i + 1] for i in range(1, N)]
#     faces = np.array([*faces, [0, N, 1]], dtype=np.int32)
#     return vertices, faces
#

file_path = "./data/ply_files/hex_patch.ply"
V, F = make_trisurface_patch(6)
save_mesh_to_ply(V, F, file_path)
# %%
