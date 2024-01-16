from mayavi import mlab
import numpy as np

default_color_dict = {
    "face_color": (0.0, 0.2667, 0.1059),
    "face_alpha": 0.8,
    "edge_color": (1.0, 0.498, 0.0),
    "vertex_color": (1.0, 0.498, 0.0),  # (0.7057, 0.0156, 0.1502)
    "normal_color": (0.7057, 0.0156, 0.1502),  # (1.0, 0.0, 0.0)
    "tangent_color": (0.2298, 0.2987, 0.7537),
}


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


def curve_plot():
    # Generate data for a 3D curve
    t = np.linspace(0, 4 * np.pi, 100)
    ex, ey, ez = np.eye(3)
    a = 0.1 * np.pi
    u1 = (ex * np.cos(a) + ez * np.sin(a)) / np.sqrt(2)
    u2 = 0.5 * (ex * np.cos(a) + ez * np.sin(a)) / np.sqrt(2)
    w = ez
    psi1 = np.array([*u1, *w])
    psi2 = np.array([*u2, *w])
    pq1 = np.array([exp_se3_quaternion(_ * psi1) for _ in t])
    pq2 = np.array([exp_se3_quaternion(_ * psi2) for _ in t])
    x1, y1, z1 = pq1[:, 0], pq1[:, 1], pq1[:, 2]
    x2, y2, z2 = pq2[:, 0], pq2[:, 1], pq2[:, 2]

    # Plot the curve
    mlab.plot3d(x1, y1, z1, color=(1, 0, 0), tube_radius=None)
    mlab.plot3d(x2, y2, z2, color=(0, 0, 1), tube_radius=None)

    # Show the plot
    mlab.show()
