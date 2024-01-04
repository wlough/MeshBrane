import numpy as np
import polyscope as ps
from src.numdiff import quaternion_to_matrix

default_color_dict = {
    "face_color": (0.0, 0.2667, 0.1059),
    "face_alpha": 0.8,
    "edge_color": (1.0, 0.498, 0.0),
    "vertex_color": (1.0, 0.498, 0.0),  # (0.7057, 0.0156, 0.1502)
    "normal_color": (0.7057, 0.0156, 0.1502),  # (1.0, 0.0, 0.0)
    "tangent_color": (0.2298, 0.2987, 0.7537),
}


def polyscope_mesh_plot(vertices, faces, frames=None):
    """register_surface_mesh variables:
    enabled=None
    color=None
    edge_color=None
    smooth_shade=None
    edge_width=None
    material=None
    back_face_policy=None
    back_face_color=None
    transparency=None

    surface_mesh.add_color_quantity(name, colors)
    surface_mesh.add_scalar_quantity(name, scalars)
    surface_mesh.add_vector_quantity(name, vectors)
    # Add a scalar quantity
    scalar_values = np.random.rand(vertices.shape[0])  # One value per vertex
    mesh.add_scalar_quantity("random scalars", scalar_values)

    # Add a vector quantity
    vector_values = np.random.rand(vertices.shape[0], 3)  # One vector per vertex
    mesh.add_vector_quantity("random vectors", vector_values)

    # Add a color quantity
    color_values = np.random.rand(vertices.shape[0], 3)  # One RGB color per vertex
    mesh.add_color_quantity("random colors", color_values)
    """
    # face_color = (0.0, 0.2667, 0.1059)
    # edge_color = (1.0, 0.498, 0.0)
    # vertex_color = (1.0, 0.498, 0.0)  # (0.7057, 0.0156, 0.1502)
    # normal_color = (0.7057, 0.0156, 0.1502)  # (1.0, 0.0, 0.0)
    # tangent_color = (0.2298, 0.2987, 0.7537)
    face_color = default_color_dict["face_color"]
    face_alpha = default_color_dict["face_alpha"]
    edge_color = default_color_dict["edge_color"]
    vertex_color = default_color_dict["vertex_color"]
    normal_color = default_color_dict["normal_color"]
    tangent_color = default_color_dict["tangent_color"]

    ps.init()

    mesh = ps.register_surface_mesh("surface", vertices, faces)
    # Set the base color of the mesh
    mesh.set_color(face_color)
    # Set the color of the edges
    mesh.set_edge_color(edge_color)
    mesh.set_edge_width(1.0)
    # Set the transparency of the mesh
    mesh.set_transparency(face_alpha)
    vertex_cloud = ps.register_point_cloud(
        "vertices", vertices, enabled=True, color=vertex_color, radius=0.0025
    )

    if frames is not None:
        # normals = frames[:, 2]
        vertex_cloud.add_vector_quantity(
            "normals",
            frames[:, :, 2],
            enabled=True,
            color=normal_color,
        )

        vertex_cloud.add_vector_quantity(
            "tangents1", frames[:, :, 0], color=tangent_color, enabled=True
        )

    ps.show()


def polyscope_plot2(vertices, faces, frames=None, pq=None):
    """ """
    # face_color = (0.0, 0.2667, 0.1059)
    # edge_color = (1.0, 0.498, 0.0)
    # vertex_color = (1.0, 0.498, 0.0)  # (0.7057, 0.0156, 0.1502)
    # normal_color = (0.7057, 0.0156, 0.1502)  # (1.0, 0.0, 0.0)
    # tangent_color = (0.2298, 0.2987, 0.7537)
    face_color = default_color_dict["face_color"]
    face_alpha = default_color_dict["face_alpha"]
    edge_color = default_color_dict["edge_color"]
    vertex_color = default_color_dict["vertex_color"]
    normal_color = default_color_dict["normal_color"]
    tangent_color = default_color_dict["tangent_color"]

    ps.init()
    ps.set_navigation_style("free")
    ps.set_up_dir("z_up")

    mesh = ps.register_surface_mesh("surface", vertices, faces)
    # Set the base color of the mesh
    mesh.set_color(face_color)
    # Set the color of the edges
    mesh.set_edge_color(edge_color)
    mesh.set_edge_width(1.0)
    # Set the transparency of the mesh
    mesh.set_transparency(face_alpha)
    vertex_cloud = ps.register_point_cloud(
        "vertices", vertices, enabled=True, color=vertex_color, radius=0.0025
    )

    if frames is not None:
        # normals = frames[:, 2]
        vertex_cloud.add_vector_quantity(
            "normals",
            frames[:, :, 2],
            enabled=True,
            color=normal_color,
        )

        vertex_cloud.add_vector_quantity(
            "tangents1", frames[:, :, 0], color=tangent_color, enabled=True
        )

    if pq is not None:
        moving_points = pq[:, :3]
        moving_quaternions = pq[:, 3:]
        moving_frames = np.array([quaternion_to_matrix(q) for q in moving_quaternions])

        moving_frame_cloud = ps.register_point_cloud(
            "points", moving_points, enabled=True, color=(0.0, 1.0, 0.0), radius=0.0025
        )
        moving_frame_cloud.add_vector_quantity(
            "e1", moving_frames[:, :, 0], color=(1.0, 0.0, 0.0), enabled=True
        )
        moving_frame_cloud.add_vector_quantity(
            "e2", moving_frames[:, :, 1], color=(0.0, 1.0, 0.0), enabled=True
        )
        moving_frame_cloud.add_vector_quantity(
            "e3", moving_frames[:, :, 2], color=(0.0, 0.0, 1.0), enabled=True
        )

    ps.show()


def polyscope_plot(surfaces_list, pq=None):
    """ """
    mesh_list = []
    vertex_cloud_list = []
    Nsurfs = len(surfaces_list)
    # normals_list = []
    ps.init()
    # ps.set_transparency_mode("simple")
    for _ in range(Nsurfs):
        surface_dict = surfaces_list[_]
        surf_keys = surface_dict.keys()
        vertices = surface_dict["vertices"]
        faces = surface_dict["faces"]
        face_color = surface_dict["face_color"]
        face_alpha = surface_dict["face_alpha"]
        edge_color = surface_dict["edge_color"]
        vertex_color = surface_dict["vertex_color"]

        mesh_list.append(ps.register_surface_mesh(f"surface{_}", vertices, faces))
        # Set the base color of the mesh
        mesh_list[-1].set_color(face_color)
        # Set the color of the edges
        mesh_list[-1].set_edge_color(edge_color)
        mesh_list[-1].set_edge_width(1.0)
        # Set the transparency of the mesh
        mesh_list[-1].set_transparency(face_alpha)
        vertex_cloud_list.append(
            ps.register_point_cloud(
                f"vertices{_}",
                vertices,
                enabled=True,
                color=vertex_color,
                radius=0.0025,
            )
        )

        if "frames" in surf_keys:
            frames = surface_dict["frames"]
            normal_color = surface_dict["normal_color"]
            tangent_color = surface_dict["tangent_color"]
            vertex_cloud_list[-1].add_vector_quantity(
                "normals",
                frames[:, :, 2],
                enabled=True,
                color=normal_color,
            )

            vertex_cloud_list[-1].add_vector_quantity(
                "tangents1", frames[:, :, 0], color=tangent_color, enabled=True
            )

    if pq is not None:
        moving_points = pq[:, :3]
        moving_quaternions = pq[:, 3:]
        moving_frames = np.array([quaternion_to_matrix(q) for q in moving_quaternions])

        moving_frame_cloud = ps.register_point_cloud(
            "points", moving_points, enabled=True, color=(0.0, 1.0, 0.0), radius=0.0025
        )
        moving_frame_cloud.add_vector_quantity(
            "e1", moving_frames[:, :, 0], color=(1.0, 0.0, 0.0), enabled=True
        )
        moving_frame_cloud.add_vector_quantity(
            "e2", moving_frames[:, :, 1], color=(0.0, 1.0, 0.0), enabled=True
        )
        moving_frame_cloud.add_vector_quantity(
            "e3", moving_frames[:, :, 2], color=(0.0, 0.0, 1.0), enabled=True
        )
    ps.show()


def polyscope_multimesh_plot(surfaces_list):
    """ """
    mesh_list = []
    vertex_cloud_list = []
    Nsurfs = len(surfaces_list)
    # normals_list = []
    ps.init()
    # ps.set_transparency_mode("simple")
    for _ in range(Nsurfs):
        surface_dict = surfaces_list[_]
        surf_keys = surface_dict.keys()
        vertices = surface_dict["vertices"]
        faces = surface_dict["faces"]
        face_color = surface_dict["face_color"]
        face_alpha = surface_dict["face_alpha"]
        edge_color = surface_dict["edge_color"]
        vertex_color = surface_dict["vertex_color"]

        mesh_list.append(ps.register_surface_mesh(f"surface{_}", vertices, faces))
        # Set the base color of the mesh
        mesh_list[-1].set_color(face_color)
        # Set the color of the edges
        mesh_list[-1].set_edge_color(edge_color)
        mesh_list[-1].set_edge_width(1.0)
        # Set the transparency of the mesh
        mesh_list[-1].set_transparency(face_alpha)
        vertex_cloud_list.append(
            ps.register_point_cloud(
                f"vertices{_}",
                vertices,
                enabled=True,
                color=vertex_color,
                radius=0.0025,
            )
        )

        if "frames" in surf_keys:
            frames = surface_dict["frames"]
            normal_color = surface_dict["normal_color"]
            tangent_color = surface_dict["tangent_color"]
            vertex_cloud_list[-1].add_vector_quantity(
                "normals",
                frames[:, :, 2],
                enabled=True,
                color=normal_color,
            )

            vertex_cloud_list[-1].add_vector_quantity(
                "tangents1", frames[:, :, 0], color=tangent_color, enabled=True
            )

    ps.show()


def example_multimesh_plot(brane, vertex_list=None):
    if vertex_list is None:
        vertex_list = [0, 93, 260, 500]
    vertices, faces = brane.position_vectors(), brane.faces
    frames = brane.orthogonal_matrices()
    surfaces_list = []
    brane_dict = {
        "vertices": vertices,
        "faces": faces,
        # "frames": frames,
    } | default_color_dict
    for vertex in vertex_list:
        v_of_e_of_v = np.array([vertex, *brane.vertices_adjacent_to_vertex(vertex)])
        # _normals = 0.1 * np.array([normals[_] for _ in v_of_e_of_v])
        # _vertices = np.array([vertices[_] for _ in v_of_e_of_v])

        mini_vertices, mini_faces = brane.mini_mesh(vertex)
        mini_frames = np.array([frames[_] for _ in v_of_e_of_v])

        mini_brane_dict = {
            "vertices": mini_vertices,
            "faces": mini_faces,
            "frames": mini_frames,
            "face_color": (0.0, 0.0, 1.0),
            "edge_color": (1.0, 0.0, 0.0),
            "vertex_color": (1.0, 0.0, 0.0),
            "normal_color": (1.0, 0.0, 0.0),
            "tangent_color": (1.0, 0.0, 0.0),
            "face_alpha": 1.0,
        }
        surfaces_list.append(mini_brane_dict)

    surfaces_list.append(brane_dict)

    polyscope_multimesh_plot(surfaces_list)


def polyscope_mesh_cloud_plot(vertices, faces, frames=None, mf_cloud=None):
    """ """
    # face_color = (0.0, 0.2667, 0.1059)
    # edge_color = (1.0, 0.498, 0.0)
    # vertex_color = (1.0, 0.498, 0.0)  # (0.7057, 0.0156, 0.1502)
    # normal_color = (0.7057, 0.0156, 0.1502)  # (1.0, 0.0, 0.0)
    # tangent_color = (0.2298, 0.2987, 0.7537)
    face_color = default_color_dict["face_color"]
    face_alpha = default_color_dict["face_alpha"]
    edge_color = default_color_dict["edge_color"]
    vertex_color = default_color_dict["vertex_color"]
    normal_color = default_color_dict["normal_color"]
    tangent_color = default_color_dict["tangent_color"]

    ps.init()

    mesh = ps.register_surface_mesh("surface", vertices, faces)
    # Set the base color of the mesh
    mesh.set_color(face_color)
    # Set the color of the edges
    mesh.set_edge_color(edge_color)
    mesh.set_edge_width(1.0)
    # Set the transparency of the mesh
    mesh.set_transparency(face_alpha)
    vertex_cloud = ps.register_point_cloud(
        "vertices", vertices, enabled=True, color=vertex_color, radius=0.0025
    )

    if frames is not None:
        # normals = frames[:, 2]
        vertex_cloud.add_vector_quantity(
            "normals",
            frames[:, :, 2],
            enabled=True,
            color=normal_color,
        )

        vertex_cloud.add_vector_quantity(
            "tangents1", frames[:, :, 0], color=tangent_color, enabled=True
        )

    if mf_cloud is not None:
        moving_points, moving_frames = mf_cloud

        moving_frame_cloud = ps.register_point_cloud(
            "points", moving_points, enabled=True, color=(0.0, 1.0, 0.0), radius=0.0025
        )
        moving_frame_cloud.add_vector_quantity(
            "e1", moving_frames, color=(1.0, 0.0, 0.0), enabled=True
        )
        # moving_frame_cloud.add_vector_quantity(
        #     "e2", moving_frames[:, :, 1], color=(0.0, 1.0, 0.0), enabled=True
        # )
        # moving_frame_cloud.add_vector_quantity(
        #     "e3", moving_frames[:, :, 2], color=(0.0, 0.0, 1.0), enabled=True
        # )

    ps.show()


def polyscope_list_plot(
    point_clouds=[],
    surfaces=[],
    vector_point_clouds=[],
    vector_surfaces=[],
    pq_point_clouds=[],
    pq_surfaces=[],
    branes=[],
):
    """
    point_cloud = {"points": points, "point_color": point_color}

    vector_point_cloud = {"points": points, "point_color": point_color,
                          "vectors": vectors, "vector_color": vector_color}

    pq_point_cloud = {"pq": pq, "point_color": point_color,
                      "vector_color1": vector_color1,
                      "vector_color2": vector_color2,
                      "vector_color3": vector_color3}

    surface = {"vertices": vertices, "faces": faces, "face_color": face_color,
               "edge_color": edge_color, "transparency": transparency,
               "face_rgb_values": face_rgb_values,
               "vertex_rgb_values": vertex_rgb_values}
    """
    for brane in branes:
        # pq_point_clouds.append({"pq": brane.pq})
        # face_rgb_values = np.random.rand(3 * len(brane.faces)).reshape(
        #     (len(brane.faces), 3)
        # )
        # vertex_rgb_values = np.random.rand(3 * len(brane.pq)).reshape(
        #     (len(brane.pq), 3)
        # )
        # face_scalar_values = np.random.rand(len(brane.faces))
        # vertex_scalar_values = brane.V_scalar  # 6 * np.random.rand(len(brane.pq)) - 3

        # brane_dict = {
        #     "vertices": brane.pq[:, :3].copy(),
        #     "faces": brane.faces.copy(),
        #     # "face_rgb_values": face_rgb_values,
        #     # "vertex_rgb_values": vertex_rgb_values,
        #     # "face_scalar_values": face_scalar_values,
        #     "vertex_scalar_values": brane.V_scalar.copy(),
        # }
        surfaces.append(
            {
                "vertices": brane.pq[:, :3].copy(),
                "faces": brane.faces.copy(),
                "vertex_scalar_values": brane.V_scalar.copy(),
            }
        )
    ps_point_clouds = []
    ps_surfaces = []
    ps_vector_point_clouds = []
    ps_vector_surfaces = []
    ps_pq_point_clouds = []
    ps_pq_surfaces = []
    ps.init()
    # ps.set_transparency_mode("simple")
    for _, point_cloud in enumerate(point_clouds):
        for key in ["point_color"]:
            if not key in point_cloud.keys():
                point_cloud[key] = None
        points = point_cloud["points"]
        color = point_cloud["point_color"]
        register_point_cloud_kwargs = {
            "name": f"point_cloud{_}",
            "points": points,
            "enabled": True,
            "radius": 0.0025,
            "color": color,
        }
        ps_point_clouds.append(ps.register_point_cloud(**register_point_cloud_kwargs))
    for _, vector_point_cloud in enumerate(vector_point_clouds):
        for key in ["point_color", "vector_color"]:
            if not key in vector_point_cloud.keys():
                vector_point_cloud[key] = None
        points = vector_point_cloud["points"]
        vectors = vector_point_cloud["vectors"]
        point_color = vector_point_cloud["point_color"]
        vector_color = vector_point_cloud["vector_color"]
        register_point_cloud_kwargs = {
            "name": f"vector_point_cloud{_}",
            "points": points,
            "enabled": True,
            "radius": 0.0025,
            "color": point_color,
        }
        add_vector_quantity_kwargs = {
            "name": f"v_vector_point_cloud{_}",
            "values": vectors,
            "enabled": True,
            # "radius": 0.0025,
            "color": vector_color,
        }
        ps_vector_point_clouds.append(
            ps.register_point_cloud(**register_point_cloud_kwargs)
        )

        ps_vector_point_clouds[-1].add_vector_quantity(**add_vector_quantity_kwargs)

    for _, pq_point_cloud in enumerate(pq_point_clouds):
        for key in ["point_color", "vector_color1", "vector_color2", "vector_color3"]:
            if not key in pq_point_cloud.keys():
                pq_point_cloud[key] = None
        point_color = pq_point_cloud["point_color"]
        v1_color = pq_point_cloud["vector_color1"]
        v2_color = pq_point_cloud["vector_color2"]
        v3_color = pq_point_cloud["vector_color3"]

        points = pq_point_cloud["pq"][:, :3]
        q = pq_point_cloud["pq"][:, 3:]
        frames = np.array([quaternion_to_matrix(qi) for qi in q])
        v1, v2, v3 = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]

        register_point_cloud_kwargs = {
            "name": f"pq_point_cloud{_}",
            "points": points,
            "enabled": True,
            "radius": 0.0025,
            "color": point_color,
        }

        add_vector_quantity_kwargs1 = {
            "name": f"v1_pq_point_cloud{_}",
            "values": v1,
            "enabled": True,
            # "radius": 0.0025,
            "color": v1_color,
        }
        add_vector_quantity_kwargs2 = {
            "name": f"v2_pq_point_cloud{_}",
            "values": v2,
            "enabled": True,
            # "radius": 0.0025,
            "color": v2_color,
        }
        add_vector_quantity_kwargs3 = {
            "name": f"v3_pq_point_cloud{_}",
            "values": v3,
            "enabled": True,
            # "radius": 0.0025,
            "color": v3_color,
        }
        ps_pq_point_clouds.append(
            ps.register_point_cloud(**register_point_cloud_kwargs)
        )

        ps_pq_point_clouds[-1].add_vector_quantity(**add_vector_quantity_kwargs1)
        ps_pq_point_clouds[-1].add_vector_quantity(**add_vector_quantity_kwargs2)
        ps_pq_point_clouds[-1].add_vector_quantity(**add_vector_quantity_kwargs3)

    for _, surface in enumerate(surfaces):
        for key in ["face_color", "edge_color", "transparency"]:
            if not key in surface.keys():
                surface[key] = None
        vertices = surface["vertices"]
        faces = surface["faces"]
        edge_color = surface["edge_color"]
        face_color = surface["face_color"]
        transparency = surface["transparency"]
        register_surface_mesh_kwargs = {
            "name": f"surfaces{_}",
            "vertices": vertices,
            "faces": faces,
            "enabled": True,
            "color": face_color,
            "edge_color": edge_color,
            "edge_width": 1.0,
            "transparency": transparency,
        }
        ps_surfaces.append(ps.register_surface_mesh(**register_surface_mesh_kwargs))
        if "face_rgb_values" in surface.keys():
            face_rgb_values = surface["face_rgb_values"]
            ps_surfaces[-1].add_color_quantity(
                "face_rgb_values", face_rgb_values, defined_on="faces", enabled=True
            )
        if "vertex_rgb_values" in surface.keys():
            vertex_rgb_values = surface["vertex_rgb_values"]
            ps_surfaces[-1].add_color_quantity(
                "vertex_rgb_values", vertex_rgb_values, enabled=True
            )
        if "vertex_scalar_values" in surface.keys():
            vertex_scalar_values = surface["vertex_scalar_values"]
            # ps_mesh.add_scalar_quantity("rand vals2", vals_face, defined_on="faces")
            # # visualize some random data per-edge (halfedges are also supported)
            # vals_edge = np.random.rand(ps_mesh.n_edges())
            # ps_mesh.add_scalar_quantity("rand vals3", vals_edge, defined_on="edges")
            # as always, we can customize the initial appearance
            ps_surfaces[-1].add_scalar_quantity(
                "vertex_scalar_values",
                vertex_scalar_values,
                enabled=True,
                # vminmax=(-3.0, 3.0),
                cmap="coolwarm",
            )

    ps.show()
    ps.remove_all_structures()
