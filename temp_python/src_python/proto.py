from temp_python.src_python.half_edge_mesh import (
    HalfEdgeMesh,
    PatchBoundary,
    HeatLaplacian,
    HalfEdgePatch,
)
from temp_python.src_python.mesh_viewer import MeshViewer
import os
import numpy as np

output_dir = f"./output/proto"
image_dir = f"{output_dir}/temp_images"
if os.path.exists(output_dir):
    os.system(f"rm -r {output_dir}")
os.system(f"mkdir -p {output_dir}")

mesh_kwargs = {"ply_path": "./data/ply/binary/unit_sphere_00162.ply"}
laplacian_kwargs = {
    # "mesh": m,
    "rtol": 1e-12,
    "atol": 1e-12,
}


m = HalfEdgeMesh.from_half_edge_ply(**mesh_kwargs)
# b = PatchBoundary.from_seed_vertex(3, m)
L = HeatLaplacian(m, **laplacian_kwargs, compute=True)
# L.mat = L.compute_matrix()
Y = m.xyz_array
lapY = L.apply(Y)
lap_scale = -1.0 / np.mean(np.linalg.norm(lapY, axis=1))
# -0.5023033566375791
######################

H = 0.5 * np.linalg.norm(lapY, axis=1)
norm_inf_weights = np.linalg.norm(H, np.inf)
norm_two_weights = np.linalg.norm(H, 2)
# %%
viewer_kwargs = {
    "image_dir": image_dir,
    "show_vertices": True,
    "v_radius": 0.01,
    "vector_field_data": [
        (Y, lap_scale * lapY),
    ],
}

mv = MeshViewer(*m.data_lists, **viewer_kwargs)
mv.plot()

# %%
