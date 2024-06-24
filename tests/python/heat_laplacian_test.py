from src.python.half_edge_mesh import HalfEdgeMesh, HalfEdgePatch, HeatLaplacian
from src.python.mesh_viewer import MeshViewer
import os
from time import time
#
source_ply = "./data/ply/binary/dumbbell.ply"
image_dir = "./output/heat_laplacian_tests/dumbbell"

# source_ply = "./data/ply/binary/torus.ply"
# image_dir = "./output/heat_laplacian_tests/torus"
#
# source_ply = "./data/ply/binary/neovius.ply"
# image_dir = "./output/heat_laplacian_tests/neovius"

v0 = 3


# def expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60):
# """
# Makes a movie of heat laplacian computations
# """
if os.path.exists(image_dir):
    os.system(f"rm -r {image_dir}")
os.system(f"mkdir -p {image_dir}")


m = HalfEdgeMesh.from_half_edge_ply(source_ply)
L = HeatLaplacian(m)


Y = m.xyz_coord_V
lapY = L.apply(Y)
vector_field_data = [lapY]
viewer_kwargs = {
    "image_dir": image_dir,
    "show_vertices": True,
    "v_radius": 0.03,
    "vector_field_data": [(Y, lapY),]
}
mv = MeshViewer(*m.data_lists, **viewer_kwargs)

# %%
