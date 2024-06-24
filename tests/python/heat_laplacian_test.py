from src.python.half_edge_mesh import HalfEdgeMesh, HalfEdgePatch, HeatLaplacian
from src.python.mesh_viewer import MeshViewer
import os
from time import time
import numpy as np
#
source_ply = "./data/ply/binary/dumbbell.ply"
image_dir = "./output/heat_laplacian_tests/dumbbell"

# source_ply = "./data/ply/binary/torus.ply"
# image_dir = "./output/heat_laplacian_tests/torus"
#
# source_ply = "./data/ply/binary/neovius.ply"
# image_dir = "./output/heat_laplacian_tests/neovius"

v0 = 3

# def heat_laplacian_compute(source_ply, image_dir, v_seed=3, iters=60):
# """
# Makes a movie of heat laplacian computations
# """
if os.path.exists(image_dir):
    os.system(f"rm -r {image_dir}")
os.system(f"mkdir -p {image_dir}")


m = HalfEdgeMesh.from_half_edge_ply(source_ply)
laplacian_kwargs = {
    "mesh": m,
    "rtol": 1e-6,
    "atol": 1e-6,
}
L = HeatLaplacian(m)
W = L.compute_weights_matrix()
# %%





Y = m.xyz_array
lapY = L.apply(Y)
lapYscale = 1/max(np.linalg.norm(lapY, axis=1))
viewer_kwargs = {
    "image_dir": image_dir,
    "show_vertices": True,
    "v_radius": 0.03,
    "vector_field_data": [(Y, -lapYscale*lapY),]
}
mv = MeshViewer(*m.data_lists, **viewer_kwargs)
mv.plot()
# %%
import cProfile
# p = HalfEdgePatch.from_seed_vertex(3, m)
def my_function():
    # Example: A function that might be slow
    # V=p.expand_boundary()
    L = HeatLaplacian(m)
    W = L.compute_weights_matrix()


# Profile the function
cProfile.run('my_function()')

import cProfile
import pstats

def function_to_profile():


# Create a Profile object
profile = cProfile.Profile()
profile.enable()

# Run the function you want to profile
function_to_profile()

profile.disable()

# Create Stats object
stats = pstats.Stats(profile)
stats.sort_stats(pstats.SortKey.TIME)  # Sort the statistics by time spent
stats.print_stats()  # Print the statistics
