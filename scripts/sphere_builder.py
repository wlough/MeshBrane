from src.python.ply_tools import SphereFactory

# from src.python.mesh_viewer import MeshViewer
# from src.python.half_edge_mesh import HalfEdgeMesh

# ply = f"./data/ply/binary/unit_sphere_10242.ply"
# m = HalfEdgeMesh.from_half_edge_ply(ply)
# V = m.xyz_array
# F = m.V_of_F
# sf = SphereFactory.from_unit_sphere_VF(V, F)
# sf.num_vertices()
# sf.refine()
# sf.write_plys(level=-1)
# b.write_plys(level=0)
# v2h = [VertTri2HalfEdgeConverter.from_source_samples(V, F[-1])]
#
SphereFactory.build_test_plys(num_refine=5)
# SphereFactory.build_noisy_test_plys(num_refine=5, noise_scale=0.01)
# [12, 42, 162, 642, 2562, 10242]
# [40968, 163872, 655488]
