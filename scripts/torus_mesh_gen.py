import sys

sys.path.append("./")

from src.python.utilities import make_output_dir
from src.python.half_edge_base_unit_torus import UnitDoughnutFactory
import numpy as np
from src.python.half_edge_base_mesh import HalfEdgeMeshBase
from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_ply_tools import MeshConverterBase

output_dir = "./output/torus_mesh_gen"
image_dir = f"{output_dir}/temp_images"
scale_phi = 3
scale_psi = 1
resolution_min = 3
resolution_max = 8
smooth_iters = 20


def generate_torii():
    make_output_dir(output_dir, overwrite=True)

    f = UnitDoughnutFactory(scale_phi=scale_phi, scale_psi=scale_psi)

    def better_mesh(m):
        num_flips = m.flip_non_delaunay()
        m.smooth_by_shifts()
        xyz_coord_V = f.project_to_torus(*m.xyz_coord_V.T)
        err = np.linalg.norm(m.xyz_coord_V - xyz_coord_V)
        m.xyz_coord_V = xyz_coord_V
        return err, num_flips

    for p in range(resolution_min, resolution_max + 1):
        print("-------------------------------------")
        print(f"initializing mesh_converter at resolution {p=}")
        f.init_mesh_converter_at_resolution_p(p)
        c = f.mesh_converter[p]
        surfname = f.name
        num_faces = f.num_faces(p)
        ply_raw = f"{output_dir}/{surfname}_raw_{num_faces:06d}_he.ply"
        samples_raw = f"{output_dir}/{surfname}_raw_{num_faces:06d}_he.npz"
        print(f"writing {ply_raw=}")
        c.write_he_ply(ply_raw, use_binary=True)
        print(f"writing {samples_raw=}")
        c.write_he_samples(
            path=samples_raw, compressed=False, chunk=False, remove_unchunked=False
        )
        image_dir = f"{output_dir}/temp_images_{num_faces:06d}"
        print(f"smoothing samples...")
        make_output_dir(image_dir, overwrite=False)
        m = HalfEdgeMeshBase(*c.he_samples)
        mv = MeshViewer(m, figsize=(480, 480), image_dir=image_dir)
        mv.apply_fun_iter(better_mesh, num_iters=smooth_iters)

        ply_smooth = f"{output_dir}/{surfname}_smooth_{num_faces:06d}_he.ply"
        samples_smooth = f"{output_dir}/{surfname}_smooth_{num_faces:06d}_he.npz"

        print(f"initializing smooth mesh_converter at resolution {p=}")
        c = MeshConverterBase.from_he_samples(*m.he_samples, compute_vf_stuff=False)
        print(f"writing {ply_smooth=}")
        c.write_he_ply(ply_smooth, use_binary=True)
        print(f"writing {samples_smooth=}")
        c.write_he_samples(
            path=samples_smooth, compressed=False, chunk=False, remove_unchunked=False
        )


# %%
# p = 5
# m0 = HalfEdgeMeshBase(*f.mesh_converter[p].he_samples)
# m = HalfEdgeMeshBase(*f.mesh_converter[p].he_samples)
#
#
# # m.flip_non_delaunay()
#
#
# err, num_flips = better_mesh(m)
# m0.num_faces
# print(f"{err=}, {num_flips=}")
# # mv0 = MeshViewer(m0)
# # mv0.plot()
# m0.total_area_of_faces()
#
# mv = MeshViewer(m)
# mv.apply_fun_iter(better_mesh, num_iters=50)
#
# ply_raw = "./output/torus_mesh_gen/unit_torus_3_1_raw_098304_he.ply"
# ply_smooth = "./output/torus_mesh_gen/unit_torus_3_1_smooth_098304_he.ply"
# ply_raw = "./output/torus_mesh_gen/unit_torus_3_1_raw_024576_he.ply"
# ply_smooth = "./output/torus_mesh_gen/unit_torus_3_1_smooth_024576_he.ply"
# # ply = "./data/ply/binary/unit_sphere_163842_he.ply"
# # m = HalfEdgeMeshBase.from_he_ply(ply_raw)
# m = HalfEdgeMeshBase.from_he_ply(ply_smooth)
# mv = MeshViewer(m, figsize=(480, 480), show_half_edges=True)
# mv.plot()
