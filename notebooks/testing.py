# from src.python.half_edge_mesh import HalfEdgeMesh
# from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_base_mesh import Brane

# from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_base_viewer import MeshViewer


# VertTri2HalfEdgeMeshConverter._oblatify_the_spheres(ratio=0.9)

vf_ply = "./data/ply/binary/torus_003072_vf.ply"
he_ply = "./data/ply/binary/torus_003072_he.ply"
he_ply = "./data/half_edge_base/ply/unit_sphere_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_003072_he.ply"

# he_ply = "./data/half_edge_base/ply/dumbbell_he.ply"
he_ply = "./data/half_edge_base/ply/neovius_coarse_he.ply"
b = Brane.from_half_edge_ply(he_ply)
# b.flip_non_delaunay()
mv = MeshViewer(*b.data_arrays, target_faces=1000)

mv.simple_plot()
# %%
_PLY_INT_ = "int32"


def samples_to_ply_data(self, *samples, use_binary=False):
    """Constructs a PlyData object using the schema"""

    (
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    ) = samples[:7]
    V_data = np.array(
        [(x, y, z, h) for (x, y, z), h in zip(xyz_coord_V, h_out_V)],
        dtype=[
            ("x", _PLY_FLOAT_),
            ("y", _PLY_FLOAT_),
            ("z", _PLY_FLOAT_),
            ("h", _PLY_INT_),
        ],
    )
    H_data = np.array(
        [(v, n, t, f) for v, n, t, f in zip(v_origin_H, h_next_H, h_twin_H, f_left_H)],
        dtype=[
            ("v", _PLY_INT_),
            ("n", _PLY_INT_),
            ("t", _PLY_INT_),
            ("f", _PLY_INT_),
        ],
    )
    F_data = np.array(h_bound_F, dtype=[("h", _PLY_INT_)])
    # # ***
    # print(V_data)
    # print(type(V_data))
    # print(self.float_type)
    # print(self.int_type)
    # # ***
    vertex_element = PlyElement.describe(V_data, "vertex")
    half_edge_element = PlyElement.describe(H_data, "half_edge")
    face_element = PlyElement.describe(F_data, "face")
    if len(samples) == 8:
        h_right_B = samples[7]
        B_data = np.array(h_right_B, dtype=[("h", _PLY_INT_)])
        boundary_element = PlyElement.describe(B_data, "boundary")
        return PlyData(
            [vertex_element, half_edge_element, face_element, boundary_element],
            text=not use_binary,
        )

    h_right_B = find_h_right_B(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )
    B_data = np.array(h_right_B, dtype=[("h", _PLY_INT_)])
    boundary_element = PlyElement.describe(B_data, "boundary")
    return PlyData(
        [vertex_element, half_edge_element, face_element, boundary_element],
        text=not use_binary,
    )


# %%

m1._gaussian_curvature_v(0)
m1.gaussian_curvature_v(0)
# %%
# m = HalfEdgeMesh.from_half_edge_ply(he_ply)
m = m1
H, K, lapH, n = m.compute_curvature_data()
H0 = 0
Kb = 0.1
Fbend = -2 * Kb * (lapH + 2 * (H - H0) * (H**2 + H0 * H - K))
Av = m.barcell_area_V()
FbendA = Fbend * Av
# %%
# mv = MeshViewer(*m.data_arrays)
# mv.plot()
scale_vec = 10.5
Fn = FbendA
# vec = scale_vec * np.einsum("i,ij->ij", Fn, n)
vec = 0.1 * n
# vec=scale_vec*n
vfdat = [m.xyz_coord_V, vec]
mv_kwargs = {
    "vector_field_data": [vfdat],
    # "V_rgba": V_rgba,
    # "color_by_V_rgba": True,
    # "E_rgba": E_rgba,
}
mv = MeshViewer(*m.data_arrays, **mv_kwargs)
mv.plot()
# %%
# from src.python.half_edge_mesh import HalfEdgeMesh
# from src.python.half_edge_patch import HalfEdgePatch
from src.python.half_edge_base_mesh import HalfEdgeMeshBase as HalfEdgeMesh
from src.python.half_edge_base_mesh import HalfEdgePatchBase as HalfEdgePatch

from src.python.mesh_viewer import MeshViewer
import os

# %%


def expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60):
    """
    Makes a movie of surface patch expanding from seed vertex
    """
    if os.path.exists(image_dir):
        os.system(f"rm -r {image_dir}")
    os.system(f"mkdir -p {image_dir}")
    viewer_kwargs = {
        "image_dir": image_dir,
        "show_vertices": True,
        "v_radius": 0.03,
    }

    m = HalfEdgeMesh.from_half_edge_ply(source_ply)
    mv = MeshViewer(*m.data_arrays, **viewer_kwargs)
    p = HalfEdgePatch.from_seed_vertex(v_seed, m)

    V = [v_seed]
    H, F = [], []
    mv.set_F_rgba(f_rgba=mv.colors["green10"])
    mv.set_E_rgba(e_rgba=mv.colors["orange10"])
    mv.set_V_rgba(v_rgba=mv.colors["transparent"])
    mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V)
    mv.save_plot()
    Vnew = list(p.V_bdry)

    for iter in range(iters):
        print(f"iter={iter}")

        V = Vnew
        H = list(p.generate_H_cw_B())
        F = list(p.generate_F_cw_B())

        mv.set_F_rgba(f_rgba=mv.colors["green10"])
        mv.set_E_rgba(e_rgba=mv.colors["orange10"])
        mv.set_V_rgba(v_rgba=mv.colors["transparent"])

        mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=H)
        mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=list(F))
        mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V)
        mv.save_plot()
        Vnew = list(p.expand_boundary())
    # V_new = list(p.V_bdry)
    #
    # for iter in range(iters):
    #     print(f"iter={iter}")
    #
    #     H_bdry = list(p.generate_H_cw_B())
    #     F_bdry = list(p.generate_F_cw_B())
    #     H = list(p.H)
    #     F = list(p.F)
    #
    #     mv.set_F_rgba(f_rgba=mv.colors["green10"])
    #     mv.set_E_rgba(e_rgba=mv.colors["orange10"])
    #     mv.set_V_rgba(v_rgba=mv.colors["transparent"])
    #
    #     mv.set_subset_E_rgba(rgba=mv.colors["orange50"], indices=H)
    #     mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=F)
    #
    #     mv.set_subset_E_rgba(rgba=mv.colors["red"], indices=H_bdry)
    #     mv.set_subset_F_rgba(rgba=mv.colors["blue"], indices=F_bdry)
    #     mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V_new)
    #     mv.save_plot()
    #     V_new = list(p.expand_boundary())

    mv.movie()


# 2 min 40 sec
# %%
source_ply = "./data/half_edge_base/ply/dumbbell_he.ply"
image_dir = "./output/expanding_patch_test/dumbbell"
expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60)

source_ply = "./data/half_edge_base/ply/torus_012288_he.ply"
image_dir = "./output/expanding_patch_test/torus"
expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60)

source_ply = "./data/half_edge_base/ply/neovius_he.ply"
image_dir = "./output/expanding_patch_test/neovius"
expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60)

# %%


def expanding_patch_vertex_count(source_ply, v_seed=3):
    """
    check new verts from HalfEdgePatch.expand_boundary() for repeats
    """

    m = HalfEdgeMesh.from_half_edge_ply(source_ply)
    Vcounts = {v: 0 for v in m.xyz_coord_V.keys()}
    p = HalfEdgePatch.from_seed_vertex(v_seed, m)

    Vnew = p.V
    for v in Vnew:
        Vcounts[v] += 1

    while Vnew:
        Vnew = p.expand_boundary()
        for v in Vnew:
            Vcounts[v] += 1

    V = [val for key, val in Vcounts.items()]
    count_max = max(V)
    count_min = min(V)
    print(f"{count_max=}, {count_min=}")


source_ply = "./data/ply/binary/torus.ply"
source_ply = "./data/ply/binary/neovius.ply"
source_ply = "./data/ply/binary/dumbbell.ply"
expanding_patch_vertex_count(source_ply, v_seed=3)


# %%
def gen(i):
    while i < 13:
        yield i
        i += 1


I = gen(3)
j = next(I)
for jrot in I:
    print(f"{j=}, {jrot=}")
    j = jrot
