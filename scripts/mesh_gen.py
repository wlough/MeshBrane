from src.python.utilities import make_output_dir
from src.python.half_edge_base_named_mesh import (
    HalfEdgeNamedMesh,
    HalfEdgeSphere,
    HalfEdgeTorus,
)
from src.python.half_edge_base_mesh import HalfEdgeMeshBase
from src.python.half_edge_base_viewer import MeshViewer

# from src.python.torus_builder import
#
# uncompress_sphere_half_edge_arrays()
# update_compressed_sphere_half_edge_arrays_with_h_right_B()
# output_dir = "./output/mesh_gen"
# make_output_dir(output_dir)
# test_ply = "./data/half_edge_base/ply/"
#
# p_Nv = [
#     [0, 12],
#     [1, 42],
#     [2, 162],
#     [3, 642],
#     [4, 2562],
#     [5, 10242],
#     [6, 40962],
#     [7, 163842],
#     [8, 655362],
#     [9, 2621442],
# ]
# he_array_dir = "./output/half_edge_arrays"
# ply_dir = "./data/half_edge_base/ply"
# # ply_path =
# m = HalfEdgeSphere.load_num(p=9)
# mv = MeshViewer(m)
# mv.plot()
#

#


def build_surf_coords_and_faces(p):
    """
    a, b = self.major_radius, self.minor_radius
    phi, psi = surfcoord_array.T
    x = (a + b * np.cos(psi)) * np.cos(phi)
    y = (a + b * np.cos(psi)) * np.sin(phi)
    z = b * np.sin(psi)
    """
    import numpy as np

    ratio_big2small = 3
    N_phi = ratio_big2small * 2**p
    N_psi = 2**p
    # Nv = N_phi * N_psi = 3*4**p
    # p = np.log(Nv//ratio_big2small)//np.log(4)
    minor_radius = 1 / 3
    major_radius = 1
    Phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Psi = np.linspace(0, 2 * np.pi, N_psi, endpoint=False)

    # Psi = np.zeros(N_psi)
    surf_coord_V = np.array([[phi, psi] for phi in Phi for psi in Psi])
    V_of_F = np.array(
        [
            [
                b * N_psi + s,
                ((b + 1) % N_phi) * N_psi + (s + 1) % N_psi,
                ((b + 1) % N_phi) * N_psi + s,
                #
                b * N_psi + s,
                b * N_psi + (s + 1) % N_psi,
                ((b + 1) % N_phi) * N_psi + (s + 1) % N_psi,
            ]
            for b in range(N_phi)
            for s in range(N_psi)
        ],
        dtype="int32",
    ).reshape((-1, 3))
    V = []
    F = []
    for b in range(N_phi):
        phi = 2 * np.pi * b / N_phi
        bp1 = (b + 1) % N_phi
        for s in range(N_psi):
            sp1 = (s + 1) % N_psi
            psi = 2 * np.pi * s / N_psi
            x = np.cos(phi) * (major_radius + np.cos(psi) * minor_radius)
            y = np.sin(phi) * (major_radius + np.cos(psi) * minor_radius)
            z = np.sin(psi) * minor_radius
            V.append(np.array([x, y, z]))
            b_s = b * N_psi + s
            b_sp1 = b * N_psi + sp1
            bp1_s = bp1 * N_psi + s
            bp1_sp1 = bp1 * N_psi + sp1
            F.append([b_s, bp1_sp1, bp1_s])
            F.append([b_s, b_sp1, bp1_sp1])
    return V, F, surf_coord_V, V_of_F


def refine(surf_coord_V_coarse, V_of_F_coarse):
    ratio_big2small = 3
    Nv_coarse = len(surf_coord_V_coarse)
    p_coarse = np.int32(np.log2(Nv_coarse // ratio_big2small) // 2)
    Nphi_coarse = ratio_big2small * 2**p_coarse
    Npsi_coarse = 2**p_coarse
    Npsi = 2 * Npsi_coarse
    Nphi = 2 * Nphi_coarse
    pow = p_coarse + 1
    # print(f"Refining {self.name}...")
    print(f"num_vertices: {Nphi_coarse*Npsi_coarse}-->{Nphi*Npsi}")

    F = []
    v_BS = []
    v_BS_coarse = self.v_BS[-1]

    for b_coarse in range(Nphi_coarse):
        ###################################################
        # add every other vertex to each ring in old mesh
        b = 2 * b_coarse
        bp1 = (b + 1) % Nphi
        phi = 2 * np.pi * b / Nphi
        for s_coarse in range(Npsi_coarse):
            # every other vertex is the same as the coarse mesh
            s = 2 * s_coarse
            b_s_coarse = b_coarse * Npsi_coarse + s_coarse
            v_b_s = v_BS_coarse[b_s_coarse]  # v index
            sp1 = (s + 1) % Npsi
            b_s = b * Npsi + s
            b_sp1 = b * Npsi + sp1
            bp1_s = bp1 * Npsi + s
            bp1_sp1 = bp1 * Npsi + sp1
            F.append([b_s, bp1_sp1, bp1_s])
            F.append([b_s, b_sp1, bp1_sp1])
            v_BS.append(v_b_s)
            # every other vertex is new
            s = 2 * s_coarse + 1
            v_b_s = len(self.xyz_coord_V)  # new v index
            psi = 2 * np.pi * s / Npsi
            x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
            y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
            z = np.sin(psi) * r_s
            self.xyz_coord_V.append(np.array([x, y, z]))
            sp1 = (s + 1) % Npsi
            b_s = b * Npsi + s
            b_sp1 = b * Npsi + sp1
            bp1_s = bp1 * Npsi + s
            bp1_sp1 = bp1 * Npsi + sp1
            # bs_V[v_b_s] = b_s
            v_BS.append(v_b_s)
            F.append([b_s, bp1_sp1, bp1_s])
            F.append([b_s, b_sp1, bp1_sp1])

        ###################################################
        # add every vertex to each new ring not in old mesh
        b = 2 * b_coarse + 1
        bp1 = (b + 1) % Nphi
        phi = 2 * np.pi * b / Nphi
        for s in range(Npsi):
            v_b_s = len(self.xyz_coord_V)  # new v index
            psi = 2 * np.pi * s / Npsi
            x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
            y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
            z = np.sin(psi) * r_s
            self.xyz_coord_V.append(np.array([x, y, z]))
            sp1 = (s + 1) % Npsi
            b_s = b * Npsi + s
            b_sp1 = b * Npsi + sp1
            bp1_s = bp1 * Npsi + s
            bp1_sp1 = bp1 * Npsi + sp1
            v_BS.append(v_b_s)
            F.append([b_s, bp1_sp1, bp1_s])
            F.append([b_s, b_sp1, bp1_sp1])

    self.F.append(F)
    self.v_BS.append(v_BS)
    if convert_to_half_edge:
        print("Converting to half-edge mesh...")
        self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))


#############################################
# make torii
# %%
from src.python.utilities import make_output_dir
from src.python.torus_builder import DoughnutFactory

output_dir = "./output/test_ply"
make_output_dir(output_dir, overwrite=True)
DoughnutFactory.build_test_plys(num_refine=2, ply_dir=output_dir)
# %%
from src.python.half_edge_base_mesh import HalfEdgeMeshBase
from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_named_mesh import HalfEdgeTorus

ply = "./output/test_ply/torus_3_1_003072_he.ply"
m = HalfEdgeTorus.from_half_edge_ply(ply)
mv = MeshViewer(m, show_half_edges=True)
mv.plot()
