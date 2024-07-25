from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer, FancyMayaviVectorField
from src.python.half_edge_ops import HeatLaplacian, HeatLaplacian2
from src.python.figs import log_log_fit
import os
import pickle
import numpy as np


def make_sphere_data(overwrite=False):
    timelike_param = [0.01 / 4, 0.04 / 4, 0.09 / 4]
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162,642]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    output_dir = "./output/sphere_tests"
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    else:
        raise ValueError("Ahhhhhh")
    os.system(f"mkdir -p {output_dir}")
    M = []
    for _ in range(len(surfs)):
        surf = surfs[_]
        print(f"running tests for {surf}")
        data_path = f"{output_dir}/{surf}"
        ply = f"./data/ply/binary/{surf}.ply"
        m = HalfEdgeMesh.from_half_edge_ply(ply)
        m.run_unit_sphere_mean_curvature_normal_tests(timelike_param)
        m.save(data_path)
        M.append(m)
    return M


def load_spheres_from_ply():
    Nverts = [12, 42, 162, 642, 2562, 10242]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    M = []
    for _ in range(len(surfs)):
        surf = surfs[_]
        ply = f"./data/ply/binary/{surf}.ply"
        m = HalfEdgeMesh.from_half_edge_ply(ply)
        M.append(m)
    return M


def load_spheres():
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162, 642, 2562]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    output_dir = "./output/sphere_tests"
    M = []
    for surf in surfs:
        data_path = f"{output_dir}/{surf}"
        with open(data_path + ".pickle", "rb") as f:
            M.append(pickle.load(f))
    return M


def get_test_data(M):
    timelike_param = np.array([m.timelike_param for m in M][0])
    mcvec_actual = [m.mcvec_actual for m in M]
    mcvec_cotan = [m.mcvec_cotan for m in M]
    mcvec_cotan_L2error = np.array([m.mcvec_cotan_L2error for m in M])
    mcvec_cotan_Lifntyerror = np.array([m.mcvec_cotan_Lifntyerror for m in M])

    _mcvec_belkin = [m.mcvec_belkin for m in M]
    _mcvec_belkin_L2error = [m.mcvec_belkin_L2error for m in M]
    _mcvec_belkin_Lifntyerror = [m.mcvec_belkin_Lifntyerror for m in M]

    num_M = len(M)
    num_timelike = len(timelike_param)
    mcvec_belkin = [[m.mcvec_belkin[_] for m in M] for _ in range(num_timelike)]
    mcvec_belkin_L2error = np.array(
        [[m.mcvec_belkin_L2error[_] for m in M] for _ in range(num_timelike)]
    )
    mcvec_belkin_Lifntyerror = np.array(
        [[m.mcvec_belkin_Lifntyerror[_] for m in M] for _ in range(num_timelike)]
    )
    return (
        timelike_param,
        mcvec_actual,
        mcvec_cotan,
        mcvec_cotan_L2error,
        mcvec_cotan_Lifntyerror,
        mcvec_belkin,
        mcvec_belkin_L2error,
        mcvec_belkin_Lifntyerror,
    )


Mall = load_spheres()
(
    timelike_param,
    mcvec_actual,
    mcvec_cotan,
    mcvec_cotan_L2error,
    mcvec_cotan_Lifntyerror,
    mcvec_belkin,
    mcvec_belkin_L2error,
    mcvec_belkin_Lifntyerror,
) = get_test_data(Mall)
# Mall = load_spheres_from_ply()
# %%
mesh_num = 5
timelike_num = 2
M = Mall
num_vertices = [m.num_vertices for m in M]
spherical_arr = [m.spherical_coord_array() for m in M]
vfdat = [[m.xyz_array, -vec] for m, vec in zip(M, mcvec_belkin[timelike_num])]
m = M[mesh_num]
vfdat = [m.xyz_array, -mcvec_belkin[timelike_num][mesh_num]]
mv = MeshViewer(*m.data_lists, vector_field_data=[vfdat])
mv.plot()
# %%
mesh_num = 0
timelike_num = 2
# err = mcvec_belkin_L2error[timelike_num][mesh_num:]
# thing = [m.num_vertices for m in Mall][mesh_num:]
err = [0.613, 0.140, 0.028, 0.015]
thing = [500, 2000, 8000, 16000]
error_ave_kwargs = {
    "X": thing,
    "Y": err,
    "Xlabel": "N",
    "Ylabel": "||$H-H*$||/||$H*$||",
    # "Ylabel": "||H-H*||_{L^2}/||H*||_{L^2}",
    "title": "Average error",
}
log_log_fit(**error_ave_kwargs)
# %%

mesh_num = 5
timelike_num = 1
s = timelike_param[timelike_num]
m = Mall[mesh_num]
mcvec0 = m.mcvec_belkin[timelike_num]
Q = m.xyz_array


def belkin_laplacian1(self, s, Q):
    V = m.xyz_array
    A = np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()])
    lapQ = (1 / (4 * np.pi * s**2)) * np.einsum(
        "y,xy,xy...->x...",
        np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()]),
        np.exp(-np.linalg.norm(np.array([V - v for v in V]), axis=-1) ** 2 / (4 * s)),
        np.array([Q - q for q in Q]),
    )
    return lapQ


def belkin_laplacian2(self, s, Q):
    V = self.xyz_array
    A = np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()])
    lapQ = np.array(
        [
            np.einsum(
                "y,y,y...->...",
                A,
                np.exp(-np.linalg.norm(V - x, axis=-1) ** 2 / (4 * s)),
                Q - q,
            )
            for x, q in zip(V, Q)
        ]
    ) / (4 * np.pi * s**2)

    return lapQ


def belkin_laplacian3(self, s, Q):
    # Nv = self.num_vertices
    V = self.xyz_array
    A = np.array([self.area_f(f) for f in self.h_bound_F.keys()]) / 3
    F = np.array([[v for v in self.generate_V_of_f(f)] for f in self.h_bound_F.keys()])
    VF = V[F]
    QF = Q[F]
    lapQ = np.array(
        [
            np.einsum(
                "f,fy,fy...->...",
                A,
                np.exp(-np.linalg.norm(VF - x, axis=-1) ** 2 / (4 * s)),
                QF - q,
            )
            for x, q in zip(V, Q)
        ]
    ) / (4 * np.pi * s**2)
    return lapQ


def belkin_laplacian4(self, s, Q):
    V = self.xyz_array  # xi
    F = np.array([[v for v in self.generate_V_of_f(f)] for f in self.h_bound_F.keys()])
    A = np.array([self.area_f(f) for f in self.h_bound_F.keys()]) / 3
    lapQ = np.array(
        [
            np.einsum(
                "f,fy,fy...->...",
                A,
                np.exp(-np.linalg.norm(V[F] - x, axis=-1) ** 2 / (4 * s)),
                Q[F] - q,
            )
            for x, q in zip(V, Q)
        ]
    ) / (4 * np.pi * s**2)
    return lapQ


def belkin_laplacian5(self, s, Q):
    V = self.xyz_array  # xi
    F = np.array([[v for v in self.generate_V_of_f(f)] for f in self.h_bound_F.keys()])
    A = np.array([self.area_f(f) for f in self.h_bound_F.keys()])  # / 3
    # V ---- xi
    # F ---- fy
    # A ---- f
    # VF --- fyi
    # QF --- fy...
    # VF_V - fyix
    # QF_Q - fy...x
    # g ---- fyx
    # lapQ - x...
    VF = V[F]
    QF = Q[F]
    VF_V = VF[:, ..., :, np.newaxis] - V.T  # fyix
    QF_Q = QF[:, ..., :, np.newaxis] - Q.T  # fy...x
    g = np.exp(-np.linalg.norm(VF_V, axis=2) ** 2 / (4 * s))
    lapQ = np.einsum("f,fyx,fy...x->x...", A, g, QF_Q) / (3 * 4 * np.pi * s**2)
    return lapQ


#
# # Profile the function and save the results to a file
# cProfile.run("belkin_laplacian5(m, s, Q)", "profile_results")
#
# # Create a Stats object
# p = pstats.Stats("profile_results")
#
# # Sort the statistics by cumulative time and print them
# _ = p.sort_stats("cumulative").print_stats(10)
# %%
mcvec1 = belkin_laplacian1(m, s, Q)
mcvec2 = belkin_laplacian2(m, s, Q)
mcvec3 = belkin_laplacian3(m, s, Q)
mcvec4 = belkin_laplacian4(m, s, Q)
# mcvec5 = belkin_laplacian5(m, s, Q)
# mcvec1 - mcvec0
# mcvec2 - mcvec0
# mcvec3 - mcvec0
# mcvec4 - mcvec0
# mcvec5 - mcvec0
# %%
# %timeit belkin_laplacian1(m, s, Q)
# %timeit belkin_laplacian2(m, s, Q)
# %timeit belkin_laplacian3(m, s, Q)
# %timeit belkin_laplacian4(m, s, Q)
# %timeit belkin_laplacian5(m, s, Q)
