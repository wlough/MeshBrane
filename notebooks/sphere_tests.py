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


def load_spheres():
    Nverts = [12, 42, 162, 642, 2562, 10242]
    Nverts = [12, 42, 162, 642]
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


# M0 = make_sphere_data(overwrite=True)


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
# %%
num = 3
timelike_num = 2
M = Mall[:]
num_vertices = [m.num_vertices for m in M]
spherical_arr = [m.spherical_coord_array() for m in M]
vfdat = [[m.xyz_array, -vec] for m, vec in zip(M, mcvec_belkin[timelike_num])]
m = M[num]
vfdat = [m.xyz_array, -mcvec_belkin[timelike_num][num]]
mv = MeshViewer(*m.data_lists, vector_field_data=[vfdat])
mv.plot()
# %%

err = mcvec_belkin_L2error[timelike_num][1:]
thing = [m.num_vertices for m in Mall][1:]

error_ave_kwargs = {
    "X": thing,
    "Y": err,
    "Xlabel": "N",
    "Ylabel": "ave|$H-H*$|",
    "title": "Average error",
}
log_log_fit(**error_ave_kwargs)
