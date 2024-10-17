# O(N) compute time

from src.python.half_edge_mesh import HalfEdgeMesh, HeatLaplacian
from src.python.utilities.misc_utils import round_to, log_log_fit
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import dill
import pickle

source_ply_dict = {
    surf: [f"./data/ply/binary/{surf}{_}.ply" for _ in ["_coarse", "", "_fine"]]
    for surf in ["sphere", "torus", "dumbbell"]
}
# 12
# 42
# 162
# 642
# 2562
# 10242
source_plys = [f"unit_sphere_{N:05d}" for N in [12, 42, 162, 642, 2562, 10242]]


def load_heat_laplacian_weights_compute(surf):
    output_dir = f"./output/heat_laplacian_weights_compute/{surf}"
    data_paths = [f"{output_dir}/{surf}{_}" for _ in ["_coarse", "", "_fine"]]
    L = []
    for dpath in data_paths:
        with open(dpath + ".pickle", "rb") as f:
            L.append(pickle.load(f))
    Nvertices = np.array([l.num_vertices for l in L])
    Tweights = np.array([l.t_weights for l in L])
    norm_inf_W = np.array(
        [np.linalg.norm(np.ravel(l.weights.todense()), np.inf) for l in L]
    )
    norm_two_W = np.array([np.linalg.norm(np.ravel(l.weights.todense()), 2) for l in L])
    return Nvertices, Tweights, norm_inf_W, norm_two_W, L


def heat_laplacian_weights_compute(surf):
    """
    Makes data for heat laplacian computations fig
    """
    output_dir = f"./output/heat_laplacian_weights_compute/{surf}"
    data_paths = [f"{output_dir}/{surf}{_}" for _ in ["_coarse", "", "_fine"]]
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    os.system(f"mkdir -p {output_dir}")
    source_plys = source_ply_dict[surf]

    M = [HalfEdgeMesh.from_half_edge_ply(_) for _ in source_plys]
    laplacian_kwargs = [
        {
            "mesh": m,
            "rtol": 1e-12,
            "atol": 1e-12,
        }
        for m in M
    ]
    L = [HeatLaplacian(**kwargs) for kwargs in laplacian_kwargs]
    data_paths = [f"{output_dir}/{surf}{_}" for _ in ["_coarse", "", "_fine"]]
    for l, dpath in zip(L, data_paths):
        l.data_path = dpath

    for l, p in zip(L, source_plys):
        l.ply = p
        t = time()
        l.weights = l.compute_weights_matrix()
        # l.weights = l.construct_matrix()
        l.t_weights = time() - t
        l.norm_inf_weights = np.linalg.norm(np.ravel(l.weights.todense()), np.inf)
        l.norm_two_weights = np.linalg.norm(np.ravel(l.weights.todense()), 2)
        print(p)
        save_npz(l.data_path + "_weights_matrix.npz", l.weights)
        with open(l.data_path + ".pickle", "wb") as f:
            pickle.dump(l, f)

    Nvertices = np.array([l.num_vertices for l in L])
    Tweights = np.array([l.t_weights for l in L])
    norm_inf_W = np.array(
        [np.linalg.norm(np.ravel(l.weights.todense()), np.inf) for l in L]
    )
    norm_two_W = np.array([np.linalg.norm(np.ravel(l.weights.todense()), 2) for l in L])
    np.save(output_dir + "/Nvertices.npy", Nvertices)
    np.save(output_dir + "/Tweights.npy", Tweights)
    np.save(output_dir + "/norm_inf_W.npy", norm_inf_W)
    np.save(output_dir + "/norm_two_W.npy", norm_two_W)

    with open(f"{output_dir}/compute_times.txt", "w") as file:
        for l in L:
            file.write(f"{l.ply=}\n")
        file.write(f"{Nvertices=}\n")
        file.write(f"{Tweights=}\n")
        file.write(f"{norm_inf_W=}\n")
        file.write(f"{norm_two_W=}\n")

    return Nvertices, Tweights, norm_inf_W, norm_two_W, L


# Nvertices, Tweights, norm_inf_W, norm_two_W, L = heat_laplacian_weights_compute("dumbbell")
Nvertices, Tweights, norm_inf_W, norm_two_W, L = load_heat_laplacian_weights_compute(
    "dumbbell"
)
# %%
N, T, W = Nvertices, Tweights, norm_inf_W

log_log_fit(N, T, Xlabel="N", Ylabel="T", title="")
x = N[:-1]
# y = [abs(W[0] - W[-1]), abs(W[1] - W[-1])]
y = [abs(w - W[-1]) for w in W[:-1]]
# g = [abs(W[0] - W[-1]) / N[0], abs(W[1] - W[-1]) / N[1]]
np.log(np.abs((W[2] - W[1]) / (W[1] - W[0])))
log_log_fit(x, y, Xlabel="N", Ylabel="W", title="")
log_log_fit(N, W, Xlabel="N", Ylabel="W", title="")
# %%
#
#
#
#
#
#
#
from src.python.half_edge_mesh import HalfEdgeMesh, HeatLaplacian
from src.python.utilities.misc_utils import round_to, log_log_fit
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

Nverts = [12, 42, 162, 642, 2562, 10242]

surfs = [f"unit_sphere_{N:05d}" for N in [162, 642, 2562]]


def load_heat_laplacian_weights_compute():
    output_dir = f"./output/heat_laplacian_weights_compute/unit_sphere"
    data_paths = [f"{output_dir}/{surf}" for surf in surfs]
    L = []
    for dpath in data_paths:
        with open(dpath + ".pickle", "rb") as f:
            L.append(pickle.load(f))
    Nvertices = np.array([l.num_vertices for l in L])
    Tweights = np.array([l.t_weights for l in L])
    norm_inf_W = np.array(
        [np.linalg.norm(np.ravel(l.weights.todense()), np.inf) for l in L]
    )
    norm_two_W = np.array([np.linalg.norm(np.ravel(l.weights.todense()), 2) for l in L])
    return Nvertices, Tweights, norm_inf_W, norm_two_W, L


def heat_laplacian_weights_compute():
    """
    Makes data for heat laplacian computations fig
    """
    output_dir = f"./output/heat_laplacian_weights_compute/unit_sphere"
    data_paths = [f"{output_dir}/{surf}" for surf in surfs]
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    os.system(f"mkdir -p {output_dir}")
    source_plys = [f"./data/ply/binary/{surf}.ply" for surf in surfs]

    M = [HalfEdgeMesh.from_half_edge_ply(_) for _ in source_plys]
    laplacian_kwargs = [
        {
            "mesh": m,
            "rtol": 1e-12,
            "atol": 1e-12,
        }
        for m in M
    ]
    L = [HeatLaplacian(**kwargs) for kwargs in laplacian_kwargs]

    for l, dpath in zip(L, data_paths):
        l.data_path = dpath

    for m, l, p in zip(M, L, source_plys):
        l.ply = p
        t = time()
        # l.weights = l.compute_weights_matrix()
        l.weights = l.compute_matrix()
        l.t_weights = time() - t
        # l.norm_inf_weights = np.linalg.norm(np.ravel(l.weights.todense()), np.inf)
        # l.norm_two_weights = np.linalg.norm(np.ravel(l.weights.todense()), 2)

        l.matrix = l.weights
        Y = m.xyz_array
        lapY = l.apply(Y)
        H = 0.5 * np.linalg.norm(lapY, axis=1)
        l.norm_inf_weights = np.linalg.norm(H, np.inf)
        l.norm_two_weights = np.linalg.norm(H, 2)

        print(p)
        save_npz(l.data_path + "_weights_matrix.npz", l.weights)
        with open(l.data_path + ".pickle", "wb") as f:
            pickle.dump(l, f)

    Nvertices = np.array([l.num_vertices for l in L])
    Tweights = np.array([l.t_weights for l in L])
    norm_inf_W = np.array(
        [np.linalg.norm(np.ravel(l.weights.todense()), np.inf) for l in L]
    )
    norm_two_W = np.array([np.linalg.norm(np.ravel(l.weights.todense()), 2) for l in L])
    np.save(output_dir + "/Nvertices.npy", Nvertices)
    np.save(output_dir + "/Tweights.npy", Tweights)
    np.save(output_dir + "/norm_inf_W.npy", norm_inf_W)
    np.save(output_dir + "/norm_two_W.npy", norm_two_W)

    with open(f"{output_dir}/compute_times.txt", "w") as file:
        for l in L:
            file.write(f"{l.ply=}\n")
        file.write(f"{Nvertices=}\n")
        file.write(f"{Tweights=}\n")
        file.write(f"{norm_inf_W=}\n")
        file.write(f"{norm_two_W=}\n")

    return Nvertices, Tweights, norm_inf_W, norm_two_W, L


# Nvertices, Tweights, norm_inf_W, norm_two_W, L = heat_laplacian_weights_compute()
Nvertices, Tweights, norm_inf_W, norm_two_W, L = load_heat_laplacian_weights_compute()
# %%
N, T, W = Nvertices, Tweights, norm_inf_W

log_log_fit(N, T, Xlabel="N", Ylabel="T", title="")
x = N[:-1]
# y = [abs(W[0] - W[-1]), abs(W[1] - W[-1])]
y = [abs(w - W[-1]) for w in W[:-1]]
# g = [abs(W[0] - W[-1]) / N[0], abs(W[1] - W[-1]) / N[1]]
np.log(np.abs((W[2] - W[1]) / (W[1] - W[0])))
log_log_fit(x, y, Xlabel="N", Ylabel="W", title="")
log_log_fit(N, W, Xlabel="N", Ylabel="W", title="")
# %%
#
#
#
#
#
#
from src.python.half_edge_mesh import HalfEdgeMesh, HeatLaplacian
from src.python.utilities.misc_utils import log_log_fit
import os
import pickle

# [12, 42, 162, 642, 2562, 10242]
surfs = [f"unit_sphere_{N:05d}" for N in [12, 42, 162, 642, 2562, 10242]]
output_dir = "./output/heat_laplacian_tests/unit_sphere_fast"


def unit_sphere_mean_curvature_normal_compute(surfs=surfs, output_dir=output_dir):

    # surfs = [f"unit_sphere_{N:05d}" for N in [12, 42, 162, 642]]
    # output_dir = "./output/heat_laplacian_weights_compute/unit_sphere_mean_curvature_normal_compute"
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    os.system(f"mkdir -p {output_dir}")
    M = []
    L = []
    for surf in surfs:
        data_path = f"{output_dir}/{surf}"
        ply = f"./data/ply/binary/{surf}.ply"
        m = HalfEdgeMesh.from_half_edge_ply(ply)
        laplacian_kwargs = {
            "mesh": m,
            "rtol": 1e-12,
            "atol": 1e-12,
            "data_path": data_path,
            "run_tests": True,
        }
        l = HeatLaplacian(**laplacian_kwargs)
        l.save()
        M.append(m)
        L.append(l)
    Nvertices = [l.num_vertices for l in L]
    T_compute_weights_matrix = [l.T_compute_weights_matrix for l in L]
    T_compute_matrix = [l.T_compute_matrix for l in L]
    T_apply = [l.T_apply for l in L]
    lapY_error_max = [l.lapY_error_max for l in L]
    lapY_error_ave = [l.lapY_error_ave for l in L]
    H = [l.H for l in L]
    H_max = [l.H_max for l in L]
    H_ave = [l.H_ave for l in L]
    return (
        Nvertices,
        T_compute_weights_matrix,
        T_compute_matrix,
        T_apply,
        lapY_error_max,
        lapY_error_ave,
        H,
        H_max,
        H_ave,
    )


def load_unit_sphere_mean_curvature_normal_compute(surfs=surfs, output_dir=output_dir):
    data_paths = [f"{output_dir}/{surf}" for surf in surfs]
    L = []
    for dpath in data_paths:
        with open(dpath + ".pickle", "rb") as f:
            L.append(pickle.load(f))
    Nvertices = [l.num_vertices for l in L]
    T_compute_weights_matrix = [l.T_compute_weights_matrix for l in L]
    T_compute_matrix = [l.T_compute_matrix for l in L]
    T_apply = [l.T_apply for l in L]
    lapY_error_max = [l.lapY_error_max for l in L]
    lapY_error_ave = [l.lapY_error_ave for l in L]
    H = [l.H for l in L]
    H_max = [l.H_max for l in L]
    H_ave = [l.H_ave for l in L]
    return (
        Nvertices,
        T_compute_weights_matrix,
        T_compute_matrix,
        T_apply,
        lapY_error_max,
        lapY_error_ave,
        H,
        H_max,
        H_ave,
    )


# %%
(
    Nvertices,
    T_compute_weights_matrix,
    T_compute_matrix,
    T_apply,
    lapY_error_max,
    lapY_error_ave,
    H,
    H_max,
    H_ave,
) = unit_sphere_mean_curvature_normal_compute()

# %%
(
    Nvertices,
    T_compute_weights_matrix,
    T_compute_matrix,
    T_apply,
    lapY_error_max,
    lapY_error_ave,
    H,
    H_max,
    H_ave,
) = load_unit_sphere_mean_curvature_normal_compute()

# %%
error_ave_kwargs = {
    "X": Nvertices,
    "Y": lapY_error_ave,
    "Xlabel": "N",
    "Ylabel": "ave|$H-H*$|",
    "title": "Average error",
}
log_log_fit(**error_ave_kwargs)
# %%
error_max_kwargs = {
    "X": Nvertices[:3],
    "Y": lapY_error_max[:3],
    "Xlabel": "N",
    "Ylabel": "max|$H-H*$|",
    "title": "Max error",
}
log_log_fit(**error_max_kwargs)
# %%
compute_time_kwargs = {
    "X": Nvertices[4:],
    "Y": T_compute_weights_matrix[4:],
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Compute time",
}

log_log_fit(**compute_time_kwargs)
