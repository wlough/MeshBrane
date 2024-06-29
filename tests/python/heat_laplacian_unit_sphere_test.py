# don't mess with this one...

from src.python.half_edge_mesh import HalfEdgeMesh, HeatLaplacian
from src.python.figs import log_log_fit
import os
import pickle
import numpy as np


def unit_sphere_mean_curvature_normal_compute(surfs=None, output_dir=None, rtol=1e-6, atol=1e-6):
    if surfs is None:
        surfs = [f"unit_sphere_{N:05d}" for N in [12, 42, 162, 642, 2562, 10242]]
    if output_dir is None:
        output_dir = "./output/heat_laplacian_tests/unit_sphere_test"
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    os.system(f"mkdir -p {output_dir}")
    M = []
    L = []
    for surf in surfs:
        data_path = f"{output_dir}/{surf}"
        ply = f"./data/ply/binary/{surf}.ply"
        m = HalfEdgeMesh.from_half_edge_ply(ply)
        laplacian_kwargs = {"mesh": m, "rtol": rtol, "atol": atol, "data_path": data_path, "run_tests": True}
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
    sparsity = [l.sparsity for l in L]
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
        sparsity,
        L,
    )


def load_unit_sphere_mean_curvature_normal_compute(surfs=None, output_dir=None):
    if surfs is None:
        surfs = [f"unit_sphere_{N:05d}" for N in [12, 42, 162, 642, 2562, 10242]]
    if output_dir is None:
        output_dir = "./output/heat_laplacian_tests/unit_sphere_1em6_precision"

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
    sparsity = [l.sparsity for l in L]
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
        sparsity,
        L,
    )


# (
#     Nvertices,
#     T_compute_weights_matrix,
#     T_compute_matrix,
#     T_apply,
#     lapY_error_max,
#     lapY_error_ave,
#     H,
#     H_max,
#     H_ave,
#     sparsity,
#     L,
# ) = unit_sphere_mean_curvature_normal_compute(
#     surfs=[f"unit_sphere_{N:05d}" for N in [12, 42, 162, 642, 2562, 10242]],
#     output_dir="./output/heat_laplacian_tests/unit_sphere_1em16_precision",
#     rtol=1e-16,
#     atol=1e-16,
# )

# %%
################################################
# 1e-12 precision
################################################
output_dir = "./output/heat_laplacian_tests/unit_sphere_1em12_precision"
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
    sparsity,
    L,
) = load_unit_sphere_mean_curvature_normal_compute(output_dir=output_dir)


# %%
error_ave_kwargs = {
    "X": Nvertices[:3],
    "Y": lapY_error_ave[:3],
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
weights_compute_time_kwargs = {
    "X": Nvertices[2:],
    "Y": T_compute_weights_matrix[2:],
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Weights compute time",
}
log_log_fit(**weights_compute_time_kwargs)
# %%
matrix_compute_time_kwargs = {
    "X": Nvertices[2:],
    "Y": T_compute_matrix[2:],
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Laplacian compute time",
}
log_log_fit(**matrix_compute_time_kwargs)
# %%
sparsity_kwargs = {
    "X": Nvertices[3:],
    "Y": sparsity[3:],
    "Xlabel": "N",
    "Ylabel": "S",
    "title": "Laplacian sparsity",
}
log_log_fit(**sparsity_kwargs)

# %%
################################################
# 1e-6 precision
################################################
output_dir = "./output/heat_laplacian_tests/unit_sphere_1em6_precision"
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
    sparsity,
    L,
) = load_unit_sphere_mean_curvature_normal_compute(output_dir=output_dir)

# %%
error_ave_kwargs = {
    "X": Nvertices[:3],
    "Y": lapY_error_ave[:3],
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
weights_compute_time_kwargs = {
    "X": Nvertices[4:],
    "Y": T_compute_weights_matrix[4:],
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Weights compute time",
}
log_log_fit(**weights_compute_time_kwargs)
# %%
matrix_compute_time_kwargs = {
    "X": Nvertices[3:],
    "Y": T_compute_matrix[3:],
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Laplacian compute time",
}
log_log_fit(**matrix_compute_time_kwargs)
# %%
sparsity_kwargs = {
    "X": Nvertices[3:],
    "Y": sparsity[3:],
    "Xlabel": "N",
    "Ylabel": "S",
    "title": "Laplacian sparsity",
}
log_log_fit(**sparsity_kwargs)

# %%
################################################
# 1e-16 precision
################################################
output_dir = "./output/heat_laplacian_tests/unit_sphere_1em16_precision"
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
    sparsity,
    L,
) = load_unit_sphere_mean_curvature_normal_compute(output_dir=output_dir)

# %%
error_ave_kwargs = {
    "X": Nvertices[:],
    "Y": lapY_error_ave[:],
    "Xlabel": "N",
    "Ylabel": "ave|$H-H*$|",
    "title": "Average error",
}
log_log_fit(**error_ave_kwargs)
# %%
error_max_kwargs = {
    "X": Nvertices,
    "Y": lapY_error_max,
    "Xlabel": "N",
    "Ylabel": "max|$H-H*$|",
    "title": "Max error",
}
log_log_fit(**error_max_kwargs)
# %%
weights_compute_time_kwargs = {
    "X": Nvertices,
    "Y": T_compute_weights_matrix,
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Weights compute time",
}
log_log_fit(**weights_compute_time_kwargs)
# %%
matrix_compute_time_kwargs = {
    "X": Nvertices,
    "Y": T_compute_matrix,
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Laplacian compute time",
}
log_log_fit(**matrix_compute_time_kwargs)
# %%
sparsity_kwargs = {
    "X": Nvertices,
    "Y": sparsity,
    "Xlabel": "N",
    "Ylabel": "S",
    "title": "Laplacian sparsity",
}
log_log_fit(**sparsity_kwargs)


# %%
