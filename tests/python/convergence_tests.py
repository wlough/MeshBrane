from src.python.half_edge_mesh import (
    HalfEdgeMesh,
    HeatLaplacian,
    MeyerHeatLaplacian,
    FixedTimelikeParamHeatLaplacian,
)
from src.python.mesh_viewer import MeshViewer
from src.python.figs import log_log_fit, get_plotsize
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def unit_sphere_mean_curvature_normal_compute(
    heatLaplacian, surfs=None, output_dir=None, rtol=1e-12, atol=1e-12
):
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
        l = heatLaplacian(**laplacian_kwargs)
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
    return {
        "Nvertices": Nvertices,
        "T_compute_weights_matrix": T_compute_weights_matrix,
        "T_compute_matrix": T_compute_matrix,
        "T_apply": T_apply,
        "lapY_error_max": lapY_error_max,
        "lapY_error_ave": lapY_error_ave,
        "H": H,
        "H_max": H_max,
        "H_ave": H_ave,
        "sparsity": sparsity,
        "L": L,
    }


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
    return {
        "Nvertices": Nvertices,
        "T_compute_weights_matrix": T_compute_weights_matrix,
        "T_compute_matrix": T_compute_matrix,
        "T_apply": T_apply,
        "lapY_error_max": lapY_error_max,
        "lapY_error_ave": lapY_error_ave,
        "H": H,
        "H_max": H_max,
        "H_ave": H_ave,
        "sparsity": sparsity,
        "L": L,
    }


# %%
output_dir = "./output/heat_laplacian_tests"
output_dir_1em6 = f"{output_dir}/unit_sphere_global_param"
surfs = [f"unit_sphere_{N:05d}" for N in [12, 42, 162, 642, 2562, 10242]][:4]
data = unit_sphere_mean_curvature_normal_compute(
    FixedTimelikeParamHeatLaplacian, surfs=surfs, output_dir=output_dir, rtol=1e-12, atol=1e-12
)
L = data["L"]
M = [l.mesh for l in data["L"]]
self = l = L[-1]
l.H_max
m = M[0]
len(m.v_origin_H.keys())
[m.h_prev_h(_) for _ in list(m.v_origin_H.keys())[:5]]


for h in m.v_origin_H.keys():
    # if not m.is_delaunay(h):
    #     m.edge_flip(h)
    h_next = m.h_next_h(h)
    if h == h_next:
        print(h)
[m.is_delaunay(h) for h in m.v_origin_H.keys()]
[np.std([m.area_f(f) for f in m.h_bound_F.keys()]) for m in M]


A1 = self.meyercell_area(3)
A2 = compute_weights_row(self, 3)

r1, i1 = (np.array(_) for _ in l.compute_weights_row(3))
r2, i2 = (np.array(_) for _ in l.compute_row(3))
r1[0] -= sum(r1)
r1

r2 - r1
# %%
data.keys()
Nvertices = data["Nvertices"]
lapY_error_ave = data["lapY_error_ave"]
lapY_error_max = data["lapY_error_max"]
T_compute_weights_matrix = data["T_compute_weights_matrix"]
T_compute_matrix = data["T_compute_matrix"]
sparsity = data["sparsity"]

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
    "X": Nvertices[:],
    "Y": lapY_error_max[:],
    "Xlabel": "N",
    "Ylabel": "max|$H-H*$|",
    "title": "Max error",
}
log_log_fit(**error_max_kwargs)
# %%
weights_compute_time_kwargs = {
    "X": Nvertices[:],
    "Y": T_compute_weights_matrix[:],
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Weights compute time",
}
log_log_fit(**weights_compute_time_kwargs)
# %%
matrix_compute_time_kwargs = {
    "X": Nvertices[:],
    "Y": T_compute_matrix[:],
    "Xlabel": "N",
    "Ylabel": "T",
    "title": "Laplacian compute time",
}
log_log_fit(**matrix_compute_time_kwargs)
# %%
sparsity_kwargs = {
    "X": Nvertices[:],
    "Y": sparsity[:],
    "Xlabel": "N",
    "Ylabel": "S",
    "title": "Laplacian sparsity",
}
log_log_fit(**sparsity_kwargs)


# %%
################################################
# precision
################################################
output_dir = "./output/heat_laplacian_tests"
output_dir_1em6 = f"{output_dir}/unit_sphere_1em6_precision"
output_dir_1em12 = f"{output_dir}/unit_sphere_1em12_precision"
output_dir_1em16 = f"{output_dir}/unit_sphere_1em16_precision"


data = dict()
data[1e-6] = load_unit_sphere_mean_curvature_normal_compute(output_dir=output_dir_1em6)
data[1e-12] = load_unit_sphere_mean_curvature_normal_compute(output_dir=output_dir_1em12)
data[1e-16] = load_unit_sphere_mean_curvature_normal_compute(output_dir=output_dir_1em16)

# M = np.array([dat["lapY_error_max"][-2] for prec, dat in data.items()])
M = [l.mesh for l in data[1e-16]["L"]]
H_max = np.array([l.H_max for l in data[1e-16]["L"]])
N = [l.num_vertices for l in data[1e-16]["L"]]

MV = [MeshViewer(*m.data_lists) for m in M]
A_faces = np.array([m.total_area_of_faces() for m in M])
A_barcells = np.array([m.total_area_of_dual_barcells() for m in M])
A_vorcells = np.array([m.total_area_of_dual_vorcells() for m in M])
A_meyercells = np.array([m.total_area_of_dual_meyercells() for m in M])
V = [m.xyz_array[500:600] for m in M[3:]]
np.linalg.norm(V[0].ravel() - V[-1].ravel())
len(M[3].xyz_array)
# %%


# MV[5].plot()
# %%
fig_path = f"{output_dir}/fig_00000.png"
fig_cols = 2
fig_frac = 1 / 5
plotsize = 2 * [get_plotsize(fig_cols, fig_frac)]
fontsize = 12
linewidth = 1
lef, rig, bot, top = 0.1, 0.975, 0.1, 0.975

fig = plt.figure(figsize=plotsize)

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$a_2$", fontsize=fontsize, labelpad=-5)
ax.set_ylabel("$a_0$", fontsize=fontsize, labelpad=-3.2)
# ax.set_xlabel('$a_2$', fontsize=fontsize, labelpad=-3.2)
# ax.set_ylabel('$a_0$', fontsize=fontsize, labelpad=-3.2)
ax.plot(A2_orb, A0_orb, linewidth=linewidth, color=c)

ax.set_box_aspect(aspect=1)
# x_ticks = ax.get_xticks()[1:-1]
# x_lbls = [f'{round_to(tic)}' for tic in x_ticks]
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_lbls, fontsize=fontsize)
# y_ticks = ax.get_yticks()[1:-1]
# y_lbls = [f'{round_to(tic)}' for tic in y_ticks]
# ax.set_yticks(y_ticks)
# ax.set_yticklabels(y_lbls, fontsize=fontsize)
ax.set_xticks([])
ax.set_yticks([])

fig.subplots_adjust(left=lef, right=rig, bottom=bot, top=top)
plt.show()

# if save:
fig.savefig(fig_path, facecolor="white", transparent=False, dpi=1800)
plt.close()
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


def a0_a2_orbit_plot(n_fil, c_num, file_name, save=False):
    out_path = f"./Figure2/temp_images/{file_name}"
    fig_cols = 2
    fig_frac = 1 / 5
    plotsize = 2 * [get_plotsize(fig_cols, fig_frac)]
    fontsize = 12
    data_dir = f"./Figure2/phase_data_0000"
    linewidth = 1
    lef, rig, bot, top = 0.1, 0.975, 0.1, 0.975

    n_fils = [_ for _ in range(50)]
    _nt_start, _dnt = a0_a2_orbit_range(n_fils)
    nt_start, dnt = _nt_start[n_fil], _dnt[n_fil]

    input_data = load_input(data_dir)
    beta_perp_list = input_data["beta_perp_list"]
    beta_perp = beta_perp_list[n_fil]
    dt = input_data["dt_sim"]
    T = dnt * dt
    s0 = input_data["s0"]
    _c = get_cmap(2, "coolwarm")
    c = _c(c_num)

    A0_orb = load_data("W2_spec_coeffs", n_fil, data_dir)[nt_start : nt_start + dnt, 0]
    A2_orb = load_data("W2_spec_coeffs", n_fil, data_dir)[nt_start : nt_start + dnt, 2]

    fig = plt.figure(figsize=plotsize)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$a_2$", fontsize=fontsize, labelpad=-5)
    ax.set_ylabel("$a_0$", fontsize=fontsize, labelpad=-3.2)
    # ax.set_xlabel('$a_2$', fontsize=fontsize, labelpad=-3.2)
    # ax.set_ylabel('$a_0$', fontsize=fontsize, labelpad=-3.2)
    ax.plot(A2_orb, A0_orb, linewidth=linewidth, color=c)

    ax.set_box_aspect(aspect=1)
    # x_ticks = ax.get_xticks()[1:-1]
    # x_lbls = [f'{round_to(tic)}' for tic in x_ticks]
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_lbls, fontsize=fontsize)
    # y_ticks = ax.get_yticks()[1:-1]
    # y_lbls = [f'{round_to(tic)}' for tic in y_ticks]
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels(y_lbls, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(left=lef, right=rig, bottom=bot, top=top)
    plt.show()

    if save:
        fig.savefig(out_path, facecolor="white", transparent=False, dpi=1800)
    plt.close()
