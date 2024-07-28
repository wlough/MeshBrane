from src.python.half_edge_mesh import HalfEdgeTestSphere
import os
import pickle
import numpy as np

np.random.normal(0, 0.01, 3)


# from src.python.mesh_viewer import MeshViewer
def make_sphere_data(overwrite=False):
    s_list = [0.0025, 0.01, 0.025]
    u = 10
    # s_list = [u**-p for p in range(1, 6)]
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162, 642]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    output_dir = "./output/sphere_tests3"
    # if os.path.exists(output_dir) and overwrite:
    #     os.system(f"rm -r {output_dir}")
    # else:
    #     raise ValueError("Ahhhhhh")
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError("Ahhhhhh")
    os.system(f"mkdir -p {output_dir}")
    M = []
    for _ in range(len(surfs)):
        surf = surfs[_]
        print(f"running tests for {surf}")
        data_path = f"{output_dir}/{surf}"
        ply = f"./data/ply/binary/{surf}.ply"
        m = HalfEdgeTestSphere.from_half_edge_ply(ply, 1.0)
        # return m
        # m.run_unit_sphere_mean_curvature_normal_tests(timelike_param)
        m.run_belkin_laplacian_mcvec_fixed_param_test(s_list)
        m.run_belkin_laplacian_mcvec_average_face_area_test()
        m.run_cotan_laplacian_mcvec_test()
        m.run_guckenberger_laplacian_mcvec_test()
        m.save(data_path)
        M.append(m)
    return M


def load_spheres():
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162, 642]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    output_dir = "./output/sphere_tests3"
    M = []
    for surf in surfs:
        data_path = f"{output_dir}/{surf}"
        with open(data_path + ".pickle", "rb") as f:
            M.append(pickle.load(f))
    return M


def get_test_data(with_M=False):
    M = load_spheres()
    # belkin_laplacian_mcvec_fixed_param_results
    # cotan_laplacian_mcvec_results
    timelike_param = np.array([m.belkin_laplacian_mcvec_fixed_param_results["s"] for m in M][0])
    mcvec_actual = [m.mcvec_actual for m in M]
    mcvec_cotan = [m.cotan_laplacian_mcvec_results["mcvec"] for m in M]
    mcvec_cotan_L2error = np.array([m.cotan_laplacian_mcvec_results["L2error"] for m in M])
    mcvec_cotan_Lifntyerror = np.array([m.cotan_laplacian_mcvec_results["Lifntyerror"] for m in M])

    # _mcvec_belkin = [m.mcvec_belkin for m in M]
    # _mcvec_belkin_L2error = [m.mcvec_belkin_L2error for m in M]
    # _mcvec_belkin_Lifntyerror = [m.mcvec_belkin_Lifntyerror for m in M]
    #
    # num_M = len(M)
    num_timelike = len(timelike_param)
    mcvec_belkin = [
        [m.belkin_laplacian_mcvec_fixed_param_results["mcvec"][_] for m in M] for _ in range(num_timelike)
    ]
    mcvec_belkin_L2error = np.array(
        [[m.belkin_laplacian_mcvec_fixed_param_results["L2error"][_] for m in M] for _ in range(num_timelike)]
    )
    mcvec_belkin_Lifntyerror = np.array(
        [
            [m.belkin_laplacian_mcvec_fixed_param_results["Lifntyerror"][_] for m in M]
            for _ in range(num_timelike)
        ]
    )
    num_vertices = np.array([m.num_vertices for m in M])
    if with_M:
        return (
            timelike_param,
            mcvec_actual,
            mcvec_cotan,
            mcvec_cotan_L2error,
            mcvec_cotan_Lifntyerror,
            mcvec_belkin,
            mcvec_belkin_L2error,
            mcvec_belkin_Lifntyerror,
            num_vertices,
            M,
        )
    else:
        return (
            timelike_param,
            mcvec_actual,
            mcvec_cotan,
            mcvec_cotan_L2error,
            mcvec_cotan_Lifntyerror,
            mcvec_belkin,
            mcvec_belkin_L2error,
            mcvec_belkin_Lifntyerror,
            num_vertices,
        )


# M = make_sphere_data(overwrite=True)
(
    timelike_param,
    mcvec_actual,
    mcvec_cotan,
    mcvec_cotan_L2error,
    mcvec_cotan_Lifntyerror,
    mcvec_belkin,
    mcvec_belkin_L2error,
    mcvec_belkin_Lifntyerror,
    num_vertices,
    M,
) = get_test_data(with_M=True)
# %%
from src.python.mesh_viewer import MeshViewer

num = 3
timelike_num = 2
# M = Mall[:]
num_vertices = [m.num_vertices for m in M]
vfdat = [[m.xyz_array, -vec] for m, vec in zip(M, mcvec_belkin[timelike_num])]
m = M[num]
vfdat = [m.xyz_array, -m.mcvec_actual + mcvec_belkin[timelike_num][num]]
mv = MeshViewer(*m.data_lists, vector_field_data=[vfdat])
mv.plot()
type(m)
# %%

# err = mcvec_belkin_L2error[timelike_num][1:]
# thing = [m.num_vertices for m in Mall][1:]
#
# error_ave_kwargs = {
#     "X": thing,
#     "Y": err,
#     "Xlabel": "N",
#     "Ylabel": "ave|$H-H*$|",
#     "title": "Average error",
# }
# log_log_fit(**error_ave_kwargs)
############################################
############################################
############################################
# %%
import numpy as np
import matplotlib.pyplot as plt
from src.python.utilities import round_to, log_log_fit
import itertools


def ten_pow(X, decimals=3):
    x = np.abs(X)
    pow = np.array([int(np.log10(_)) for _ in x])
    coeff = [xx / 10.0**p for xx, p in zip(x, pow)]
    for _ in range(len(coeff)):
        if coeff[_] < 1:
            coeff[_] *= 10
            pow[_] -= 1
        if X[_] < 0:
            coeff[_] *= -1
        if int(coeff[_]) == coeff[_]:
            coeff[_] = int(coeff[_])
        else:
            coeff[_] = np.round(coeff[_], decimals=decimals)
    ten_pow_tex = [r"$10^{" + f"{p}" + r"}$" for p in pow]
    return coeff, pow, ten_pow_tex


def to_scinotation_tex(X, decimals=3):
    x = np.abs(X)
    pow = np.array([int(np.log10(_)) for _ in x])
    coeff = [xx / 10.0**p for xx, p in zip(x, pow)]
    for _ in range(len(coeff)):
        if coeff[_] < 1:
            coeff[_] *= 10
            pow[_] -= 1
        if X[_] < 0:
            coeff[_] *= -1
        if int(coeff[_]) == coeff[_]:
            coeff[_] = int(coeff[_])
        else:
            coeff[_] = np.round(coeff[_], decimals=decimals)
    xlabels = [r"$" + f"{c}" + r" \times " + r"10^{" + f"{p}" + r"}$" for c, p in zip(coeff, pow)]
    return xlabels


# Define extended lists of possible colors, markers, and line styles
colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
markers = ["o", "s", "^", "D", "v", "p", "*", "h", "H", "+", "x", "d", "|", "_"]
line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (5, 1)), (0, (3, 5, 1, 5))]

# Create a list of all possible combinations
combinations = list(itertools.product(colors, markers, line_styles))


# Function to get unique sequence of length n
def get_unique_combinations(n):
    if n > len(combinations):
        raise ValueError("Requested more unique combinations than available")
    return combinations[:n]


def fig_single(
    num_vertices,
    error_belkin,
    error_cot,
    error_guckenberger,
    parameter_belkin,
    parameter_guckenberger,
    suptitle="",
    Xlabel="",
    Ylabel="",
    legendlabel="",
):
    pream = ""
    # colors = ["#EE6666", "#3388BB", "#9988DD", "#EECC55", "#88BB44", "#FFBBBB"]
    # colors = [
    #     "#1f77b4", #b
    #     "#ff7f0e", #o
    #     "#2ca02c", #g
    #     "#d62728", #r
    # ]
    colors = 2 * [
        "#2ca02c",  # g
        "#1f77b4",  # b
        "#d62728",  # r
        "#ff7f0e",  # o
    ]
    markers = 2 * ["o", "*", "^", "s"]
    linestyles = 2 * ["solid", "dashed", "dashdot", "dotted"]
    textsize = 14
    plt.rcParams.update(
        {
            "text.latex.preamble": pream,
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            "xtick.labeltop": False,
            "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "text.usetex": False,
            "figure.titlesize": 18,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            "lines.linewidth": 3,
            "lines.markersize": 14,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.loc": "best",
        }
    )
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(r"$" + f"{suptitle}" + r"$")
    ax = fig.add_subplot(1, 1, 1)

    ymin = np.inf
    ymax = -np.inf
    errmin = np.inf
    errmax = -np.inf
    n_fit = 0

    error = error_belkin
    parameter = parameter_belkin
    for _ in range(len(error)):
        param = parameter[_]
        err = error[_]
        color = colors[_]
        marker = markers[_]
        linestyle = linestyles[_]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]
        fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
        ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
        ax.plot(x, y, marker, color=color, label=r"$s_B=" + f"{param}" + r"$")

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])
        n_fit += 1

    err = error_guckenberger
    parameter = parameter_guckenberger
    # for _ in range(n_fit+len(error)):

    color = colors[n_fit]
    marker = markers[n_fit]
    linestyle = linestyles[n_fit]
    fit_dict = log_log_fit(num_vertices, err)
    m = fit_dict["m"]
    y_fit = fit_dict["F"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]
    fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
    sample_label = r" $s_G=("
    for param in parameter:
        sample_label += f"{round_to(param, n=3)}, "
    sample_label = sample_label[:-2]
    sample_label += r")$"
    ax.plot(x, y, marker, color=color, label=sample_label)

    ymin = np.min([ymin, *y])
    ymax = np.max([ymax, *y])
    errmin = np.min([errmin, *err])
    errmax = np.max([errmax, *err])
    n_fit += 1

    ax.set_xlabel(r"$\text{Vertex count}\quad " + Xlabel + r"$")
    ax.set_xticks(x)
    ax.set_xticklabels(to_scinotation_tex(num_vertices))
    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")

    err = error_cot
    color = colors[n_fit]
    marker = markers[n_fit]
    linestyle = linestyles[n_fit]
    fit_dict = log_log_fit(num_vertices, err)
    m = fit_dict["m"]
    y_fit = fit_dict["F"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]

    ymin = np.min([ymin, *y])
    ymax = np.max([ymax, *y])
    errmin = np.min([errmin, *err])
    errmax = np.max([errmax, *err])
    # errminmax = np.linspace(errmin, errmax, 4)
    yminmax = np.linspace(ymin, ymax, 4)
    errminmax = np.exp(yminmax)
    ax.set_yticks(yminmax)
    ax.set_yticklabels(to_scinotation_tex(errminmax, decimals=2))
    # ax.set_yticks([ymin, ymax])
    # ax.set_yticklabels(to_scinotation_tex([errmin, errmax]))

    fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"

    ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
    ax.plot(x, y, marker, color=color, label="cotan")
    n_fit += 1

    ax.legend()
    plt.tight_layout()
    plt.show()


(
    timelike_param,
    mcvec_actual,
    mcvec_cotan,
    mcvec_cotan_L2error,
    mcvec_cotan_Lifntyerror,
    mcvec_belkin,
    mcvec_belkin_L2error,
    mcvec_belkin_Lifntyerror,
    N_vertices,
    M,
) = get_test_data(with_M=True)
# %%
###############################################################
# Normalized L2 error for spherical mesh.
###############################################################

fun = r"{\bf{r}}(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{2}}{||\Delta {\bf{r}}||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
n0 = 3
num_vertices = N_vertices[n0:]

error_belkin = mcvec_belkin_L2error[:, n0:]
error_cot = np.array([*mcvec_cotan_L2error[n0:]])
error_guckenberger = [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M[n0:]]

parameter_belkin = timelike_param
parameter_guckenberger = [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M[n0:]]

# error_heat = np.array([*error_belkin, error_guckenberger])

# error_belkin[-1:] = [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M[n0:]]
# error_belkin = [*_error_belkin, [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M]]
plot_kwargs = {
    "num_vertices": num_vertices,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "error_guckenberger": error_guckenberger,
    "parameter_belkin": parameter_belkin,
    "parameter_guckenberger": parameter_guckenberger,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
fig_single(**plot_kwargs)

# %%
###############################################################
# Normalized Linf error for spherical mesh.
###############################################################
parameter = timelike_param
num_vertices = N_vertices[3:]

fun = r"{\bf{r}}(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{\infty}}{||\Delta {\bf{r}}||_{\infty}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = mcvec_cotan_Lifntyerror[3:]
error_belkin = mcvec_belkin_Lifntyerror[:, 3:]
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
fig_single(**plot_kwargs)
