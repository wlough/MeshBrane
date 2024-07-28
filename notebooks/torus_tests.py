from src.python.half_edge_mesh import HalfEdgeMesh
import os
import pickle
import numpy as np


def make_torus_data(overwrite=False):
    timelike_param = [0.0025, 0.00025, 0.000025]
    Nverts = [192, 768, 3072, 12288, 49152, 196608]
    # Nverts = [192, 768, 3072, 12288]
    surfs = [f"torus_{N:06d}" for N in Nverts]
    output_dir = "./output/torus_tests"
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
        ply = f"./data/ply/binary/{surf}_he.ply"
        m = HalfEdgeMesh.from_half_edge_ply(ply)
        m.run_unit_torus_mean_curvature_normal_tests(timelike_param)
        m.save(data_path)
        M.append(m)
    return M


def load_tori():
    Nverts = [192, 768, 3072, 12288, 49152, 196608]
    Nverts = [192, 768, 3072, 12288, 49152]
    surfs = [f"torus_{N:06d}" for N in Nverts]
    output_dir = "./output/torus_tests"
    M = []
    for surf in surfs:
        data_path = f"{output_dir}/{surf}"
        with open(data_path + ".pickle", "rb") as f:
            M.append(pickle.load(f))
    return M


def get_test_data():
    M = load_tori()
    timelike_param = np.array([m.timelike_param for m in M][0])
    mcvec_actual = [m.mcvec_actual for m in M]
    mcvec_cotan = [m.mcvec_cotan for m in M]
    mcvec_cotan_L2error = np.array([m.mcvec_cotan_L2error for m in M])
    mcvec_cotan_Lifntyerror = np.array([m.mcvec_cotan_Lifntyerror for m in M])

    # _mcvec_belkin = [m.mcvec_belkin for m in M]
    # _mcvec_belkin_L2error = [m.mcvec_belkin_L2error for m in M]
    # _mcvec_belkin_Lifntyerror = [m.mcvec_belkin_Lifntyerror for m in M]
    #
    # num_M = len(M)
    num_timelike = len(timelike_param)
    mcvec_belkin = [[m.mcvec_belkin[_] for m in M] for _ in range(num_timelike)]
    mcvec_belkin_L2error = np.array([[m.mcvec_belkin_L2error[_] for m in M] for _ in range(num_timelike)])
    mcvec_belkin_Lifntyerror = np.array(
        [[m.mcvec_belkin_Lifntyerror[_] for m in M] for _ in range(num_timelike)]
    )
    num_vertices = np.array([m.num_vertices for m in M])
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


# M = make_torus_data(overwrite=True)


M = load_tori()
M[0].mcvec_belkin_L2error
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
) = get_test_data()
[m.average_face_area() for m in M]
# %%
from src.python.mesh_viewer import MeshViewer

num = 2
timelike_num = 1

num_vertices = [m.num_vertices for m in M]

# mcvec_belkin = [[m.xyz_array, -vec] for m, vec in zip(M, mcvec_belkin[timelike_num])]
m = M[num]

vfdat = [m.xyz_array, -0.1 * m.mcvec_cotan]
vfdat = [m.xyz_array, -0.1 * m.mcvec_actual]
vfdat = [m.xyz_array, -0.1 * mcvec_belkin[timelike_num][num]]


mv = MeshViewer(*m.data_lists, vector_field_data=[vfdat])
mv.plot()
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
m = M[1]
a, b = 1.0, 1.0 / 3.0
V = m.xyz_array
X, Y, Z = V.T
Rho = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)
Psi = np.arctan2(Z, Rho - a)
x = (a + b * np.cos(Psi)) * np.cos(Phi)
y = (a + b * np.cos(Psi)) * np.sin(Phi)
z = b * np.sin(Psi)
X - x
Y - y
Z - z
# %%
import numpy as np
import matplotlib.pyplot as plt
from src.python.utilities import round_to, log_log_fit


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


def fig_subplots(
    num_vertices,
    parameter,
    error_belkin,
    error_cot,
    suptitle="",
    Xlabel="",
    Ylabel="",
    legendlabel="",
):
    pream = ""

    plt.rcParams.update(
        {
            "text.latex.preamble": pream,
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": 12.0,
            "xtick.labeltop": False,
            "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": 12.0,
            "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "text.usetex": False,
            "figure.titlesize": 18,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "lines.linewidth": 3,
            "lines.markersize": 10,
            "legend.frameon": False,
            # "legend.loc": "best",
        }
    )
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(r"$" + f"{suptitle}" + r"$")
    axs = [fig.add_subplot(2, 2, _ + 1) for _ in range(4)]

    for _ in range(3):
        param = parameter[_]
        err = error_belkin[_]
        ax = axs[_]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]
        fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
        ax.plot(
            x,
            y_fit,
            label=fit_label,
        )
        ax.plot(x, y, "*")
        ax.set_title(r"$s=" + f"{param}" + r"$")
        ax.set_xlabel(r"$\text{Vertex count}\quad " + Xlabel + r"$")
        ax.set_xticks(x)
        ax.set_xticklabels(to_scinotation_tex(num_vertices))

        ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")
        ax.set_yticks(y)
        ax.set_yticklabels(to_scinotation_tex(err))
        ax.legend()

    err = error_cot
    ax = axs[-1]
    fit_dict = log_log_fit(num_vertices, err)
    m = fit_dict["m"]
    y_fit = fit_dict["F"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]
    title = "cotan"
    # fit_label = (
    #     f"${Ylabel}=O\\left({Xlabel}" + "^{" + f"{round_to(m, n=3)}" + "}\\right)$"
    # )
    fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    ax.plot(
        x,
        y_fit,
        label=fit_label,
    )
    ax.plot(x, y, "*")
    ax.set_title(title)
    ax.set_xlabel(r"$\text{Vertex count}\quad " + Xlabel + r"$")
    ax.set_xticks(x)
    ax.set_xticklabels(to_scinotation_tex(num_vertices))
    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")
    ax.set_yticks(y)
    ax.set_yticklabels(to_scinotation_tex(err))
    ax.legend()

    plt.tight_layout()
    plt.show()


def fig_single(
    num_vertices,
    parameter,
    error_belkin,
    error_cot,
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
    colors = [
        "#2ca02c",
        "#1f77b4",
        "#d62728",
        "#ff7f0e",
        "#88BB44",
        "#FFBBBB",
    ]  # g  # b  # r  # o
    markers = ["o", "*", "^", "s", "x", "o"]
    linestyles = ["solid", "dashed", "dashdot", "dotted", "dotted", "dotted"]
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
    for _ in range(len(error_belkin)):
        param = parameter[_]
        err = error_belkin[_]
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
        ax.plot(x, y, marker, color=color, label=r"$s=" + f"{param}" + r"$")

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])

    ax.set_xlabel(r"$\text{Vertex count}\quad " + Xlabel + r"$")
    ax.set_xticks(x)
    ax.set_xticklabels(to_scinotation_tex(num_vertices))

    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")

    err = error_cot
    color = colors[-1]
    marker = markers[-1]
    linestyle = linestyles[-1]
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
) = get_test_data()
# %%
###############################################################
# Normalized L2 error for spherical mesh.
###############################################################
parameter = timelike_param
num_vertices = N_vertices[1:]

fun = r"{\bf{r}}(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{2}}{||\Delta {\bf{r}}||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = mcvec_cotan_L2error[1:]
error_belkin = mcvec_belkin_L2error[:, 1:]
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
# plt.plot(np.log(num_vertices), np.log(error_cot))
# %%
###############################################################
# Normalized Linf error for spherical mesh.
###############################################################
parameter = timelike_param
num_vertices = N_vertices[1:]

fun = r"{\bf{r}}(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{\infty}}{||\Delta {\bf{r}}||_{\infty}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = mcvec_cotan_Lifntyerror[1:]
error_belkin = mcvec_belkin_Lifntyerror[:, 1:]
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
