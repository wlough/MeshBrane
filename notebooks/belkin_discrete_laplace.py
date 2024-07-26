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
    xlabels = [
        r"$" + f"{c}" + r" \times " + r"10^{" + f"{p}" + r"}$"
        for c, p in zip(coeff, pow)
    ]
    return xlabels


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
        "#2ca02c",  # g
        "#1f77b4",  # b
        "#d62728",  # r
        "#ff7f0e",  # o
    ]
    markers = ["o", "*", "^", "s"]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
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


###############################################################
# Table 1: Normalized L2 error for planar [-1, 1]x[-1, 1] mesh.
###############################################################
# %%
############
# f(x)=x^2 #
############
parameter = np.array([0.0025, 0.01, 0.0225])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=x^2"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.220, 0.173, 0.197, 0.207])
error_belkin = np.array(
    [
        [0.450, 0.146, 0.040, 0.022],
        [0.126, 0.038, 0.010, 0.005],
        [0.069, 0.017, 0.004, 0.002],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
# %%
######################
# f(x, y)=exp(x + y) #
######################
parameter = np.array([0.0025, 0.01, 0.0225])
num_vertices = np.array([500, 2000, 8000, 16000])
fun = "f(x, y)=exp(x + y)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.198, 0.188, 0.190, 0.202])
error_belkin = np.array(
    [
        [0.875, 0.128, 0.055, 0.027],
        [0.189, 0.037, 0.022, 0.016],
        [0.099, 0.033, 0.027, 0.025],
    ]
)
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


###############################################################
# Table 2: Normalized L2 error for spherical meshes with
# uniform sampling.
###############################################################
# %%
############
# f(x)=x #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=x"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.058, 0.030, 0.015, 0.011])
error_belkin = np.array(
    [
        [0.606, 0.142, 0.034, 0.017],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
# %%
############
# f(x)=x^2 #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=x^2"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.171, 0.157, 0.158, 0.155])
error_belkin = np.array(
    [
        [0.488, 0.115, 0.013, 0.005],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
# %%
############
# f(x, y)=exp(x,y) #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=exp(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.124, 0.101, 0.102, 0.099])
error_belkin = np.array(
    [
        [0.613, 0.140, 0.028, 0.015],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)

###############################################################
# Table 3: Normalized Linf error for spherical meshes with
# uniform sampling.
###############################################################
# %%
############
# f(x)=x #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=x"
err_tex = (
    r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{\infty}}{||\Delta f||_{\infty}}"
)
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.229, 0.185, 0.083, 0.062])
error_belkin = np.array(
    [
        [2.097, 0.492, 0.081, 0.037],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
# %%
############
# f(x)=x^2 #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=x^2"
err_tex = (
    r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{\infty}}{||\Delta f||_{\infty}}"
)
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([1.147, 1.577, 1.478, 1.375])
error_belkin = np.array(
    [
        [2.915, 0.838, 0.062, 0.025],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
# %%
############
# f(x, y)=exp(x,y) #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=exp(x)"
err_tex = (
    r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{\infty}}{||\Delta f||_{\infty}}"
)
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.798, 0.887, 0.928, 0.849])
error_belkin = np.array(
    [
        [4.000, 0.873, 0.112, 0.054],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)


###############################################################
# Table 4: Normalized L2 error for the experiments on the
# sphere with noise.
###############################################################
# %%
############
# f(x)=x #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=x"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.398, 1.532, 3.015, 0.936])
error_belkin = np.array(
    [
        [0.599, 0.155, 0.051, 0.022],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
# %%
############
# f(x)=x^2 #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=x^2"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.267, 0.914, 1.840, 0.545])
error_belkin = np.array(
    [
        [0.484, 0.128, 0.028, 0.006],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
# %%
############
# f(x, y)=exp(x,y) #
############
parameter = np.array([0.01])
num_vertices = np.array([500, 2000, 8000, 16000])

fun = r"f(x)=exp(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} f-\Delta f||_{2}}{||\Delta f||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = np.array([0.308, 1.271, 2.631, 0.817])
error_belkin = np.array(
    [
        [0.612, 0.153, 0.043, 0.018],
    ]
)
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
# fig_subplots(**plot_kwargs)
fig_single(**plot_kwargs)
