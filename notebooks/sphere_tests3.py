from src.python.mesh_viewer import MeshViewer

# %%
from src.python.half_edge_mesh import HalfEdgeTestSphere
from matplotlib import colormaps as plt_cmap
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.python.utilities import round_to, log_log_fit
import itertools

# Define extended lists of possible colors, markers, and line styles
colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
markers = ["o", "s", "^", "D", "v", "p", "*", "h", "H", "+", "x", "d", "|", "_"]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (5, 1)), (0, (3, 5, 1, 5))]

# Create a list of all possible combinations
combinations = list(itertools.product(colors, markers, linestyles))


# Function to get unique sequence of length n
def get_unique_combinations(n):
    if n > len(combinations):
        raise ValueError("Requested more unique combinations than available")
    return combinations[:n]


def make_sphere_data(overwrite=False):
    # s_list = [0.0025, 0.01, 0.025]
    u = 10
    s_list = [u**-p for p in range(1, 6)]
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


def make_noisy_sphere_data(overwrite=False, loc=0, scale=0.01):
    # s_list = [0.0025, 0.01, 0.025]
    u = 10
    s_list = [u**-p for p in range(1, 6)]
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162, 642]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    output_dir = "./output/noisy_sphere_tests"
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

        m.run_noisy_belkin_laplacian_mcvec_fixed_param_test(s_list, loc=loc, scale=scale)
        m.run_noisy_belkin_laplacian_mcvec_average_face_area_test(loc=loc, scale=scale)
        m.run_noisy_cotan_laplacian_mcvec_test(loc=loc, scale=scale)
        m.run_noisy_guckenberger_laplacian_mcvec_test(loc=loc, scale=scale)
        m.save(data_path)
        M.append(m)
    return M


def load_spheres(output_dir="./output/sphere_tests3"):
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162, 642]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    # output_dir = "./output/sphere_tests3"
    M = []
    for surf in surfs:
        data_path = f"{output_dir}/{surf}"
        with open(data_path + ".pickle", "rb") as f:
            M.append(pickle.load(f))
    return M


def get_test_data(with_M=False, output_dir="./output/sphere_tests3"):
    M = load_spheres(output_dir=output_dir)
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

    mcvec_belkin_afe = [m.belkin_laplacian_mcvec_average_face_area_results["mcvec"] for m in M]
    mcvec_belkin_afe_L2error = np.array(
        [m.belkin_laplacian_mcvec_average_face_area_results["L2error"] for m in M]
    )
    mcvec_belkin_afe_Lifntyerror = np.array(
        [m.belkin_laplacian_mcvec_average_face_area_results["Lifntyerror"] for m in M]
    )

    # error_belkin_afe = run_noisy_belkin_laplacian_mcvec_average_face_area_test

    num_vertices = np.array([m.num_vertices for m in M])
    if with_M:
        return {
            "timelike_param": timelike_param,
            "mcvec_actual": mcvec_actual,
            "mcvec_cotan": mcvec_cotan,
            "mcvec_cotan_L2error": mcvec_cotan_L2error,
            "mcvec_cotan_Lifntyerror": mcvec_cotan_Lifntyerror,
            "mcvec_belkin": mcvec_belkin,
            "mcvec_belkin_L2error": mcvec_belkin_L2error,
            "mcvec_belkin_Lifntyerror": mcvec_belkin_Lifntyerror,
            "mcvec_belkin_afe": mcvec_belkin_afe,
            "mcvec_belkin_afe_L2error": mcvec_belkin_afe_L2error,
            "mcvec_belkin_afe_Lifntyerror": mcvec_belkin_afe_Lifntyerror,
            "N_vertices": num_vertices,
            "M": M,
        }
    else:
        return {
            "timelike_param": timelike_param,
            "mcvec_actual": mcvec_actual,
            "mcvec_cotan": mcvec_cotan,
            "mcvec_cotan_L2error": mcvec_cotan_L2error,
            "mcvec_cotan_Lifntyerror": mcvec_cotan_Lifntyerror,
            "mcvec_belkin": mcvec_belkin,
            "mcvec_belkin_L2error": mcvec_belkin_L2error,
            "mcvec_belkin_Lifntyerror": mcvec_belkin_Lifntyerror,
            "N_vertices": num_vertices,
        }


def get_crange(samps, Nstd=2):
    c0 = np.mean(samps)
    sig = np.std(samps)
    cmin = c0 - Nstd * sig
    cmax = c0 + Nstd * sig
    samps_clipped = np.clip(samps, cmin, cmax)
    return samps_clipped, cmin, cmax


def get_cmap(cmin=0.0, cmax=1.0, name="hsv"):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
    'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
    'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
    'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
    'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
    'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
    'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
    'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
    'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
    'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
    'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
    'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
    'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
    'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot',
    'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma',
    'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',
    'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
    'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
    'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
    'winter_r'

    """
    if cmax > cmin:
        cnum = lambda x: (x - cmin) / (cmax - cmin)
    else:
        cnum = lambda x: 0 * x
        print("bad min-max range")
    cmap01 = plt_cmap[name]
    my_cmap = lambda x: cmap01(cnum(float(x)))
    return my_cmap


def scalars_to_rgba(samples, cmin=None, cmax=None, name="coolwarm"):
    if cmin is None:
        cmin = np.min(samples)
    if cmax is None:
        cmax = np.max(samples)
    # Nsamps = len(samples)
    cmap = get_cmap(cmin=cmin, cmax=cmax, name=name)
    rgba = np.array([cmap(_) for _ in samples])
    return rgba


def plot_surf_belkin_error(numv, nums):

    m = M[numv]
    vec_belkin = M[numv].belkin_laplacian_mcvec_fixed_param_results["mcvec"][nums]
    vec_guckenberger = M[numv].guckenberger_laplacian_mcvec_results["mcvec"]
    vec_cotan = M[numv].cotan_laplacian_mcvec_results["mcvec"]

    vec = vec_cotan
    vec = vec_belkin
    vec = vec_guckenberger
    scale_vec = 2.0
    alpha = 0.4
    # err_belkin = np.linalg.norm(vec_belkin - m.mcvec_actual, axis=-1)
    # err_cotan = np.linalg.norm(vec_cotan - m.mcvec_actual, axis=-1)
    err = np.linalg.norm(vec - m.mcvec_actual, axis=-1)
    V_rgba = scalars_to_rgba(err)
    # V_rgba = scalars_to_rgba([-m.valence_v(v) for v in m.Vkeys])
    V_rgba[:, -1] = alpha
    E_rgba = np.zeros((len(m._v_origin_H), 4))

    vfdat = [m.xyz_array, scale_vec * (vec - m.mcvec_actual)]
    mv_kwargs = {
        "vector_field_data": [vfdat],
        "V_rgba": V_rgba,
        "color_by_V_rgba": True,
        "E_rgba": E_rgba,
        # "color_by_E_rgba": True,
        "show_halfedges": False,
    }
    mv = MeshViewer(*m.data_lists, **mv_kwargs)
    mv.plot()


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
    # colors = 2 * [
    #     "#2ca02c",  # g
    #     "#1f77b4",  # b
    #     "#d62728",  # r
    #     "#ff7f0e",  # o
    # ]
    # markers = 2 * ["o", "*", "^", "s"]
    # linestyles = 2 * ["solid", "dashed", "dashdot", "dotted"]
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

    if True:
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

    if True:
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

    if True:
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


# M = make_sphere_data(overwrite=True)
# test_data = get_test_data(with_M=True)
# M = make_noisy_sphere_data(overwrite=True, loc=0, scale=0.01)
output_dir = "./output/noisy_sphere_tests"
test_data = get_test_data(with_M=True, output_dir=output_dir)

timelike_param = test_data["timelike_param"]
mcvec_actual = test_data["mcvec_actual"]
mcvec_cotan = test_data["mcvec_cotan"]
mcvec_cotan_L2error = test_data["mcvec_cotan_L2error"]
mcvec_cotan_Lifntyerror = test_data["mcvec_cotan_Lifntyerror"]
mcvec_belkin = test_data["mcvec_belkin"]
mcvec_belkin_L2error = test_data["mcvec_belkin_L2error"]
mcvec_belkin_Lifntyerror = test_data["mcvec_belkin_Lifntyerror"]
mcvec_belkin_afe = test_data["mcvec_belkin_afe"]
mcvec_belkin_afe_L2error = test_data["mcvec_belkin_afe_L2error"]
mcvec_belkin_afe_Lifntyerror = test_data["mcvec_belkin_afe_Lifntyerror"]
N_vertices = test_data["N_vertices"]
M = test_data["M"]
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
error_belkin_afe = run_noisy_belkin_laplacian_mcvec_average_face_area_test
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


#########################################################
# %%
