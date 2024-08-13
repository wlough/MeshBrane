import numpy as np
import matplotlib.pyplot as plt
from src.python.utilities import round_to, log_log_fit
from matplotlib import colormaps as plt_cmap
import pickle

_COLORS_ = [
    "b",  # Blue
    "g",  # Green
    "r",  # Red
    "c",  # Cyan
    "m",  # Magenta
    "y",  # Yellow
    "k",  # Black
    "orange",  # Orange
    "purple",  # Purple
    "brown",  # Brown
    "pink",  # Pink
    "gray",  # Gray
    "olive",  # Olive
    "cyan",  # Cyan
    "navy",  # Navy
    "teal",  # Teal
    "lime",  # Lime
    "indigo",  # Indigo
    "gold",  # Gold
    "coral",  # Coral
    "turquoise",  # Turquoise
    "violet",  # Violet
    "plum",  # Plum
    "salmon",  # Salmon
    "chocolate",  # Chocolate
    "tan",  # Tan
    "orchid",  # Orchid
    "azure",  # Azure
    "lavender",  # Lavender
]
_MARKERS_ = ["o", "s", "^", "D", "v", "p", "*", "h", "H", "+", "x", "d", "|", "_"]
_LINESTYLES_ = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 10)),
    (0, (5, 1)),
    (0, (3, 5, 1, 5)),
    (0, (1, 1)),  # Dotted line
    (0, (1, 10)),  # Dotted line with large gaps
    (0, (1, 1, 1, 1)),  # Dotted line with small gaps
    (0, (5, 5)),  # Dashed line with equal gaps
    (0, (5, 1, 1, 1, 1, 1)),  # Dash-dot-dot line
    (0, (3, 5, 1, 5, 1, 5)),  # Dash-dot-dash-dot line
    (0, (1, 2, 3, 4)),  # Custom pattern
    (0, (2, 2, 10, 2)),  # Custom pattern with long dash
]

_CMAP_NAMES_ = [
    "Accent",
    "Accent_r",
    "Blues",
    "Blues_r",
    "BrBG",
    "BrBG_r",
    "BuGn",
    "BuGn_r",
    "BuPu",
    "BuPu_r",
    "CMRmap",
    "CMRmap_r",
    "Dark2",
    "Dark2_r",
    "GnBu",
    "GnBu_r",
    "Greens",
    "Greens_r",
    "Greys",
    "Greys_r",
    "OrRd",
    "OrRd_r",
    "Oranges",
    "Oranges_r",
    "PRGn",
    "PRGn_r",
    "Paired",
    "Paired_r",
    "Pastel1",
    "Pastel1_r",
    "Pastel2",
    "Pastel2_r",
    "PiYG",
    "PiYG_r",
    "PuBu",
    "PuBuGn",
    "PuBuGn_r",
    "PuBu_r",
    "PuOr",
    "PuOr_r",
    "PuRd",
    "PuRd_r",
    "Purples",
    "Purples_r",
    "RdBu",
    "RdBu_r",
    "RdGy",
    "RdGy_r",
    "RdPu",
    "RdPu_r",
    "RdYlBu",
    "RdYlBu_r",
    "RdYlGn",
    "RdYlGn_r",
    "Reds",
    "Reds_r",
    "Set1",
    "Set1_r",
    "Set2",
    "Set2_r",
    "Set3",
    "Set3_r",
    "Spectral",
    "Spectral_r",
    "Wistia",
    "Wistia_r",
    "YlGn",
    "YlGnBu",
    "YlGnBu_r",
    "YlGn_r",
    "YlOrBr",
    "YlOrBr_r",
    "YlOrRd",
    "YlOrRd_r",
    "afmhot",
    "afmhot_r",
    "autumn",
    "autumn_r",
    "binary",
    "binary_r",
    "bone",
    "bone_r",
    "brg",
    "brg_r",
    "bwr",
    "bwr_r",
    "cividis",
    "cividis_r",
    "cool",
    "cool_r",
    "coolwarm",
    "coolwarm_r",
    "copper",
    "copper_r",
    "cubehelix",
    "cubehelix_r",
    "flag",
    "flag_r",
    "gist_earth",
    "gist_earth_r",
    "gist_gray",
    "gist_gray_r",
    "gist_heat",
    "gist_heat_r",
    "gist_ncar",
    "gist_ncar_r",
    "gist_rainbow",
    "gist_rainbow_r",
    "gist_stern",
    "gist_stern_r",
    "gist_yarg",
    "gist_yarg_r",
    "gnuplot",
    "gnuplot2",
    "gnuplot2_r",
    "gnuplot_r",
    "gray",
    "gray_r",
    "hot",
    "hot_r",
    "hsv",
    "hsv_r",
    "inferno",
    "inferno_r",
    "jet",
    "jet_r",
    "magma",
    "magma_r",
    "nipy_spectral",
    "nipy_spectral_r",
    "ocean",
    "ocean_r",
    "pink",
    "pink_r",
    "plasma",
    "plasma_r",
    "prism",
    "prism_r",
    "rainbow",
    "rainbow_r",
    "seismic",
    "seismic_r",
    "spring",
    "spring_r",
    "summer",
    "summer_r",
    "tab10",
    "tab10_r",
    "tab20",
    "tab20_r",
    "tab20b",
    "tab20b_r",
    "tab20c",
    "tab20c_r",
    "terrain",
    "terrain_r",
    "turbo",
    "turbo_r",
    "twilight",
    "twilight_r",
    "twilight_shifted",
    "twilight_shifted_r",
    "viridis",
    "viridis_r",
    "winter",
    "winter_r",
]


def get_plt_combos(n):
    """
    Returns a list of n combinations of matplotlib color, marker, and linestyle.
    """
    len_colors = len(_COLORS_)
    len_markers = len(_MARKERS_)
    len_linestyles = len(_LINESTYLES_)
    return [
        [
            _COLORS_[k % len_colors],
            _MARKERS_[k % len_markers],
            _LINESTYLES_[k % len_linestyles],
        ]
        for k in range(n)
    ]


def get_crange(samples, Nstd=2):
    """
    Clip the samples to ramain within Nstd standard deviations of the mean.
    """
    c0 = np.mean(samples)
    sig = np.std(samples)
    cmin = c0 - Nstd * sig
    cmax = c0 + Nstd * sig
    return np.clip(samples, cmin, cmax), [cmin, cmax]


def get_cmap(cmin, cmax, name="hsv"):
    """
    Returns a function that maps the interval [cmin, cmax] to distinct rgba colors.

    Args:
        cmin (float): minimum value of the interval
        cmax (float): maximum value of the interval
        name (str): name of the matplotlib colormap to use
    Returns:
        function: a cmap on [cmin, cmax]
    Keyword argument 'name' must be one of the standard matplotlib colormaps listed below:

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
    if cmin >= cmax:
        raise ValueError("cmax must be greater than cmin")
    cmap01 = plt_cmap[name]  # cmap on [0, 1]
    return lambda x: cmap01((float(x) - cmin) / (cmax - cmin))


def scalars_to_rgba(samples, cminmax=None, name="coolwarm"):
    """
    Assigns a color to each scalar value in samples using a colormap. See get_cmap() for details.
    """
    if cminmax is None:
        cmap = get_cmap(np.min(samples), np.max(samples), name=name)
        return np.array([cmap(_) for _ in samples])
    else:
        cmap = get_cmap(*cminmax, name=name)
        return np.array([cmap(_) for _ in np.clip(samples, *cminmax)])


def to_scinotation_tex(X, decimals=3, mode="inline"):
    """
    Makes a list of numbers into a list of strings
    in latex scientific notation.
    """
    if mode == "inline":
        left, right = r"$", r"$"
    if mode == "plain":
        left, right = r"", r""
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
        left + f"{c}" + r" \times " + r"10^{" + f"{p}" + r"}" + right
        for c, p in zip(coeff, pow)
    ]
    return xlabels


def get_test_data(with_M=False, output_dir="./output/sphere_tests3"):
    M = load_spheres(output_dir=output_dir)
    # belkin_laplacian_mcvec_fixed_param_results
    # cotan_laplacian_mcvec_results
    timelike_param = np.array(
        [m.belkin_laplacian_mcvec_fixed_param_results["s"] for m in M][0]
    )
    mcvec_actual = [m.mcvec_actual for m in M]

    mcvec_cotan = [m.cotan_laplacian_mcvec_results["mcvec"] for m in M]
    mcvec_cotan_L2error = np.array(
        [m.cotan_laplacian_mcvec_results["L2error"] for m in M]
    )
    mcvec_cotan_Linftyerror = np.array(
        [m.cotan_laplacian_mcvec_results["Linftyerror"] for m in M]
    )

    # _mcvec_belkin = [m.mcvec_belkin for m in M]
    # _mcvec_belkin_L2error = [m.mcvec_belkin_L2error for m in M]
    # _mcvec_belkin_Linftyerror = [m.mcvec_belkin_Linftyerror for m in M]
    #
    # num_M = len(M)
    num_timelike = len(timelike_param)
    mcvec_belkin = [
        [m.belkin_laplacian_mcvec_fixed_param_results["mcvec"][_] for m in M]
        for _ in range(num_timelike)
    ]
    mcvec_belkin_L2error = np.array(
        [
            [m.belkin_laplacian_mcvec_fixed_param_results["L2error"][_] for m in M]
            for _ in range(num_timelike)
        ]
    )
    mcvec_belkin_Linftyerror = np.array(
        [
            [m.belkin_laplacian_mcvec_fixed_param_results["Linftyerror"][_] for m in M]
            for _ in range(num_timelike)
        ]
    )

    mcvec_belkin_afe = [
        m.belkin_laplacian_mcvec_average_face_area_results["mcvec"] for m in M
    ]
    mcvec_belkin_afe_L2error = np.array(
        [m.belkin_laplacian_mcvec_average_face_area_results["L2error"] for m in M]
    )
    mcvec_belkin_afe_Linftyerror = np.array(
        [m.belkin_laplacian_mcvec_average_face_area_results["Linftyerror"] for m in M]
    )

    # error_belkin_afe = run_noisy_belkin_laplacian_mcvec_average_face_area_test

    num_vertices = np.array([m.num_vertices for m in M])
    if with_M:
        return {
            "timelike_param": timelike_param,
            "mcvec_actual": mcvec_actual,
            "mcvec_cotan": mcvec_cotan,
            "mcvec_cotan_L2error": mcvec_cotan_L2error,
            "mcvec_cotan_Linftyerror": mcvec_cotan_Linftyerror,
            "mcvec_belkin": mcvec_belkin,
            "mcvec_belkin_L2error": mcvec_belkin_L2error,
            "mcvec_belkin_Linftyerror": mcvec_belkin_Linftyerror,
            "mcvec_belkin_afe": mcvec_belkin_afe,
            "mcvec_belkin_afe_L2error": mcvec_belkin_afe_L2error,
            "mcvec_belkin_afe_Linftyerror": mcvec_belkin_afe_Linftyerror,
            "N_vertices": num_vertices,
            "M": M,
        }
    else:
        return {
            "timelike_param": timelike_param,
            "mcvec_actual": mcvec_actual,
            "mcvec_cotan": mcvec_cotan,
            "mcvec_cotan_L2error": mcvec_cotan_L2error,
            "mcvec_cotan_Linftyerror": mcvec_cotan_Linftyerror,
            "mcvec_belkin": mcvec_belkin,
            "mcvec_belkin_L2error": mcvec_belkin_L2error,
            "mcvec_belkin_Linftyerror": mcvec_belkin_Linftyerror,
            "N_vertices": num_vertices,
        }


class ConvergenceTestData:
    """
    Data structure to store the results of a convergence test for a given method.
    Attributes:
        name (str): name of the method/test
        samples_numerical (list of array): samples computed using numerical method
        samples_actual (list of array): samples computed using analytical formula
        independent_var (array): independent variable for calculating convergence rate (e.g. number of vertices)
        params (dict): any parameters used to compute numerical samples
        fun_actual (callable): function to compute the actual samples at vertex positions
        half_edge_mesh_arrays (list of dict): half-edge mesh arrays


        normalized_L2_error (array): normalized L2 error
        Linfinity_error (array): Linfinity (max) error
    """

    def __init__(
        self,
        name="ctest",
        samples_numerical=None,
        samples_actual=None,
        independent_var=None,
        params=None,
        fun_actual=None,
        half_edge_mesh_arrays=None,
        data_path=None,
    ):
        self.name = name
        self.data_path = data_path
        self.samples_numerical = samples_numerical
        self.samples_actual = samples_actual
        self.params = params
        self.fun_actual = fun_actual
        self.independent_var = independent_var
        self.half_edge_mesh_arrays = half_edge_mesh_arrays
        if samples_actual is not None and samples_numerical is not None:
            self.normalized_L2_error = self.compute_normalized_Lp_error(2)
            self.Linfinity_error = self.compute_absolut_Lp_error(np.inf)

    def compute_samples_actual(self):
        Nsamples = len(self.samples_numerical)
        samples_actual = []
        for n in range(Nsamples):
            xyz_coord_V = self.half_edge_mesh_arrays[n]["xyz_coord_V"]
            samples_actual.append(self.fun_actual(xyz_coord_V))

        return samples_actual

    def compute_normalized_Lp_error(self, order):
        error = []
        for s_num, s_act in zip(self.samples_numerical, self.samples_actual):
            error.append(
                np.linalg.norm((s_num - s_act).ravel(), order)
                / np.linalg.norm(s_act.ravel(), order)
            )
        return error

    def compute_absolut_Lp_error(self, order):
        error = []
        for s_num, s_act in zip(self.samples_numerical, self.samples_actual):
            error.append(np.linalg.norm((s_num - s_act).ravel(), order))
        return error

    def save(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
        with open(self.data_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, data_path):
        with open(data_path, "rb") as f:
            return pickle.load(f)
