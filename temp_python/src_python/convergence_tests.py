import numpy as np
import matplotlib.pyplot as plt
from temp_python.src_python.utilities.misc_utils import round_to, log_log_fit
from matplotlib import colormaps as plt_cmap
import pickle
import os
from temp_python.src_python.utilities.misc_utils import (
    make_output_dir,
    load_npz,
    save_npz,
    unchunk_file_with_cat,
)


def uncompress_sphere_half_edge_arrays(output_dir="./output/half_edge_arrays"):
    _NUM_VERTS_ = [
        12,
        42,
        162,
        642,
        2562,
        10242,
        40962,
        163842,
        655362,
        2621442,
    ]
    make_output_dir(output_dir)
    npz_paths = [f"{output_dir}/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_]
    compressed_npz_paths = [
        f"./data/half_edge_arrays/compressed_unit_sphere_{N:07d}.npz"
        for N in _NUM_VERTS_
    ]
    chunked_npz_path = compressed_npz_paths[-1]
    unchunked_npz_path = compressed_npz_paths[-1]
    unchunk_file_with_cat(chunked_npz_path, unchunked_npz_path)
    he_arrays = [load_npz(p) for p in compressed_npz_paths]
    os.system(f"rm {unchunked_npz_path}")
    for data, filename in zip(he_arrays, npz_paths):
        print("saving " + filename)
        save_npz(data, filename, compressed=False, chunk=False, remove_unchunked=False)
    print("done")


MATPLOTLIB_COLORS = [
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
    len_colors = len(MATPLOTLIB_COLORS)
    len_markers = len(_MARKERS_)
    len_linestyles = len(_LINESTYLES_)
    return [
        [
            MATPLOTLIB_COLORS[k % len_colors],
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


class ConvergenceTestBase:
    """
    Attributes
    ----------
    test_dir : str
        output directory to store test results
    test_names : list of str
        names of tests to run
    test_keys : list
        list of keys for indexing test results. used for keys of dicts in test_data_paths
    test_data_paths : dict of dict
        dictionary of paths to data files to tests path=test_data_paths[test_name][test_key]
    npz_paths : list
        paths to half-edge mesh arrays used to initialized HalfEdgeMesh objects
    M : list
        list of HalfEdgeMesh objects



    """

    def __init__(
        self,
        test_dir="./output/sphere_tests_base",
        test_names=[
            "mean_curvature",
            "unit_normal",
            "lap_x",
            "lap_x_squared",
            "lap_exp_x_y",
            # "laplacian_mean_curvature",
            # "bending_force",
        ],
        load_half_edge_meshes=False,
        # surface/laplacian specific parameters
        # npz_dir="./output/half_edge_arrays",
        # num_vertices=[
        #     # 12,
        #     # 42,
        #     162,
        #     642,
        #     2562,
        #     10242,
        #     40962,
        #     # 163842,
        #     # 655362,
        #     # 2621442,
        # ],
    ):
        #######################################
        # surface/laplacian specific attributes
        self.test_keys = [0]
        # self.npz_paths = [f"{npz_dir}/unit_sphere_{N:07d}.npz" for N in num_vertices]
        self.laplacian_kwargs = {key: dict() for key in self.test_keys}
        #######################################
        self.test_names = test_names
        self.test_dir = test_dir
        if load_half_edge_meshes:
            self.M = self.load_meshes()
        else:
            self.M = None

        self.test_data_paths = {
            test_name: {
                key: f"{test_dir}/{test_name}/test_{self.key2index_str(key)}.pkl"
                for key in self.test_keys
            }
            for test_name in self.test_names
        }
        self.Tdict = {
            test_name: {key: dict() for key in self.test_keys}
            for test_name in self.test_names
        }

    #######################################
    # laplacian specific methods
    def key2index_str(self, key):
        k = 0
        return f"{k:03d}"

    def apply_laplacian(self, m, Q, *args):
        return m.cotan_laplacian(Q, *args)

    #######################################
    # surface specific methods
    def load_meshes(self):
        print("loading half-edge meshes")
        from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

        num_vertices = (
            [
                # 12,
                # 42,
                162,
                642,
                2562,
                10242,
                40962,
                # 163842,
                # 655362,
                # 2621442,
            ],
        )

        npz_paths = [f"{npz_dir}/unit_sphere_{N:07d}.npz" for N in num_vertices]
        M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)
        print("done")
        return M

    def lap_x(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return -2 * np.sin(theta) * np.cos(phi)

    def lap_x_squared(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return 6 * np.sin(phi) ** 2 * np.sin(theta) ** 2 - 6 * np.sin(theta) ** 2 + 2

    def lap_exp_x_y(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return (
            (
                -np.sin(2 * phi) / 2
                + np.sin(2 * (phi - theta)) / 4
                + np.sin(2 * (phi + theta)) / 4
                + np.cos(2 * theta) / 2
                - np.sqrt(2) * np.cos(phi - theta + np.pi / 4)
                + np.sqrt(2) * np.cos(phi + theta + np.pi / 4)
                + 3 / 2
            )
            * np.exp(-np.sin(phi - theta) / 2 + np.sin(phi + theta) / 2)
            * np.exp(np.cos(phi - theta) / 2 - np.cos(phi + theta) / 2)
        )

    def mean_curvature_actual(self, xyz_array):
        num_vertices = xyz_array.shape[0]
        return -np.ones(num_vertices)

    def unit_normal_actual(self, xyz_array):
        num_vertices = xyz_array.shape[0]
        return xyz_array.copy()

    ##################################################################
    def run_tests(self, overwrite=False):
        make_output_dir(output_dir=self.test_dir, overwrite=overwrite)

        for test_name in self.test_names:
            make_output_dir(
                output_dir=self.test_dir + "/" + test_name, overwrite=overwrite
            )
        if self.M is None:
            self.M = self.load_meshes()

        he_keys = [
            "xyz_coord_V",
            "h_out_V",
            "v_origin_H",
            "h_next_H",
            "h_twin_H",
            "f_left_H",
            "h_bound_F",
            "h_comp_B",
        ]
        half_edge_mesh_arrays = [
            {k: v for k, v in zip(he_keys, m.data_arrays)} for m in self.M
        ]
        num_faces = np.array([m.num_faces for m in self.M])
        test_kwargs = dict()
        #################################################################
        # lap of fun tests
        run_lap_fun_tests = any(
            [
                "lap_x" in self.test_names,
                "lap_x_squared" in self.test_names,
                "lap_exp_x_y" in self.test_names,
            ]
        )
        if "lap_x" in self.test_names:
            lap_x_actual = [self.lap_x(m.xyz_array) for m in self.M]
            test_kwargs["lap_x"] = {
                "samples_actual": lap_x_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        if "lap_x_squared" in self.test_names:
            lap_x_squared_actual = [self.lap_x_squared(m.xyz_array) for m in self.M]
            test_kwargs["lap_x_squared"] = {
                "samples_actual": lap_x_squared_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        if "lap_exp_x_y" in self.test_names:
            lap_exp_x_y_actual = [self.lap_exp_x_y(m.xyz_array) for m in self.M]
            test_kwargs["lap_exp_x_y"] = {
                "samples_actual": lap_exp_x_y_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        #################################################################
        # mcvec tests
        run_mcvec_tests = (
            "mean_curvature" in self.test_names or "unit_normal" in self.test_names
        )
        if "mean_curvature" in self.test_names:
            mean_curvature_actual = [
                self.mean_curvature_actual(m.xyz_array) for m in self.M
            ]
            test_kwargs["mean_curvature"] = {
                "samples_actual": mean_curvature_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        if "unit_normal" in self.test_names:
            unit_normal_actual = [self.unit_normal_actual(m.xyz_array) for m in self.M]
            test_kwargs["unit_normal"] = {
                "samples_actual": unit_normal_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        # self.xyz = dict()
        # self.lap_xyz = dict()
        # self.H = dict()
        # self.n = dict()
        print(f"running tests:\n{self.test_names}")
        print(f"test keys:\n{self.test_keys}")
        for key in self.test_keys:
            print("--------------------")
            print(f"key={key}")
            lap_kwargs = self.laplacian_kwargs[key]

            # samples_numerical = []
            lap_x_numerical = []
            lap_x_squared_numerical = []
            lap_exp_x_y_numerical = []
            mean_curvature_numerical = []
            unit_normal_numerical = []
            for m in self.M:
                print(f"-- num_faces={m.num_faces}")
                xyz = m.xyz_array
                if run_mcvec_tests:
                    # lap_xyz = m.order_p_belkin_laplacian(xyz, s, n_findiff)

                    lap_xyz = self.apply_laplacian(m, xyz, **lap_kwargs)
                    mean_curvature = -np.linalg.norm(lap_xyz, axis=-1) / 2
                    mean_curvature_numerical.append(mean_curvature)
                    unit_normal = np.einsum("ix,i->ix", lap_xyz, 1 / mean_curvature) / 2
                    unit_normal_numerical.append(unit_normal)

                if "lap_x" in self.test_names:
                    x = xyz[:, 0]
                    lap_x = self.apply_laplacian(m, x, **lap_kwargs)
                    lap_x_numerical.append(lap_x)
                if "lap_x_squared" in self.test_names:
                    x_squared = xyz[:, 0] ** 2
                    lap_x_squared = self.apply_laplacian(m, x_squared, **lap_kwargs)
                    lap_x_squared_numerical.append(lap_x_squared)
                if "lap_exp_x_y" in self.test_names:
                    x = xyz[:, 0]
                    y = xyz[:, 1]
                    exp_x_y = np.exp(x + y)
                    lap_exp_x_y = self.apply_laplacian(m, exp_x_y, **lap_kwargs)
                    lap_exp_x_y_numerical.append(lap_exp_x_y)

            if "mean_curvature" in self.test_names:
                # mean_curvature
                test_kwargs["mean_curvature"].update(
                    {
                        "name": f"mean_curvature_{self.key2index_str(key)}",
                        "samples_numerical": mean_curvature_numerical,
                        "params": lap_kwargs,
                        "data_path": self.test_data_paths["mean_curvature"][key],
                    }
                )
                T = ConvergenceTestData(**test_kwargs["mean_curvature"])
                T.save()
                self.Tdict["mean_curvature"][key] = T
            if "unit_normal" in self.test_names:
                # unit_normal
                test_kwargs["unit_normal"].update(
                    {
                        "name": f"unit_normal_{self.key2index_str(key)}",
                        "samples_numerical": unit_normal_numerical,
                        "params": lap_kwargs,
                        "data_path": self.test_data_paths["unit_normal"][key],
                    }
                )
                T = ConvergenceTestData(**test_kwargs["unit_normal"])
                T.save()
                self.Tdict["unit_normal"][key] = T
            if "lap_x" in self.test_names:
                # lap_x
                test_kwargs["lap_x"].update(
                    {
                        "name": f"lap_x_{self.key2index_str(key)}",
                        "samples_numerical": lap_x_numerical,
                        "params": lap_kwargs,
                        "data_path": self.test_data_paths["lap_x"][key],
                    }
                )
                T = ConvergenceTestData(**test_kwargs["lap_x"])
                T.save()
                self.Tdict["lap_x"][key] = T
            if "lap_x_squared" in self.test_names:
                # lap_x_squared
                test_kwargs["lap_x_squared"].update(
                    {
                        "name": f"lap_x_squared_{self.key2index_str(key)}",
                        "samples_numerical": lap_x_squared_numerical,
                        "params": lap_kwargs,
                        "data_path": self.test_data_paths["lap_x_squared"][key],
                    }
                )
                T = ConvergenceTestData(**test_kwargs["lap_x_squared"])
                T.save()
                self.Tdict["lap_x_squared"][key] = T
            if "lap_exp_x_y" in self.test_names:
                # lap_exp_x_y
                test_kwargs["lap_exp_x_y"].update(
                    {
                        "name": f"lap_exp_x_y_{self.key2index_str(key)}",
                        "samples_numerical": lap_exp_x_y_numerical,
                        "params": lap_kwargs,
                        "data_path": self.test_data_paths["lap_exp_x_y"][key],
                    }
                )
                T = ConvergenceTestData(**test_kwargs["lap_exp_x_y"])
                T.save()
                self.Tdict["lap_exp_x_y"][key] = T

        #################################################################
        #################################################################
        #################################################################

        print("Done.")

        return self.Tdict

    def load_test_results(
        self,
        test_names=[
            "mean_curvature",
            "unit_normal",
            "lap_x",
            "lap_x_squared",
            "lap_exp_x_y",
        ],
    ):
        Tdict = {test_name: dict() for test_name in test_names}
        for test_name in test_names:
            for test_key in self.test_keys:
                data_path = self.test_data_paths[test_name][test_key]
                Tdict[test_name][test_key] = ConvergenceTestData.load(data_path)
        self.Tdict = Tdict
        return Tdict

    ############################################################


class SphereBelkinTest:
    """
    Attributes
    ----------
    test_dir : str
        output directory to store test results
    test_names : list of str
        names of tests to run
    test_keys : list
        list of keys for indexing test results. used for keys of dicts in test_data_paths
    test_data_paths : dict of dict
        dictionary of paths to data files to tests path=test_data_paths[test_name][test_key]
    npz_paths : list
        paths to half-edge mesh arrays used to initialized HalfEdgeMesh objects
    M : list
        list of HalfEdgeMesh objects

    time_step : array
        step sizes for the timelike parameter used in Belkin Laplacian
    time_step_num : list
        indices of time_step for naming test data files
    tfindiff_order : list
        order of finite difference wrt timelike parameter used in Belkin Laplacian

    """

    def __init__(
        self,
        test_dir="./output/sphere_belkin_tests",
        npz_dir="./output/half_edge_arrays",
        num_vertices=[
            # 12,
            # 42,
            162,
            642,
            2562,
            10242,
            40962,
            # 163842,
            # 655362,
            # 2621442,
        ],
        load_half_edge_meshes=True,
        test_names=[
            "mean_curvature",
            "unit_normal",
            "lap_x",
            "lap_x_squared",
            "lap_exp_x_y",
            # "laplacian_mean_curvature",
            # "bending_force",
        ],
    ):
        self.test_names = test_names
        self.test_dir = test_dir
        self.time_step = np.array(
            [0.03853078, 0.0097707, 0.00245144, 0.00061341, 0.00015339]
        )
        self.time_step_num = [_ for _, m in enumerate(self.time_step)]
        self.tfindiff_order = [1, 2, 3]  # findiff order for Belkin Laplacian
        self.test_keys = [
            (n_findiff, n_tstep)
            for n_findiff in self.tfindiff_order
            for n_tstep in self.time_step_num
        ]
        self.npz_paths = [f"{npz_dir}/unit_sphere_{N:07d}.npz" for N in num_vertices]
        if load_half_edge_meshes:
            print("loading half-edge meshes")
            from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

            self.M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)
            print("done")
        else:
            self.M = None

        self.test_data_paths = {
            test_name: {
                (
                    n_findiff,
                    n_tstep,
                ): f"{test_dir}/{test_name}/test_{n_findiff:03d}_{n_tstep:03d}.pkl"
                for (n_findiff, n_tstep) in self.test_keys
            }
            for test_name in self.test_names
        }

    def key2index_str(self, key):
        (n_findiff, n_tstep) = key
        return f"{n_findiff:03d}_{n_tstep:03d}"

    def lap_x(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return -2 * np.sin(theta) * np.cos(phi)

    def lap_x_squared(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return 6 * np.sin(phi) ** 2 * np.sin(theta) ** 2 - 6 * np.sin(theta) ** 2 + 2

    def lap_exp_x_y(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return (
            (
                -np.sin(2 * phi) / 2
                + np.sin(2 * (phi - theta)) / 4
                + np.sin(2 * (phi + theta)) / 4
                + np.cos(2 * theta) / 2
                - np.sqrt(2) * np.cos(phi - theta + np.pi / 4)
                + np.sqrt(2) * np.cos(phi + theta + np.pi / 4)
                + 3 / 2
            )
            * np.exp(-np.sin(phi - theta) / 2 + np.sin(phi + theta) / 2)
            * np.exp(np.cos(phi - theta) / 2 - np.cos(phi + theta) / 2)
        )

    def run_tests(self, overwrite=False):
        make_output_dir(output_dir=self.test_dir, overwrite=overwrite)

        for test_name in self.test_names:
            make_output_dir(
                output_dir=self.test_dir + "/" + test_name, overwrite=overwrite
            )
        if self.M is None:
            from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

            self.M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)

        he_keys = [
            "xyz_coord_V",
            "h_out_V",
            "v_origin_H",
            "h_next_H",
            "h_twin_H",
            "f_left_H",
            "h_bound_F",
            "h_comp_B",
        ]
        half_edge_mesh_arrays = [
            {k: v for k, v in zip(he_keys, m.data_arrays)} for m in self.M
        ]
        num_faces = np.array([m.num_faces for m in self.M])
        test_kwargs = dict()
        Tdict = dict()
        #################################################################
        # lap of fun tests
        run_lap_fun_tests = any(
            [
                "lap_x" in self.test_names,
                "lap_x_squared" in self.test_names,
                "lap_exp_x_y" in self.test_names,
            ]
        )
        if "lap_x" in self.test_names:
            Tdict["lap_x"] = dict()
            lap_x_actual = [self.lap_x(m.xyz_array) for m in self.M]
            test_kwargs["lap_x"] = {
                "samples_actual": lap_x_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        if "lap_x_squared" in self.test_names:
            Tdict["lap_x_squared"] = dict()
            lap_x_squared_actual = [self.lap_x_squared(m.xyz_array) for m in self.M]
            test_kwargs["lap_x_squared"] = {
                "samples_actual": lap_x_squared_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        if "lap_exp_x_y" in self.test_names:
            Tdict["lap_exp_x_y"] = dict()
            lap_exp_x_y_actual = [self.lap_exp_x_y(m.xyz_array) for m in self.M]
            test_kwargs["lap_exp_x_y"] = {
                "samples_actual": lap_exp_x_y_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        #################################################################
        # mcvec tests
        run_mcvec_tests = (
            "mean_curvature" in self.test_names or "unit_normal" in self.test_names
        )
        if "mean_curvature" in self.test_names:
            mcvec_actual = [-2 * np.ones(m.num_vertices) for m in self.M]
            Tdict["mean_curvature"] = dict()
            mean_curvature_actual = [-np.ones(m.num_vertices) for m in self.M]
        if "unit_normal" in self.test_names:
            Tdict["unit_normal"] = dict()
            unit_normal_actual = [m.xyz_array for m in self.M]
            test_kwargs["unit_normal"] = {
                "samples_actual": unit_normal_actual,
                "independent_var": num_faces,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
            }
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        # self.xyz = dict()
        # self.lap_xyz = dict()
        # self.H = dict()
        # self.n = dict()
        #################################################################
        #################################################################
        #################################################################
        for n_findiff in self.tfindiff_order:
            print("--------------------")
            print(f"findiff order={n_findiff}")
            for n_tstep in self.time_step_num:
                s = self.time_step[n_tstep]
                print(f"- time_step={s}")
                # samples_numerical = []
                lap_x_numerical = []
                lap_x_squared_numerical = []
                lap_exp_x_y_numerical = []
                mean_curvature_numerical = []
                unit_normal_numerical = []
                for m in self.M:
                    print(f"-- num_faces={m.num_faces}")
                    xyz = m.xyz_array
                    if run_mcvec_tests:
                        lap_xyz = m.order_p_belkin_laplacian(xyz, s, n_findiff)
                        mean_curvature = -np.linalg.norm(lap_xyz, axis=-1) / 2
                        mean_curvature_numerical.append(mean_curvature)
                        unit_normal = (
                            np.einsum("ix,i->ix", lap_xyz, 1 / mean_curvature) / 2
                        )
                        #################################################################
                        #################################################################
                        #################################################################
                        # self.xyz[(n_findiff, n_tstep)] = xyz
                        # self.lap_xyz[(n_findiff, n_tstep)] = lap_xyz
                        # self.H[(n_findiff, n_tstep)] = mean_curvature
                        # self.n[(n_findiff, n_tstep)] = unit_normal
                        #################################################################
                        #################################################################
                        #################################################################
                        unit_normal_numerical.append(unit_normal)

                    if "lap_x" in self.test_names:
                        x = xyz[:, 0]
                        lap_x = m.order_p_belkin_laplacian(x, s, n_findiff)
                        lap_x_numerical.append(lap_x)
                    if "lap_x_squared" in self.test_names:
                        x_squared = xyz[:, 0] ** 2
                        lap_x_squared = m.order_p_belkin_laplacian(
                            x_squared, s, n_findiff
                        )
                        lap_x_squared_numerical.append(lap_x_squared)
                    if "lap_exp_x_y" in self.test_names:
                        x = xyz[:, 0]
                        y = xyz[:, 1]
                        exp_x_y = np.exp(x + y)
                        lap_exp_x_y = m.order_p_belkin_laplacian(exp_x_y, s, n_findiff)
                        lap_exp_x_y_numerical.append(lap_exp_x_y)

                if "mean_curvature" in self.test_names:
                    # mean_curvature
                    test_kwargs["mean_curvature"] = {
                        "name": "mean_curvature",
                        "samples_actual": mean_curvature_actual,
                        "samples_numerical": mean_curvature_numerical,
                        "independent_var": num_faces,
                        "half_edge_mesh_arrays": half_edge_mesh_arrays,
                        "params": {"s": s},
                        "data_path": self.test_data_paths["mean_curvature"][
                            (n_findiff, n_tstep)
                        ],
                    }
                    T = ConvergenceTestData(**test_kwargs["mean_curvature"])
                    T.save()
                    Tdict["mean_curvature"][(n_findiff, n_tstep)] = T
                if "unit_normal" in self.test_names:
                    # unit_normal
                    test_kwargs["unit_normal"] = {
                        "name": "unit_normal",
                        "samples_actual": unit_normal_actual,
                        "samples_numerical": unit_normal_numerical,
                        "independent_var": num_faces,
                        "half_edge_mesh_arrays": half_edge_mesh_arrays,
                        "params": {"s": s},
                        "data_path": self.test_data_paths["unit_normal"][
                            (n_findiff, n_tstep)
                        ],
                    }
                    T = ConvergenceTestData(**test_kwargs["unit_normal"])
                    T.save()
                    Tdict["unit_normal"][(n_findiff, n_tstep)] = T
                if "lap_x" in self.test_names:
                    # lap_x
                    test_kwargs["lap_x"] = {
                        "name": "lap_x",
                        "samples_actual": lap_x_actual,
                        "samples_numerical": lap_x_numerical,
                        "independent_var": num_faces,
                        "half_edge_mesh_arrays": half_edge_mesh_arrays,
                        "params": {"s": s},
                        "data_path": self.test_data_paths["lap_x"][
                            (n_findiff, n_tstep)
                        ],
                    }
                    T = ConvergenceTestData(**test_kwargs["lap_x"])
                    T.save()
                    Tdict["lap_x"][(n_findiff, n_tstep)] = T
                if "lap_x_squared" in self.test_names:
                    # lap_x_squared
                    test_kwargs["lap_x_squared"] = {
                        "name": "lap_x_squared",
                        "samples_actual": lap_x_squared_actual,
                        "samples_numerical": lap_x_squared_numerical,
                        "independent_var": num_faces,
                        "half_edge_mesh_arrays": half_edge_mesh_arrays,
                        "params": {"s": s},
                        "data_path": self.test_data_paths["lap_x_squared"][
                            (n_findiff, n_tstep)
                        ],
                    }
                    T = ConvergenceTestData(**test_kwargs["lap_x_squared"])
                    T.save()
                    Tdict["lap_x_squared"][(n_findiff, n_tstep)] = T
                if "lap_exp_x_y" in self.test_names:
                    # lap_exp_x_y
                    test_kwargs["lap_exp_x_y"] = {
                        "name": "lap_exp_x_y",
                        "samples_actual": lap_exp_x_y_actual,
                        "samples_numerical": lap_exp_x_y_numerical,
                        "independent_var": num_faces,
                        "half_edge_mesh_arrays": half_edge_mesh_arrays,
                        "params": {"s": s},
                        "data_path": self.test_data_paths["lap_exp_x_y"][
                            (n_findiff, n_tstep)
                        ],
                    }
                    T = ConvergenceTestData(**test_kwargs["lap_exp_x_y"])
                    T.save()
                    Tdict["lap_exp_x_y"][(n_findiff, n_tstep)] = T
        print("Done.")
        self.Tdict = Tdict
        return Tdict

    def load_test_results(
        self,
        test_names=[
            "mean_curvature",
            "unit_normal",
            "lap_x",
            "lap_x_squared",
            "lap_exp_x_y",
        ],
    ):
        Tdict = {test_name: dict() for test_name in test_names}
        for test_name in test_names:
            for test_key in self.test_keys:
                data_path = self.test_data_paths[test_name][test_key]
                Tdict[test_name][test_key] = ConvergenceTestData.load(data_path)
        self.Tdict = Tdict
        return Tdict


class SphereMcvecBelkinTest:
    """
    Attributes
    ----------
    test_dir : str
        output directory to store test results
    test_data_paths : dict
        dictionary of paths to test data files
    test_names : dict
        dictionary of names for ConvergenceTestData objects
    npz_paths : list
        paths to half-edge mesh arrays used to initialized HalfEdgeMesh objects
    test_keys : list
        list of keys for test_data_paths and test_names
    M : list
        list of HalfEdgeMesh objects

    time_step : array
        step sizes for the timelike parameter used in Belkin Laplacian
    time_step_num : list
        indices of time_step for naming test data files
    tfindiff_order : list
        order of finite difference wrt timelike parameter used in Belkin Laplacian

    """

    def __init__(
        self,
        test_dir="./output/sphere_belkin_tests",
        npz_dir="./output/half_edge_arrays",
        num_vertices=[
            # 12,
            # 42,
            162,
            642,
            2562,
            10242,
            40962,
            # 163842,
            # 655362,
            # 2621442,
        ],
        load_half_edge_meshes=True,
    ):
        self.test_dir = test_dir
        self.time_step = np.array(
            [0.03853078, 0.0097707, 0.00245144, 0.00061341, 0.00015339]
        )
        self.time_step_num = [_ for _, m in enumerate(self.time_step)]
        self.tfindiff_order = [1, 2, 3]  # findiff order for Belkin Laplacian
        self.test_keys = [
            (n_findiff, n_tstep)
            for n_findiff in self.tfindiff_order
            for n_tstep in self.time_step_num
        ]
        self.test_data_paths = {
            (n_findiff, n_tstep): test_dir
            + "/"
            + f"test_data_{n_findiff:03d}_{n_tstep:03d}"
            + ".pkl"
            for (n_findiff, n_tstep) in self.test_keys
        }
        self.test_names = {
            (n_findiff, n_tstep): f"sphere_mcvec_belkin_{n_findiff:03d}_{n_tstep:03d}"
            for (n_findiff, n_tstep) in self.test_keys
        }
        self.npz_paths = [f"{npz_dir}/unit_sphere_{N:07d}.npz" for N in num_vertices]

        if load_half_edge_meshes:
            from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

            self.M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)
        else:
            self.M = None

    def run_tests(self, overwrite=False):
        make_output_dir(output_dir=self.test_dir, overwrite=overwrite)
        if self.M is None:
            from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

            self.M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)

        he_keys = [
            "xyz_coord_V",
            "h_out_V",
            "v_origin_H",
            "h_next_H",
            "h_twin_H",
            "f_left_H",
            "h_bound_F",
            "h_comp_B",
        ]
        test_kwargs = {
            "samples_actual": [-np.ones(m.num_vertices) for m in self.M],
            "independent_var": np.array([m.num_faces for m in self.M]),
            "half_edge_mesh_arrays": [
                {k: v for k, v in zip(he_keys, m.data_arrays)} for m in self.M
            ],
        }
        Tdict = dict()
        for n_findiff in self.tfindiff_order:
            print("--------------------")
            print(f"findiff order={n_findiff}")
            for n_tstep in self.time_step_num:
                s = self.time_step[n_tstep]
                print(f"- time_step={s}")
                samples_numerical = []
                for m in self.M:
                    print(f"-- num_faces={m.num_faces}")
                    Q = m.xyz_array
                    lapQ = m.order_p_belkin_laplacian(Q, s, n_findiff)
                    H = -np.linalg.norm(lapQ, axis=-1) / 2
                    samples_numerical.append(H)

                test_kwargs["name"] = self.test_names[(n_findiff, n_tstep)]
                test_kwargs["samples_numerical"] = samples_numerical
                test_kwargs["params"] = params = {"s": s}
                test_kwargs["data_path"] = self.test_data_paths[(n_findiff, n_tstep)]
                T = ConvergenceTestData(**test_kwargs)
                T.save()
                Tdict[(n_findiff, n_tstep)] = T
        print("Done.")
        self.Tdict = Tdict
        return Tdict

    def load_test_results(self):
        Tdict = dict()
        for test_key in self.test_keys:
            data_path = self.test_data_paths[test_key]
            Tdict[test_key] = ConvergenceTestData.load(data_path)
        self.Tdict = Tdict
        return Tdict


class SphereMcvecGuckenbergerTest:
    """
    Attributes
    ----------
    test_dir : str
        output directory to store test results
    test_data_paths : dict
        dictionary of paths to test data files
    test_names : dict
        dictionary of names for ConvergenceTestData objects
    npz_paths : list
        paths to half-edge mesh arrays used to initialized HalfEdgeMesh objects
    test_keys : list
        list of keys for test_data_paths and test_names
    M : list
        list of HalfEdgeMesh objects

    tfindiff_order : list
        order of finite difference wrt timelike parameter used in Belkin Laplacian

    """

    def __init__(
        self,
        test_dir="./output/sphere_mcvec_guckenberger_tests",
        npz_dir="./output/half_edge_arrays",
        num_vertices=[
            # 12,
            # 42,
            162,
            642,
            2562,
            10242,
            40962,
            # 163842,
            # 655362,
            # 2621442,
        ],
        load_half_edge_meshes=True,
    ):
        self.test_dir = test_dir
        self.time_step = np.array(
            [0.03853078, 0.0097707, 0.00245144, 0.00061341, 0.00015339]
        )
        self.tfindiff_order = [1, 2, 3]  # findiff order for Laplacian kernel
        self.test_keys = [n_findiff for n_findiff in self.tfindiff_order]
        self.test_data_paths = {
            key: test_dir + "/" + f"test_data_{key:03d}" + ".pkl"
            for key in self.test_keys
        }
        self.test_names = {
            key: f"sphere_mcvec_guckenberger_{key:03d}" for key in self.test_keys
        }
        self.npz_paths = [f"{npz_dir}/unit_sphere_{N:07d}.npz" for N in num_vertices]

        if load_half_edge_meshes:
            from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

            self.M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)
        else:
            self.M = None

    def run_tests(self, overwrite=False):
        make_output_dir(output_dir=self.test_dir, overwrite=overwrite)
        if self.M is None:
            from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

            self.M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)

        he_keys = [
            "xyz_coord_V",
            "h_out_V",
            "v_origin_H",
            "h_next_H",
            "h_twin_H",
            "f_left_H",
            "h_bound_F",
            "h_comp_B",
        ]
        test_kwargs = {
            "samples_actual": [-np.ones(m.num_vertices) for m in self.M],
            "independent_var": np.array([m.num_faces for m in self.M]),
            "half_edge_mesh_arrays": [
                {k: v for k, v in zip(he_keys, m.data_arrays)} for m in self.M
            ],
        }
        Tdict = dict()
        for key in self.test_keys:
            n_findiff = key
            print("--------------------")
            print(f"findiff order={n_findiff}")
            samples_numerical = []
            for m in self.M:
                print(f"-- num_faces={m.num_faces}")
                Q = m.xyz_array
                lapQ = m.order_p_guckenberger_laplacian(Q, n_findiff)
                H = -np.linalg.norm(lapQ, axis=-1) / 2
                samples_numerical.append(H)

            test_kwargs["name"] = self.test_names[key]
            test_kwargs["samples_numerical"] = samples_numerical
            # test_kwargs["params"] = params = {"s": s}
            test_kwargs["data_path"] = self.test_data_paths[key]
            T = ConvergenceTestData(**test_kwargs)
            T.save()
            Tdict[key] = T
        print("Done.")
        self.Tdict = Tdict
        return Tdict

    def load_test_results(self):
        Tdict = dict()
        for test_key in self.test_keys:
            data_path = self.test_data_paths[test_key]
            Tdict[test_key] = ConvergenceTestData.load(data_path)
        self.Tdict = Tdict
        return Tdict


class SphereCotanTest(ConvergenceTestBase):
    def __init__(
        self,
        test_dir="./output/sphere_cotan_tests",
        test_names=[
            "mean_curvature",
            "unit_normal",
            "lap_x",
            "lap_x_squared",
            "lap_exp_x_y",
            # "laplacian_mean_curvature",
            # "bending_force",
        ],
        load_half_edge_meshes=True,
        # surface/laplacian specific parameters
        npz_dir="./output/half_edge_arrays",
        num_vertices=[
            # 12,
            # 42,
            162,
            642,
            2562,
            10242,
            40962,
            # 163842,
            # 655362,
            # 2621442,
        ],
    ):
        super().__init__(
            test_dir=test_dir,
            test_names=test_names,
            load_half_edge_meshes=False,
        )
        #######################################
        # surface/laplacian specific attributes
        self.test_keys = [0]
        self.npz_paths = [f"{npz_dir}/unit_sphere_{N:07d}.npz" for N in num_vertices]
        self.laplacian_kwargs = {key: dict() for key in self.test_keys}
        #######################################
        if load_half_edge_meshes:
            self.M = self.load_meshes()
        else:
            self.M = None

    #######################################
    # laplacian specific methods
    def key2index_str(self, key):
        k = key
        return f"{k:03d}"

    def apply_laplacian(self, m, Q, *args):
        return m.cotan_laplacian(Q)

    #######################################
    # surface specific methods
    def load_meshes(self):
        print("loading half-edge meshes")
        from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

        M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=self.npz_paths)
        print("done")
        return M

    def lap_x(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return -2 * np.sin(theta) * np.cos(phi)

    def lap_x_squared(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return 6 * np.sin(phi) ** 2 * np.sin(theta) ** 2 - 6 * np.sin(theta) ** 2 + 2

    def lap_exp_x_y(self, xyz_array):
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return (
            (
                -np.sin(2 * phi) / 2
                + np.sin(2 * (phi - theta)) / 4
                + np.sin(2 * (phi + theta)) / 4
                + np.cos(2 * theta) / 2
                - np.sqrt(2) * np.cos(phi - theta + np.pi / 4)
                + np.sqrt(2) * np.cos(phi + theta + np.pi / 4)
                + 3 / 2
            )
            * np.exp(-np.sin(phi - theta) / 2 + np.sin(phi + theta) / 2)
            * np.exp(np.cos(phi - theta) / 2 - np.cos(phi + theta) / 2)
        )

    def mean_curvature_actual(self, xyz_array):
        num_vertices = xyz_array.shape[0]
        return -np.ones(num_vertices)

    def unit_normal_actual(self, xyz_array):
        num_vertices = xyz_array.shape[0]
        return xyz_array.copy()


# def run_sphere_mcvec_belkin_tests(test_dir="./output/sphere_mcvec_belkin_tests"):
#     from temp_python.src_python.jit_brane import HalfEdgeMeshBuilder

#     make_output_dir(output_dir=test_dir)
#     _NUM_VERTS_ = [
#         # 12,
#         # 42,
#         162,
#         642,
#         2562,
#         # 10242,
#         # 40962,
#         # 163842,
#         # 655362,
#         # 2621442,
#     ]
#     npz_paths = [
#         f"./output/half_edge_arrays/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_
#     ]
#     M = HalfEdgeMeshBuilder.load_test_spheres(npz_paths=npz_paths)
#     S = np.array([0.03853078, 0.0097707, 0.00245144, 0.00061341, 0.00015339])
#     Snum = [_ for _, m in enumerate(S)]
#     Hnum = [1, 2, 3]
#     Tdict = dict()
#     for heat_order in Hnum:
#         print("--------------------")
#         print(f"{heat_order=}")
#         for s_num in Snum:
#             print(f"- {s_num=}")
#             s = S[s_num]
#             name = f"sphere_mcvec_belkin_{heat_order:03d}_{s_num:03d}"
#             data_path = test_dir + "/" + name + ".pkl"
#             params = {"s": s}
#             independent_var = np.array([m.num_faces for m in M])
#             he_keys = [
#                 "xyz_coord_V",
#                 "h_out_V",
#                 "v_origin_H",
#                 "h_next_H",
#                 "h_twin_H",
#                 "f_left_H",
#                 "h_bound_F",
#                 "h_comp_B",
#             ]
#             half_edge_mesh_arrays = [
#                 {k: v for k, v in zip(he_keys, m.data_arrays)} for m in M
#             ]
#             samples_numerical = []
#             samples_actual = []
#             Nsamps = len(M)
#             for n in range(Nsamps):
#                 m = M[n]
#                 num_faces = m.num_faces
#                 print(f"-- {num_faces=}")
#                 Q = m.xyz_array
#                 lapQ = m.order_p_belkin_laplacian(Q, s, heat_order)
#                 H = -np.linalg.norm(lapQ, axis=-1) / 2
#                 samples_numerical.append(H)
#                 samples_actual.append(-np.ones_like(H))
#             test_kwargs = {
#                 "name": name,
#                 "samples_numerical": samples_numerical,
#                 "samples_actual": samples_actual,
#                 "independent_var": independent_var,
#                 "params": params,
#                 # "fun_actual": lambda xyz: -1.0,
#                 "half_edge_mesh_arrays": half_edge_mesh_arrays,
#                 "data_path": data_path,
#             }
#             T = ConvergenceTestData(**test_kwargs)
#             T.save()
#             Tdict[(heat_order, s_num)] = T
#     print("Done.")
#     return Tdict


# def load_sphere_mcvec_belkin_test_results(
#     test_dir="./output/sphere_mcvec_belkin_tests",
# ):
#     S = np.array([0.03853078, 0.0097707, 0.00245144, 0.00061341, 0.00015339])
#     Snum = [_ for _, m in enumerate(S)]
#     Hnum = [1, 2, 3]
#     Tdict = dict()
#     for heat_order in Hnum:
#         # print("--------------------")
#         # print(f"{heat_order=}")
#         for s_num in Snum:
#             # print(f"- {s_num=}")
#             name = f"sphere_mcvec_belkin_{heat_order:03d}_{s_num:03d}"
#             data_path = test_dir + "/" + name + ".pkl"
#             Tdict[(heat_order, s_num)] = ConvergenceTestData.load(data_path)
#     # print("Done.")
#     return Tdict
