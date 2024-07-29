import itertools
from matplotlib import colormaps as plt_cmap
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
from src.python.ply_tools import VertTri2HalfEdgeConverter
from src.python.utilities import round_to, log_log_fit

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


######################################
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


######################################
class HalfEdgeTestSurf(HalfEdgeMesh):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        *surface_params,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
        self.surface_params = [*surface_params]
        # self.surfcoord_array = self.compute_surfcoord_from_xyz()
        # self.mean_curvature = self.compute_mean_curvature()
        # self.gaussian_curvature = self.compute_gaussian_curvature()
        # self.unit_normal = self.compute_unit_normal()
        # self.mcvec_actual = np.einsum(
        #     "v,vi->vi", 2 * self.mean_curvature, self.unit_normal
        # )

    #######################################################
    # Initilization methods
    @classmethod
    def from_half_edge_ply(cls, ply_path, *surface_params):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(
            *VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples,
            *surface_params,
        )

    #######################################################
    def save(self, data_path=None):
        import pickle

        if data_path is not None:
            self.data_path = data_path
        if self.data_path is None:
            raise ValueError("No path to save data.")
        with open(self.data_path + ".pickle", "wb") as f:
            pickle.dump(self, f)

    def recompute_from_xyz(self, xyz_array=None):
        if xyz_array is None:
            xyz_array = self.xyz_array
        else:
            self.xyz_coord_V = xyz_array
        self.surfcoord_array = self.compute_surfcoord_from_xyz()
        self.mean_curvature = self.compute_mean_curvature()
        self.unit_normal = self.compute_unit_normal()

    def recompute_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        else:
            self.surfcoord_array = surfcoord_array.copy()
        self.xyz_coord_V = self.compute_xyz_from_surfcoord()
        self.mean_curvature = self.compute_mean_curvature()
        self.unit_normal = self.compute_unit_normal()

    # Perturbations and noise
    def perturbation_gaussian_xyz(self, loc=0, scale=0.01):
        """Adds a Gaussian perturbation to vertex cartesian coordinates"""
        self.xyz_coord_V = np.array(
            [
                self.xyz_coord_v(v) + np.random.normal(loc, scale, 3)
                for v in self.xyz_coord_V.keys()
            ]
        )

    def perturbation_gaussian_surfcoords(self, loc=0, scale=0.01):
        """
        Adds a Gaussian perturbation to vertex surface coordinates
        """
        self.surfcoord_array = np.array(
            [
                self.surfcoord_array[v] + np.random.normal(loc, scale, 2)
                for v in self.xyz_coord_V.keys()
            ]
        )
        # self.recompute_from_surfcoord()

    def perturbation_edge_flip(self, p=0.1):
        Nh = self.num_edges
        Hlist = list(self._v_origin_H.keys())
        for _ in range(Nh):
            h = np.random.choice(Hlist)
            if np.random.rand() < p:
                if self.h_is_flippable(h):
                    self.flip_edge(h)

    # Laplace operators
    def cotan_laplacian(self, Y):
        """
        Computes the cotan Laplacian of Y at each vertex
        """
        Nv = self.num_vertices
        lapY = np.zeros_like(Y)
        for vi in range(Nv):
            Atot = 0.0
            ri = self.xyz_coord_v(vi)
            yi = Y[vi]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            for hij in self.generate_H_out_v_clockwise(vi):
                hijm1 = self.h_next_h(self.h_twin_h(hij))
                hijp1 = self.h_twin_h(self.h_prev_h(hij))
                vjm1 = self.v_head_h(hijm1)
                vj = self.v_head_h(hij)
                vjp1 = self.v_head_h(hijp1)

                yj = Y[vj]

                rjm1 = self.xyz_coord_v(vjm1)
                rj = self.xyz_coord_v(vj)
                rjp1 = self.xyz_coord_v(vjp1)

                rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
                rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
                rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
                ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
                ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
                rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
                ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
                rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

                Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
                Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
                Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
                Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
                Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

                cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

                cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

                cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
                cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

                Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
                lapY[vi] += (cot_thetam + cot_thetap) * (yj - yi) / 2
            lapY[vi] /= Atot

        return lapY

    def belkin_laplacian(self, s, Q):
        """
        Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
        defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
        constant timelike parameter s.
        """
        V = self.xyz_array
        A = np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()])
        lapQ = np.array(
            [
                np.einsum(
                    "y,y,y...->...",
                    A,
                    np.exp(-np.linalg.norm(V - x, axis=-1) ** 2 / (4 * s)),
                    Q - q,
                )
                for x, q in zip(V, Q)
            ]
        ) / (4 * np.pi * s**2)

        return lapQ

    def guckenberger_laplacian(self, Q):
        """
        Computes the heat kernel Laplacian of Q at each vertex using 'Method D' from
        Guckenberger et al 2016 'On the bending algorithms for soft objects in flows'.
        This is a modification of Belkin et al's which replaces the constant time-like
        parameter s with the mixed area of the dual cell at each vertex.
        """
        V = self.xyz_array
        A = np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()])
        Amixed = np.array([self.meyercell_area(v) for v in self.xyz_coord_V.keys()])
        lapQ = np.array(
            [
                np.einsum(
                    "y,y,y...->...",
                    A / (4 * np.pi * s**2),
                    np.exp(-np.linalg.norm(V - x, axis=-1) ** 2 / (4 * s)),
                    Q - q,
                )
                for x, q, s in zip(V, Q, Amixed)
            ]
        )

        return lapQ

    def tdiff_laplacian(self, s, Q):
        """
        Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
        defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
        constant timelike parameter s.
        """
        V = self.xyz_array
        A = np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()])
        lapQ = np.array(
            [
                np.einsum(
                    "y,y,y...->...",
                    A,
                    np.exp(-np.linalg.norm(V - x, axis=-1) ** 2 / (4 * s)),
                    Q - q,
                )
                for x, q in zip(V, Q)
            ]
        ) / (4 * np.pi * s**2)

        return lapQ

    # Gradient operators
    def belkin_gradient(self, s, Q):
        """computes the heat kernel gradient of Q at each vertex"""
        V = self.xyz_array
        A = np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()])
        gradQ = np.array(
            [
                np.einsum(
                    "y,y,yj...->j...",
                    A,
                    np.exp(-np.linalg.norm(V - x, axis=-1) ** 2 / (4 * s)),
                    V - x,
                    Q - q,
                )
                for x, q in zip(V, Q)
            ]
        ) / (4 * np.pi * s**2)

        return gradQ

    def run_belkin_laplacian_mcvec_fixed_heat_param_test(
        self, heat_param_vals, loc=0, scale=0
    ):
        if scale != 0:
            self.perturbation_gaussian_surfcoords(loc, scale)
            self.xyz_coord_V = self.compute_xyz_from_surfcoord()
            self.mean_curvature = self.compute_mean_curvature()
            self.unit_normal = self.compute_unit_normal()
        mcvec = np.array(
            [self.belkin_laplacian(s, self.xyz_array) for s in heat_param_vals]
        )
        L2error = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel())
                / np.linalg.norm(self.mcvec_actual.ravel())
                for _ in mcvec
            ]
        )
        Linftyerror = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel(), np.inf)
                / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
                for _ in mcvec
            ]
        )
        results = {
            "heat_param": np.array(heat_param_vals),
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
            "noise_scale": scale,
        }
        self.belkin_laplacian_mcvec_fixed_heat_param_results = results
        return results

    def run_belkin_laplacian_mcvec_average_face_area_test(self, loc=0, scale=0):
        if scale != 0:
            self.perturbation_gaussian_surfcoords(loc, scale)
            self.xyz_coord_V = self.compute_xyz_from_surfcoord()
            self.mean_curvature = self.compute_mean_curvature()
            self.unit_normal = self.compute_unit_normal()
        Af = self.average_face_area()
        heat_param_vals = np.array([np.sqrt(Af), Af, Af**2])
        mcvec = np.array(
            [self.belkin_laplacian(s, self.xyz_array) for s in heat_param_vals]
        )
        L2error = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel())
                / np.linalg.norm(self.mcvec_actual.ravel())
                for _ in mcvec
            ]
        )
        Linftyerror = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel(), np.inf)
                / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
                for _ in mcvec
            ]
        )
        results = {
            "heat_param": np.array(heat_param_vals),
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
            "noise_scale": scale,
        }
        self.belkin_laplacian_mcvec_average_face_area_results = results
        return results

    def run_cotan_laplacian_mcvec_test(self, loc=0, scale=0):
        if scale != 0:
            self.perturbation_gaussian_surfcoords(loc, scale)
            self.xyz_coord_V = self.compute_xyz_from_surfcoord()
            self.mean_curvature = self.compute_mean_curvature()
            self.unit_normal = self.compute_unit_normal()
        mcvec = self.cotan_laplacian(self.xyz_array)
        L2error = np.linalg.norm((mcvec - self.mcvec_actual).ravel()) / np.linalg.norm(
            self.mcvec_actual.ravel()
        )
        Linftyerror = np.linalg.norm(
            (mcvec - self.mcvec_actual).ravel(), np.inf
        ) / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
        results = {
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
            "noise_scale": scale,
        }
        self.cotan_laplacian_mcvec_results = results
        return results

    def run_guckenberger_laplacian_mcvec_test(self, loc=0, scale=0):
        if scale != 0:
            self.perturbation_gaussian_surfcoords(loc, scale)
            self.xyz_coord_V = self.compute_xyz_from_surfcoord()
            self.mean_curvature = self.compute_mean_curvature()
            self.unit_normal = self.compute_unit_normal()
        mcvec = self.guckenberger_laplacian(self.xyz_array)
        L2error = np.linalg.norm((mcvec - self.mcvec_actual).ravel()) / np.linalg.norm(
            self.mcvec_actual.ravel()
        )
        Linftyerror = np.linalg.norm(
            (mcvec - self.mcvec_actual).ravel(), np.inf
        ) / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
        results = {
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
            "noise_scale": scale,
        }
        self.guckenberger_laplacian_mcvec_results = results
        return results

    ##########################################################
    # Overwrite these functions #
    #############################

    # Coordinate expressions/computations
    def compute_surfcoord_from_xyz(self, xyz_array=None):
        if xyz_array is None:
            xyz_array = self.xyz_array
        a, b = self.surface_params
        # xyz_array = self.xyz_array
        r = np.linalg.norm(xyz_array, axis=-1)
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        # rho = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        psi = np.arctan2(xyz_array[:, 2], rho - a)
        return np.array([phi, psi]).T

    def compute_xyz_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        x = (a + b * np.cos(psi)) * np.cos(phi)
        y = (a + b * np.cos(psi)) * np.sin(phi)
        z = b * np.sin(psi)
        return np.array([x, y, z]).T

    def compute_mean_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        return -(a + 2 * b * np.cos(psi)) / (2 * b * (a + b * np.cos(psi)))

    def compute_gaussian_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        return np.cos(psi) / (b * (a + b * np.cos(psi)))

    def compute_unit_normal(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        nx = np.cos(phi) * np.cos(psi)
        ny = np.sin(phi) * np.cos(psi)
        nz = np.sin(psi)
        return np.array([nx, ny, nz]).T

    # Tests
    # def run_laplacian_tests(self, _timelike_param):
    #     timelike_param = np.array(
    #         [
    #             *_timelike_param,
    #             self.average_face_area(),
    #             # self.average_dual_barcell_area(),
    #         ]
    #     )
    #     self.timelike_param = timelike_param
    #     self.mcvec_cotan = self.cotan_laplacian(self.xyz_array)
    #     self.mcvec_belkin = [
    #         self.belkin_laplacian(s, self.xyz_array) for s in timelike_param
    #     ]

    #     phi = np.arctan2(self.xyz_array[:, 1], self.xyz_array[:, 0])
    #     rho = np.sqrt(self.xyz_array[:, 0] ** 2 + self.xyz_array[:, 1] ** 2)
    #     a, b = 1, 1 / 3
    #     psi = np.arctan2(self.xyz_array[:, 2], rho - a)
    #     H = -(a + 2 * b * np.cos(psi)) / (2 * b * (a + b * np.cos(psi)))
    #     nx = np.cos(phi) * np.cos(psi)
    #     ny = np.sin(phi) * np.cos(psi)
    #     nz = np.sin(psi)
    #     self.mcvec_actual = np.einsum("v,iv->vi", 2 * H, np.array([nx, ny, nz]))

    #     self.mcvec_cotan_L2error = np.linalg.norm(
    #         (self.mcvec_cotan - self.mcvec_actual).ravel()
    #     ) / np.linalg.norm(self.mcvec_actual.ravel())
    #     self.mcvec_cotan_Linftyerror = np.linalg.norm(
    #         (self.mcvec_cotan - self.mcvec_actual).ravel(), np.inf
    #     ) / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)

    #     self.mcvec_belkin_L2error = [
    #         np.linalg.norm((mcvec - self.mcvec_actual).ravel())
    #         / np.linalg.norm(self.mcvec_actual.ravel())
    #         for mcvec in self.mcvec_belkin
    #     ]
    #     self.mcvec_belkin_Linftyerror = [
    #         np.linalg.norm((mcvec - self.mcvec_actual).ravel(), np.inf)
    #         / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
    #         for mcvec in self.mcvec_belkin
    #     ]

    ######################################################
    ######################################################
    # to be deprecated
    def spherical_coord_v(self, v):
        r = np.linalg.norm(self.xyz_coord_v(v))
        theta = np.arccos(self.xyz_coord_v(v)[2] / r)
        phi = np.arctan2(self.xyz_coord_v(v)[1], self.xyz_coord_v(v)[0])
        return r, theta, phi

    def spherical_coord_array(self, sorted=False):
        if sorted:
            return np.array(
                [self.spherical_coord_v(v) for v in sorted(self.xyz_coord_V.keys())]
            )
        else:
            return np.array(
                [self.spherical_coord_v(v) for v in self.xyz_coord_V.keys()]
            )

    @property
    def surfcoord1(self):
        return self.surfcoord_array[:, 0]

    @property
    def surfcoord2(self):
        return self.surfcoord_array[:, 1]

    def belkin_gradient_slow(self, s, Q):
        """computes the heat kernel gradient of Q at each vertex"""
        Nv = self.num_vertices
        Nf = self.num_faces
        Qshape = Q.shape
        is_scalar = len(Qshape) == 1
        if is_scalar:
            nabQshape = (Nv, 3)
        else:
            vec_shape = Qshape[1:]
            nabQshape = (Nv, 3, *vec_shape)
        nabQ = np.zeros(nabQshape)
        for i in range(Nv):
            x = self.xyz_coord_v(i)
            qx = Q[i]
            for f in range(Nf):
                Af = self.area_f(f)
                for j in self.generate_V_of_f(f):
                    y = self.xyz_coord_v(j)
                    qy = Q[j]
                    if is_scalar:
                        nabQ[i] += (
                            (1 / (8 * np.pi * s**2))
                            * (Af / 3)
                            * np.exp(-np.linalg.norm(x - y) ** 2 / (4 * s))
                            * (y - x)
                            * (qy - qx)
                        )
                    else:
                        nabQ[i] += np.einsum(
                            "3,...->3...",
                            (1 / (4 * np.pi * s**2))
                            * (Af / 3)
                            * np.exp(-np.linalg.norm(x - y) ** 2 / (4 * s))
                            * (y - x),
                            (qy - qx),
                        )

        return nabQ

    def belkin_laplacian_slow(self, s, Q):
        """computes the heat kernel laplacian of Q at each vertex"""
        Nv = self.num_vertices
        Nf = self.num_faces
        lapQ = np.zeros_like(Q)
        for i in range(Nv):
            x = self.xyz_coord_v(i)
            qx = Q[i]
            for f in range(Nf):
                Af = self.area_f(f)
                for j in self.generate_V_of_f(f):
                    y = self.xyz_coord_v(j)
                    qy = Q[j]
                    lapQ[i] += (
                        (1 / (4 * np.pi * s**2))
                        * (Af / 3)
                        * np.exp(-np.linalg.norm(x - y) ** 2 / (4 * s))
                        * (qy - qx)
                    )

        return lapQ

    def _run_belkin_laplacian_mcvec_fixed_param_test(self, heat_param_vals):
        mcvec = np.array(
            [self.belkin_laplacian(s, self.xyz_array) for s in heat_param_vals]
        )
        L2error = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel())
                / np.linalg.norm(self.mcvec_actual.ravel())
                for _ in mcvec
            ]
        )
        Linftyerror = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel(), np.inf)
                / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
                for _ in mcvec
            ]
        )
        results = {
            "heat_param": np.array(heat_param_vals),
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
        }
        self.belkin_laplacian_mcvec_fixed_param_results = results
        return results

    def _run_belkin_laplacian_mcvec_average_face_area_test(self):
        Af = self.average_face_area()
        heat_param_vals = np.array([Af**2, Af, np.sqrt(Af)])
        mcvec = np.array(
            [self.belkin_laplacian(s, self.xyz_array) for s in heat_param_vals]
        )
        L2error = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel())
                / np.linalg.norm(self.mcvec_actual.ravel())
                for _ in mcvec
            ]
        )
        Linftyerror = np.array(
            [
                np.linalg.norm((_ - self.mcvec_actual).ravel(), np.inf)
                / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
                for _ in mcvec
            ]
        )
        results = {
            "heat_param": np.array(heat_param_vals),
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
        }
        self.belkin_laplacian_mcvec_average_face_area_results = results
        return results

    def _run_cotan_laplacian_mcvec_test(self):
        mcvec = self.cotan_laplacian(self.xyz_array)
        L2error = np.linalg.norm((mcvec - self.mcvec_actual).ravel()) / np.linalg.norm(
            self.mcvec_actual.ravel()
        )
        Linftyerror = np.linalg.norm(
            (mcvec - self.mcvec_actual).ravel(), np.inf
        ) / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
        results = {
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
        }
        self.cotan_laplacian_mcvec_results = results
        return results

    def _run_guckenberger_laplacian_mcvec_test(self):
        mcvec = self.guckenberger_laplacian(self.xyz_array)
        L2error = np.linalg.norm((mcvec - self.mcvec_actual).ravel()) / np.linalg.norm(
            self.mcvec_actual.ravel()
        )
        Linftyerror = np.linalg.norm(
            (mcvec - self.mcvec_actual).ravel(), np.inf
        ) / np.linalg.norm(self.mcvec_actual.ravel(), np.inf)
        results = {
            "mcvec": mcvec,
            "L2error": L2error,
            "Linftyerror": Linftyerror,
        }
        self.guckenberger_laplacian_mcvec_results = results
        return results


class HalfEdgeTestSphere(HalfEdgeTestSurf):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        *surface_params,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            *surface_params,
        )
        self.surface_params = [*surface_params]
        self.surfcoord_array = self.compute_surfcoord_from_xyz()
        self.mean_curvature = self.compute_mean_curvature()
        self.gaussian_curvature = self.compute_gaussian_curvature()
        self.unit_normal = self.compute_unit_normal()
        self.mcvec_actual = np.einsum(
            "v,vi->vi", 2 * self.mean_curvature, self.unit_normal
        )

    # Coordinate expressions/computations
    def compute_surfcoord_from_xyz(self, xyz_array=None):
        if xyz_array is None:
            xyz_array = self.xyz_array
        a = self.surface_params[0]
        # r = np.linalg.norm(xyz_array, axis=-1)
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        # rho = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        theta = np.arctan2(rho, xyz_array[:, 2])
        return np.array([theta, phi]).T

    def compute_xyz_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a = self.surface_params[0]
        theta, phi = surfcoord_array.T
        x = a * np.sin(theta) * np.cos(phi)
        y = a * np.sin(theta) * np.sin(phi)
        z = a * np.cos(theta)
        return np.array([x, y, z]).T

    def compute_mean_curvature(self, surfcoord_array=None):
        # if surfcoord_array is None:
        #     surfcoord_array = self.surfcoord_array
        a = self.surface_params[0]
        return -np.ones(self.num_vertices) / a

    def compute_gaussian_curvature(self, surfcoord_array=None):
        # if surfcoord_array is None:
        #     surfcoord_array = self.surfcoord_array
        a = self.surface_params[0]
        return np.ones(self.num_vertices) / a**2

    def compute_unit_normal(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        theta, phi = surfcoord_array.T
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        return np.array([nx, ny, nz]).T


class HalfEdgeTestTorus(HalfEdgeTestSurf):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        *surface_params,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            *surface_params,
        )
        # self.recompute_from_xyz()
        self.surface_params = [*surface_params]
        self.surfcoord_array = self.compute_surfcoord_from_xyz()
        self.mean_curvature = self.compute_mean_curvature()
        self.gaussian_curvature = self.compute_gaussian_curvature()
        self.unit_normal = self.compute_unit_normal()
        self.mcvec_actual = np.einsum(
            "v,vi->vi", 2 * self.mean_curvature, self.unit_normal
        )

    # Coordinate expressions/computations
    def compute_surfcoord_from_xyz(self, xyz_array=None):
        if xyz_array is None:
            xyz_array = self.xyz_array
        a, b = self.surface_params
        xyz_array = self.xyz_array
        r = np.linalg.norm(xyz_array, axis=-1)
        phi = np.arctan2(xyz_array[:, 1], xyz_array[:, 0])
        # rho = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        rho = np.linalg.norm(xyz_array[:, :2], axis=-1)
        psi = np.arctan2(xyz_array[:, 2], rho - a)
        return np.array([phi, psi]).T

    def compute_xyz_from_surfcoord(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        x = (a + b * np.cos(psi)) * np.cos(phi)
        y = (a + b * np.cos(psi)) * np.sin(phi)
        z = b * np.sin(psi)
        return np.array([x, y, z]).T

    def compute_mean_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        return -(a + 2 * b * np.cos(psi)) / (2 * b * (a + b * np.cos(psi)))

    def compute_gaussian_curvature(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        return np.cos(psi) / (b * (a + b * np.cos(psi)))

    def compute_unit_normal(self, surfcoord_array=None):
        if surfcoord_array is None:
            surfcoord_array = self.surfcoord_array
        a, b = self.surface_params
        phi, psi = surfcoord_array.T
        nx = np.cos(phi) * np.cos(psi)
        ny = np.sin(phi) * np.cos(psi)
        nz = np.sin(psi)
        return np.array([nx, ny, nz]).T
