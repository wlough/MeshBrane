import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_cmaps
import numpy as np

MATPLOTLIB_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
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
)
MATPLOTLIB_MARKERS = (
    "o",
    "^",
    "s",
    "D",
    "v",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "d",
    "|",
    "_",
)
MATPLOTLIB_HATCHSTYLES = (
    "/",
    ".",
    "\\",
    "O",
    "x",
    "*",
    "+",
    "|",
    "-",
    "O",
    "/o",
    "\\|",
    "|*",
    "-\\",
    "+o",
    "x*",
    "o-",
    "O|",
    "O.",
    "*-",
)
MATPLOTLIB_LINESTYLES = (
    "-",
    "--",
    ":",
    "-.",
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
)
MATPLOTLIB_COLORMAPS = (
    "magma",
    "inferno",
    "plasma",
    "viridis",
    "cividis",
    "twilight",
    "twilight_shifted",
    "turbo",
    "berlin",
    "managua",
    "vanimo",
    "Blues",
    "BrBG",
    "BuGn",
    "BuPu",
    "CMRmap",
    "GnBu",
    "Greens",
    "Greys",
    "OrRd",
    "Oranges",
    "PRGn",
    "PiYG",
    "PuBu",
    "PuBuGn",
    "PuOr",
    "PuRd",
    "Purples",
    "RdBu",
    "RdGy",
    "RdPu",
    "RdYlBu",
    "RdYlGn",
    "Reds",
    "Spectral",
    "Wistia",
    "YlGn",
    "YlGnBu",
    "YlOrBr",
    "YlOrRd",
    "afmhot",
    "autumn",
    "binary",
    "bone",
    "brg",
    "bwr",
    "cool",
    "coolwarm",
    "copper",
    "cubehelix",
    "flag",
    "gist_earth",
    "gist_gray",
    "gist_heat",
    "gist_ncar",
    "gist_rainbow",
    "gist_stern",
    "gist_yarg",
    "gnuplot",
    "gnuplot2",
    "gray",
    "hot",
    "hsv",
    "jet",
    "nipy_spectral",
    "ocean",
    "pink",
    "prism",
    "rainbow",
    "seismic",
    "spring",
    "summer",
    "terrain",
    "winter",
    "Accent",
    "okabe_ito",
    "Dark2",
    "Paired",
    "Pastel1",
    "Pastel2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "grey",
    "gist_grey",
    "gist_yerg",
    "Grays",
    "magma_r",
    "inferno_r",
    "plasma_r",
    "viridis_r",
    "cividis_r",
    "twilight_r",
    "twilight_shifted_r",
    "turbo_r",
    "berlin_r",
    "managua_r",
    "vanimo_r",
    "Blues_r",
    "BrBG_r",
    "BuGn_r",
    "BuPu_r",
    "CMRmap_r",
    "GnBu_r",
    "Greens_r",
    "Greys_r",
    "OrRd_r",
    "Oranges_r",
    "PRGn_r",
    "PiYG_r",
    "PuBu_r",
    "PuBuGn_r",
    "PuOr_r",
    "PuRd_r",
    "Purples_r",
    "RdBu_r",
    "RdGy_r",
    "RdPu_r",
    "RdYlBu_r",
    "RdYlGn_r",
    "Reds_r",
    "Spectral_r",
    "Wistia_r",
    "YlGn_r",
    "YlGnBu_r",
    "YlOrBr_r",
    "YlOrRd_r",
    "afmhot_r",
    "autumn_r",
    "binary_r",
    "bone_r",
    "brg_r",
    "bwr_r",
    "cool_r",
    "coolwarm_r",
    "copper_r",
    "cubehelix_r",
    "flag_r",
    "gist_earth_r",
    "gist_gray_r",
    "gist_heat_r",
    "gist_ncar_r",
    "gist_rainbow_r",
    "gist_stern_r",
    "gist_yarg_r",
    "gnuplot_r",
    "gnuplot2_r",
    "gray_r",
    "hot_r",
    "hsv_r",
    "jet_r",
    "nipy_spectral_r",
    "ocean_r",
    "pink_r",
    "prism_r",
    "rainbow_r",
    "seismic_r",
    "spring_r",
    "summer_r",
    "terrain_r",
    "winter_r",
    "Accent_r",
    "okabe_ito_r",
    "Dark2_r",
    "Paired_r",
    "Pastel1_r",
    "Pastel2_r",
    "Set1_r",
    "Set2_r",
    "Set3_r",
    "tab10_r",
    "tab20_r",
    "tab20b_r",
    "tab20c_r",
    "grey_r",
    "gist_grey_r",
    "gist_yerg_r",
    "Grays_r",
)


def get_plt_combos(n):
    """
    Returns a list of n combinations of matplotlib color, marker, and linestyle.
    """
    len_colors = len(MATPLOTLIB_COLORS)
    len_markers = len(MATPLOTLIB_MARKERS)
    len_linestyles = len(MATPLOTLIB_LINESTYLES)
    return [
        [
            MATPLOTLIB_COLORS[k % len_colors],
            MATPLOTLIB_MARKERS[k % len_markers],
            MATPLOTLIB_LINESTYLES[k % len_linestyles],
        ]
        for k in range(n)
    ]


def get_crange(samples, Nstd=2):
    """
    Clip the samples to remain within Nstd standard deviations of the mean.
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

    'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight',
    'twilight_shifted', 'turbo', 'berlin', 'managua', 'vanimo', 'Blues',
    'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd',
    'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples',
    'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia',
    'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone',
    'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag',
    'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow',
    'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv',
    'jet', 'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic',
    'spring', 'summer', 'terrain', 'winter', 'Accent', 'okabe_ito', 'Dark2',
    'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',
    'tab20b', 'tab20c', 'grey', 'gist_grey', 'gist_yerg', 'Grays', 'magma_r',
    'inferno_r', 'plasma_r', 'viridis_r', 'cividis_r', 'twilight_r',
    'twilight_shifted_r', 'turbo_r', 'berlin_r', 'managua_r', 'vanimo_r',
    'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r', 'GnBu_r', 'Greens_r',
    'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r',
    'PuOr_r', 'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r',
    'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r',
    'YlOrBr_r', 'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r',
    'brg_r', 'bwr_r', 'cool_r', 'coolwarm_r', 'copper_r', 'cubehelix_r',
    'flag_r', 'gist_earth_r', 'gist_gray_r', 'gist_heat_r', 'gist_ncar_r',
    'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r',
    'gray_r', 'hot_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'ocean_r',
    'pink_r', 'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r',
    'terrain_r', 'winter_r', 'Accent_r', 'okabe_ito_r', 'Dark2_r', 'Paired_r',
    'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r',
    'tab20_r', 'tab20b_r', 'tab20c_r', 'grey_r', 'gist_grey_r', 'gist_yerg_r',
    'Grays_r'
    """
    if cmin >= cmax:
        raise ValueError("cmax must be greater than cmin")
    cmap01 = mpl_cmaps[name]  # cmap on [0, 1]
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
