# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import colormaps as plt_cmap
import numpy as np
from temp_python.src_python.utilities.misc_utils import log_log_fit, round_to
import os
import subprocess
from sympy import latex

RGBA_DICT = {
    "black": (0.0, 0.0, 0.0, 1.0),
    "white": (1.0, 1.0, 1.0, 1.0),
    "transparent": (0.0, 0.0, 0.0, 0.0),
    "red": (0.8392, 0.1529, 0.1569, 1.0),
    "red10": (0.8392, 0.1529, 0.1569, 0.1),
    "red20": (0.8392, 0.1529, 0.1569, 0.2),
    "red50": (0.8392, 0.1529, 0.1569, 0.5),
    "red80": (0.8392, 0.1529, 0.1569, 0.8),
    "green": (0.0, 0.6745, 0.2784, 1.0),
    "green10": (0.0, 0.6745, 0.2784, 0.1),
    "green20": (0.0, 0.6745, 0.2784, 0.2),
    "green30": (0.0, 0.6745, 0.2784, 0.3),
    "green40": (0.0, 0.6745, 0.2784, 0.4),
    "green50": (0.0, 0.6745, 0.2784, 0.5),
    "green80": (0.0, 0.6745, 0.2784, 0.8),
    "blue": (0.0, 0.4471, 0.6980, 1.0),
    "blue10": (0.0, 0.4471, 0.6980, 0.1),
    "blue20": (0.0, 0.4471, 0.6980, 0.2),
    "blue50": (0.0, 0.4471, 0.6980, 0.5),
    "blue80": (0.0, 0.4471, 0.6980, 0.8),
    "yellow": (1.0, 0.8431, 0.0, 1.0),
    "yellow10": (1.0, 0.8431, 0.0, 0.1),
    "yellow20": (1.0, 0.8431, 0.0, 0.2),
    "yellow50": (1.0, 0.8431, 0.0, 0.5),
    "yellow80": (1.0, 0.8431, 0.0, 0.8),
    "cyan": (0.0, 0.8431, 0.8431, 1.0),
    "cyan10": (0.0, 0.8431, 0.8431, 0.1),
    "cyan20": (0.0, 0.8431, 0.8431, 0.2),
    "cyan50": (0.0, 0.8431, 0.8431, 0.5),
    "cyan80": (0.0, 0.8431, 0.8431, 0.8),
    "magenta": (0.8784, 0.0, 0.8784, 1.0),
    "magenta10": (0.8784, 0.0, 0.8784, 0.1),
    "magenta20": (0.8784, 0.0, 0.8784, 0.2),
    "magenta50": (0.8784, 0.0, 0.8784, 0.5),
    "magenta80": (0.8784, 0.0, 0.8784, 0.8),
    "orange": (1.0, 0.5490, 0.0, 1.0),
    "orange10": (1.0, 0.5490, 0.0, 0.1),
    "orange20": (1.0, 0.5490, 0.0, 0.2),
    "orange50": (1.0, 0.5490, 0.0, 0.5),
    "orange80": (1.0, 0.5490, 0.0, 0.8),
    "purple": (0.5804, 0.0, 0.8275, 1.0),
    "purple10": (0.5804, 0.0, 0.8275, 0.1),
    "purple20": (0.5804, 0.0, 0.8275, 0.2),
    "purple30": (0.5804, 0.0, 0.8275, 0.3),
    "purple40": (0.5804, 0.0, 0.8275, 0.4),
    "purple50": (0.5804, 0.0, 0.8275, 0.5),
    "purple60": (0.5804, 0.0, 0.8275, 0.6),
    "purple70": (0.5804, 0.0, 0.8275, 0.7),
    "purple80": (0.5804, 0.0, 0.8275, 0.8),
    "purple90": (0.5804, 0.0, 0.8275, 0.9),
    "meshbrane_red": (0.7057, 0.0156, 0.1502, 1.0),
    "meshbrane_orange": (1.0, 0.498, 0.0, 1.0),
    "meshbrane_green": (0.0, 0.63335, 0.05295, 0.65),
    "meshbrane_blue": (0.0, 0.4471, 0.6980, 1.0),
}
MATPLOTLIB_COLORS = (
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
    "s",
    "^",
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
MATPLOTLIB_LINESTYLES = (
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
)
MATPLOTLIB_CMAPS = (
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


def eq_tex_str(lhs, rhs, mode="inline"):
    if mode == "inline":
        tex_str = r"$" + latex(lhs) + " = " + latex(rhs) + r"$"
    elif mode == "plain":
        tex_str = latex(lhs) + " = " + latex(rhs)
    elif mode == "equation":
        tex_str = (
            r"\begin{equation}" + latex(lhs) + " = " + latex(rhs) + r"\end{equation}"
        )
    elif mode == "equation*":
        tex_str = (
            r"\begin{equation*}" + latex(lhs) + " = " + latex(rhs) + r"\end{equation*}"
        )

    return tex_str


def get_plotsize(fig_cols=1, fig_frac=0.25):
    """
    prl column width = 3.375 inches
    fig_cols: 1 or 2 columns of fig in paper
    fig_frac: ratio of plot with to total figure width
    """
    # col_width = 3.375  # prl column width
    col_width = 6.75
    plot_width = fig_cols * col_width * fig_frac
    return plot_width


def plot_log_log_fit(
    X,
    Y,
    Xlabel="X",
    Ylabel="Y",
    title="log-log fit",
    show=True,
    fig_path=None,
    rcparams=None,
):
    """
    Make a log-log plot of X vs Y and fit a power law to the data.
    """
    rcparams0 = dict(plt.rcParams)  # save original rcparams
    if rcparams is None:
        rcparams = {
            "font.size": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
        }
    plt.rcParams.update(rcparams)
    fit = log_log_fit(X, Y)
    x = fit["logX"]
    y = fit["logY"]
    p = round_to(fit["slope"], n=3)
    f = fit["fit_samples"]

    fit_label = r"$" + Ylabel + r"=O\left(" + Xlabel + r"^{" + f"{p}" + r"}\right)$"
    plt.plot(
        x,
        f,
        label=fit_label,
        linewidth=3.5,
    )
    plt.plot(x, y, "*", markersize=10)
    plt.title(title, fontsize=16)
    plt.xlabel(f"log({Xlabel})", fontsize=16)
    plt.ylabel(f"log({Ylabel})", fontsize=16)
    plt.legend()

    if fig_path is not None:
        plt.savefig(fig_path)
    if show:
        plt.show()
    plt.close()
    plt.rcParams.update(rcparams0)  # restore original rcparams


def movie(
    image_dir,
    image_format="png",
    image_prefix="frame",
    index_length=6,
    movie_name="movie",
    movie_dir=None,
    movie_format="mp4",
):
    """
    Create a movie from a sequence of images in a directory using ffmpeg.
    Images must be named with a common prefix followed by a zero-padded index.
    The movie is saved in the same directory as the images by default.
    """
    image_name = f"{image_prefix}_%0{index_length}d.{image_format}"
    ###############################################################
    image_path = os.path.join(image_dir, image_name)
    if movie_dir is None:
        movie_dir = image_dir
    movie_path = os.path.join(movie_dir, f"{movie_name}.{movie_format}")
    ###############################################################
    wkdir = image_dir
    relative_movie_path = os.path.relpath(movie_path, wkdir)
    relative_image_path = os.path.relpath(image_path, wkdir)
    ###############################################################
    run_command = [
        "ffmpeg",
        # overwrite output file without asking if it already exists
        "-y",
        # frame rate (Hz)
        "-r",
        "20",
        # frame size (width x height)
        "-s",
        "1080x720",
        # input files
        "-i",
        relative_image_path,
        # video codec
        "-vcodec",
        "libx264",
        # video quality, lower means better
        "-crf",
        "25",
        # pixel format
        "-pix_fmt",
        "yuv420p",
        # output file
        relative_movie_path,
    ]

    # Start the process
    process = subprocess.Popen(
        run_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=image_dir,
    )
    for line in iter(process.stdout.readline, b""):
        print(line.decode(), end="")
    process.stdout.close()
    process.wait()
    print(f"Movie saved at {movie_path}")
    return movie_path
