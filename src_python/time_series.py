import struct
import numpy as np
import matplotlib.pyplot as plt


def read_time_series(filepath, verbose=False):
    """
    Read output files (.dat) from rigid spindle sims.

    Args
    ----
        filepath (str) : path to .dat file
    Returns
    -------
        ndarray : time series

    """

    try:
        with open(filepath, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            rows = struct.unpack("<q", f.read(8))[0]
            cols = struct.unpack("<q", f.read(8))[0]
            data = np.fromfile(f, dtype=np.float64)
        # reshape into (n, rows, cols) because we wrote row-major blocks
        data = data.reshape((n, rows, cols))
        # data[i] is the Samples2d for frame i
        if cols == 1:
            data = data.reshape((n, rows))
        elif rows == 1:
            data = data.reshape((n, cols))
        else:
            data = data.reshape((n, rows, cols))
        if verbose:
            print(f"Opened array-valued time series {filepath}")
        return data

    except ValueError:

        with open(filepath, "rb") as f:
            # Read the size of the vector (size_t is typically 8 bytes on 64-bit systems)
            size = np.fromfile(f, dtype=np.uint64, count=1)[0]
            # Read the data
            data = np.fromfile(f, dtype=np.float64, count=size)
        if verbose:
            print(f"Opened scalar-valued time series {filepath}")
        return data


def round_to(x, n=3):
    if x == 0:
        return 0.0
    else:
        return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def log_log_fit(X, Y):
    """
    Computes linear best fit for log(X)-log(Y)
    """
    logX, logY = np.log(X), np.log(Y)
    a11 = logX @ logX
    a12 = sum(logX)
    a21 = a12
    a22 = len(logX)
    u1 = logX @ logY
    u2 = sum(logY)
    u = np.array([u1, u2])
    detA = a11 * a22 - a12 * a21
    Ainv = np.array([[a22, -a12], [-a21, a11]]) / detA
    m, b = Ainv @ u
    F = m * logX + b
    fun = lambda x: np.exp(b) * x**m
    return {
        "logX": logX,
        "logY": logY,
        "slope": m,
        "intercept": b,
        "fit_samples": F,
        "fit_fun": fun,
    }


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
