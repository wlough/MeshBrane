import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def get_sweep_num_rows_cols(sweep_dir):
    sweep_dir = Path(sweep_dir)
    num_rows = 0
    num_cols = 0
    while True:
        run_dir = sweep_dir / f"run_{num_rows:0>2}_{num_cols:0>2}"
        if not run_dir.exists():
            break
        num_rows += 1
    while True:
        run_dir = sweep_dir / f"run_{0:0>2}_{num_cols:0>2}"
        if not run_dir.exists():
            break
        num_cols += 1
    return num_rows, num_cols


def plot_ne_point_cloud2d_run(
    output_dir,
    t_start=0.01,
    dt_sample=0.01,
    markersize=2.5,
):
    output_dir = Path(output_dir)
    big_t = read_time_series(output_dir / "raw_data/t.dat")
    dt0 = big_t[1] - big_t[0]
    nt_skip = int(dt_sample / dt0) + 1
    nt_start = int(t_start / dt0) + 1
    big_t = big_t[nt_start::nt_skip]
    big_t -= big_t[0]
    dt = big_t[1] - big_t[0]

    big_xyz_coord_V = read_time_series(
        output_dir / "raw_data/envelope_xyz_coord_V.dat"
    )[nt_start::nt_skip]

    Nt, Nv, _ = big_xyz_coord_V.shape
    big_rhoz_coord_V = np.zeros((Nt, Nv, 2))
    big_rhoz_coord_V[:, :, 1] = big_xyz_coord_V[:, :, 2]
    big_rhoz_coord_V[:, :, 0] = np.linalg.norm(big_xyz_coord_V[:, :, :2], axis=2)

    for nt in range(Nt):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
        things = []
        pt_cloud = ax.plot(
            big_rhoz_coord_V[nt, :, 1],
            big_rhoz_coord_V[nt, :, 0],
            ".",
            markersize=markersize,
            color="blue",
        )[0]
        things.append(pt_cloud)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        ax.set_title(f"t={big_t[nt]:.2g}")
        plt.show()
        # for thing in things:
        #     thing.remove()


def plot_ne_point_cloud2d_sweep(
    sweep_dir,
    t_start=0.01,
    dt_sample=0.01,
    markersize=2.5,
    # num_points=300,
):
    sweep_dir = Path(sweep_dir)
    num_rows, num_cols = get_sweep_num_rows_cols(sweep_dir)
    run_dirs = [
        sweep_dir / f"run_{row:0>2}_{col:0>2}"
        for row in range(num_rows)
        for col in range(num_cols)
    ]

    run_names = [
        f"run_{row:0>2}_{col:0>2}" for row in range(num_rows) for col in range(num_cols)
    ]

    run_data_list = []
    for run_dir in run_dirs:
        big_t = read_time_series(run_dir / "raw_data/t.dat")
        dt0 = big_t[1] - big_t[0]
        nt_skip = int(dt_sample / dt0) + 1
        nt_start = int(t_start / dt0) + 1
        big_t = big_t[nt_start::nt_skip]
        big_t -= big_t[0]
        dt = big_t[1] - big_t[0]

        big_xyz_coord_V = read_time_series(
            run_dir / "raw_data/envelope_xyz_coord_V.dat"
        )[nt_start::nt_skip]

        Nt, Nv, _ = big_xyz_coord_V.shape
        # V = np.random.choice(np.arange(Nv), num_points)
        # big_xyz_coord_V = big_xyz_coord_V[:, V, :]
        # Nt, Nv, _ = big_xyz_coord_V.shape
        big_rhoz_coord_V = np.zeros((Nt, Nv, 2))
        big_rhoz_coord_V[:, :, 1] = big_xyz_coord_V[:, :, 2]
        big_rhoz_coord_V[:, :, 0] = np.linalg.norm(big_xyz_coord_V[:, :, :2], axis=2)

        run_data = {
            "big_t": big_t,
            "dt": dt,
            "Nt": Nt,
            "Nv": Nv,
            "big_rhoz_coord_V": big_rhoz_coord_V,
        }
        run_data_list.append(run_data)

    Nt_list = [data["Nt"] for data in run_data_list]
    Nt_max = max(Nt_list)
    run_with_max_Nt = Nt_list.index(Nt_max)
    Nruns = len(run_data_list)
    for nt0 in range(Nt_max):
        fig, axes = plt.subplots(
            nrows=Nruns,
            ncols=1,
            figsize=(6, Nruns),
            sharex=True,
        )
        t = run_data_list[run_with_max_Nt]["big_t"][nt0]
        fig.suptitle(f"t={t:.2g}")
        for run in range(Nruns):
            ax = axes[run]
            label = run_names[run]
            data = run_data_list[run]
            nt = min(nt0, data["Nt"] - 1)
            rho_coord_V = data["big_rhoz_coord_V"][nt, :, 0]
            z_coord_V = data["big_rhoz_coord_V"][nt, :, 1]
            ax.plot(
                z_coord_V,
                rho_coord_V,
                ".",
                markersize=markersize,
                # color="blue",
                label=label,
            )
            ax.legend()
            # ax.set_aspect("equal", adjustable="box")
            ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
