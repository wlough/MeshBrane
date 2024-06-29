import matplotlib.pyplot as plt
import numpy as np
from src.python.utilities import round_to


def get_plotsize(fig_cols=1, fig_frac=0.25):
    """
    prl column width = 3.375 inches
    fig_cols: 1 or 2 columns of fig in paper
    fig_frac: ratio of plot with to total figure width
    """
    col_width = 3.375  # prl column width
    # col_width2 = 6.75
    plot_width = fig_cols * col_width * fig_frac
    return plot_width


def log_log_fit(X, Y, Xlabel="X", Ylabel="Y", title="", show=True, fig_path=None):
    x, y = np.log(X), np.log(Y)
    a11 = x @ x
    a12 = sum(x)
    a21 = a12
    a22 = len(x)
    u1 = x @ y
    u2 = sum(y)
    u = np.array([u1, u2])
    detA = a11 * a22 - a12 * a21
    Ainv = np.array([[a22, -a12], [-a21, a11]]) / detA
    p, c = Ainv @ u
    f = p * x + c
    fit_label = f"${Ylabel}=O\\left({Xlabel}" + "^{" + f"{round_to(p,n=3)}" + "}\\right)$"
    # fit_label = f"${Ylabel}\\sim{round_to(np.exp(c),n=3)}{Xlabel}" + "^{" + f"{round_to(p,n=3)}" + "}$"
    # fit_label = f"${round_to(c,n=3)}{Xlabel}" + "^{" + f"{round_to(p,n=3)}" + "}$"
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
