import tracemalloc
from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
from temp_python.src_python.utilities.misc_utils import round_to
from temp_python.src_python.combinatorics import argsort


def bounded_factorial(n, bound=int(2e4)):
    if n > bound:
        return np.math.factorial(int(bound))
    else:
        return np.math.factorial(int(n))


def bounded_2pow(n, bound=int(20)):
    if n > bound:
        return 2**bound
    else:
        return 2**n


def factorial_compare(n):
    if hasattr(n, "__len__"):
        return np.array([bounded_factorial(_) for _ in n])
    return bounded_factorial(n)


def twopow_compare(n):
    if hasattr(n, "__len__"):
        return np.array([bounded_2pow(_) for _ in n])
    return bounded_2pow(n)


def comparison_fun_to_log_log(Xin, f):
    X = np.array(sorted(Xin))
    if np.min(X) <= 0:
        X = X - np.min(X) + 1
    # Y = [Yin[i] for i in argsort(Xin)]
    # if Y[0] <= 0:
    #     Y = Y - np.min(Y) + 1
    logX = np.log(X)
    fX = f(X)
    if np.min(fX) <= 0:
        fX = fX - np.min(fX) + 1
    logY = np.log(fX)
    # max_logY = np.max([ly for ly in logY if all([ly != -np.inf, ly != np.inf])])
    # min_logY = np.min([ly for ly in logY if all([ly != -np.inf, ly != np.inf])])
    # for i in range(len(logY)):
    #     if logY[i] == -np.inf:
    #         logY[i] = min_logY
    #     if logY[i] == np.inf:
    #         logY[i] = max_logY
    return logY - np.min(logY)


class ComplexityTest:
    """
    Args:
        H (list): List of values to analyze time/mem against (e.g. input_sizes).
        fun_input (list): List of tuples. Each tuple contains args for fun().
        fun (function): The function to test.
        output_dir (str): The directory to save the output files.
        overwrite_output (bool): If True, overwrite the output directory if it already exists.
        comment (str): A comment to add to the output files.
        time (bool): If True, measure the run time.
        memory (bool): If True, measure the peak memory usage.
    """

    comparison_funs = {
        "log(n)": lambda n: np.log(n),
        "n": lambda n: n,
        "n*log(n)": lambda n: n * np.log(n),
        "n^2": lambda n: n**2,
        "n^3": lambda n: n**3,
        "2^n": twopow_compare,
        "n!": factorial_compare,
    }
    # comparison_funs = {
    #     key: lambda n: comparison_fun_to_log_log(n, f)
    #     for key, f in comparison_funs0.items()
    # }

    comparison_funs_bigO_tex = {
        "log(n)": lambda n: f"$O\\left(\\log {n}\\right)$",
        "n": lambda n: f"$O\\left({n}\\right)$",
        "n*log(n)": lambda n: f"$O\\left({n} \\log {n}\\right)$",
        "n^2": lambda n: f"$O\\left({n}" + "^{2}\\right)$",
        "n^3": lambda n: f"$O\\left({n}" + "^{3}\\right)$",
        "2^n": lambda n: "$O\\left(2^{" f"{n}" + "}\\right)$",
        "n!": lambda n: "$O\\left(" f"{n}" + "!\\right)$",
    }

    def __init__(
        self,
        H,
        fun_input,
        fun,
        output_dir="./output/complexity_test",
        overwrite_output=False,
        comment="",
        time=True,
        memory=True,
        indep_var_name="h",
        fun_name="fun_name",
    ):
        self.n_inputs = len(H)
        self.H = H
        self.indep_var_name = indep_var_name
        self.fun_input = fun_input
        self.fun = fun
        self.output_dir = output_dir
        self.fun_outputs = []
        self.T = []
        self.M = []
        if os.path.exists(output_dir):
            if overwrite_output:
                os.system(f"rm -r {output_dir}")
            else:
                raise Exception(
                    "Output directory already exists. Set overwrite_output=True to overwrite it."
                )
        os.system(f"mkdir -p {output_dir}")
        with open(f"{output_dir}/results.txt", "w") as f:
            f.write("ComplexityTest\n")
            f.write(f"start_comment\n")
            f.write(f"{comment}\n")
            f.write(f"end_comment\n")
            f.write(f"fun {fun_name}\n")
            f.write(f"n_inputs {self.n_inputs}\n")
            format_str = "independent_variable"
            if time:
                format_str += " run_time"
            if memory:
                format_str += " peak_mem_usage"
            f.write(format_str + "\n")
            f.write("end_header\n")

    @property
    def data(self):
        return (
            np.array(self.H),
            np.array(self.T),
            np.array(self.M),
        )

    def run(self, plot=True):

        for input_num in range(self.n_inputs):
            print(f"Running input {input_num+1} of {self.n_inputs}")
            h = self.H[input_num]

            tracemalloc.start()
            t = time()
            fun_output = self.fun(*self.fun_input[input_num])
            t = time() - t
            _, m = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"({self.indep_var_name}, run_time, peak_memory)=({h}, {t}, {m})")

            self.fun_outputs.append(fun_output)
            self.T.append(t)
            self.M.append(m)
            with open(f"{self.output_dir}/results.txt", "a") as f:
                f.write(f"{h} {t} {m}\n")

        print("Done.")
        if plot:
            self.bigOplot(
                show=True,
                save=True,
                comparisons=["log(n)", "n", "n*log(n)", "n^2", "n^3", "2^n", "n!"],
            )

    def bigOplot(self, show=True, save=False, fig_path=None, comparisons=[]):
        fontsize = 16
        marker_size = 10
        linewidth = 3.5

        H, T, M = self.data
        t_fit = log_log_fit(H, T, Xname=self.indep_var_name)
        m_fit = log_log_fit(H, M, Xname=self.indep_var_name)
        # comparison_fits = {
        #     key: log_log_fit(H, self.comparison_funs[key](H), Xname=self.indep_var_name)
        #     for key in comparisons
        # }
        comparison_vals = {
            key: comparison_fun_to_log_log(H, self.comparison_funs[key])
            for key in comparisons
        }

        fig = plt.figure(figsize=(10, 5))
        ax0 = fig.add_subplot(1, 2, 1)
        ax0.set_title("Time Complexity")
        ax0.set_xlabel(f"{self.indep_var_name}", fontsize=fontsize)
        ax0.set_ylabel("time", fontsize=fontsize)
        ax0.plot(
            # t_fit["logX"]-t_fit["logX"][0],
            t_fit["logY"] - t_fit["logY"][0],
            "o",
            markersize=marker_size,
        )
        ax0.plot(
            # t_fit["logX"],
            t_fit["logY_pred"] - t_fit["logY_pred"][0],
            label=t_fit["bigO"],
            linewidth=linewidth,
        )
        for key in comparisons:
            ax0.plot(
                # comparison_fits[key]["logY"] - comparison_fits[key]["logY"][0],
                comparison_vals[key],
                "--",
                label=self.comparison_funs_bigO_tex[key](self.indep_var_name),
                linewidth=linewidth,
            )
        ax0.legend()

        dy = t_fit["logY"][-1] - t_fit["logY"][0]
        ax0.set_xticks([])
        ax0.set_ylim(-0.1 * dy, dy * 1.1)

        ax1 = fig.add_subplot(1, 2, 2)
        ax1.set_title("Space Complexity")
        ax1.set_xlabel(f"{self.indep_var_name}", fontsize=fontsize)
        ax1.set_ylabel("memory", fontsize=fontsize)
        ax1.plot(
            m_fit["logX"],
            m_fit["logY"],
            "o",
            markersize=marker_size,
        )
        ax1.plot(
            m_fit["logX"],
            m_fit["logY_pred"],
            label=m_fit["bigO"],
            linewidth=linewidth,
        )
        # for key in comparisons:
        #     ax1.plot(
        #         comparison_fits[key]["logX"],
        #         comparison_fits[key]["logY"],
        #         "--",
        #         label=f"{key}",
        #         linewidth=linewidth,
        #     )
        ax1.legend()

        plt.tight_layout()
        if save:
            if fig_path is None:
                fig_path = f"{self.output_dir}/complexity_bigO_plot.png"
            plt.savefig(fig_path)
        if show:
            plt.show()
        plt.close()
        return comparison_vals


def log_log_fit(Xin, Yin, Xname="h", Yname="t", pow_round=3):
    X = np.array(sorted(Xin))
    Y = np.array([Yin[i] for i in argsort(Xin)])
    if np.min(X) <= 0:
        X = X - np.min(X) + 1
    if np.min(Y) <= 0:
        Y = Y - np.min(Y) + 1
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
    slope, intercept = Ainv @ u

    logY_pred = slope * logX + intercept

    mean_logY = np.mean(logY)
    SST = np.sum((logY - mean_logY) ** 2)
    SSR = np.sum((logY - logY_pred) ** 2)
    R_squared = 1 - (SSR / SST)
    MSE = SSR / len(logY)
    RMSE = np.sqrt(MSE)
    bigO = f"$O\\left({Xname}" + "^{" + f"{round_to(slope,n=pow_round)}" + "}\\right)$"
    return {
        "logX": logX,
        "logY": logY,
        "slope": slope,
        "intercept": intercept,
        "logY_pred": logY_pred,
        "RMSE": RMSE,
        "R_squared": R_squared,
        "bigO": bigO,
    }


def estimate_power_law(Hin, Yin, inv):
    H = sorted(Hin, reverse=True)
    Y = [Yin[i] for i in argsort(Hin, reverse=True)]
