# import matplotlib.pyplot as plt
# from src.python.utilities import round_to
# from src.python.combinatorics import argsort
from src.python.complexity import ComplexityTest
import numpy as np

ps = 3
pt = 2


def fun(n):
    # Step 1: Create a 2D array of size n x n, contributing to O(n^2) space complexity
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    # Step 2: Perform a triple nested loop, contributing to O(n^3) time complexity
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Example operation: Update matrix[i][j] based on some computation involving i, j, and k
                matrix[i][j] += i + j + k

    # The function doesn't need to return anything for this example
    # This is just to demonstrate the complexities
    return matrix


H = [2**k for k in range(3, 8)]
2 ** np.array(H)
fun_input = [(h,) for h in H]
ct = ComplexityTest(
    H,
    fun_input,
    fun,
    output_dir="./output/complexity_test",
    overwrite_output=True,
    comment="",
    time=True,
    memory=True,
    indep_var_name="h",
)
ct.run(plot=False)

# %%
ct.comparison_funs.keys()
a = ct.bigOplot(comparisons=["log(n)", "n", "n*log(n)", "2^n"])
a
