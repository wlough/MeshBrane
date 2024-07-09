import sympy as sp
from sympy.core.symbol import uniquely_named_symbol
from sympy.abc import a, b, c, i, j, k, l, m, n, s, t, u, v, w, x, y, z
from sympy.printing.latex import (
    accepted_latex_functions,
    tex_greek_dictionary,
    modifier_dict,
    greek_letters_set,
)
from sympy.printing.conventions import split_super_sub

from sympy import Lambda
from sympy.sets.sets import Interval, ProductSet, imageset, simplify_union, simplify_intersection

X = sp.Interval(a, b, left_open=True)
Y = sp.Interval(a, 3 * b, left_open=True, right_open=True)

X.closure
X.left_open
X.interior.subs({"a": 5, "b": 77})
X.n(5)
dir(X)
XY = sp.ProductSet(X, Y)
xvalue_at_index = sp.Lambda(x, 3 * x + y)

sp.ProductSet(I, J)

p = uniquely_named_symbol(a, sp.sympify("a+a0+a1+a2+a3+a4+a5+a6+a7+a8+a9"))
sp.Range(x).subs({"x": 5})
x = sp.Symbol("x", latex="$\\displaystyle \\xi$")
sp.latex("x", mode="plain")

split_super_sub("ijk")


# %%
