import sympy as sp

x, y = sp.symbols("x y")
x_range = (x, sp.sympify("-1/2"), sp.sympify("sqrt(3)/2"))
y_range = (y, sp.sympify("-1/2"), sp.sympify("sqrt(3)/2") - x)
(1 / 2 + 1) ** 2
N = sp.sqrt(27) / 4
p0 = sp.sympify(1)
p2 = sp.sympify("sqrt(5/3)") * (4 * x**2 + 4 * y**2 - 1)
p3 = sp.sympify("(2/3)**(5/2)") * (4 - 30 * x**2 + 35 * x**3 - 30 * y**2 - 105 * x * y**2)
p4 = sp.sympify("4*sqrt(7/243)") * (
    1 - 12 * (x**2 + y**2) + 36 * (x**2 + y**2) ** 2 + 16 * x * (-(x**2) + 3 * y**2)
)


def inner(p, q):
    return sp.integrate(sp.integrate(p * q, y_range), x_range)


inner(p2, p2).simplify()
