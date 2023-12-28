import numpy as np
import sympy as sp
from numba import njit
import dill

factorial_table = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype="int64",
)

ex = np.array([1.0, 0.0, 0.0])
ey = np.array([0.0, 1.0, 0.0])
ez = np.array([0.0, 0.0, 1.0])

exhat = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
eyhat = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
ezhat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


@njit
def factorial(n):
    """
    n! for n<=20
    """
    facn = 1
    for _ in range(2, n + 1):
        facn *= _
    return facn


@njit
def factorial_lookup(n):
    """
    n! for n<=20
    """
    return factorial_table[n]


def Ylm(l, m, theta, phi):
    B = (
        1j ** (m + abs(m))
        * np.sqrt((2 * l + 1) * factorial(l + m) * factorial(l - m) + 0.0)
        / (2 ** (abs(m) + 1) * np.sqrt(np.pi))
    )
    return np.exp(1j * m * phi) * sum(
        (
            (-1) ** k
            * B
            / (
                4**k
                * factorial(l - abs(m) - 2 * k)
                * factorial(k + abs(m))
                * factorial(k)
            )
        )
        * np.cos(theta) ** (l - abs(m) - 2 * k)
        * np.sin(theta) ** (abs(m) + 2 * k)
        for k in range(1 + int((l - abs(m)) / 2))
    )


@njit("f8[:,:](f8[:])")
def exp_so3(angles):
    """..."""
    theta1, theta2, theta3 = angles
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        A = 1.0 - theta**2 / 6 + theta**4 / 120 - theta**6 / 540
        B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720 - theta**6 / 40320
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
    # theta_hat = np.array(
    #     [[0.0, -theta3, theta2], [theta3, 0.0, -theta1], [-theta2, theta1, 0.0]]
    # )
    # theta_hat_squared = np.array(
    #     [
    #         [-(theta2**2) - theta3**2, theta1 * theta2, theta1 * theta3],
    #         [theta1 * theta2, -(theta1**2) - theta3**2, theta2 * theta3],
    #         [theta1 * theta3, theta2 * theta3, -(theta1**2) - theta2**2],
    #     ]
    # )
    exp_theta = np.array(
        [
            [1.0, -A * theta3, A * theta2],
            [A * theta3, 1.0, -A * theta1],
            [-A * theta2, A * theta1, 1.0],
        ]
    )
    exp_theta += B * np.array(
        [
            [-(theta2**2) - theta3**2, theta1 * theta2, theta1 * theta3],
            [theta1 * theta2, -(theta1**2) - theta3**2, theta2 * theta3],
            [theta1 * theta3, theta2 * theta3, -(theta1**2) - theta2**2],
        ]
    )
    return exp_theta


@njit("f8[:](f8[:,:])")
def log_so3(R):
    """Returns the components of log(R) in the canonical basis for Lie algebra so(3)"""
    # Ensure the matrix is square
    assert R.shape[0] == R.shape[1] == 3, "Matrix must be 3x3"

    # Compute the value to be passed to np.arccos
    cos_theta = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2

    # Ensure the value is within the valid range [-1, 1]
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1

    # Compute the rotation angle
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        theta_over_two_sin_theta = (
            0.5 + theta**2 / 12 + 7 * theta**4 / 720 + 31 * theta**6 / 30240
        )
    else:
        theta_over_two_sin_theta = 0.5 * theta / np.sin(theta)

    # R_RT_x = -R[1, 2] + R[2, 1]
    # R_RT_y = R[0, 2] - R[2, 0]
    # R_RT_z = -R[0, 1] + R[1, 0]
    thetax = (-R[1, 2] + R[2, 1]) * theta_over_two_sin_theta
    thetay = (R[0, 2] - R[2, 0]) * theta_over_two_sin_theta
    thetaz = (-R[0, 1] + R[1, 0]) * theta_over_two_sin_theta

    return np.array([thetax, thetay, thetaz])


@njit("f8[:,:](f8[:])")
def quaternion_to_matrix(q):
    qw, qx, qy, qz = q
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )


@njit("f8[:](f8[:,:])")
def matrix_to_quaternion(R):
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])


@njit("f8[:](f8[:])")
def exp_quaternion(angles):
    """..."""
    theta1, theta2, theta3 = angles
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        D = 1 / 2 - theta**2 / 48 + theta**4 / 3840 - theta**6 / 645120
        qw = 1 - theta**2 / 8 + theta**4 / 384 - theta**6 / 46080
    else:
        D = np.sin(theta / 2) / theta
        qw = np.cos(theta / 2)

    qx, qy, qz = D * theta1, D * theta2, D * theta3
    exp_theta = np.array([qw, qx, qy, qz])
    return exp_theta


@njit("f8[:](f8[:])")
def log_quaternion(q):
    """..."""
    qw, qx, qy, qz = q

    theta = 2 * np.arccos(qw)
    if theta < 1e-6:
        theta_over_sin_half_theta = (
            2 + theta**2 / 12 + 7 * theta**4 / 2880 + 31 * theta**6 / 483840
        )
    else:
        theta_over_sin_half_theta = theta / np.sin(theta / 2)

    thetax = qx * theta_over_sin_half_theta
    thetay = qy * theta_over_sin_half_theta
    thetaz = qz * theta_over_sin_half_theta

    return np.array([thetax, thetay, thetaz])


#######################################
# numbafied numerical stuff functions #
def my_vectorize(jitfun, out_dim=1, args=None):
    """
    vectorizes jited functions

    fun(X)=Y --> fun_vec([X0,X1,...])=[Y0,Y1...]=[fun(X0),fun(X1),...]
    """
    if out_dim == 1:

        @njit()
        def jitfun_vec(X):
            N = len(X)
            out_array = np.zeros(N)
            for s in range(N):
                out_array[s] = jitfun(X[s])
            return out_array

    elif out_dim == 3:

        @njit
        def jitfun_vec(X):
            N = len(X)
            out_array = np.zeros((N, 3))
            for s in range(N):
                out_array[s] = jitfun(X[s])
            return out_array

    return jitfun_vec


#######################################
# numbafied numerical stuff functions #
def my_vectorize_args(jitfun, out_dim=1, args=None):
    """
    vectorizes jited functions

    fun(X)=Y --> fun_vec([X0,X1,...])=[Y0,Y1...]=[fun(X0),fun(X1),...]
    """
    if out_dim == 1:

        @njit()
        def jitfun_vec(X, args):
            N = len(X)
            out_array = np.zeros(N)
            for s in range(N):
                out_array[s] = jitfun(X[s], args)
            return out_array

    elif out_dim == 3:

        @njit
        def jitfun_vec(X, args):
            N = len(X)
            out_array = np.zeros((N, 3))
            for s in range(N):
                out_array[s] = jitfun(X[s], args)
            return out_array

    return jitfun_vec


@njit("f8[:](f8[:],f8[:])")
def jitcross(u, v):
    w = np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )
    return w


@njit
def fib_disc(Npoints=100):
    xy = np.zeros((Npoints, 2))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(Npoints):
        rad = np.sqrt(i / (Npoints - 1))  # radius at z

        theta = ga * i  # angle increment
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xy[i] = np.array([x, y])

    return xy


@njit
def weighted_fib_disc(Npoints=100):
    xy = np.zeros((Npoints, 2))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    # r_fun = lambda r: np.exp(-((1 - i) ** 2))

    for i in range(Npoints):
        rad = np.sqrt(i / (Npoints - 1))  # radius at z

        theta = ga * i  # angle increment
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xy[i] = np.array([x, y])

    return xy


@njit
def fib_sphere(Npoints=100):
    xyz = np.zeros((Npoints, 3))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(Npoints):
        z = 1.0 - 2 * i / (Npoints - 1)  # -1<=z<=1
        rad = np.sqrt(1.0 - z**2)  # radius at z

        theta = ga * i  # angle increment

        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xyz[i] = np.array([x, y, z])

    return xyz


def my_findiff_coeffs(deriv, h):
    """
    computes finite difference coefficients.
    offsets: h = [h_0,...,h_j,...]
        * one of the h_j's should be 0
    deriv: oder of derivative

    a = [a_0,...,a_j,...]
    X = [x+h_0,...,x+h_j,...]
    Y = [y(x+h_0),...,y(x+)]
    (d_dx)^deriv y(x) ~= a[0]*y[0]+...+a[j]*y[j]+...

    """
    N = len(h)
    H = np.array([h**j for j in range(N)])
    b = np.array([factorial(n) if n == deriv else 0.0 for n in range(N)])
    a = np.linalg.solve(H, b)
    return a


def sym_findiff_weights(deriv_order, sample_number):
    """
    makes sympy expressions of weights for finite difference of order
    'deriv_order' from 'sample_number' samples.

    derivative Df(x) ~= (d/dx)^{deriv_order}f is at x
    using points [x+h[0],...,x+h[j],...]
    """
    h = sp.Matrix([sp.symbols(f"h_{j}") for j in range(sample_number)])
    weights = sp.Array(sp.finite_diff_weights(deriv_order, h, x0=0))
    a = weights[deriv_order, -1].simplify()
    return h, a


def save_findiff_weight_funs():
    """
    lambdifies 2nd order difference coefficients for 1st-4th derivatives
    """
    out_dir = "./numdiff"
    h1, a1 = sym_findiff_weights(1, 3)
    h2, a2 = sym_findiff_weights(2, 4)
    h3, a3 = sym_findiff_weights(3, 5)
    h4, a4 = sym_findiff_weights(4, 6)
    findiff_dict = {}
    findiff_dict["findiff_weights1"] = njit(sp.lambdify([h1], a1))
    findiff_dict["findiff_weights2"] = njit(sp.lambdify([h2], a2))
    findiff_dict["findiff_weights3"] = njit(sp.lambdify([h3], a3))
    findiff_dict["findiff_weights4"] = njit(sp.lambdify([h4], a4))
    for key, fun in findiff_dict.items():
        out_path = f"{out_dir}/{key}.pickle"
        with open(out_path, "wb") as _f:
            dill.dump(fun, _f, recurse=True)


##############################################################################
# save/load finite difference weight functions ###############################
##############################################################################
try:
    with open("./numdiff/findiff_weights1.pickle", "rb") as _f:
        findiff_weights1 = dill.load(_f)
    with open("./numdiff/findiff_weights2.pickle", "rb") as _f:
        findiff_weights2 = dill.load(_f)
    with open("./numdiff/findiff_weights3.pickle", "rb") as _f:
        findiff_weights3 = dill.load(_f)
    with open("./numdiff/findiff_weights4.pickle", "rb") as _f:
        findiff_weights4 = dill.load(_f)
except FileNotFoundError:
    save_findiff_weight_funs()
# def findiff_test(h, a, expand2order):
#     """
#     ***uses dot() from symdiff package***
#     verifies sym_findiff_weights() gives correct weights
#
#     returns coefficients of derivatives in taylor expansion using weights 'a'
#     with offsets 'h'
#
#     h1,a1 = sym_findiff_weights(1,3)
#     h2,a2 = sym_findiff_weights(2,4)
#     h3,a3 = sym_findiff_weights(3,5)
#     h4,a4 = sym_findiff_weights(4,6)
#     using these ^^^ should get you 2nd order
#
#     Df1 = findiff_test(h1,a1,4)
#     Df2 = findiff_test(h2,a2,5)
#     Df3 = findiff_test(h3,a3,6)
#     Df4 = findiff_test(h4,a4,7)
#     """
#     # h,a = sym_weights(deriv,Nsamps)
#     # Nsamps = len(h)
#     _df = sp.Array([sp.symbols(f"f_{j}") for j in range(expand2order)])
#     x = sp.symbols("x")
#     _x = sp.Array([x**j / sp.factorial(j) for j in range(expand2order)])
#     _f = dot(_x, _df)
#     f = sp.Array([_f.subs({x: hj}) for hj in h])
#     _Df = dot(a, f)
#     Df = sp.Array([_Df.diff(dfj).simplify() for dfj in _df])
#     return Df
#


@njit
def diff(z, ds, n=1):
    """
    0th,1st,2nd,3rd,4th order finite difference on uniform grid with spacing ds
    """
    if n == 0:
        return z
    elif n == 1:
        dz = np.zeros(z.shape)
        dz[0] = (-1.5 * z[0] + 2.0 * z[1] - 0.5 * z[2]) / ds
        dz[-1] = (0.5 * z[-3] - 2.0 * z[-2] + 1.5 * z[-1]) / ds
        dz[1:-1] = (-0.5 * z[:-2] + 0.5 * z[2:]) / ds
    elif n == 2:
        dz = np.zeros(z.shape)
        dz[0] = (2.0 * z[0] - 5.0 * z[1] + 4.0 * z[2] - z[3]) / ds**2
        dz[-1] = (-z[-4] + 4.0 * z[-3] - 5.0 * z[-2] + 2.0 * z[-1]) / ds**2

        dz[1:-1] = (z[:-2] - 2 * z[1:-1] + z[2:]) / ds**2
    elif n == 3:
        dz = np.zeros(z.shape)
        dz[0] = (
            -2.5 * z[0] + 9.0 * z[1] - 12.0 * z[2] + 7.0 * z[3] - 1.5 * z[4]
        ) / ds**3
        dz[1] = (
            -1.5 * z[0] + 5.0 * z[1] - 6.0 * z[2] + 3.0 * z[3] - 0.5 * z[4]
        ) / ds**3
        dz[-2] = (
            0.5 * z[-5] - 3.0 * z[-4] + 6.0 * z[-3] - 5.0 * z[-2] + 1.5 * z[-1]
        ) / ds**3
        dz[-1] = (
            1.5 * z[-5] - 7.0 * z[-4] + 12.0 * z[-3] - 9.0 * z[-2] + 2.5 * z[-1]
        ) / ds**3

        dz[2:-2] = (-0.5 * z[:-4] + z[1:-3] - z[3:-1] + 0.5 * z[4:]) / ds**3
    elif n == 4:
        dz = np.zeros(z.shape)
        dz[0] = (
            3 * z[0] - 14 * z[1] + 26 * z[2] - 24 * z[3] + 11 * z[4] - 2 * z[5]
        ) / ds**4
        dz[1] = (
            2 * z[0] - 9 * z[1] + 16 * z[2] - 14 * z[3] + 6 * z[4] - z[5]
        ) / ds**4
        dz[-2] = (
            -z[-6] + 6 * z[-5] - 14 * z[-4] + 16 * z[-3] - 9 * z[-2] + 2 * z[-1]
        ) / ds**4
        dz[-1] = (
            -2 * z[-6] + 11 * z[-5] - 24 * z[-4] + 26 * z[-3] - 14 * z[-2] + 3 * z[-1]
        ) / ds**4
        dz[2:-2] = (z[:-4] - 4 * z[1:-3] + 6 * z[2:-2] - 4 * z[3:-1] + z[4:]) / ds**4

    return dz


@njit
def diff_nu(f, deriv, x):
    """
    2nd order accurate finite difference on grid x
    """

    # fdim = len(f.shape)
    # # es_str = 'i,i'#+'abcde'[:fdim-1]
    if deriv == 0:
        return f
    elif deriv == 1:
        df = np.zeros(f.shape)
        h = x[:3] - x[0]
        a = findiff_weights1(h)
        df[0] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2]
        for j in range(1, len(f) - 1):
            h = x[j - 1 : j + 2] - x[j]
            a = findiff_weights1(h)
            df[j] = (
                a[0] * f[j - 1] + a[1] * f[j] + a[2] * f[j + 1]
            )  # np.einsum('i,i...', a,f[j-1:j+2])
        h = x[-3:] - x[-1]
        a = findiff_weights1(h)
        df[-1] = (
            a[0] * f[-3] + a[1] * f[-2] + a[2] * f[-1]
        )  # np.einsum('i,i...', a,f[-3:])

    elif deriv == 2:
        df = np.zeros(f.shape)
        h = x[:4] - x[0]
        a = findiff_weights2(h)
        df[0] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3]

        h = x[:4] - x[1]
        a = findiff_weights2(h)
        df[1] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3]
        for j in range(2, len(f) - 1):
            h = x[j - 2 : j + 2] - x[j]
            a = findiff_weights2(h)
            df[j] = a[0] * f[j - 2] + a[1] * f[j - 1] + a[2] * f[j] + a[3] * f[j + 1]
        h = x[-4:] - x[-1]
        a = findiff_weights2(h)
        df[-1] = a[0] * f[-4] + a[1] * f[-3] + a[2] * f[-2] + a[3] * f[-1]

    elif deriv == 3:
        df = np.zeros(f.shape)
        h = x[:5] - x[0]
        a = findiff_weights3(h)
        df[0] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3] + a[4] * f[4]

        h = x[:5] - x[1]
        a = findiff_weights3(h)
        df[1] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3] + a[4] * f[4]

        for j in range(2, len(f) - 2):
            h = x[j - 2 : j + 3] - x[j]
            a = findiff_weights3(h)
            df[j] = (
                a[0] * f[j - 2]
                + a[1] * f[j - 1]
                + a[2] * f[j]
                + a[3] * f[j + 1]
                + a[4] * f[j + 2]
            )

        h = x[-5:] - x[-2]
        a = findiff_weights3(h)
        df[-2] = (
            a[0] * f[-5] + a[1] * f[-4] + a[2] * f[-3] + a[3] * f[-2] + a[4] * f[-1]
        )

        h = x[-5:] - x[-1]
        a = findiff_weights3(h)
        df[-1] = (
            a[0] * f[-5] + a[1] * f[-4] + a[2] * f[-3] + a[3] * f[-2] + a[4] * f[-1]
        )

    elif deriv == 4:
        df = np.zeros(f.shape)
        h = x[:6] - x[0]
        a = findiff_weights4(h)
        df[0] = (
            a[0] * f[0]
            + a[1] * f[1]
            + a[2] * f[2]
            + a[3] * f[3]
            + a[4] * f[4]
            + a[5] * f[5]
        )

        h = x[:6] - x[1]
        a = findiff_weights4(h)
        df[1] = (
            a[0] * f[0]
            + a[1] * f[1]
            + a[2] * f[2]
            + a[3] * f[3]
            + a[4] * f[4]
            + a[5] * f[5]
        )

        h = x[:6] - x[2]
        a = findiff_weights4(h)
        df[2] = (
            a[0] * f[0]
            + a[1] * f[1]
            + a[2] * f[2]
            + a[3] * f[3]
            + a[4] * f[4]
            + a[5] * f[5]
        )

        for j in range(3, len(f) - 2):
            h = x[j - 3 : j + 3] - x[j]
            a = findiff_weights4(h)
            df[j] = (
                a[0] * f[j - 3]
                + a[1] * f[j - 2]
                + a[2] * f[j - 1]
                + a[3] * f[j]
                + a[4] * f[j + 1]
                + a[5] * f[j + 2]
            )

        h = x[-6:] - x[-2]
        a = findiff_weights4(h)
        df[-2] = (
            a[0] * f[-6]
            + a[1] * f[-5]
            + a[2] * f[-4]
            + a[3] * f[-3]
            + a[4] * f[-2]
            + a[5] * f[-1]
        )

        h = x[-6:] - x[-1]
        a = findiff_weights4(h)
        df[-1] = (
            a[0] * f[-6]
            + a[1] * f[-5]
            + a[2] * f[-4]
            + a[3] * f[-3]
            + a[4] * f[-2]
            + a[5] * f[-1]
        )

    return df


@njit
def trapint(f, x):
    """
    integrates f over x using trapezoid rule
    """
    int_f = 0.5 * (x[1] - x[0]) * f[0] + 0.5 * (x[-1] - x[-2]) * f[-1]
    for j in range(1, len(x) - 1):
        int_f += 0.5 * (x[j + 1] - x[j - 1]) * f[j]
    return int_f
