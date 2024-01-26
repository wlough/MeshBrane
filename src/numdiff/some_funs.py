import numpy as np
import sympy as sp
from numba import njit
import dill


#######################################
# tested
@njit
def matrix_to_quaternion(Q):
    """Converts a 3-by-3 rotation matrix to a unit quaternion. Safe to use with
    arbitrary rotation angle"""
    q = np.zeros(4)
    diagQ = np.array([Q[0, 0], Q[1, 1], Q[2, 2]])
    trQ = Q[0, 0] + Q[1, 1] + Q[2, 2]
    diagQmax = max(diagQ)

    use_alt_form = diagQmax > trQ
    if use_alt_form:
        i = index_of(diagQ, diagQmax)
        j = (i + 1) % 3  # index of the next elements
        k = (j + 1) % 3  # index of the next next element

        # multipliying by np.sign(Qkj_Qjk) ensures qs >=0
        qi = np.sqrt(1 + 2 * diagQmax - trQ) / 2
        Qkj_Qjk = Q[k, j] - Q[j, k]
        if Qkj_Qjk < 0:
            qi *= -1
        qj = (Q[i, j] + Q[j, i]) / (4 * qi)
        qk = (Q[i, k] + Q[k, i]) / (4 * qi)
        qs = Qkj_Qjk / (4 * qi)
    else:
        i, j, k = 0, 1, 2
        qs = np.sqrt(1 + trQ) / 2
        # cos_theta = 2 * qw**2 - 1
        qi = (Q[2, 1] - Q[1, 2]) / (4 * qs)
        qj = (Q[0, 2] - Q[2, 0]) / (4 * qs)
        qk = (Q[1, 0] - Q[0, 1]) / (4 * qs)

    q[0] = qs
    q[i + 1] = qi
    q[j + 1] = qj
    q[k + 1] = qk
    return q


@njit("f8[:,:](f8[:])")
def quaternion_to_matrix(q):
    qw, qx, qy, qz = q
    # qhat = np.array([[0.0, -qz, qy], [qz, 0.0, -qx], [-qy, qx, 0.0]])
    # qvec = np.array([qx, qy, qz])
    # I = np.eye(3)
    # q_qT = np.array(
    #     [
    #         [qx * qx, qx * qy, qx * qz],
    #         [qy * qx, qy * qy, qy * qz],
    #         [qz * qx, qz * qy, qz * qz],
    #     ]
    # )
    # Q = (qw**2 - qvec @ qvec) * I + 2 * qw * qhat + 2 * q_qT
    Q = np.array(
        [
            [
                qw**2 + qx**2 - qy**2 - qz**2,
                2 * qx * qy - 2 * qw * qz,
                2 * qw * qy + 2 * qx * qz,
            ],
            [
                2 * qw * qz + 2 * qx * qy,
                qw**2 - qx**2 + qy**2 - qz**2,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qx * qz - 2 * qw * qy,
                2 * qw * qx + 2 * qy * qz,
                qw**2 - qx**2 - qy**2 + qz**2,
            ],
        ]
    )
    return Q


@njit("f8[:,:,:](f8[:,:])")
def quaternion_to_matrix_vectorized(q_samps):
    Nsamps = len(q_samps)
    Q_samps = np.zeros((Nsamps, 3, 3))
    for i in range(Nsamps):
        qw, qx, qy, qz = q_samps[i]

        Q_samps[i] = np.array(
            [
                [
                    qw**2 + qx**2 - qy**2 - qz**2,
                    2 * qx * qy - 2 * qw * qz,
                    2 * qw * qy + 2 * qx * qz,
                ],
                [
                    2 * qw * qz + 2 * qx * qy,
                    qw**2 - qx**2 + qy**2 - qz**2,
                    2 * qy * qz - 2 * qw * qx,
                ],
                [
                    2 * qx * qz - 2 * qw * qy,
                    2 * qw * qx + 2 * qy * qz,
                    qw**2 - qx**2 - qy**2 + qz**2,
                ],
            ]
        )
    return Q_samps


#######################################
# special functions #
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


@njit
def index_of(array, element):
    """
    gets the index of element in array
    """

    for i, x in enumerate(array):
        if x == element:
            return i

    msg = "element is not in array"
    raise ValueError(msg)


@njit
def index_of_nested(array, element):
    """
    gets the index of subarray in array
    """

    for i, x in enumerate(array):
        if (x == element).all():
            return i

    msg = "element is not in array"
    raise ValueError(msg)


@njit  # ("int32(int32)")
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


# @njit("f8(int32,int32,f8,f8)")
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


#######################################
# linear algebra, quaternions, rotatations, and euclidean transformations  #
@njit
def csr_to_csc(csr_data, csr_indices, csr_indptr):
    # Compute the row indices
    row = np.empty_like(csr_indices)
    for i in range(len(csr_indptr) - 1):
        row[csr_indptr[i] : csr_indptr[i + 1]] = i

    # Sort the elements by column
    order = np.argsort(csr_indices)
    row = row[order]
    col = csr_indices[order]
    data = csr_data[order]

    # Compute the CSC indptr
    csc_indptr = np.zeros(len(csr_indptr), dtype=int)
    for j in range(len(col)):
        csc_indptr[col[j] + 1] += 1
    for j in range(1, len(csc_indptr)):
        csc_indptr[j] += csc_indptr[j - 1]

    return data, row, csc_indptr


@njit
def transpose_csr(data, indices, indptr):
    # Compute the number of non-zero entries and the number of columns
    # n = len(data)
    m = np.max(indices) + 1

    # Initialize the data, indices, and indptr for the transpose
    data_T = np.empty_like(data)
    indices_T = np.empty_like(indices)
    # indptr_T = np.zeros(m + 1, dtype=indptr.dtype)
    _indptr_T = np.zeros(m + 1, dtype=indptr.dtype)
    indptr_T = np.zeros(m + 1, dtype=np.int32)
    # _indptr_T[0] = 0
    # _indptr_T[j+1] = indptr_T[j]

    # Compute the column counts
    for index in indices:
        # indptr_T[index + 1] += 1
        _indptr_T[index + 1] += 1

    # Compute the column pointers
    # indptr_T = np.cumsum(indptr_T)
    _indptr_T = np.cumsum(_indptr_T)

    for i in range(len(indptr) - 1):
        # For each non-zero in the row...
        for data_index in range(indptr[i], indptr[i + 1]):
            # Get the column index
            j = indices[data_index]

            # Get the insertion index
            insert_index = _indptr_T[j]
            # _insert_index = _indptr_T[j + 1]

            # Insert the data and row index
            data_T[insert_index] = data[data_index]
            indices_T[insert_index] = i

            # Increment the column pointer
            _indptr_T[j] += 1
            # _indptr_T[j + 1] += 1

    indptr_T[1:] = _indptr_T[:-1]
    return data_T, indices_T, indptr_T


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


@njit("f8(f8[:],f8[:])")
def jitdot(u, v):
    u_dot_v = 0.0
    Ndim = len(u)
    for i in range(Ndim):
        u_dot_v += u[i] * v[i]
    return u_dot_v


@njit("f8(f8[:])")
def jitnorm(u):
    normu = 0.0
    # Ndim = len(u)
    for ui in u:
        normu += ui**2
    normu = np.sqrt(normu)
    return normu


@njit("f8(f8[:],f8[:], f8[:])")
def triprod(u, v, w):
    uvw = (
        u[1] * v[2] * w[0]
        - u[2] * v[1] * w[0]
        + u[2] * v[0] * w[1]
        - u[0] * v[2] * w[1]
        + u[0] * v[1] * w[2]
        - u[1] * v[0] * w[2]
    )
    return uvw


#######################################
# actual quaternion operations
@njit("f8[:](f8[:], f8[:])")
def mul_quaternion(q1, q2):
    """quaternion multiplication"""
    qw1, qx1, qy1, qz1 = q1
    qw2, qx2, qy2, qz2 = q2
    qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
    qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
    qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
    qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2

    return np.array([qw, qx, qy, qz])


@njit("f8[:](f8[:])")
def inv_quaternion(q):
    """quaternion multiplication"""
    qw, qx, qy, qz = q
    normq2 = qw**2 + qx**2 + qy**2 + qz**2
    return np.array([qw, -qx, -qy, -qz]) / normq2


@njit("f8[:](f8[:])")
def con_quaternion(q):
    """quaternion conjugate"""
    qw, qx, qy, qz = q
    return np.array([qw, -qx, -qy, -qz])


@njit("f8[:](f8[:])")
def exp_quaternion(q):
    """..."""
    exp_q = np.zeros_like(q)
    q0, qvec = q[0], q[1:]
    alpha = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    # theta = 2 * normqvec
    if alpha < 1e-6:
        sinc_alpha = 1 - alpha**2 / 6 + alpha**4 / 120 - alpha**6 / 5040
        cos_alpha = 1 - alpha**2 / 2 + alpha**4 / 24 - alpha**6 / 720
    else:
        sinc_alpha = np.sin(alpha) / alpha
        cos_alpha = np.cos(alpha)

    # exp_q = np.exp(q0)*np.array([])
    exp_q0 = np.exp(q0)
    exp_q[0] = exp_q0 * cos_alpha
    exp_q[1:] = exp_q0 * sinc_alpha * qvec
    return exp_q


@njit("f8[:](f8[:])")
def log_quaternion(q):
    """..."""
    log_q = np.zeros_like(q)
    q0, qvec = q[0], q[1:]
    norm_q = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    norm_qvec = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    cos_alpha = q0 / norm_q
    alpha = np.arccos(cos_alpha)

    log_q[0] = np.log(norm_q)
    log_q[1:] = alpha * qvec / norm_qvec
    return log_q


@njit("f8[:](f8[:], f8[:])")
def pow_quaternion(q1, q2):
    """q1**q2"""
    log_q1 = log_quaternion(q1)
    log_q1_q2 = mul_quaternion(log_q1, q2)
    q1_to_the_q2 = exp_quaternion(log_q1_q2)
    return q1_to_the_q2


#######################################
# UNIT quaternion operations
@njit("f8[:](f8[:])")
def exp_unit_quaternion(a):
    """(pure imaginary quaternion)->(unit quaternion)
    [qx,qy,qz]->[exp(q)_w,exp(q)_x,exp(q)_y,exp(q)_z]"""
    exp_a = np.zeros(4)
    norm_a = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if norm_a < 1e-6:
        sinc_norm_a = 1 - norm_a**2 / 6 + norm_a**4 / 120  # - norm_a**6 / 5040
        cos_norm_a = 1 - norm_a**2 / 2 + norm_a**4 / 24  # - norm_a**6 / 720
    else:
        sinc_norm_a = np.sin(norm_a) / norm_a
        cos_norm_a = np.cos(norm_a)
    exp_a[0] = cos_norm_a
    exp_a[1:] = sinc_norm_a * a
    return exp_a


@njit("f8[:](f8[:])")
def log_unit_quaternion(q):
    """(unit quaternion)->(pure imaginary quaternion)"""
    # log_q = np.zeros_like(q[1:])
    q0, qvec = q[0], q[1:]
    norm_qvec = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    # cos_alpha = q0
    # sin_alpha = norm_qvec
    alpha = np.arccos(q0)
    alpha_vec = alpha * qvec / norm_qvec
    return alpha_vec


@njit("f8[:](f8[:])")
def exp_unit_quaternion2(a):
    """(pure imaginary quaternion)->(unit quaternion)
    [qx,qy,qz]->[exp(q)_w,exp(q)_x,exp(q)_y,exp(q)_z]"""
    exp_a = np.zeros(4)
    norm_a = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if norm_a < 1e-6:
        sinc_norm_a = 1 - norm_a**2 / 6 + norm_a**4 / 120  # - norm_a**6 / 5040
        cos_norm_a = 1 - norm_a**2 / 2 + norm_a**4 / 24  # - norm_a**6 / 720
    else:
        sinc_norm_a = np.sin(norm_a) / norm_a
        cos_norm_a = np.cos(norm_a)
    exp_a[0] = cos_norm_a
    exp_a[1:] = sinc_norm_a * a
    return exp_a


#
# # @njit
# def rcm_quaternion(G):
#     iters = 4
#     Nsamps = len(G)
#     g0 = G[0]
#     mu_g = np.zeros_like(g0)
#     mu_g[:] = g0
#     mu_g_inv = np.zeros_like(g0)
#     for iter in range(iters):
#         mu_g_inv = inv_quaternion(
#             mu_g
#         )  # np.array([mu_g[0], -mu_g[1], -mu_g[2], -mu_g[3]])
#         Psi = np.zeros(3)
#         for i in range(Nsamps):
#             g = G[i]
#             mu_g_inv_g = mul_quaternion(mu_g_inv, g)
#             print(mu_g_inv_g@mu_g_inv_g)
#             mu_g_inv_g /= np.linalg.norm(mu_g_inv_g)
#             Psi += log_unit_quaternion(mu_g_inv_g) / Nsamps
#
#             # Psi += log_se3_quaternion(mul_se3_quaternion(g, mu_g_inv)) / Nsamps
#         mu_g = mul_quaternion(mu_g, exp_unit_quaternion(Psi))
#         # mu_g = mul_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
#     return mu_g
#
#
# for _ in range(100):
#     Nsamps = 13
#     G = np.zeros((Nsamps, 4))
#     for i in range(Nsamps):
#         q = np.random.rand(4) - 0.5
#         q *= q[0]
#         q /= np.linalg.norm(q)
#         G[i] = q
#
#     g = rcm_quaternion(G)
#     print(g @ g)
#     # g - sum(G)/Nsamps
# (np.array([mul_quaternion(inv_quaternion(g), _) for _ in G]))


@njit("f8[:](f8[:], i8)")
def log_unit_quaternion2(q, m):
    """(unit quaternion)->(pure imaginary quaternion)"""
    # log_q = np.zeros_like(q[1:])
    q0, qvec = q[0], q[1:]
    norm_qvec = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    # cos_alpha = q0
    # sin_alpha = norm_qvec
    alpha = np.arccos(q0) + 2 * np.pi * m
    alpha_vec = alpha * qvec / norm_qvec
    return alpha_vec


# A = np.pi * np.random.rand(33).reshape((11, 3))
# a = A[0]
# q = exp_unit_quaternion(a)
# _a = log_unit_quaternion(q)
# _a2 = log_unit_quaternion2(q, -1)
# exp_unit_quaternion(_a)-exp_unit_quaternion(_a2)
# for a in A:
#     q = exp_unit_quaternion(a)
#     _a = log_unit_quaternion(q)
#     norma = np.linalg.norm(a)
#     _norma = np.linalg.norm(_a)
#     err = a - _a
#
#     print(f"|a|/pi={np.round(norma/np.pi, 3)}")
#     print(f"|_a|={np.round(_norma/np.pi, 3)}")
#     print(f"err={err}")
#     print(f"q={q}")
#
#     print("------------------------------")


@njit("f8[:](f8[:])")
def cay_quaternion(a):
    """cay(eta)=(1-a)^{-1}(1+a)"""
    q = np.zeros(4)
    norm_a2 = a[0] ** 2 + a[1] ** 2 + a[2] ** 2
    q[0] = (1 - norm_a2) / (1 + norm_a2)
    q[1:] = 2 * a / (1 + norm_a2)
    # if q[0] < 0:
    #     q *= -1
    return q


@njit("f8[:](f8[:])")
def cayinv_quaternion(q):
    """inverse of cay(eta)=(1-a)^{-1}(1+a)

    *Assumes q is normalized
    *Breaks when q=[-1,0,0,0]
    """

    qs, qv = q[0], q[1:]
    # if qs < 0:
    #     qs *= -1
    #     qv *= -1
    norm_qv2 = qv[0] ** 2 + qv[1] ** 2 + qv[2] ** 2

    N = qs**2 + 2 * qs + 1 + norm_qv2
    #############################
    a = 2 * qv / N
    return a


#######################################
#######################################
# rotations as quaternions
#######################################
#######################################
# tested


@njit("f8[:](f8[:,:])")
def matrix_to_quaternion2(R):
    """3-by-3 orthogonal matrix to unit quaternion. valid for rotation angle
    0<=theta<pi"""
    trR = R[0, 0] + R[1, 1] + R[2, 2]  # =1+2cos(theta)
    qw = np.sqrt(1 + trR) / 2
    # cos_theta = 2 * qw**2 - 1
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])


@njit("f8[:,:](f8[:])")
def quaternion_to_matrix2(q):
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


#######################################
# untested
@njit("f8[:](f8[:])")
def exp_so3_quaternion(angles):
    """..."""
    theta1, theta2, theta3 = angles
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        D = 1 / 2 - theta**2 / 48 + theta**4 / 3840  # - theta**6 / 645120
        qw = 1 - theta**2 / 8 + theta**4 / 384  # - theta**6 / 46080
    else:
        D = np.sin(theta / 2) / theta
        qw = np.cos(theta / 2)

    qx, qy, qz = D * theta1, D * theta2, D * theta3
    exp_theta = np.array([qw, qx, qy, qz])
    return exp_theta


@njit("f8[:](f8[:])")
def log_so3_quaternion(q):
    """2*(unit quaternion log of q)"""
    qw, qx, qy, qz = q

    theta = 2 * np.arccos(qw)
    if theta < 1e-6:
        theta_over_sin_half_theta = (
            2 + theta**2 / 12 + 7 * theta**4 / 2880  # + 31 * theta**6 / 483840
        )
    else:
        theta_over_sin_half_theta = theta / np.sin(theta / 2)

    thetax = qx * theta_over_sin_half_theta
    thetay = qy * theta_over_sin_half_theta
    thetaz = qz * theta_over_sin_half_theta

    return np.array([thetax, thetay, thetaz])


@njit("f8[:](f8[:])")
def safe_log_so3_quaternion(q):
    qw, qx, qy, qz = q
    # norm_q = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    # R00 = qw**2 + qx**2 - qy**2 - qz**2
    # R11 = qw**2 - qx**2 + qy**2 - qz**2
    # R22 = qw**2 - qx**2 - qy**2 + qz**2
    cos_theta = 2 * qw**2 - 1
    # Ensure the value is within the valid range [-1, 1]
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    # Compute the rotation angle
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        theta_over_two_sin_theta = (
            0.5 + theta**2 / 12 + 7 * theta**4 / 720  # + 31 * theta**6 / 30240
        )
    else:
        theta_over_two_sin_theta = 0.5 * theta / np.sin(theta)
    # R12 = 2 * qy * qz - 2 * qw * qx
    # R21 = 2 * qw * qx + 2 * qy * qz
    # R02 = 2 * qw * qy + 2 * qx * qz
    # R20 = 2 * qx * qz - 2 * qw * qy
    # R01 = 2 * qx * qy - 2 * qw * qz
    # R10 = 2 * qw * qz + 2 * qx * qy
    # thetax = (-R[1, 2] + R[2, 1]) * theta_over_two_sin_theta
    # thetay = (R[0, 2] - R[2, 0]) * theta_over_two_sin_theta
    # thetaz = (-R[0, 1] + R[1, 0]) * theta_over_two_sin_theta
    thetax = (4 * qw * qx) * theta_over_two_sin_theta
    thetay = (4 * qw * qy) * theta_over_two_sin_theta
    thetaz = (4 * qw * qz) * theta_over_two_sin_theta
    # q0, qvec = q[0], q[1:]
    # norm_qvec = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    # cos_theta = 2 * q0**2 - 1
    # theta = np.arccos(cos_theta)
    # thetavec = theta * qvec / norm_qvec
    return np.array([thetax, thetay, thetaz])


@njit("f8[:](f8[:], f8[:])")
def rotate_by_quaternion(q, r):
    """applies rotation representated by unit quaternion q=[qw,qx,qy,qz]
    to verctor r=[x,y,z]"""
    # qw1, qx1, qy1, qz1 = q
    # qx2, qy2, qz2 = r
    qw, qx, qy, qz = q
    x, y, z = r

    qrw = -qx * x - qy * y - qz * z
    qrx = qw * x + qy * z - qz * y
    qry = qw * y - qx * z + qz * x
    qrz = qw * z + qx * y - qy * x

    # qw = qrw * qw + qrx * qx + qry * qy + qrz * qz
    rotated_x = -qrw * qx + qrx * qw - qry * qz + qrz * qy
    rotated_y = -qrw * qy + qrx * qz + qry * qw - qrz * qx
    rotated_z = -qrw * qz - qrx * qy + qry * qx + qrz * qw

    # _r = np.zeros_like(q)
    # _r[1:] = r
    # q_inv = np.zeros_like(q)
    # q_inv[0] = q[0]
    # q_inv[1:] = -q[1:]
    # qrq_inv = mul_quaternion(q, mul_quaternion(_r, q_inv))
    # return qrq_inv[1:]
    return np.array([rotated_x, rotated_y, rotated_z])


#########################
# so3 operations
@njit("f8[:,:](f8[:])")
def exp_so3(angles):
    """..."""
    theta1, theta2, theta3 = angles
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        A = 1.0 - theta**2 / 6 + theta**4 / 120  # - theta**6 / 540
        B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720  # - theta**6 / 40320
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2

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
            0.5 + theta**2 / 12 + 7 * theta**4 / 720  # + 31 * theta**6 / 30240
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


#########################
# se3 operations
# @njit("f8[:,:](f8[:])")
def exp_se3_slow(psi):
    """slow but has formula for V"""
    theta1, theta2, theta3 = psi[3:]
    phi_vec = psi[:3]
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        A = 1.0 - theta**2 / 6 + theta**4 / 120  # - theta**6 / 540
        B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720  # - theta**6 / 40320
        C = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040  # - theta**6 / 362880
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        C = (theta - np.sin(theta)) / theta**3

    Q = np.array(
        [
            [1.0, -A * theta3, A * theta2],
            [A * theta3, 1.0, -A * theta1],
            [-A * theta2, A * theta1, 1.0],
        ]
    )
    Q += B * np.array(
        [
            [-(theta2**2) - theta3**2, theta1 * theta2, theta1 * theta3],
            [theta1 * theta2, -(theta1**2) - theta3**2, theta2 * theta3],
            [theta1 * theta3, theta2 * theta3, -(theta1**2) - theta2**2],
        ]
    )

    V = np.array(
        [
            [1.0, -B * theta3, B * theta2],
            [B * theta3, 1.0, -B * theta1],
            [-B * theta2, B * theta1, 1.0],
        ]
    )
    V += C * np.array(
        [
            [-(theta2**2) - theta3**2, theta1 * theta2, theta1 * theta3],
            [theta1 * theta2, -(theta1**2) - theta3**2, theta2 * theta3],
            [theta1 * theta3, theta2 * theta3, -(theta1**2) - theta2**2],
        ]
    )

    p = V @ phi_vec
    f = np.zeros((4, 4))
    f[3, 3] = 1.0
    f[:3, 3] = p
    f[:3, :3] = Q
    return f


@njit("f8[:,:](f8[:])")
def exp_se3(psi):
    """..."""
    theta1, theta2, theta3 = psi[3:]
    phi_vec = psi[:3]
    theta_vec = psi[3:]
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        A = 1.0 - theta**2 / 6 + theta**4 / 120  # - theta**6 / 540
        B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720  # - theta**6 / 40320
        C = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040  # - theta**6 / 362880
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        C = (theta - np.sin(theta)) / theta**3

    Q = np.array(
        [
            [1.0, -A * theta3, A * theta2],
            [A * theta3, 1.0, -A * theta1],
            [-A * theta2, A * theta1, 1.0],
        ]
    )
    Q += B * np.array(
        [
            [-(theta2**2) - theta3**2, theta1 * theta2, theta1 * theta3],
            [theta1 * theta2, -(theta1**2) - theta3**2, theta2 * theta3],
            [theta1 * theta3, theta2 * theta3, -(theta1**2) - theta2**2],
        ]
    )

    theta_cross_phi = jitcross(theta_vec, phi_vec)
    theta_cross_theta_cross_phi = jitcross(theta_vec, theta_cross_phi)
    p = phi_vec + B * theta_cross_phi + C * theta_cross_theta_cross_phi

    f = np.zeros((4, 4))
    f[3, 3] = 1.0
    f[:3, 3] = p
    f[:3, :3] = Q
    return f


@njit("f8[:](f8[:], f8[:])")
def adjoint_se3(psi1, psi2):
    """se3 adjoint theta=rotational,phi=translational"""
    psi = np.zeros_like(psi1)
    theta1, phi1 = psi1[3:], psi1[:3]
    theta2, phi2 = psi2[3:], psi2[:3]
    psi[:3] = jitcross(theta1, phi2) - jitcross(theta2, phi1)
    psi[3:] = jitcross(theta1, theta2)
    return psi


@njit("f8[:](f8[:], f8[:])")
def coadjoint_se3(psi1, psi2):
    """theta=rotational,phi=translational"""
    psi = np.zeros_like(psi1)
    theta1, phi1 = psi1[3:], psi1[:3]
    theta2, phi2 = psi2[3:], psi2[:3]
    psi[:3] = -jitcross(theta1, phi2)
    psi[3:] = -jitcross(theta1, theta2) - jitcross(phi1, phi2)
    return psi


#########################
# se3 operations with rotations as quaternions
@njit("f8[:](f8[:])")
def log_se3_quaternion(pq):
    """phi_vec = translational part"""
    # x, y, z, qw, qx, qy, qz = pq
    psi = np.zeros(6)
    p, q = pq[:3], pq[3:]
    q0, qvec = q[0], q[1:]
    # norm_qvec = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    cos_theta = 2 * q0**2 - 1
    if cos_theta > 1:
        cos_theta = 1.0
    elif cos_theta < -1:
        cos_theta = -1.0
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        theta_over_sin_half_theta = (
            2 + theta**2 / 12 + 7 * theta**4 / 2880  # + 31 * theta**6 / 483840
        )
        ##################################
        # A = 1.0 - theta**2 / 6 + theta**4 / 120 - theta**6 / 540
        # B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720 - theta**6 / 40320
        # C = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040 - theta**6 / 362880
        D = 1.0 / 12.0 + theta**2 / 720 + theta**4 / 30240  # + theta**6 / 1209600
        ##################################
    else:
        theta_over_sin_half_theta = theta / np.sin(theta / 2)
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        # C = (theta - np.sin(theta)) / theta**3
        D = (1 - 0.5 * A / B) / theta**2
    theta_vec = theta_over_sin_half_theta * qvec
    theta_cross_r = jitcross(theta_vec, p)
    theta_cross_theta_cross_r = jitcross(theta_vec, theta_cross_r)
    phi_vec = p - 0.5 * theta_cross_r + D * theta_cross_theta_cross_r

    psi[:3] = phi_vec
    psi[3:] = theta_vec
    return psi


@njit("f8[:](f8[:])")
def exp_se3_quaternion(psi):
    """phi_vec = translational part"""
    # x, y, z, qw, qx, qy, qz = pq
    pq = np.zeros(7)
    phi_vec, theta_vec = psi[:3], psi[3:]
    theta1, theta2, theta3 = theta_vec
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        # A = 1.0 - theta**2 / 6 + theta**4 / 120  # - theta**6 / 540
        B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720  # - theta**6 / 40320
        C = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040  # - theta**6 / 362880
        D = 1 / 2 - theta**2 / 48 + theta**4 / 3840  # - theta**6 / 645120
        qw = 1 - theta**2 / 8 + theta**4 / 384  # - theta**6 / 46080
    else:
        # A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        C = (theta - np.sin(theta)) / theta**3
        D = np.sin(theta / 2) / theta
        qw = np.cos(theta / 2)

    qx, qy, qz = D * theta1, D * theta2, D * theta3
    pq[3:] = np.array([qw, qx, qy, qz])

    theta_cross_phi = jitcross(theta_vec, phi_vec)
    theta_cross_theta_cross_phi = jitcross(theta_vec, theta_cross_phi)
    pq[:3] = phi_vec + B * theta_cross_phi + C * theta_cross_theta_cross_phi

    return pq


@njit("f8[:](f8[:])")
def _log_se3_quaternion(pq):
    """phi_vec = translational part"""
    # x, y, z, qw, qx, qy, qz = pq
    psi = np.zeros(6)
    p, q = pq[:3], pq[3:]
    # Ensure the value is within the valid range [-1, 1]
    qw = pq[3]
    if qw > 1:
        qw = 1.0
    elif qw < -1:
        qw = -1.0
    theta = 2 * np.arccos(qw)
    if theta < 1e-6:
        theta_over_sin_half_theta = (
            2 + theta**2 / 12 + 7 * theta**4 / 2880  # + 31 * theta**6 / 483840
        )
        ##################################
        # A = 1.0 - theta**2 / 6 + theta**4 / 120 - theta**6 / 540
        # B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720 - theta**6 / 40320
        # C = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040 - theta**6 / 362880
        D = 1.0 / 12.0 + theta**2 / 720 + theta**4 / 30240  # + theta**6 / 1209600
        ##################################
    else:
        theta_over_sin_half_theta = theta / np.sin(theta / 2)
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        # C = (theta - np.sin(theta)) / theta**3
        D = (1 - 0.5 * A / B) / theta**2
    theta_vec = theta_over_sin_half_theta * q[1:]
    theta_cross_r = jitcross(theta_vec, p)
    theta_cross_theta_cross_r = jitcross(theta_vec, theta_cross_r)
    phi_vec = p - 0.5 * theta_cross_r + D * theta_cross_theta_cross_r

    psi[:3] = phi_vec
    psi[3:] = theta_vec
    return psi


@njit("f8[:](f8[:])")
def _exp_se3_quaternion(psi):
    """phi_vec = translational part"""
    # x, y, z, qw, qx, qy, qz = pq
    pq = np.zeros(7)
    phi_vec, theta_vec = psi[:3], psi[3:]
    theta1, theta2, theta3 = theta_vec
    theta = np.sqrt(theta1**2 + theta2**2 + theta3**2)
    if theta < 1e-6:
        # A = 1.0 - theta**2 / 6 + theta**4 / 120  # - theta**6 / 540
        B = 1.0 / 2.0 - theta**2 / 24 + theta**4 / 720  # - theta**6 / 40320
        C = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040  # - theta**6 / 362880
        D = 1 / 2 - theta**2 / 48 + theta**4 / 3840  # - theta**6 / 645120
        qw = 1 - theta**2 / 8 + theta**4 / 384  # - theta**6 / 46080
    else:
        # A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        C = (theta - np.sin(theta)) / theta**3
        D = np.sin(theta / 2) / theta
        qw = np.cos(theta / 2)

    qx, qy, qz = D * theta1, D * theta2, D * theta3
    pq[3:] = np.array([qw, qx, qy, qz])

    theta_cross_phi = jitcross(theta_vec, phi_vec)
    theta_cross_theta_cross_phi = jitcross(theta_vec, theta_cross_phi)
    pq[:3] = phi_vec + B * theta_cross_phi + C * theta_cross_theta_cross_phi

    return pq


@njit("f8[:](f8[:], f8[:])")
def mul_se3_quaternion(pq1, pq2):
    """qr1*qr2"""
    # qw1, qx1, qy1, qz1, x1, y1, z1 = qr1
    # qw2, qx2, qy2, qz2, x2, y2, z2 = qr2
    x1, y1, z1, qw1, qx1, qy1, qz1 = pq1
    x2, y2, z2, qw2, qx2, qy2, qz2 = pq2

    qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
    qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
    qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
    qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2

    x, y, z = pq1[:3] + rotate_by_quaternion(pq1[3:], pq2[:3])

    return np.array([x, y, z, qw, qx, qy, qz])


@njit("f8[:](f8[:])")
def inv_se3_quaternion(pq):
    """..."""
    pq_inv = np.zeros_like(pq)
    # p, q = pq[:3], pq[3:]
    # qw_inv = q[0]
    # qx_inv, qy_inv, qz_inv = -q[1:]
    pq_inv[3] = pq[3]
    pq_inv[4:] = -pq[4:]

    # q_inv = np.array([qw_inv, qx_inv, qx_inv, qy_inv, qz_inv])
    # p_inv = rotate_by_quaternion(q_inv, -p)
    # (Qinv, -Qinv*pinv) * ()
    pq_inv[:3] = -rotate_by_quaternion(pq_inv[3:], pq[:3])
    return pq_inv


@njit("f8[:,:](f8[:])")
def se3_quaternion_to_matrix(pq):
    p, q = pq[:3], pq[3:]
    f = np.zeros((4, 4))
    f[3, 3] = 1.0
    f[:3, 3] = p
    f[:3, :3] = quaternion_to_matrix(q)
    return f


@njit("f8[:](f8[:,:])")
def se3_matrix_to_quaternion(f):
    R, p = f[:3, :3], f[:3, 3]
    pq = np.zeros(7)
    pq[:3] = p
    pq[3:] = matrix_to_quaternion(R)
    return pq


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


# def my_findiff_coeffs(deriv, h):
#     """
#     computes finite difference coefficients.
#     offsets: h = [h_0,...,h_j,...]
#         * one of the h_j's should be 0
#     deriv: oder of derivative
#
#     a = [a_0,...,a_j,...]
#     X = [x+h_0,...,x+h_j,...]
#     Y = [y(x+h_0),...,y(x+)]
#     (d_dx)^deriv y(x) ~= a[0]*y[0]+...+a[j]*y[j]+...
#
#     """
#     N = len(h)
#     H = np.array([h**j for j in range(N)])
#     b = np.array([factorial(n) if n == deriv else 0.0 for n in range(N)])
#     a = np.linalg.solve(H, b)
#     return a

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


def save_findiff_weight_funs(out_dir="./data/findiff_weights"):
    """
    lambdifies 2nd order difference coefficients for 1st-4th derivatives
    """

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


@njit
def findiff_weights1(h):
    """
    findiff_weights{deriv_order}
    O(2) accurate finite difference weights

    see:
    h, a = sym_findiff_weights(deriv_order, sample_number)
    h.__str__()
    a.__str__()
    """
    h_0, h_1, h_2 = h
    a = np.array(
        [
            (-h_1 - h_2) / ((h_0 - h_1) * (h_0 - h_2)),
            (h_0 + h_2) / ((h_0 - h_1) * (h_1 - h_2)),
            (-h_0 - h_1) / ((h_0 - h_2) * (h_1 - h_2)),
        ]
    )
    return a


@njit
def findiff_weights2(h):
    """
    findiff_weights{deriv_order}
    O(2) accurate finite difference weights

    see:
    h, a = sym_findiff_weights(deriv_order, sample_number)
    h.__str__()
    a.__str__()
    """
    h_0, h_1, h_2, h_3 = h
    a = np.array(
        [
            2 * (-h_1 - h_2 - h_3) / ((h_0 - h_1) * (h_0 - h_2) * (h_0 - h_3)),
            2 * (h_0 + h_2 + h_3) / ((h_0 - h_1) * (h_1 - h_2) * (h_1 - h_3)),
            2 * (-h_0 - h_1 - h_3) / ((h_0 - h_2) * (h_1 - h_2) * (h_2 - h_3)),
            2
            * (h_2 * (-h_0 + h_1) - (h_0 - h_1) * (h_0 + h_1))
            / ((-h_0 + h_1) * (h_0 - h_3) * (h_1 - h_3) * (h_2 - h_3)),
        ]
    )
    return a


@njit
def findiff_weights3(h):
    """
    findiff_weights{deriv_order}
    O(2) accurate finite difference weights

    see:
    h, a = sym_findiff_weights(deriv_order, sample_number)
    h.__str__()
    a.__str__()
    """
    h_0, h_1, h_2, h_3, h_4 = h
    a = np.array(
        [
            6
            * (
                h_4 * (-h_0 + h_1) ** 2
                + (h_0 - h_1) * (-h_3 * (-h_0 + h_1) + (h_0 - h_1) * (h_1 + h_2))
            )
            / ((-h_0 + h_1) ** 3 * (h_0 - h_2) * (h_0 - h_3) * (h_0 - h_4)),
            6
            * (
                h_4 * (-h_0 + h_1) * (-h_1 + h_2)
                + (h_0 - h_1) * (h_1 - h_2) * (h_0 + h_2 + h_3)
            )
            / ((-h_0 + h_1) ** 2 * (-h_1 + h_2) ** 2 * (h_1 - h_3) * (h_1 - h_4)),
            6
            * (-h_3 * (-h_0 + h_1) - h_4 * (-h_0 + h_1) + (h_0 - h_1) * (h_0 + h_1))
            / ((-h_0 + h_1) * (h_0 - h_2) * (h_1 - h_2) * (h_2 - h_3) * (h_2 - h_4)),
            6
            * (
                h_4 * (-h_0 + h_2) * (-h_1 + h_2)
                + (h_0 - h_2) * (h_1 - h_2) * (h_0 + h_1 + h_2)
            )
            / (
                (-h_0 + h_2)
                * (h_0 - h_3)
                * (-h_1 + h_2)
                * (h_1 - h_3)
                * (h_2 - h_3)
                * (h_3 - h_4)
            ),
            6
            * (
                -h_3 * (-h_0 + h_2) * (-h_1 + h_2)
                - (h_0 - h_2) * (h_1 - h_2) * (h_0 + h_1 + h_2)
            )
            / (
                (-h_0 + h_2)
                * (h_0 - h_4)
                * (-h_1 + h_2)
                * (h_1 - h_4)
                * (h_2 - h_4)
                * (h_3 - h_4)
            ),
        ]
    )
    return a


@njit
def findiff_weights4(h):
    """
    findiff_weights{deriv_order}
    O(2) accurate finite difference weights

    see:
    h, a = sym_findiff_weights(deriv_order, sample_number)
    h.__str__()
    a.__str__()
    """
    h_0, h_1, h_2, h_3, h_4, h_5 = h
    a = np.array(
        [
            24
            * (
                h_5 * (-h_0 + h_1) ** 3
                - (h_0 - h_1)
                * (
                    h_4 * (-h_0 + h_1) ** 2
                    + (h_0 - h_1) * (-h_3 * (-h_0 + h_1) + (h_0 - h_1) * (h_1 + h_2))
                )
            )
            / (
                (-h_0 + h_1) ** 4
                * (h_0 - h_2)
                * (h_0 - h_3)
                * (h_0 - h_4)
                * (h_0 - h_5)
            ),
            24
            * (
                h_5 * (-h_0 + h_1) ** 2 * (-h_1 + h_2) ** 2
                + (h_0 - h_1)
                * (h_1 - h_2)
                * (
                    h_4 * (-h_0 + h_1) * (-h_1 + h_2)
                    + (h_0 - h_1) * (h_1 - h_2) * (h_0 + h_2 + h_3)
                )
            )
            / (
                (-h_0 + h_1) ** 3
                * (-h_1 + h_2) ** 3
                * (h_1 - h_3)
                * (h_1 - h_4)
                * (h_1 - h_5)
            ),
            24
            * (
                -h_3 * (-h_0 + h_1)
                - h_4 * (-h_0 + h_1)
                - h_5 * (-h_0 + h_1)
                + (h_0 - h_1) * (h_0 + h_1)
            )
            / (
                (-h_0 + h_1)
                * (h_0 - h_2)
                * (h_1 - h_2)
                * (h_2 - h_3)
                * (h_2 - h_4)
                * (h_2 - h_5)
            ),
            24
            * (
                h_4 * (-h_0 + h_2) * (-h_1 + h_2)
                + h_5 * (-h_0 + h_2) * (-h_1 + h_2)
                + (h_0 - h_2) * (h_1 - h_2) * (h_0 + h_1 + h_2)
            )
            / (
                (-h_0 + h_2)
                * (h_0 - h_3)
                * (-h_1 + h_2)
                * (h_1 - h_3)
                * (h_2 - h_3)
                * (h_3 - h_4)
                * (h_3 - h_5)
            ),
            24
            * (
                -h_3 * (-h_0 + h_2) * (-h_1 + h_2)
                - h_5 * (-h_0 + h_2) * (-h_1 + h_2)
                - (h_0 - h_2) * (h_1 - h_2) * (h_0 + h_1 + h_2)
            )
            / (
                (-h_0 + h_2)
                * (h_0 - h_4)
                * (-h_1 + h_2)
                * (h_1 - h_4)
                * (h_2 - h_4)
                * (h_3 - h_4)
                * (h_4 - h_5)
            ),
            24
            * (
                h_4 * (-h_0 + h_3) * (-h_1 + h_3) * (-h_2 + h_3)
                - (h_0 - h_3) * (h_1 - h_3) * (h_2 - h_3) * (h_0 + h_1 + h_2 + h_3)
            )
            / (
                (-h_0 + h_3)
                * (h_0 - h_5)
                * (-h_1 + h_3)
                * (h_1 - h_5)
                * (-h_2 + h_3)
                * (h_2 - h_5)
                * (h_3 - h_5)
                * (h_4 - h_5)
            ),
        ]
    )
    return a


##############################################################################
# save/load finite difference weight functions ###############################
##############################################################################
# try:
#     with open("./data/findiff_weights/findiff_weights1.pickle", "rb") as _f:
#         findiff_weights1 = dill.load(_f)
#     with open("./data/findiff_weights/findiff_weights2.pickle", "rb") as _f:
#         findiff_weights2 = dill.load(_f)
#     with open("./data/findiff_weights3.pickle", "rb") as _f:
#         findiff_weights3 = dill.load(_f)
#     with open("./data/findiff_weights/findiff_weights4.pickle", "rb") as _f:
#         findiff_weights4 = dill.load(_f)
# except FileNotFoundError:
#     print("ehhh")
#     save_findiff_weight_funs()


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
