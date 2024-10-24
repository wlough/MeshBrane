import numpy as np
from numba import jit


def power_method(A, x, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        y = A @ x
        x = y / np.linalg.norm(y)
        if np.linalg.norm(A @ x - x) < tol:
            break
    return x, np.linalg.norm(y) / np.linalg.norm(x)


def inverse_power_method(A, x, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        y = np.linalg.solve(A, x)
        x = y / np.linalg.norm(y)
        if np.linalg.norm(A @ x - x) < tol:
            break
    return x, np.linalg.norm(y) / np.linalg.norm(x)


#######################################


def matrix_to_quaternion(Q):
    """
    Converts a 3-by-3 rotation matrix to a unit quaternion. Safe to use with
    arbitrary rotation angle. Not vectorized.
    """
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


def _quaternion_to_matrix(q):
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


def _quaternion_to_matrix_vectorized(q_samps):
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


def quaternion_to_matrix(q, normalize=False):
    if normalize:
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    qw2, qx2, qy2, qz2 = qw**2, qx**2, qy**2, qz**2
    qxqy, qwqz, qwqy, qxqz, qyqz, qwqx = (
        qx * qy,
        qw * qz,
        qw * qy,
        qx * qz,
        qy * qz,
        qw * qx,
    )

    Q = np.empty(q.shape[:-1] + (3, 3))
    Q[..., 0, 0] = qw2 + qx2 - qy2 - qz2
    Q[..., 0, 1] = 2 * (qxqy - qwqz)
    Q[..., 0, 2] = 2 * (qwqy + qxqz)
    Q[..., 1, 0] = 2 * (qwqz + qxqy)
    Q[..., 1, 1] = qw2 - qx2 + qy2 - qz2
    Q[..., 1, 2] = 2 * (qyqz - qwqx)
    Q[..., 2, 0] = 2 * (qxqz - qwqy)
    Q[..., 2, 1] = 2 * (qwqx + qyqz)
    Q[..., 2, 2] = qw2 - qx2 - qy2 + qz2
    return Q


@jit
def quaternion_to_matrix_numba(q, normalize=False):

    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    qw2, qx2, qy2, qz2 = qw**2, qx**2, qy**2, qz**2
    if normalize:
        qnorm = np.sqrt(qw2 + qx2 + qy2 + qz2)
        qw, qx, qy, qz = qw / qnorm, qx / qnorm, qy / qnorm, qz / qnorm
        qw2, qx2, qy2, qz2 = qw**2, qx**2, qy**2, qz**2
    qxqy, qwqz, qwqy, qxqz, qyqz, qwqx = (
        qx * qy,
        qw * qz,
        qw * qy,
        qx * qz,
        qy * qz,
        qw * qx,
    )

    Q = np.empty(q.shape[:-1] + (3, 3))
    Q[..., 0, 0] = qw2 + qx2 - qy2 - qz2
    Q[..., 0, 1] = 2 * (qxqy - qwqz)
    Q[..., 0, 2] = 2 * (qwqy + qxqz)
    Q[..., 1, 0] = 2 * (qwqz + qxqy)
    Q[..., 1, 1] = qw2 - qx2 + qy2 - qz2
    Q[..., 1, 2] = 2 * (qyqz - qwqx)
    Q[..., 2, 0] = 2 * (qxqz - qwqy)
    Q[..., 2, 1] = 2 * (qwqx + qyqz)
    Q[..., 2, 2] = qw2 - qx2 - qy2 + qz2
    return Q


def rotate_by_quaternion(q, r):
    """
    applies rotation representated by unit quaternion q
    to vector r
    """
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    qrw = -qx * x - qy * y - qz * z
    qrx = qw * x + qy * z - qz * y
    qry = qw * y - qx * z + qz * x
    qrz = qw * z + qx * y - qy * x
    rot_r = np.zeros_like(r)
    rot_r[..., 0] = -qrw * qx + qrx * qw - qry * qz + qrz * qy
    rot_r[..., 1] = -qrw * qy + qrx * qz + qry * qw - qrz * qx
    rot_r[..., 2] = -qrw * qz - qrx * qy + qry * qx + qrz * qw
    return rot_r


def mul_quaternion(q1, q2):
    """quaternion multiplication"""
    qw1, qx1, qy1, qz1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    qw2, qx2, qy2, qz2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    q = np.zeros_like(q1)
    q[..., 0] = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
    q[..., 1] = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
    q[..., 2] = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
    q[..., 3] = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
    return q


def mul_se3_quaternion(pq1, pq2):
    """qr1*qr2"""
    p1 = pq1[..., :3]
    q1 = pq1[..., 3:]
    p2 = pq2[..., :3]
    q2 = pq2[..., 3:]
    pq = np.zeros_like(pq1)
    pq[..., :3] = p1 + rotate_by_quaternion(q1, p2)
    pq[..., 3:] = mul_quaternion(q1, q2)
    return pq


def exp_so3_quaternion(angle_vec, small_angle=False):
    """..."""
    q = np.empty(angle_vec.shape[:-1] + (4,))
    ax, ay, az = angle_vec[..., 0], angle_vec[..., 1], angle_vec[..., 2]
    a_sqr = ax**2 + ay**2 + az**2
    a = np.sqrt(a_sqr)
    if not small_angle:
        q[..., 0] = np.cos(a / 2)
        D = np.sin(a / 2) / a
        q[..., 1] = D * ax
        q[..., 2] = D * ay
        q[..., 3] = D * az
        return q
    else:
        I_small = np.abs(a) < 1e-6
        I_big = np.logical_not(I_small)
        a_fourth_small = a_sqr[I_small] ** 2
        q[..., 0][I_small] = (
            1 - a_sqr[I_small] / 8 + a_fourth_small / 384
        )  # - theta**6 / 46080
        D_small = 1 / 2 - a_sqr[I_small] / 48 + a_fourth_small / 3840  # - a**6 / 645120
        q[..., 1][I_small] = D_small * ax[I_small]
        q[..., 2][I_small] = D_small * ay[I_small]
        q[..., 3][I_small] = D_small * az[I_small]

        q[..., 0][I_big] = np.cos(a[I_big] / 2)
        D_big = np.sin(a[I_big] / 2) / a[I_big]
        q[..., 1][I_big] = D_big * ax[I_big]
        q[..., 2][I_big] = D_big * ay[I_big]
        q[..., 3][I_big] = D_big * az[I_big]
        return q


def rigid_transform(translation, angle_vec, X, origin=None):
    """
    Applies a rigid transformation to 3D point(s)
    """
    q = exp_so3_quaternion(angle_vec)
    if origin is None:
        return translation + rotate_by_quaternion(q, X)
    else:
        return orgin + translation + rotate_by_quaternion(q, X - origin)
