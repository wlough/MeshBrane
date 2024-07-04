import numpy as np


#######################################
# tested
# @njit
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


# @njit("f8[:,:](f8[:])")
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


# @njit("f8[:,:,:](f8[:,:])")
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


#########################
# so3 operations
# @njit("f8[:,:](f8[:])")
def exp_so3(theta1, theta2, theta3):
    """..."""
    # theta1, theta2, theta3 = angles
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


# @njit("f8[:](f8[:,:])")
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


# @njit("f8[:,:](f8[:])")
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


# @njit("f8[:](f8[:], f8[:])")
def adjoint_se3(psi1, psi2):
    """se3 adjoint theta=rotational,phi=translational"""
    psi = np.zeros_like(psi1)
    theta1, phi1 = psi1[3:], psi1[:3]
    theta2, phi2 = psi2[3:], psi2[:3]
    psi[:3] = jitcross(theta1, phi2) - jitcross(theta2, phi1)
    psi[3:] = jitcross(theta1, theta2)
    return psi


# @njit("f8[:](f8[:], f8[:])")
def coadjoint_se3(psi1, psi2):
    """theta=rotational,phi=translational"""
    psi = np.zeros_like(psi1)
    theta1, phi1 = psi1[3:], psi1[:3]
    theta2, phi2 = psi2[3:], psi2[:3]
    psi[:3] = -jitcross(theta1, phi2)
    psi[3:] = -jitcross(theta1, theta2) - jitcross(phi1, phi2)
    return psi
