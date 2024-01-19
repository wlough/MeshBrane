import numpy as np

rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# %%
num_rotations = 1  # We have one rotation matrix

decision_matrix = np.empty((num_rotations, 4))
decision_matrix[
    :, :3
] = rot_matrix.diagonal()  # The diagonal elements of the rotation matrix
decision_matrix[:, -1] = decision_matrix[
    :, :3
].sum()  # The trace of the rotation matrix
# %%
choices = decision_matrix.argmax(axis=1)  # The index of the maximum element in each row

# %%
quat = np.empty((num_rotations, 4))

ind = np.nonzero(choices != 3)[
    0
]  # The indices where the maximum element is not the last one
i = choices[ind]  # The indices of the maximum elements
j = (i + 1) % 3  # The indices of the next elements
k = (j + 1) % 3  # The indices of the next next elements

# %%
quat[ind, i] = (
    1 - decision_matrix[ind, -1] + 2 * rot_matrix[ind, i, i]
)  # The quaternion component corresponding to the maximum element
quat[ind, j] = (
    rot_matrix[ind, j, i] + rot_matrix[ind, i, j]
)  # The next quaternion component
quat[ind, k] = (
    rot_matrix[ind, k, i] + rot_matrix[ind, i, k]
)  # The next next quaternion component
quat[ind, 3] = (
    rot_matrix[ind, k, j] - rot_matrix[ind, j, k]
)  # The scalar quaternion component

# %%


# %%
import numpy as np

# rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

num_rotations = 1  # We have one rotation matrix

decision_matrix = np.empty((num_rotations, 4))
decision_matrix[
    :, :3
] = rot_matrix.diagonal()  # The diagonal elements of the rotation matrix
decision_matrix[:, -1] = decision_matrix[
    :, :3
].sum()  # The trace of the rotation matrix

choices = decision_matrix.argmax(axis=1)  # The index of the maximum element in each row

quat = np.empty((num_rotations, 4))

ind = np.nonzero(choices != 3)[
    0
]  # The indices where the maximum element is not the last one
i = choices[ind]  # The indices of the maximum elements
j = (i + 1) % 3  # The indices of the next elements
k = (j + 1) % 3  # The indices of the next next elements

quat[ind, i] = (
    np.sqrt(1 + 2 * rot_matrix[i, i] - decision_matrix[ind, -1]) / 2
)  # The quaternion component corresponding to the maximum element
quat[ind, j] = (rot_matrix[j, i] + rot_matrix[i, j]) / (
    4 * quat[ind, i]
)  # The next quaternion component
quat[ind, k] = (rot_matrix[k, i] + rot_matrix[i, k]) / (
    4 * quat[ind, i]
)  # The next next quaternion component
quat[ind, 3] = (rot_matrix[k, j] - rot_matrix[j, k]) / (
    4 * quat[ind, i]
)  # The scalar quaternion component

print(quat)


# %%
# rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

num_rotations = 1  # We have one rotation matrix

# decision_matrix = np.empty((num_rotations, 4))
decision_matrix = np.empty(4)
# decision_matrix[:, :3] = rot_matrix.diagonal()  # The diagonal elements of the rotation matrix
decision_matrix[
    :3
] = rot_matrix.diagonal()  # The diagonal elements of the rotation matrix
decision_matrix[-1] = decision_matrix[:3].sum()  # The trace of the rotation matrix

choices = decision_matrix.argmax(axis=0)  # The index of the maximum element in each row


quat = np.empty(4)

ind = np.nonzero(choices != 3)[
    0
]  # The indices where the maximum element is not the last one
i = choices[ind]  # The indices of the maximum elements
j = (i + 1) % 3  # The indices of the next elements
k = (j + 1) % 3  # The indices of the next next elements

quat[ind, i] = (
    np.sqrt(1 + 2 * rot_matrix[i, i] - decision_matrix[ind, -1]) / 2
)  # The quaternion component corresponding to the maximum element
quat[ind, j] = (rot_matrix[j, i] + rot_matrix[i, j]) / (
    4 * quat[ind, i]
)  # The next quaternion component
quat[ind, k] = (rot_matrix[k, i] + rot_matrix[i, k]) / (
    4 * quat[ind, i]
)  # The next next quaternion component
quat[ind, 3] = (rot_matrix[k, j] - rot_matrix[j, k]) / (
    4 * quat[ind, i]
)  # The scalar quaternion component

print(quat)


# %%
from src.numdiff import (
    jitcross,
    index_of,
)
from numba import njit


@njit
def matrix_to_quaternion(Q):
    diagQ = np.array([Q[0, 0], Q[1, 1], Q[2, 2]])
    trQ = Q[0, 0] + Q[1, 1] + Q[2, 2]
    # cos_theta = (trQ-1)/2
    # theta = np.arccos(cos_theta)
    diagQmax = max(diagQ)

    use_alt_form = diagQmax > trQ
    if use_alt_form:
        i = index_of(diagQ, diagQmax)
        j = (i + 1) % 3  # index of the next elements
        k = (j + 1) % 3  # index of the next next element

    qi = np.sqrt(1 + 2 * diagQmax - trQ) / 2
    qj = (Q[i, j] + Q[j, i]) / 4 * qi
    qk = (Q[i, k] + Q[k, i]) / 4 * qi
    qs = (Q[k, j] - Q[j, k]) / 4 * qi

    # qi = (R[k, j] - R[j, k]) / (4 * qw)

    q = np.zeros(4)
    q[0] = qs
    q[i + 1] = qi
    q[j + 1] = qj
    q[k + 1] = qk
    return q


Q = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
matrix_to_quaternion(Q)

# %%
# decision_matrix = np.empty((num_rotations, 4))
# decision_matrix = np.empty(4)
# # decision_matrix[:, :3] = rot_matrix.diagonal()  # The diagonal elements of the rotation matrix
# decision_matrix[:3] = rot_matrix.diagonal()  # The diagonal elements of the rotation matrix
# decision_matrix[-1] = decision_matrix[:3].sum()  # The trace of the rotation matrix

choices = decision_matrix.argmax(axis=0)  # The index of the maximum element in each row


quat = np.empty(4)

ind = np.nonzero(choices != 3)[
    0
]  # The indices where the maximum element is not the last one
i = choices  # [ind]  # The indices of the maximum elements
j = (i + 1) % 3  # The indices of the next elements
k = (j + 1) % 3  # The indices of the next next elements

quat[i] = (
    np.sqrt(1 + 2 * rot_matrix[i, i] - decision_matrix[-1]) / 2
)  # The quaternion component corresponding to the maximum element
quat[j] = (rot_matrix[j, i] + rot_matrix[i, j]) / (
    4 * quat[i]
)  # The next quaternion component
quat[k] = (rot_matrix[k, i] + rot_matrix[i, k]) / (
    4 * quat[i]
)  # The next next quaternion component
quat[3] = (rot_matrix[k, j] - rot_matrix[j, k]) / (
    4 * quat[i]
)  # The scalar quaternion component


# %%
th = np.random.rand(12) * 2 * np.pi
f = np.cos(th)
ff = 2 * np.cos(th / 2) ** 2 - 1
fff = 1 - 2 * np.sin(th / 2) ** 2
fff - f
