try:
    import sys
    from pathlib import Path

    nb_dir = Path().resolve()
    proj_dir = (nb_dir / "..").resolve()
    sys.path.insert(0, str(proj_dir))

    import numpy as np
    import matplotlib.pyplot as plt
    from src_python.time_series import read_time_series, plot_log_log_fit
except:
    pass

# %%
# Curvature and bending stress convergence

output_dir = "../output/bending_force_test_angle_defect"
# output_dir = "../output/bending_force_test_laplacian"
# output_dir = "../output"

Nf = np.array(
    [
        320,
        1280,
        5120,
        20480,
    ]
)

L_paths = [
    f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_average_edge_length.dat"
    for _ in Nf
]
H_paths = [
    f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_mean_curvature_V.dat"
    for _ in Nf
]
lapH_paths = [
    f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_lap_mean_curvature_V.dat"
    for _ in Nf
]
K_paths = [
    f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_gaussian_curvature_V.dat"
    for _ in Nf
]

Ne = 3 * Nf / 2
Nv = 2 + Nf / 2
L = np.array([read_time_series(path)[0] for path in L_paths])


H = [read_time_series(path)[0] for path in H_paths]
lapH = [read_time_series(path)[0] for path in lapH_paths]
K = [read_time_series(path)[0] for path in K_paths]
F = [-2 * (laph + 2 * h * (h**2 - k)) for h, laph, k in zip(H, lapH, K)]

H0 = -1.0
lapH0 = 0.0
K0 = 1.0
F0 = -2 * (lapH0 + 2 * H0 * (H0**2 - K0))


eps_H = [np.abs(h - H0) for h in H]
eps_lapH = [np.abs(laph - lapH0) for laph in lapH]
eps_K = [np.abs(k - K0) for k in K]
eps_F = [np.abs(f - F0) for f in F]

# normalizedL2_err_H used by Belkin
normalizedL2_err_H = [
    np.linalg.norm(h - H0) / np.linalg.norm(H0 * np.ones_like(h)) for h in H
]

L2_eps_H = [np.linalg.norm(e, 2) for e in eps_H]
L2_eps_lapH = [np.linalg.norm(e, 2) for e in eps_lapH]
L2_eps_K = [np.linalg.norm(e, 2) for e in eps_K]
L2_eps_F = [np.linalg.norm(e, 2) for e in eps_F]

Linf_eps_H = [np.linalg.norm(e, np.inf) for e in eps_H]
Linf_eps_lapH = [np.linalg.norm(e, np.inf) for e in eps_lapH]
Linf_eps_K = [np.linalg.norm(e, np.inf) for e in eps_K]
Linf_eps_F = [np.linalg.norm(e, np.inf) for e in eps_F]

mean_eps_H = np.array([np.mean(e) for e in eps_H])
mean_eps_lapH = np.array([np.mean(e) for e in eps_lapH])
mean_eps_K = np.array([np.mean(e) for e in eps_K])
mean_eps_F = np.array([np.mean(e) for e in eps_F])

plot_log_log_fit(1 / L, normalizedL2_err_H)

# %%
# Area and volume conservation SPHERE


output_dir = "../output/area_volume_test_sphere"
output_dir = "../output"
Nf = np.array(
    [
        320,
        1280,
        5120,
    ]
)
t_max = 0.05

time_paths = [
    f"{output_dir}/area_volume_test_sphere_{nf:0>6}/raw_data/t.dat" for nf in Nf
]
area_paths = [
    f"{output_dir}/area_volume_test_sphere_{nf:0>6}/raw_data/envelope_area.dat"
    for nf in Nf
]
volume_paths = [
    f"{output_dir}/area_volume_test_sphere_{nf:0>6}/raw_data/envelope_volume.dat"
    for nf in Nf
]

time = [read_time_series(path) for path in time_paths]
area = [read_time_series(path) for path in area_paths]
volume = [read_time_series(path) for path in volume_paths]

dA_A = [(A - A[0]) / A[0] for A in area]
dV_V = [(V - V[0]) / V[0] for V in volume]


# len(time[0][time[0]<t_max])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.25, 3.25))

for nf, t, da_a, dv_v in zip(Nf, time, dA_A, dV_V):
    t_mask = t <= t_max
    axes[0].plot(t[t_mask], da_a[t_mask], label=r"$N_f=" + f"{nf}" + r"$")
    axes[1].plot(t[t_mask], dv_v[t_mask], label=r"$N_f=" + f"{nf}" + r"$")
plt.legend()

plt.show()
# %%
# Area and volume conservation MITOSIS


output_dir = "../output/area_volume_test_mitosis"
rows = [
    0,
    1,
    2,
    3,
]
cols = [0]
num_rows = 4
num_cols = 1
t_max = 5.0
err_min = 0.001

R_contact = np.array(
    [
        0.65,
        0.45,
        0.25,
        0.15,
    ]
)

time_paths = [
    f"{output_dir}/run_{row:0>2}_{col:0>2}/raw_data/t.dat"
    for row in rows
    for col in cols
]
area_paths = [
    f"{output_dir}/run_{row:0>2}_{col:0>2}/raw_data/envelope_area.dat"
    for row in rows
    for col in cols
]
volume_paths = [
    f"{output_dir}/run_{row:0>2}_{col:0>2}/raw_data/envelope_volume.dat"
    for row in rows
    for col in cols
]

time = [read_time_series(path) for path in time_paths]
area = [read_time_series(path) for path in area_paths]
volume = [read_time_series(path) for path in volume_paths]

dA_A = [(A - A[0]) / A[0] for A in area]
dV_V = [(V - V[0]) / V[0] for V in volume]


# len(time[0][time[0]<t_max])
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(8, 4),
    # sharey=True,
)

for r, t, da_a, dv_v in zip(R_contact, time, dA_A, dV_V):
    t_mask = t <= t_max
    err_mask = da_a >= err_min
    plot_mask = np.logical_and(t_mask, err_mask)
    tt = t[plot_mask]
    tt -= tt[0]
    a_err = da_a[plot_mask]
    v_err = dv_v[plot_mask]
    # axes[0].plot(t[t_mask], da_a[t_mask], label=r"$\rho=" + f"{r}" + r"$")
    # axes[1].plot(t[t_mask], dv_v[t_mask], label=r"$\rho=" + f"{r}" + r"$")
    axes[0].plot(tt, a_err, label=r"$\rho=" + f"{r}" + r"$")
    axes[1].plot(tt, v_err, label=r"$\rho=" + f"{r}" + r"$")
plt.legend()
plt.show()
