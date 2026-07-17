import sys
from pathlib import Path

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))

import numpy as np
from src_python.time_series import read_time_series, plot_log_log_fit


# %%
# Curvature and bending stress convergence

output_dir = "../output/bending_force_test_angle_defect"
# output_dir = "../output/bending_force_test_laplacian"
# output_dir = "../output"

L_paths = [f"{output_dir}/bending_force_test{_}/raw_data/envelope_average_edge_length.dat" for _ in range(3)]
H_paths = [f"{output_dir}/bending_force_test{_}/raw_data/envelope_mean_curvature_V.dat" for _ in range(3)]
lapH_paths = [f"{output_dir}/bending_force_test{_}/raw_data/envelope_lap_mean_curvature_V.dat" for _ in range(3)]
K_paths = [f"{output_dir}/bending_force_test{_}/raw_data/envelope_gaussian_curvature_V.dat" for _ in range(3)]

Nf = np.array([1280, 5120, 20480])
Ne = 3*Nf/2
Nv = 2 + Nf/2
L = np.array([read_time_series(path)[0] for path in L_paths])


H = [read_time_series(path)[0] for path in H_paths]
lapH = [read_time_series(path)[0] for path in lapH_paths]
K = [read_time_series(path)[0] for path in K_paths]
F = [-2*(laph +2*h*(h**2-k)) for h, laph, k in zip(H, lapH, K)]

H0 = -1.0
lapH0 = 0.0
K0 = 1.0
F0 = -2*(lapH0 +2*H0*(H0**2-K0))


eps_H = [np.abs(h-H0) for h in H]
eps_lapH = [np.abs(laph-lapH0) for laph in lapH]
eps_K = [np.abs(k-K0) for k in K]
eps_F = [np.abs(f-F0) for f in F]

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

plot_log_log_fit(Nf, Linf_eps_lapH)

# %%
# Area and volume conservation
