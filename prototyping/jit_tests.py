from src.python.ply_tools import SphereFactory

# import numpy as np
# from time import time
# import matplotlib.pyplot as plt
# from src.python.utilities import round_to, log_log_fit

# from src.python.half_edge_test import (
#     # get_plt_combos,
#     # scalars_to_rgba,
#     to_scinotation_tex,
# )

# sfpy = SphereFactory(jit=False)
# sfjit = SphereFactory(jit=True)
# sfjit.v2h
# sfjit.refine()
SphereFactory.build_test_plys(num_refine=8, jit=True, name="unit_sphere")
# %%
sf = sfjit
num_refine = 6
Tpy = np.zeros(num_refine)
Tjit = np.zeros(num_refine)
tSamples = []
Nv = sfjit._NUM_VERTICES_[: num_refine + 1]
# %%
print("-----------------")
for _ in range(len(Nv)):
    # nv = Nv[_]
    # print(f"num_vertices={nv}")
    # Tpy[_] = time()
    # sfpy.refine()
    # Tpy[_] = time() - Tpy[_]
    Tjit[_] = time()
    sfjit.refine()
    Tjit[_] = time() - Tjit[_]
    # print(f"tpy = {Tpy[_]}")
    print(f"tjit = {Tjit[_]}")
    # print(f"tpy/tjit = {Tpy[_]/Tjit[_]}\n-----------------")
# %%
n0, nf = 4, len(Nv)
T = Tjit
X, Y = Nv[n0 + 1 : nf + 1], T[n0 : nf + 1]
# [12, 42, 162, 642, 2562, 10242, 40962, 163842]
fit = log_log_fit(X, Y)
x = np.array(fit["logX"])
y = np.array(fit["logY"])
f = np.array(fit["F"])
p = fit["m"]
fun = fit["fun"]
Npred = np.array([12, 42, 162, 642, 2562, 10242, 40962, 163842, 163842 * 4])[n0 + 1 :]
Tpred = np.array([fun(_) for _ in Npred])
xpred, ypred = np.log(Npred), np.log(Tpred)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, "o")
ax.plot(x, f, label=f"{p=}")
ax.plot(xpred, ypred, "*", label="predicted")

ax.set_xlabel(r"$\text{Vertex count}$")
ax.set_xticks(xpred)
ax.set_xticklabels(to_scinotation_tex(Npred))
ax.set_ylabel(r"$\text{Convert time}$")
ax.set_yticks(ypred)
ax.set_yticklabels(to_scinotation_tex(Tpred, decimals=2))

ax.legend()
plt.tight_layout()
plt.show()
hour = Tpred // 3600
min = (Tpred - 3600 * hour) // 60
sec = Tpred - 3600 * hour - 60 * min
for nv, t in zip(Npred, Tpred):
    hour = t // 3600
    min = (t - 3600 * hour) // 60
    sec = np.round(t - 3600 * hour - 60 * min)

    print(f"---------\nNv={nv}")
    print(f"{hour}h {min}m {sec}s")
# %%
# from src.python.jit_utils import jit_get_index_of_twin, jit_source_samples_to_target_samples
# from src.python.ply_tools import VertTri2HalfEdgeConverter
# from src.python.half_edge_mesh import HalfEdgeMesh
# from time import time
# import matplotlib.pyplot as plt
# from src.python.utilities import round_to, log_log_fit
# import numpy as np
# from src.python.half_edge_test import (
#     get_plt_combos,
#     scalars_to_rgba,
#     to_scinotation_tex,
# )
#
# _NUM_VERTS_ = [
#     162,
#     642,
#     2562,
#     10242,
#     # 40962,
# ]  # [12, 42, 162, 642, 2562, 10242, 40962, 163842]
# _SURF_NAMES_ = [f"unit_sphere_{N:06d}_he" for N in _NUM_VERTS_]
#
# plys = [f"./data/ply/binary/{_}.ply" for _ in _SURF_NAMES_]
# M = [HalfEdgeMesh.from_half_edge_ply(_) for _ in plys]
# T = len(M) * [0]
# tSamples = []
# Nm = len(M)
# Nv = [m.num_vertices for m in M]
# # %%
# for _ in range(Nm):
#     m = M[_]
#     print(f"Nv = {m.num_vertices}")
#     V = m.xyz_array
#     F = m.V_of_F
#     H = m.V_of_H
#     h = 14
#     h_twin = jit_get_index_of_twin(H, h)
#     T[_] = time()
#     target_samples = jit_source_samples_to_target_samples(V, F)
#     T[_] = time() - T[_]
#     print(f"t = {T[_]}")
#     tSamples.append(target_samples)
#
#
# # %%
# n0, nf = 0, len(M)
# X, Y = Nv[n0 : nf + 1], T[n0 : nf + 1]
# # [12, 42, 162, 642, 2562, 10242, 40962, 163842]
# fit = log_log_fit(X, Y)
# x = np.array(fit["logX"])
# y = np.array(fit["logY"])
# f = np.array(fit["F"])
# p = fit["m"]
# fun = fit["fun"]
# Npred = np.array([162, 642, 2562, 10242, 40962, 163842])
# Tpred = np.array([fun(_) for _ in Npred])
# xpred, ypred = np.log(Npred), np.log(Tpred)
#
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x, y, "o")
# ax.plot(x, f, label=f"{p=}")
# ax.plot(xpred, ypred, "*", label="predicted")
#
# ax.set_xlabel(r"$\text{Vertex count}$")
# ax.set_xticks(xpred)
# ax.set_xticklabels(to_scinotation_tex(Npred))
# ax.set_ylabel(r"$\text{Convert time}$")
# ax.set_yticks(ypred)
# ax.set_yticklabels(to_scinotation_tex(Tpred, decimals=2))
#
# ax.legend()
# plt.tight_layout()
# plt.show()
# Tpred[-1] // 60
