# os.path.exists
# os.path.join
# os.path.relpath
# os.getcwd
# os.chdir
# os.path.basename
import numpy as np
import matplotlib.pyplot as plt
from src.python.pretty_pictures import get_plt_combos, scalars_to_rgba, movie
from src.python.special_functions import (
    unit_bump,
    bump3,
    tethering_potential_vutukuri,
    tethering_potential_vutukuri0,
)
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer

# ply = "./data/half_edge_base/ply/oblate_005120_he.ply"
# params = Brane.vutukuri_params(num_faces=5120, radius=1.0)
ply = "./data/half_edge_base/ply/unit_torus_3_1_raw_006144_he.ply"
m = Brane.from_he_ply(ply)
m.default_params()
# %%
m.default_spontaneous_edge_length()
np.mean(m.length_H())
# %%
mv = MeshViewer(m)
points = m.xyz_coord_V
# vectors = .01*F_harmonic
vectors = 0.1 * F_analytic
mv.add_vector_field(points, vectors)
mv.plot()
# %%


def _Fbend_harmonic(self):
    Nv = self.num_vertices
    for vrt in range(Nv):
        sum_lsqT = 0.0
        sum_del_lsqT = np.zeros(3)
        sum_rT = np.zeros(3)
        sum_del_rT = np.zeros((3, 3))

        for i_neighb in range(vrt.n_neighbs_):
            i_plus = 0 if i_neighb == (vrt.n_neighbs_ - 1) else i_neighb + 1
            i_minus = vrt.n_neighbs_ - 1 if i_neighb == 0 else i_neighb - 1
            neighb = vrt.neighbs_[i_neighb]
            neighb_plus = vrt.neighbs_[i_plus]
            neighb_minus = vrt.neighbs_[i_minus]

            r_ij = vrt.pos_ - neighb.pos_
            r_ij_plus = vrt.pos_ - neighb_plus.pos_
            r_ij_minus = vrt.pos_ - neighb_minus.pos_
            r_jj_plus = neighb.pos_ - neighb_plus.pos_
            r_jj_minus = neighb.pos_ - neighb_minus.pos_
            l_ij = np.linalg.norm(r_ij)
            l_ij_plus = np.linalg.norm(r_ij_plus)
            l_ij_minus = np.linalg.norm(r_ij_minus)
            l_jj_plus = np.linalg.norm(r_jj_plus)
            l_jj_minus = np.linalg.norm(r_jj_minus)
            chi_minus = np.dot(r_ij_minus, r_jj_minus) / (l_ij_minus * l_jj_minus)
            chi_plus = np.dot(r_ij_plus, r_jj_plus) / (l_ij_plus * l_jj_plus)
            T_ij = (chi_minus / np.sqrt(1.0 - chi_minus**2)) + (chi_plus / np.sqrt(1.0 - chi_plus**2))
            grad_lsq = 2 * r_ij
            grad_chi_plus = (1.0 / (l_ij_plus * l_jj_plus)) * (
                r_jj_plus - (l_jj_plus / l_ij_plus) * chi_plus * r_ij_plus
            )
            grad_chi_minus = (1.0 / (l_ij_minus * l_jj_minus)) * (
                r_jj_minus - (l_jj_minus / l_ij_minus) * chi_minus * r_ij_minus
            )
            grad_T = grad_chi_plus / (1.0 - chi_plus**2) ** 1.5 + grad_chi_minus / (1.0 - chi_minus**2) ** 1.5
            sum_lsqT += l_ij**2 * T_ij
            sum_rT += r_ij * T_ij
            sum_del_lsqT += T_ij * grad_lsq + l_ij**2 * grad_T
            for i_dim in range(3):
                sum_del_rT[i_dim] += r_ij[i_dim] * grad_T
                if i_dim == i_dim:
                    sum_del_rT[i_dim] += T_ij
        f_bend = np.zeros(3)
        for i_dim in range(3):
            f_bend[i_dim] += 2 * np.dot(sum_rT, sum_rT) / sum_lsqT**2 * sum_del_lsqT[i_dim]
            f_bend[i_dim] -= 4 * np.dot(sum_rT, sum_del_rT[i_dim]) / sum_lsqT
            f_bend[i_dim] *= self.kappa_


# %%
# kbT=1

# tether
# mesh_kB: 16.1 # 80.5
# kappa_B_ = params_->mesh_kB;
length_reg_stiffness = 16.1

# bend
# mesh_k: 0.4 # 4.0 # 20.1
# kappa_ = params_->mesh_k;
bending_modulus = 0.4

# area
# mesh_kl: 0.1 # 1 # 5.03
# kappa_l_ = params_->mesh_kl;
area_reg_stiffness = 0.1

# volume
# mesh_kV: 0.3 # 3.0 # 15.2
# kappa_v_ = params_->mesh_kV;
volume_reg_stiffness = 0.3


# mesh_node_gamma: 0.2 # 0.02
# gamma_ = params_->node_gamma;
linear_drag_coeff = 0.2


# r_sys_ = params_->system_radius;
R = 40

Nf = 5120
# vesicle radius at equilibrium
# R = radius
# thermal energy unit
# kBT = 0.2
kBT = 1
# time scale
tau = 1.28e5

# friction coefficient
linear_drag_coeff = 0.4 * kBT * tau / R**2

##########################################
# KMC parameters
# flipping frequency
flip_freq = 6.4e6 / tau
# flipping probability
flip_prop = 0.3

##########################################
# Bending force parameters
# bending rigidity
bending_modulus = 20 * kBT
# spontaneous curvature
spontaneous_curvature = 0.0
# splay modulus
splay_modulus = 0.0

##########################################
# Area constraint/penalty parameters
# desired vesicle area
A = 4 * np.pi * R**2
# desired face area
spontaneous_face_area = A / Nf
# local area stiffness
area_reg_stiffness = 6.43e6 * kBT / A

##########################################
# Volume constraint/penalty parameters
# desired vesicle volume
spontaneous_volume = 4 * np.pi * R**3 / 3
# volume stiffness
volume_reg_stiffness = 1.6e7 * kBT / R**3

################################################
# Edge length and tethering potential parameters
# bond stiffness
length_reg_stiffness = 80 * kBT
# average bond length
spontaneous_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))
# minimum bond length
min_edge_length = 0.6 * spontaneous_edge_length
# potential cutoff lengths
tether_cutoff_length1 = 0.8 * spontaneous_edge_length  # onset of repulsion
tether_cutoff_length0 = 1.2 * spontaneous_edge_length  # onset of attraction
# maximum bond length
max_edge_length = 1.4 * spontaneous_edge_length

Lscale = 1.0  # length scale
Dl = 0.5 * (max_edge_length - min_edge_length)  # = 0.4 * spontaneous_edge_length
mu = (spontaneous_edge_length - tether_cutoff_length1) / Dl  # = .5
lam = Lscale / Dl
nu = Lscale / Dl

brane_kwargs = {
    "length_reg_stiffness": length_reg_stiffness,
    "area_reg_stiffness": area_reg_stiffness,
    "volume_reg_stiffness": volume_reg_stiffness,
    "bending_modulus": bending_modulus,
    "splay_modulus": splay_modulus,
    "spontaneous_curvature": spontaneous_curvature,
    "linear_drag_coeff": linear_drag_coeff,
    "spontaneous_edge_length": spontaneous_edge_length,
    "spontaneous_face_area": spontaneous_face_area,
    "spontaneous_volume": spontaneous_volume,
    "tether_Dl": Dl,
    "tether_mu": mu,
    "tether_lam": lam,
    "tether_nu": nu,
}
