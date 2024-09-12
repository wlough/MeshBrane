# os.path.exists
# os.path.join
# os.path.relpath
# os.getcwd
# os.chdir
# os.path.basename
import numpy as np
import matplotlib.pyplot as plt
from src.python.pretty_pictures import get_plt_combos, scalars_to_rgba, movie
from src.python.special_functions import unit_bump, bump3
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer


def Fbend_harmonic(self):
    def delta(i, j):
        return 1 if i == j else 0

    def L(i, j):
        return np.linalg.norm(self.xyz_coord_v(i) - self.xyz_coord_v(j))

    def dL2_dx(i, j, l):
        xi = self.xyz_coord_v(i)
        xj = self.xyz_coord_v(j)
        dil = delta(i, l)
        djl = delta(j, l)
        return 2 * (xi - xj) * (dil - djl)

    def chi(i, j, m):
        xi = self.xyz_coord_v(i)
        xj = self.xyz_coord_v(j)
        xm = self.xyz_coord_v(m)
        Lim = L(i, m)
        Ljm = L(j, m)
        return np.dot(xi - xm, xj - xm) / (Lim * Ljm)

    def T(i, j, jm1, jp1):
        chi_ijjp1 = chi(i, j, jp1)
        chi_ijjm1 = chi(i, j, jm1)
        return chi_ijjp1 / np.sqrt(1 - chi_ijjp1**2) + chi_ijjm1 / np.sqrt(
            1 - chi_ijjm1**2
        )

    def dchi_dx(i, j, m, l):
        xi = self.xyz_coord_v(i)
        xj = self.xyz_coord_v(j)
        xm = self.xyz_coord_v(m)
        Lim = L(i, m)
        Ljm = L(j, m)
        dil = delta(i, l)
        dml = delta(m, l)
        djl = delta(j, l)
        chi_ijm = chi(i, j, m)

        return (
            (dil - dml) * (xj - xm)
            + (djl - dml) * (xi - xm)
            - (dil - dml) * (xi - xm) * chi_ijm * Ljm / Lim
            - (djl - dml) * (xj - xm) * chi_ijm * Lim / Ljm
        ) / (Lim * Ljm)

    def dT_dx(i, j, jm1, jp1, l):
        chi_ijjp1 = chi(i, j, jp1)
        chi_ijjm1 = chi(i, j, jm1)
        dchi_ijjp1_dxl = dchi_dx(i, j, jp1, l)
        dchi_ijjm1_dxl = dchi_dx(i, j, jm1, l)
        return (
            dchi_ijjm1_dxl / (1 - chi_ijjm1**2) ** 1.5
            + dchi_ijjp1_dxl / (1 - chi_ijjp1**2) ** 1.5
        )

    kappa = self.bending_modulus
    F = np.zeros_like(self.xyz_coord_V)
    I = np.eye(3)
    Nv = self.num_vertices
    for l in range(Nv):
        Fl = np.zeros(3)
        #################
        i = l
        #################
        dil = delta(i, l)
        xi = self.xyz_coord_v(i)
        sum_Lij2_Tij = 0.0
        sum_xi_xj_Tij = np.zeros(3)
        # sum_xi_xj_dTij_dxl = np.zeros(3, 3)
        sum_xi_xj_dTij_dxl_Tij_dil_dij = np.zeros((3, 3))
        sum_Tij_dLij2_dxl_Lij2_dTij_dxl = np.zeros(3)
        for hij in self.generate_H_out_v_clockwise(i):
            hijp1 = self.h_twin_h(self.h_prev_h(hij))
            hijm1 = self.h_next_h(self.h_twin_h(hij))
            j = self.v_head_h(hij)
            jp1 = self.v_head_h(hijp1)
            jm1 = self.v_head_h(hijm1)

            djl = delta(j, l)
            xj = self.xyz_coord_v(j)

            Lij = L(i, j)
            dLij2_dxl = dL2_dx(i, j, l)
            Tij = T(i, j, jm1, jp1)
            dTij_dxl = dT_dx(i, j, jm1, jp1, l)

            sum_Lij2_Tij += Lij**2 * Tij
            sum_xi_xj_Tij += (xi - xj) * Tij
            sum_xi_xj_dTij_dxl_Tij_dil_dij += (
                np.outer((xi - xj) * Tij, dTij_dxl) + Tij * (dil - djl) * I
            )
            sum_Tij_dLij2_dxl_Lij2_dTij_dxl += Tij * dLij2_dxl + Lij**2 * dTij_dxl

        Fl += 4 * np.dot(sum_xi_xj_Tij, sum_xi_xj_dTij_dxl_Tij_dil_dij) / sum_Lij2_Tij
        Fl += (
            -2
            * np.linalg.norm(sum_xi_xj_Tij) ** 2
            * sum_Tij_dLij2_dxl_Lij2_dTij_dxl
            / sum_Lij2_Tij**2
        )
        #################
        # i in j(l)
        #################
        for hli in self.generate_H_out_v_clockwise(l):
            #################
            i = self.v_head_h(hli)
            #################
            dil = delta(i, l)
            xi = self.xyz_coord_v(i)
            sum_Lij2_Tij = 0.0
            sum_xi_xj_Tij = np.zeros(3)
            # sum_xi_xj_dTij_dxl = np.zeros(3, 3)
            sum_xi_xj_dTij_dxl_Tij_dil_dij = np.zeros((3, 3))
            sum_Tij_dLij2_dxl_Lij2_dTij_dxl = np.zeros(3)
            for hij in self.generate_H_out_v_clockwise(i):
                hijp1 = self.h_twin_h(self.h_prev_h(hij))
                hijm1 = self.h_next_h(self.h_twin_h(hij))
                j = self.v_head_h(hij)
                jp1 = self.v_head_h(hijp1)
                jm1 = self.v_head_h(hijm1)

                djl = delta(j, l)
                xj = self.xyz_coord_v(j)

                Lij = L(i, j)
                dLij2_dxl = dL2_dx(i, j, l)
                Tij = T(i, j, jm1, jp1)
                dTij_dxl = dT_dx(i, j, jm1, jp1, l)

                sum_Lij2_Tij += Lij**2 * Tij
                sum_xi_xj_Tij += (xi - xj) * Tij
                sum_xi_xj_dTij_dxl_Tij_dil_dij += (
                    np.outer((xi - xj) * Tij, dTij_dxl) + Tij * (dil - djl) * I
                )
                sum_Tij_dLij2_dxl_Lij2_dTij_dxl += Tij * dLij2_dxl + Lij**2 * dTij_dxl

            Fl += (
                4 * np.dot(sum_xi_xj_Tij, sum_xi_xj_dTij_dxl_Tij_dil_dij) / sum_Lij2_Tij
            )
            Fl += (
                -2
                * np.linalg.norm(sum_xi_xj_Tij) ** 2
                * sum_Tij_dLij2_dxl_Lij2_dTij_dxl
                / sum_Lij2_Tij**2
            )

        ####################################
        Fl *= -0.5 * kappa
        F[l] = Fl
    return F


def Fbend_harmonic2(self):
    def delta(i, j):
        return 1 if i == j else 0

    def L(i, j):
        return np.linalg.norm(self.xyz_coord_v(i) - self.xyz_coord_v(j))

    def dL2_dx(i, j, l):
        xi = self.xyz_coord_v(i)
        xj = self.xyz_coord_v(j)
        dil = delta(i, l)
        djl = delta(j, l)
        return 2 * (xi - xj) * (dil - djl)

    def chi(i, j, m):
        xi = self.xyz_coord_v(i)
        xj = self.xyz_coord_v(j)
        xm = self.xyz_coord_v(m)
        Lim = L(i, m)
        Ljm = L(j, m)
        return np.dot(xi - xm, xj - xm) / (Lim * Ljm)

    def T(i, j, jm1, jp1):
        chi_ijjp1 = chi(i, j, jp1)
        chi_ijjm1 = chi(i, j, jm1)
        return chi_ijjp1 / np.sqrt(1 - chi_ijjp1**2) + chi_ijjm1 / np.sqrt(
            1 - chi_ijjm1**2
        )

    def dchi_dx(i, j, m, l):
        xi = self.xyz_coord_v(i)
        xj = self.xyz_coord_v(j)
        xm = self.xyz_coord_v(m)
        Lim = L(i, m)
        Ljm = L(j, m)
        dil = delta(i, l)
        dml = delta(m, l)
        djl = delta(j, l)
        chi_ijm = chi(i, j, m)

        return (
            (dil - dml) * (xj - xm)
            + (djl - dml) * (xi - xm)
            - (dil - dml) * (xi - xm) * chi_ijm * Ljm / Lim
            - (djl - dml) * (xj - xm) * chi_ijm * Lim / Ljm
        ) / (Lim * Ljm)

    def dT_dx(i, j, jm1, jp1, l):
        chi_ijjp1 = chi(i, j, jp1)
        chi_ijjm1 = chi(i, j, jm1)
        dchi_ijjp1_dxl = dchi_dx(i, j, jp1, l)
        dchi_ijjm1_dxl = dchi_dx(i, j, jm1, l)
        return (
            dchi_ijjm1_dxl / (1 - chi_ijjm1**2) ** 1.5
            + dchi_ijjp1_dxl / (1 - chi_ijjp1**2) ** 1.5
        )

    kappa = self.bending_modulus
    F = np.zeros_like(self.xyz_coord_V)
    I = np.eye(3)
    Nv = self.num_vertices
    for l in range(Nv):
        Fl = np.zeros(3)
        #################
        i = l
        #################
        # dil = delta(i, l)
        # dil = 1
        xi = self.xyz_coord_v(i)
        sum_Lij2_Tij = 0.0
        sum_xi_xj_Tij = np.zeros(3)
        # sum_xi_xj_dTij_dxl = np.zeros(3, 3)
        sum_xi_xj_dTij_dxl_Tij_dil_dij = np.zeros((3, 3))
        sum_Tij_dLij2_dxl_Lij2_dTij_dxl = np.zeros(3)
        for hij in self.generate_H_out_v_clockwise(i):
            hijp1 = self.h_twin_h(self.h_prev_h(hij))
            hijm1 = self.h_next_h(self.h_twin_h(hij))
            j = self.v_head_h(hij)
            jp1 = self.v_head_h(hijp1)
            jm1 = self.v_head_h(hijm1)

            # djl = delta(j, l)
            # djl = 0
            xj = self.xyz_coord_v(j)

            Lij = L(i, j)
            dLij2_dxl = dL2_dx(i, j, l)
            Tij = T(i, j, jm1, jp1)
            dTij_dxl = dT_dx(i, j, jm1, jp1, l)

            sum_Lij2_Tij += Lij**2 * Tij
            sum_xi_xj_Tij += (xi - xj) * Tij
            sum_xi_xj_dTij_dxl_Tij_dil_dij += (
                np.outer((xi - xj) * Tij, dTij_dxl) + Tij * I
            )
            sum_Tij_dLij2_dxl_Lij2_dTij_dxl += Tij * dLij2_dxl + Lij**2 * dTij_dxl

        Fl += 4 * np.dot(sum_xi_xj_Tij, sum_xi_xj_dTij_dxl_Tij_dil_dij) / sum_Lij2_Tij
        Fl += (
            -2
            * np.linalg.norm(sum_xi_xj_Tij) ** 2
            * sum_Tij_dLij2_dxl_Lij2_dTij_dxl
            / sum_Lij2_Tij**2
        )
        #################
        # i in j(l)
        #################
        for hli in self.generate_H_out_v_clockwise(l):
            #################
            i = self.v_head_h(hli)
            #################
            # dil = delta(i, l)
            # dil = 0
            xi = self.xyz_coord_v(i)
            sum_Lij2_Tij = 0.0
            sum_xi_xj_Tij = np.zeros(3)
            # sum_xi_xj_dTij_dxl = np.zeros(3, 3)
            sum_xi_xj_dTij_dxl_Tij_dil_dij = np.zeros((3, 3))
            sum_Tij_dLij2_dxl_Lij2_dTij_dxl = np.zeros(3)
            for hij in self.generate_H_out_v_clockwise(i):
                hijp1 = self.h_twin_h(self.h_prev_h(hij))
                hijm1 = self.h_next_h(self.h_twin_h(hij))
                j = self.v_head_h(hij)
                jp1 = self.v_head_h(hijp1)
                jm1 = self.v_head_h(hijm1)

                djl = delta(j, l)
                xj = self.xyz_coord_v(j)

                Lij = L(i, j)
                dLij2_dxl = dL2_dx(i, j, l)
                Tij = T(i, j, jm1, jp1)
                dTij_dxl = dT_dx(i, j, jm1, jp1, l)

                sum_Lij2_Tij += Lij**2 * Tij
                sum_xi_xj_Tij += (xi - xj) * Tij
                sum_xi_xj_dTij_dxl_Tij_dil_dij += (
                    np.outer((xi - xj) * Tij, dTij_dxl) + Tij * (-djl) * I
                )
                sum_Tij_dLij2_dxl_Lij2_dTij_dxl += Tij * dLij2_dxl + Lij**2 * dTij_dxl

            Fl += (
                4 * np.dot(sum_xi_xj_Tij, sum_xi_xj_dTij_dxl_Tij_dil_dij) / sum_Lij2_Tij
            )
            Fl += (
                -2
                * np.linalg.norm(sum_xi_xj_Tij) ** 2
                * sum_Tij_dLij2_dxl_Lij2_dTij_dxl
                / sum_Lij2_Tij**2
            )

        ####################################
        Fl *= -0.5 * kappa
        F[l] = Fl
    return F


def Fbend_shane(self):
    Fbend = np.zeros_like(self.xyz_coord_V)
    Nv = self.num_vertices
    for vrt in range(Nv):
        sum_lsqT = 0.0
        sum_del_lsqT = np.zeros(3)
        sum_rT = np.zeros(3)
        sum_del_rT = np.zeros((3, 3))
        for h in self.generate_H_out_v_clockwise(vrt):
            h_plus = self.h_twin_h(self.h_prev_h(h))
            h_minus = self.h_next_h(self.h_twin_h(h))
            neighb = self.v_head_h(h)
            neighb_plus = self.v_head_h(h_plus)
            neighb_minus = self.v_head_h(h_minus)

            r_ij = self.xyz_coord_v(vrt) - self.xyz_coord_v(neighb)
            r_ij_plus = self.xyz_coord_v(vrt) - self.xyz_coord_v(neighb_plus)
            r_ij_minus = self.xyz_coord_v(vrt) - self.xyz_coord_v(neighb_minus)
            r_jj_plus = self.xyz_coord_v(neighb) - self.xyz_coord_v(neighb_plus)
            r_jj_minus = self.xyz_coord_v(neighb) - self.xyz_coord_v(neighb_minus)
            #########################################
            l_ij = np.linalg.norm(r_ij)
            l_ij_plus = np.linalg.norm(r_ij_plus)
            l_ij_minus = np.linalg.norm(r_ij_minus)
            l_jj_plus = np.linalg.norm(r_jj_plus)
            l_jj_minus = np.linalg.norm(r_jj_minus)
            chi_minus = np.dot(r_ij_minus, r_jj_minus) / (l_ij_minus * l_jj_minus)
            chi_plus = np.dot(r_ij_plus, r_jj_plus) / (l_ij_plus * l_jj_plus)
            T_ij = (chi_minus / np.sqrt(1.0 - chi_minus**2)) + (
                chi_plus / np.sqrt(1.0 - chi_plus**2)
            )
            grad_lsq = 2 * r_ij
            grad_chi_plus = (1.0 / (l_ij_plus * l_jj_plus)) * (
                r_jj_plus - (l_jj_plus / l_ij_plus) * chi_plus * r_ij_plus
            )
            grad_chi_minus = (1.0 / (l_ij_minus * l_jj_minus)) * (
                r_jj_minus - (l_jj_minus / l_ij_minus) * chi_minus * r_ij_minus
            )
            grad_T = (
                grad_chi_plus / (1.0 - chi_plus**2) ** 1.5
                + grad_chi_minus / (1.0 - chi_minus**2) ** 1.5
            )
            sum_lsqT += l_ij**2 * T_ij
            sum_rT += r_ij * T_ij
            sum_del_lsqT += T_ij * grad_lsq + l_ij**2 * grad_T
            for i_dim in range(3):
                sum_del_rT[i_dim] += r_ij[i_dim] * grad_T
                if i_dim == i_dim:
                    sum_del_rT[i_dim] += T_ij
        # f_bend = np.zeros(3)
        for i_dim in range(3):
            Fbend[vrt, i_dim] += (
                2 * np.dot(sum_rT, sum_rT) / sum_lsqT**2 * sum_del_lsqT[i_dim]
            )
            Fbend[vrt, i_dim] -= 4 * np.dot(sum_rT, sum_del_rT[i_dim]) / sum_lsqT
            Fbend[vrt, i_dim] *= self.bending_modulus
    return Fbend


# ply = "./data/half_edge_base/ply/oblate_005120_he.ply"
# params = Brane.vutukuri_params(num_faces=5120, radius=1.0)
ply = "./data/half_edge_base/ply/unit_torus_3_1_raw_006144_he.ply"
R = np.sqrt(1 / (4 * np.pi))
params = Brane.vutukuri_params(num_faces=6144, radius=R)
# m = Brane.unit_vutukuri_vesicle()
m = Brane.from_he_ply(ply)
np.linalg.norm(m.xyz_coord_V, axis=-1)
m.update_params(**params)
F_harmonic = Fbend_harmonic(m)
F_harmonic2 = Fbend_harmonic2(m)
F_analytic = m.Fbend_analytic()
F_shane = Fbend_shane(m)
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
            T_ij = (chi_minus / np.sqrt(1.0 - chi_minus**2)) + (
                chi_plus / np.sqrt(1.0 - chi_plus**2)
            )
            grad_lsq = 2 * r_ij
            grad_chi_plus = (1.0 / (l_ij_plus * l_jj_plus)) * (
                r_jj_plus - (l_jj_plus / l_ij_plus) * chi_plus * r_ij_plus
            )
            grad_chi_minus = (1.0 / (l_ij_minus * l_jj_minus)) * (
                r_jj_minus - (l_jj_minus / l_ij_minus) * chi_minus * r_ij_minus
            )
            grad_T = (
                grad_chi_plus / (1.0 - chi_plus**2) ** 1.5
                + grad_chi_minus / (1.0 - chi_minus**2) ** 1.5
            )
            sum_lsqT += l_ij**2 * T_ij
            sum_rT += r_ij * T_ij
            sum_del_lsqT += T_ij * grad_lsq + l_ij**2 * grad_T
            for i_dim in range(3):
                sum_del_rT[i_dim] += r_ij[i_dim] * grad_T
                if i_dim == i_dim:
                    sum_del_rT[i_dim] += T_ij
        f_bend = np.zeros(3)
        for i_dim in range(3):
            f_bend[i_dim] += (
                2 * np.dot(sum_rT, sum_rT) / sum_lsqT**2 * sum_del_lsqT[i_dim]
            )
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
