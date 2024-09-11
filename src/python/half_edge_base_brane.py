from src.python.half_edge_base_mesh import HalfEdgeMeshBase
import numpy as np
from src.python.global_vars import _INT_TYPE_, _FLOAT_TYPE_


# Forces
# -bending force
# -area force
# -volume force
# -tether force
# -stochastic flips
# -external force
# Compute data for forces fun()
# Timestepping
# -Velocity Verlet
class Brane(HalfEdgeMeshBase):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
        #
        length_reg_stiffness=None,
        area_reg_stiffness=None,
        volume_reg_stiffness=None,
        bending_modulus=None,
        splay_modulus=None,
        spontaneous_curvature=None,
        linear_drag_coeff=None,
        spontaneous_edge_length=None,
        spontaneous_face_area=None,
        spontaneous_volume=None,
        tether_Dl=None,
        tether_mu=None,
        tether_lam=None,
        tether_nu=None,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        self.length_reg_stiffness = length_reg_stiffness
        self.area_reg_stiffness = area_reg_stiffness
        self.volume_reg_stiffness = volume_reg_stiffness
        self.bending_modulus = bending_modulus
        self.splay_modulus = splay_modulus
        self.spontaneous_curvature = spontaneous_curvature
        self.linear_drag_coeff = linear_drag_coeff

        # (
        #     default_spontaneous_edge_length,
        #     default_spontaneous_face_area,
        #     default_spontaneous_volume,
        # ) = self.default_spontaneous_length_area_volume()
        # if spontaneous_edge_length is None:
        #     spontaneous_edge_length = default_spontaneous_edge_length
        # if spontaneous_face_area is None:
        #     spontaneous_face_area = default_spontaneous_face_area
        # if spontaneous_volume is None:
        # spontaneous_volume = default_spontaneous_volume
        self.spontaneous_edge_length = spontaneous_edge_length
        self.spontaneous_face_area = spontaneous_face_area
        self.spontaneous_volume = spontaneous_volume
        self.tether_Dl = tether_Dl
        self.tether_mu = tether_mu
        self.tether_lam = tether_lam
        self.tether_nu = tether_nu

    @classmethod
    def vutukuri_vesicle(cls):

        # number of vertices/edges/faces
        # Nf-Ne+Nv=2
        # 2*Ne = 3*Nf
        # => Nf = 2*Nv - 4, Ne = 3*Nv - 6
        # Vutukuri actually uses Nv=30000
        # Nv = 30000
        Nv = 40962
        Ne = 3 * Nv - 6
        Nf = 2 * Nv - 4
        ply_path = "./data/half_edge_base/ply/vutukuri_vesicle_he.ply"

        # vesicle radius at equilibrium
        R = 32
        # thermal energy unit
        kBT = 0.2
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
        Dl = 0.5 * (
            max_edge_length - min_edge_length
        )  # = 0.4 * spontaneous_edge_length
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

        m = cls.from_he_ply(ply_path, **brane_kwargs)

        return m

    @classmethod
    def unscaled_vutukuri_vesicle(cls):

        # number of vertices/edges/faces
        # Nf-Ne+Nv=2
        # 2*Ne = 3*Nf
        # => Nf = 2*Nv - 4, Ne = 3*Nv - 6
        # Vutukuri actually uses Nv=30000
        # Nv = 30000
        # Nv = 40962
        # Ne = 3 * Nv - 6
        # Nf = 2 * Nv - 4
        # ply_path = "./data/half_edge_base/ply/vutukuri_vesicle_he.ply"

        Nv = 5120
        Ne = 3 * Nv - 6
        Nf = 2 * Nv - 4
        ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"

        # vesicle radius at equilibrium
        R = 32
        # thermal energy unit
        kBT = 0.2
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
        Dl = 0.5 * (
            max_edge_length - min_edge_length
        )  # = 0.4 * spontaneous_edge_length
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

        m = cls.from_he_ply(ply_path, **brane_kwargs)

        return m

    def default_spontaneous_length_area_volume(self):
        Nf = self.num_faces
        Nv = self.num_vertices
        volume = self.total_volume()
        area = self.total_area_of_faces()
        Rv = (3 * volume / (4 * np.pi)) ** (1 / 3)
        Ra = np.sqrt(area / (4 * np.pi))
        w = 0.75
        R = (1 - w) * Ra + w * Rv
        spontaneous_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))
        spontaneous_face_area = 4 * np.pi * R**2 / Nf
        spontaneous_volume = 4 * np.pi * R**3 / 3
        return spontaneous_edge_length, spontaneous_face_area, spontaneous_volume

    ######################################################
    # experimental stuff
    ######################

    def laplacian(self, Q):
        """
        Computes the cotan Laplacian of Q at each vertex
        """

        # lapQ = np.zeros_like(Q)
        # for vi in range(self.num_vertices):
        #     Atot = 0.0
        #     ri = self.xyz_coord_v(vi)
        #     qi = Q[vi]
        #     ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        #     for hij in self.generate_H_out_v_clockwise(vi):
        #         hijm1 = self.h_next_h(self.h_twin_h(hij))
        #         hijp1 = self.h_twin_h(self.h_prev_h(hij))
        #         vjm1 = self.v_head_h(hijm1)
        #         vj = self.v_head_h(hij)
        #         vjp1 = self.v_head_h(hijp1)

        #         qj = Q[vj]

        #         rjm1 = self.xyz_coord_v(vjm1)
        #         rj = self.xyz_coord_v(vj)
        #         rjp1 = self.xyz_coord_v(vjp1)

        #         rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
        #         rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
        #         rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
        #         ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
        #         ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
        #         rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
        #         ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
        #         rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

        #         Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
        #         Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
        #         Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
        #         Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
        #         Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

        #         cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

        #         cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

        #         cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
        #         cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

        #         Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
        #         lapQ[vi] += (cot_thetam + cot_thetap) * (qj - qi) / 2
        #     lapQ[vi] /= Atot

        # return lapQ
        return self.cotan_laplacian(Q)

    #####################
    # unit normal methods
    def normal_some_face_of_v(self, i):
        h = self.h_out_v(i)
        f = self.f_left_h(h)
        if f < 0:
            h = self.h_rotcw_h(h)
            f = self.f_left_h(h)
        avec = self.vec_area_f(f)
        n = avec / np.linalg.norm(avec)
        return n

    def normal_some_face_of_V(self):
        n = np.zeros((self.num_vertices, 3), dtype=_FLOAT_TYPE_)
        for i in range(self.num_vertices):
            n[i] = self.normal_some_face_of_v(i)
        return n

    def normal_other_weighted_v(self, i):
        """Weights for Computing Vertex Normals from Facet Normals Max99"""
        n = np.zeros(3)
        x = self.xyz_coord_v(i)
        h = self.h_out_v(i)
        rrot = self.xyz_coord_v(self.v_head_h(h)) - x
        h = self.h_rotcw_h(h)
        for hrot in self.generate_H_out_v_clockwise(i, h_start=h):
            r = rrot
            jrot = self.v_head_h(hrot)
            rrot = self.xyz_coord_v(jrot) - x
            if self.negative_boundary_contains_h(hrot):
                continue
            n += np.cross(rrot, r) / (np.dot(r, r) * np.dot(rrot, rrot))
        n /= np.linalg.norm(n)
        return n

    def normal_other_weighted_V(self):
        n = np.zeros((self.num_vertices, 3), dtype=_FLOAT_TYPE_)
        for i in range(self.num_vertices):
            n[i] = self.normal_other_weighted_v(i)
        return n

    def normal_laplacian_V(self):
        """
        Compute unit normals from mean curvature vector at all vertices
        """
        X = self.xyz_coord_V
        lapX = self.laplacian(X)
        n = np.zeros_like(X)
        for i in range(self.num_vertices):

            mcvec = lapX[i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)

        return n

    ###################
    # Curvature methods
    def gaussian_curvature_v(self, i):
        """
        2*pi - sum_f (angle_f)
        """
        area = 0.0
        defect = 2 * np.pi
        x = self.xyz_coord_v(i)
        h = self.h_out_v(i)
        rrot = self.xyz_coord_v(self.v_head_h(h)) - x
        norm_rrot = np.linalg.norm(rrot)
        h = self.h_rotcw_h(h)
        # for jrot in self.generate_V_nearest_v_clockwise(i, h_start=h):
        for hrot in self.generate_H_out_v_clockwise(i, h_start=h):
            r = rrot
            norm_r = norm_rrot
            jrot = self.v_head_h(hrot)
            rrot = self.xyz_coord_v(jrot) - x
            norm_rrot = np.linalg.norm(rrot)
            if self.negative_boundary_contains_h(hrot):
                # do boundary geodesic curvature stuff
                continue
            # r_dot_rrot = np.dot(r, rrot)
            cos_angle = np.dot(r, rrot) / (norm_r * norm_rrot)
            defect -= np.arccos(cos_angle)
            area += norm_r * norm_rrot * np.sqrt(1 - cos_angle**2) / 6

        return defect / area

    def compute_curvature_data(self):
        """
        Compute (H, K, lapH, n) at all vertices
        """

        X = self.xyz_coord_V
        lapX = self.laplacian(X)
        H = np.zeros_like(X[:, 0])
        K = np.zeros_like(X[:, 0])
        n = np.zeros_like(X)
        for i in range(self.num_vertices):

            mcvec = lapX[i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)
            H[i] = np.dot(n[i], mcvec) / 2
            K[i] = self.gaussian_curvature_v(i)

        lapH = self.laplacian(H)
        return H, K, lapH, n

    ######################################
    ############### Forces ###############
    ######################################
    def Fbend_density(self):
        H0 = self.spontaneous_curvature
        B = self.bending_modulus

        X = self.xyz_coord_V
        lapX = self.laplacian(X)
        H = np.zeros_like(X[:, 0])
        K = np.zeros_like(X[:, 0])
        n = np.zeros_like(X)
        for i in range(self.num_vertices):

            mcvec = lapX[i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)
            H[i] = np.dot(n[i], mcvec) / 2
            K[i] = self.gaussian_curvature_v(i)

        lapH = self.laplacian(H)
        Fdensity = -2 * B * (lapH + 2 * (H - H0) * (H**2 + H0 * H - K))

        return Fdensity

    def Fbend_analytic(self):
        H0 = self.spontaneous_curvature
        B = self.bending_modulus

        X = self.xyz_coord_V
        lapX = self.laplacian(X)
        H = np.zeros_like(X[:, 0])
        K = np.zeros_like(X[:, 0])
        n = np.zeros_like(X)
        for i in range(self.num_vertices):

            mcvec = lapX[i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)
            H[i] = np.dot(n[i], mcvec) / 2
            K[i] = self.gaussian_curvature_v(i)

        lapH = self.laplacian(H)
        Fdensity = -2 * B * (lapH + 2 * (H - H0) * (H**2 + H0 * H - K))
        Av = self.barcell_area_V()
        Fbend = np.einsum("i,i,ij->ij", Av, Fdensity, n)

        return Fbend

    def Farea_harmonic(self):
        """local cell area regulation"""
        Nv = self.num_vertices
        F = np.zeros((Nv, 3))
        A0 = self.spontaneous_face_area
        k_a = self.area_reg_stiffness
        for i in range(Nv):
            p = self.xyz_coord_v(i)
            h = self.h_out_v(i)
            xrot = self.xyz_coord_v(self.v_head_h(h))
            h = self.h_rotcw_h(h)
            for hrot in self.generate_H_out_v_clockwise(i, h_start=h):
                x = xrot
                jrot = self.v_head_h(hrot)
                xrot = self.xyz_coord_v(jrot)
                if self.negative_boundary_contains_h(hrot):
                    continue
                f = self.f_left_h(hrot)
                vecAf = self.vec_area_f(f)
                Af = np.linalg.norm(vecAf)
                F[i] += -k_a * (Af - A0) * np.cross(vecAf, x - xrot) / (2 * A0 * Af)
        return F

    def Fvolume_harmonic(self):
        Nv = self.num_vertices
        F = np.zeros((Nv, 3))
        V0 = self.spontaneous_volume
        V = self.total_volume()
        k_v = self.volume_reg_stiffness
        for i in range(Nv):
            p = self.xyz_coord_v(i)
            h = self.h_out_v(i)
            xrot = self.xyz_coord_v(self.v_head_h(h))
            h = self.h_rotcw_h(h)
            for hrot in self.generate_H_out_v_clockwise(i, h_start=h):
                x = xrot
                jrot = self.v_head_h(hrot)
                xrot = self.xyz_coord_v(jrot)
                F[i] += -k_v * (V - V0) * np.cross(xrot, x) / (6 * V0)
        return F

    ##
    def compute_data_for_forces(self):
        X = self.xyz_coord_V
        lapX = self.laplacian(X)
        H = np.zeros_like(X[:, 0])
        K = np.zeros_like(X[:, 0])
        n = np.zeros_like(X)
        for i in range(self.num_vertices):

            mcvec = lapX[i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)
            H[i] = np.dot(n[i], mcvec) / 2
            K[i] = self.gaussian_curvature_v(i)

        lapH = self.laplacian(H)
        self.H = H
        self.lapH = lapH
        self.K = K
        self.unit_normal = n

    # Bending
    def Fbend_harmonic(self, Nsmooth=0):
        """from Tu"""
        Kbend = self.bending_modulus
        H, K = self.get_angle_weighted_arc_curvatures()
        for _ in range(Nsmooth):
            H = self.gaussian_smooth_samples(H, 1, a)
            K = self.gaussian_smooth_samples(K, 1, a)
        Nv = H.shape[0]
        lapH = self.cotan_laplacian(H)
        F = np.zeros((Nv, 3))
        Fdensity = -2 * B * (lapH + 2 * H * (H**2 - K))
        Fn = -2 * B * (lapH + 2 * H * (H**2 - K))

        for v in range(Nv):
            n = self.other_weighted_vertex_normal(v)
            Av = self.vorcell_area(v)
            F[v] = Fn[v] * n * Av
        return F

    # def Flength(self):
    #     Nv = self.num_vertices
    #     F = np.zeros((Nv, 3))
    #     k_l = self.length_reg_stiffness
    #     L0 = self.spontaneous_edge_length
    #     for v in range(Nv):
    #         r = self.xyz_coord_v(v)
    #         h_start = self.V_hedge[v]
    #         h = h_start
    #         while True:
    #             v0 = self.H_vertex[h]
    #             r0 = self.V_pq[v0, :3]
    #             u = r - r0
    #             L = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    #             gradL = u / L
    #             F[v] += -Ke * (L - L0) * gradL / L0
    #             h = self.H_twin[self.H_prev[h]]
    #             if h == h_start:
    #                 break
    #     return F
    def Utether_OG(self, s, _a=None):
        l0 = self.spontaneous_edge_length
        if _a is None:
            a = l0
        else:
            a = _a
        Kl = self.length_reg_stiffness
        Dl = 0.8 * l0
        normDs = np.abs(s - l0)
        if normDs > Dl / 4 and normDs < Dl / 2:
            # U = a * Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
            U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
        else:
            U = 0.0
        return U

    def Utether(self, s, _a=None):
        l0 = self.spontaneous_edge_length
        if _a is None:
            a = l0
        else:
            a = _a
        Kl = self.length_reg_stiffness
        Dl = 0.8 * l0
        normDs = np.abs(s - l0)
        if normDs <= Dl / 4:
            U = 0.0
        elif normDs < Dl / 2:
            U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
        else:
            U = np.inf
        return U

    def Ftether_OG(self, _a=None):
        Nv = self.num_vertices
        Fl = np.zeros((Nv, 3))
        l0 = self.spontaneous_edge_length
        Kl = self.length_reg_stiffness
        if _a is None:
            a = l0
        else:
            a = _a
        Dl = 0.8 * l0
        for i in range(Nv):
            ri = self.xyz_coord_v(i)
            for hk in self.generate_H_out_v_clockwise(i):
                vk = self.v_head_h(hk)
                rk = self.xyz_coord_v(vk)
                Drki = ri - rk
                s = np.sqrt(Drki[0] ** 2 + Drki[1] ** 2 + Drki[2] ** 2)
                Ds = s - l0
                normDs = np.abs(Ds)
                if normDs > Dl / 4 and normDs < Dl / 2:
                    # U = a * Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
                    U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
                    Fl[i] += -(
                        (1 / (Dl / 2 - normDs) + a / (normDs - Dl / 4) ** 2)
                        * U
                        * (Ds / normDs)
                        * Drki
                        / s
                    )
                else:
                    pass
        return Fl

    # def Ftether(self, _a=None):
    #     Nv = self.num_vertices
    #     Fl = np.zeros((Nv, 3))
    #     l0 = self.spontaneous_edge_length
    #     Kl = self.length_reg_stiffness
    #     if _a is None:
    #         a = l0
    #     else:
    #         a = _a
    #     Dl = 0.8 * l0

    #     for i in range(Nv):
    #         ri = self.xyz_coord_v(i)
    #         for hk in self.generate_H_out_v_clockwise(i):
    #             vk = self.v_head_h(hk)
    #             rk = self.xyz_coord_v(vk)
    #             Drki = ri - rk
    #             s = np.sqrt(Drki[0] ** 2 + Drki[1] ** 2 + Drki[2] ** 2)
    #             Ds = s - l0
    #             normDs = np.abs(Ds)
    #             if normDs <= Dl / 4:
    #                 pass
    #             else:
    #                 U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
    #                 Fl[i] += -(
    #                     (1 / (Dl / 2 - normDs) + a / (normDs - Dl / 4) ** 2)
    #                     * U
    #                     * (Ds / normDs)
    #                     * Drki
    #                     / s
    #                 )
    #             ############
    #     return Fl
    def Ftether(self, _a=None, cutoff=0.99):
        Nv = self.num_vertices
        Fl = np.zeros((Nv, 3))
        l0 = self.spontaneous_edge_length
        Kl = self.length_reg_stiffness
        if _a is None:
            a = l0
        else:
            a = _a
        Dl = 0.8 * l0

        for i in range(Nv):
            ri = self.xyz_coord_v(i)
            for hk in self.generate_H_out_v_clockwise(i):
                vk = self.v_head_h(hk)
                rk = self.xyz_coord_v(vk)
                Drki = ri - rk
                s = np.sqrt(Drki[0] ** 2 + Drki[1] ** 2 + Drki[2] ** 2)
                Ds = s - l0
                normDs = np.abs(Ds)
                if normDs <= Dl / 4:
                    pass
                elif normDs < Dl / 2:
                    U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
                    Fl[i] += -(
                        (1 / (Dl / 2 - normDs) + a / (normDs - Dl / 4) ** 2)
                        * U
                        * (Ds / normDs)
                        * Drki
                        / s
                    )
                else:
                    normDs = cutoff * Dl / 2
                    U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
                    Fl[i] += -(
                        (1 / (Dl / 2 - normDs) + a / (normDs - Dl / 4) ** 2)
                        * U
                        * (Ds / normDs)
                        * Drki
                        / s
                    )
                ############
        return Fl

    def euler_step(self, dt):
        """
        Euler step
        """
        Fb = self.Fbend_analytic()
        Fa = self.Farea_harmonic()
        Fv = self.Fvolume_harmonic()
        Ft = self.Ftether()
        F = Fb + Fa + Fv + Ft
        self.xyz_coord_V += dt * F / self.linear_drag_coeff

    ######################################################
    # To be deprecated
    def h_in_cw_from_h(self, h):
        return self.h_twin_h(self.h_next_h(h))

    def generate_H_in_cw_from_h(self, h):
        """ """
        h_start = h
        while True:
            yield h
            h = self.h_in_cw_from_h(h)
            if h == h_start:
                break

    def xyz_com_f(self, f):
        h0 = self.h_bound_f(f)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        r0 = self.xyz_coord_v(self.v_origin_h(h0))
        r1 = self.xyz_coord_v(self.v_origin_h(h1))
        r2 = self.xyz_coord_v(self.v_origin_h(h2))
        return (r0 + r1 + r2) / 3

    def _angle_defect_v(self, v):
        """
        2*pi - sum_f (angle_f)
        """
        r0 = self.xyz_coord_v(v)
        defect = 2 * np.pi
        for h in self.generate_H_out_v_clockwise(v):
            h_rot = self.h_next_h(self.h_twin_h(h))
            r1 = self.xyz_coord_v(self.v_head_h(h))
            r2 = self.xyz_coord_v(self.v_head_h(h_rot))
            e1 = r1 - r0
            e2 = r2 - r0
            norm_e1 = np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
            norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
            cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
                norm_e1 * norm_e2
            )
            defect -= np.arccos(cos_angle)

        return defect

    def _gaussian_curvature_v(self, v):
        """
        Compute the Gaussian curvature at vertex v
        """
        area_v = self.barcell_area(v)
        angle_defect_v = self.angle_defect_v(v)
        return angle_defect_v / area_v
