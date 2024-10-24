from temp_python.src_python.half_edge_mesh import HalfEdgeMeshBase
import numpy as np
from temp_python.src_python.global_vars import INT_TYPE, FLOAT_TYPE

# from temp_python.src_python.combinatorics import argsort
# from scipy.sparse import lil_matrix


class ReferenceTriangle:
    def __init__(self, a=1, b=1, c=1):
        # phi = np.pi*np.array([0, 1, 2])/3
        self.xyz_coord_V = np.zeros((3, 3))
        self.xyz_coord_V[:, 0] = np.cos(phi)
        self.xyz_coord_V[:, 1] = np.sin(phi)
        # self.h_out_V = np.array([0, 1, 2], dtype=INT_TYPE)
        # self.v_origin_H = np.array([0, 1, 2], dtype=INT_TYPE)
        # self.h_next_H = np.array([1, 2, 0], dtype=INT_TYPE)
        # self.h_twin_H
        # self.f_left_H
        # self.h_bound_F
        # self.h_gen_B

    def affine_coodinates_of_nodes(self, resolution):
        """
        Compute the barycentric coordinates of a point (x, y) in the reference triangle
        """
        return np.array([1 - x - y, x, y])


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
    """
    Membrane model

    HalfEdgeMesh parameters
    -----------------------
    ...

    Parameters
    ----------
    preferred_area : float (Length^2)
        preferred surface area
    preferred_volume : float (Length^3)
        preferred interior volume
    spontaneous_curvature : float (1/Length)
        spontaneous mean curvature
    bending_modulus : float (Energy)
        bending rigidity
    splay_modulus : float (Energy)
        splay modulus
    volume_reg_stiffness : float (Energy/Length^3)
        volume constraint stiffness
    area_reg_stiffness : float (Energy/Length^2)
        area constraint stiffness

    tether_stiffness : float (Energy)
        edge length constraint stiffness
    tether_repulsive_onset : float (Dimensionless)
        onset of repulsive penalty
    tether_repulsive_singularity : float (Dimensionless)
        tethering potential minimum length
    tether_attractive_onset : float (Dimensionless)
        onset of attractive penalty
    tether_attractive_singularity : float (Dimensionless)
        tethering potential max length

    drag_coefficient : float
        friction coefficient
    flipping_frequency : float
        flipping frequency
    flipping_probability : float
        flipping probability

    length_unit : float
        characteristic length scale
    energy_unit : float
        characteristic energy scale
    time_unit : float
        characteristic time scale
    """

    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_gen_B,
        #
        surface_tension_model="local_harmonic",  # "constant"
        pressure_model="harmonic",  # "constant"
        constant_surface_tension=None,
        constant_pressure=None,
        #
        preferred_area=None,
        preferred_volume=None,
        #
        spontaneous_curvature=None,
        bending_modulus=None,
        splay_modulus=None,
        #
        volume_reg_stiffness=None,
        area_reg_stiffness=None,
        #
        tether_stiffness=None,
        tether_repulsive_onset=None,
        tether_repulsive_singularity=None,
        tether_attractive_onset=None,
        tether_attractive_singularity=None,
        tether_well_slope=None,
        #
        drag_coefficient=None,
        #
        kBT=0.2,
        flipping_frequency=None,
        flipping_probability=None,
        #
        samples_per_face=1,
        # length_unit=1.0,
        # energy_unit=1.0,
        # time_unit=1.0,
        **kwargs
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_gen_B,
        )
        if preferred_area is None:
            preferred_area = self.total_area_of_faces()
        if preferred_volume is None:
            preferred_volume = self.total_volume()
        self.preferred_area = preferred_area
        self.preferred_volume = preferred_volume
        #
        self.spontaneous_curvature = spontaneous_curvature
        self.bending_modulus = bending_modulus
        self.splay_modulus = splay_modulus
        #
        self.volume_reg_stiffness = volume_reg_stiffness
        self.area_reg_stiffness = area_reg_stiffness
        #
        self.tether_stiffness = tether_stiffness
        self.tether_repulsive_onset = tether_repulsive_onset
        self.tether_repulsive_singularity = tether_repulsive_singularity
        self.tether_attractive_onset = tether_attractive_onset
        self.tether_attractive_singularity = tether_attractive_singularity
        self.tether_well_slope = tether_well_slope
        #
        self.drag_coefficient = drag_coefficient
        #
        self.kBT = kBT
        self.flipping_frequency = flipping_frequency
        self.flipping_probability = flipping_probability
        ################################################
        self.preferred_face_area = self.preferred_area / self.num_faces
        self.preferred_edge_length = np.sqrt(4 * self.preferred_face_area / np.sqrt(3))
        ################################################
        self.surface_tension_model = surface_tension_model
        self.pressure_model = pressure_model
        self.constant_surface_tension = constant_surface_tension
        self.constant_pressure = constant_pressure
        self.surface_tension_F = np.zeros(self.num_faces)
        self.pressure_F = np.zeros(self.num_faces)
        self.laplacian_matrix = None
        self.helfrich_stress_V = np.zeros(self.num_vertices)

    def mesh_state(self):
        return {
            "xyz_coord_V": self.xyz_coord_V,
            "h_out_V": self.h_out_V,
            "v_origin_H": self.v_origin_H,
            "h_next_H": self.h_next_H,
            "h_twin_H": self.h_twin_H,
            "f_left_H": self.f_left_H,
            "h_bound_F": self.h_bound_F,
            "h_gen_B": self.h_gen_B,
        }

    def apply_laplacian_matrix(self, x):
        return self.laplacian_matrix.dot(x)

    def get_local_harmonic_surface_tension_F(self):
        """
        Compute the effective surface tension
        """
        Karea = self.area_reg_stiffness
        A0 = self.preferred_face_area
        A = self.area_F()
        return Karea * (A - A0) / A0

    def get_harmonic_pressure(self):
        """
        Compute the effective pressure
        """
        Kvol = self.volume_reg_stiffness
        V0 = self.preferred_volume
        V = self.total_volume()
        return Kvol * (V - V0) / V0

    def get_constant_surface_tension(self):
        return self.constant_surface_tension

    def get_constant_pressure(self):
        return self.constant_pressure

    def get_helfrich_stress_V(self):
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

    def refresh_laplacian(self):
        self.laplacian_lil = self.get_cotan_laplacian_lil()

    def refresh_pressure(self):
        self.pressure_F = self.harmonic_pressure()

    def refresh_geometric_data(self):
        """
        associate with each h the portion of f=f_left_h(h) that is closer to v=v_origin_h(h) than other vertices of f.

        Fbend
        Fpressure
        Fsurface_tension
        Ftether
        """
        num_vertices = self.num_vertices
        num_half_edges = self.num_half_edges
        num_faces = self.num_faces
        area_F = np.zeros(num_faces, dtype=FLOAT_TYPE)
        area_V = np.zeros(num_vertices, dtype=FLOAT_TYPE)
        unit_normal_V = np.zeros((num_vertices, 3), dtype=FLOAT_TYPE)
        edge_vec_H = np.zeros((num_half_edges, 3), dtype=FLOAT_TYPE)
        cotan_opposite_H = np.zeros(num_half_edges, dtype=FLOAT_TYPE)
        # unit_normal_F = np.zeros((num_faces, 3), dtype=FLOAT_TYPE)
        cotan_weights_H = np.zeros(num_half_edges, dtype=FLOAT_TYPE)

        hhh = np.zeros(3, dtype=INT_TYPE)
        eee = np.zeros((3, 3))
        vvv = np.zeros(3, dtype=INT_TYPE)
        xxx = np.zeros((3, 3))
        ddd = np.zeros(3)
        for f in range(num_faces):
            # h1 = self.h_bound_f(f)
            # h2 = self.h_next_h(h1)
            # h3 = self.h_next_h(h2)
            hhh[0] = self.h_bound_f(f)
            hhh[1] = self.h_next_h(hhh[0])
            hhh[2] = self.h_next_h(hhh[1])
            # v1 = self.v_origin_h(h1)
            # v2 = self.v_origin_h(h2)
            # v3 = self.v_origin_h(h3)
            vvv = np.array([self.v_origin_h(h) for h in hhh])
            # x1 = self.xyz_coord_v(v1)
            # x2 = self.xyz_coord_v(v2)
            # x3 = self.xyz_coord_v(v3)
            xxx = np.array([self.xyz_coord_v(v) for v in vvv])

            # e1 = x2 - x1
            # e2 = x3 - x2
            # e3 = x1 - x3
            eee = np.roll(xxx, -1) - xxx
            # d1 = np.linalg.norm(e1)
            # d2 = np.linalg.norm(e2)
            # d3 = np.linalg.norm(e3)
            ddd = np.linalg.norm(eee, axis=1)
            sort_indices = np.argsort(ddd)
            h3, h2, h1 = hhh[sort_indices]
            v3, v2, v1 = vvv[sort_indices]
            x3, x2, x1 = xxx[sort_indices]
            e3, e2, e1 = eee[sort_indices]
            d3, d2, d1 = ddd[sort_indices]

            # area_f = (1 / 4) * np.sqrt(
            #     (d1 + (d2 + d3))
            #     * (d3 - (d1 - d2))
            #     * (d3 + (d1 - d2))
            #     * (d1 + (d2 - d3))
            # )
            a, b, c = ddd[sort_indices]

            area_f = (1 / 4) * np.sqrt(
                (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
            )

            mu_c = c - (a - b)
            theta_c = np.arctan2((b**2 + c**2 - a**2), (2 * b * c))
            ###############################

            cos_theta1 = -np.dot(e2, e3) / (d2 * d3)
            cos_theta2 = -np.dot(e3, e1) / (d3 * d1)
            cos_theta3 = -np.dot(e1, e2) / (d1 * d2)
            sin_theta1 = np.sqrt(1 - cos_theta1**2)
            sin_theta2 = np.sqrt(1 - cos_theta1**2)
            sin_theta3 = np.sqrt(1 - cos_theta1**2)

            edge_vec_H[h1] = e1
            edge_vec_H[h2] = e2
            edge_vec_H[h3] = e3
            cotan_opposite_H[h1] = cos_theta1 / sin_theta1
            cotan_opposite_H[h2] = cos_theta2 / sin_theta2
            cotan_opposite_H[h3] = cos_theta3 / sin_theta3

            area_V[v1] += d1 * d2 * sin_theta3 / 6
            area_F[f] = (
                d1 * d2 * sin_theta3 + d2 * d3 * sin_theta1 + d3 * d1 * sin_theta2
            ) / 6

            vec_area_F[f] = (np.cross(x1, x2) + np.cross(x2, x3) + np.cross(x3, x1)) / 2
            area_F[f] = np.linalg.norm(vec_area_F[f])
            unit_normal_F[f] = vec_area_F[f] / area_F[f]
            area_V[v1] += area_F[f] / 3
            unit_normal_V[v1] += vec_area_F[f] / 3
            unit_normal_V[v2] += vec_area_F[f] / 3
            unit_normal_V[v3] += vec_area_F[f] / 3

        unit_normal_V /= np.linalg.norm(unit_normal_V, axis=1)[:, np.newaxis]
        lap_rows = np.empty(self.num_vertices, dtype=object)
        lap_data = np.empty(self.num_vertices, dtype=object)
        for vi in range(num_vertices):
            rows[vi] = [vi]
            data[vi] = [0.0]
            for hij in self.generate_H_out_v_clockwise(vi):
                hji = self.h_twin_H[hij]
                vj = self.v_origin_h(hji)
                w = (cotan_opposite_H[hij] + cotan_opposite_H[hji]) / 2
                lap_data[vi].append(w)
                lap_data[vi][0] -= w
                lap_rows[vi].append(vj)
            # sort nonzero column indices for row i
            argsort_row = argsort(lap_rows[vi])
            lap_rows[vi] = [lap_rows[vi][_] for _ in argsort_row]
            lap_data[vi] = [lap_data[vi][_] for _ in argsort_row]

        lap_mat = lil_matrix((self.num_vertices, self.num_vertices), dtype=FLOAT_TYPE)
        lap_mat.lap_rows = lap_rows
        lap_mat.lap_data = lap_data

    def refresh_soft_penalty_A_soft_penalty_V(self, **kwargs):
        num_vertices = self.num_vertices
        num_half_edges = self.num_half_edges
        num_faces = self.num_faces

        surface_tension_F = np.zeros(num_faces, dtype=FLOAT_TYPE)
        pressure_F = np.zeros(num_faces, dtype=FLOAT_TYPE)
        area_F = np.zeros(num_faces, dtype=FLOAT_TYPE)

        area_V = np.zeros(num_vertices, dtype=FLOAT_TYPE)
        vec_area_V = np.zeros((num_vertices, 3), dtype=FLOAT_TYPE)

        force_helfrich_V = np.zeros((num_vertices, 3), dtype=FLOAT_TYPE)
        force_pressure_V = np.zeros((num_vertices, 3), dtype=FLOAT_TYPE)
        force_surface_tension_V = np.zeros((num_vertices, 3), dtype=FLOAT_TYPE)
        force_tether_V = np.zeros((num_vertices, 3), dtype=FLOAT_TYPE)
        total_volume = 0.0
        for f in range(num_faces):
            for h in self.generate_H_bound_f(f):
                v = self.v_origin_h(h)

    def flip_energy_area_penalty_tether_vutukuri(self, h):
        h0 = h
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        h3 = self.h_twin_h(h0)
        v0 = self.v_origin_h(h)
        f_left = self.f_left_h(h)
        f_right = self.f_left_h(self.h_twin_h(h))

        f1 = self.f_left_h(h)
        ht = self.h_twin_h(h)

    def dimensionless_tethering_potential_vutukuri(self, s):
        """
        s = actual_length / preferred_length
        """
        Lmin = self.tether_repulsive_singularity
        Lmax = self.tether_attractive_singularity
        Lrep = self.tether_repulsive_onset
        Latt = self.tether_attractive_onset

        if s <= Latt:
            if s >= Lrep:  # free region
                return 0.0
            elif s > Lmin:  # repulsive region
                return np.exp(1 / (s - Lrep)) / (s - Lmin)
            else:
                raise ValueError("s < Lmin")
        elif s < Lmax:  # attractive region
            return np.exp(1 / (Latt - s)) / (Lmax - s)
        else:
            raise ValueError("s > Lmax")

    def dimensionless_tethering_potential_siggel(self, s):
        """
                s = actual_length / preferred_length
                TriMem: A parallelized hybrid Monte Carlo software for
        efficient simulations of lipid membranes
        Marc Sigge
        """
        r = self.tether_well_slope
        Lrep = self.tether_repulsive_onset
        Latt = self.tether_attractive_onset

        if s <= Latt:
            if s >= Lrep:  # free region  Lrep <= s <= Latt
                return 0.0
            else:  # repulsive region s < Lrep
                return np.exp(1 / (s - Lrep)) / s**r
        else:  # attractive region                     Latt < s
            return r ** (r + 1) * (s - Latt) ** r

    def flip_check(self, h):
        r"""
        h cannot be on boundary!
                v2                           v2                                                      
              /  |  \                      /    \                                                                      
             /   |   \                    /      \                                                                        
            /h2  |  h1\                  /h2    h1\                                                                       
           /     |     \                /    ft    \                                                                    
          /  f   | ft   \              /            \                                                                         
         /       |       \            /      ht      \                                                                     
        v3     h |ht     v1 |----->  v3--------------v1                                              
         \       |       /            \      h       /                                                                    
          \      |      /              \            /                                                                    
           \     |     /                \    f     /                                                                        
            \h3  |  h0/                  \h3    h0/                                                                             
             \   |   /                    \      /                                                                         
              \  |  /                      \    /                                                                             
                v0                           v0                                                                            
        """
        energy = 0.0

        # get involved half-edges/vertices/faces
        f = self.f_left_h(h)
        ht = self.h_twin_h(h)
        ft = self.f_left_h(ht)
        h0 = self.h_next_h(ht)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h)
        h3 = self.h_twin_h(h2)
        H = np.array([h0, h1, h2, h3])
        v_origin_H = self.v_origin_h(H)
        x0, x1, x2, x3 = self.xyz_coord_v(v_origin_H)
        d0, d1, d2, d3, dpre, dpost = np.linalg.norm(
            [x1 - x0, x2 - x1, x3 - x2, x0 - x3, x2 - x0, x3 - x1], axis=1
        )
        # tether penalty
        k_tether = self.tether_stiffness
        preUtether = k_tether * self.dimensionless_tethering_potential_siggel(
            dpre / self.preferred_edge_length
        )
        postUtether = k_tether * self.dimensionless_tethering_potential_siggel(
            dpost / self.preferred_edge_length
        )
        # area penalty
        c, b, a = sorted([dpre, d2, d3])
        preA = (1 / 4) * np.sqrt(
            (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
        )
        c, b, a = sorted([dpre, d0, d1])
        preAt = (1 / 4) * np.sqrt(
            (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
        )
        c, b, a = sorted([dpost, d3, d0])
        postA = (1 / 4) * np.sqrt(
            (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
        )
        c, b, a = sorted([dpost, d1, d2])
        postAt = (1 / 4) * np.sqrt(
            (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
        )
        preUarea = self.area_reg_stiffness * (
            (preA - self.preferred_face_area) ** 2
            + (preAt - self.preferred_face_area) ** 2
        )
        postUarea = self.area_reg_stiffness * (
            (postA - self.preferred_face_area) ** 2
            + (postAt - self.preferred_face_area) ** 2
        )

        # area_f = (1 / 4) * np.sqrt(
        #     (d1 + (d2 + d3))
        #     * (d3 - (d1 - d2))
        #     * (d3 + (d1 - d2))
        #     * (d1 + (d2 - d3))
        # )

        # H = np.array([h0, h1, h2, h3, h4, h5])
        # v0 = self.v_origin_h(h0)
        # v1 = self.v_origin_h(h1)
        # v2 = self.v_origin_h(h2)
        # v3 = self.v_origin_h(h5)

        f0 = self.f_left_h(h0)
        f1 = self.f_left_h(h1)
        V = self.v_origin_h([h0, h1, h2, h5])
        x0, x1, x2, x3 = self.xyz_coord_v(V)
        # xyz_coord_V=self.xyz_coord_v(V)

        # e0 = x1 - x0
        # e1 = x2 - x1
        # e2 = x0 - x2
        # e4 = x3 - x0
        # e5 = x1 - x3

        d0 = np.linalg.norm([x1 - x0, x2 - x1, x0 - x2, x3 - x0, x1 - x3], axis=1)

        # things h0/h1, f0/f1 refer to

        # things that refer to h0/h1, f0/f1

        # update vertices
        if self.h_out_v(v0) == h1:
            self.update_vertex(v0, h_out=h2)
        if self.h_out_v(v2) == h0:
            self.update_vertex(v2, h_out=h4)
        # update half-edges
        self.update_half_edge(h0, v_origin=v3, h_next=h3)
        self.update_half_edge(h1, v_origin=v1, h_next=h5)
        self.update_half_edge(h2, h_next=h1, f_left=f1)
        self.update_half_edge(h3, h_next=h4)
        self.update_half_edge(h4, h_next=h0, f_left=f0)
        self.update_half_edge(h5, h_next=h2)
        # update faces
        if self.h_bound_f(f0) == h2:
            self.update_face(f0, h_bound=h3)
        if self.h_bound_f(f1) == h4:
            self.update_face(f1, h_bound=h5)

    def update_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

        if "preferred_area" not in kwargs:
            self.preferred_area = self.total_area_of_faces()
        if "preferred_volume" not in kwargs:
            self.preferred_volume = self.total_volume()
        self.preferred_face_area = self.preferred_area / self.num_faces
        self.preferred_edge_length = np.sqrt(4 * self.preferred_face_area / np.sqrt(3))

    @staticmethod
    def default_params():
        length_unit = 1.0
        energy_unit = 6.25e-3  # .2/32
        time_unit = 1e5

        spontaneous_curvature = 0.0 / length_unit
        bending_modulus = 20.0 * energy_unit
        splay_modulus = 0.0 * energy_unit
        #
        volume_reg_stiffness = 1.6e7 * energy_unit / length_unit**3
        area_reg_stiffness = 6.43e6 * energy_unit / length_unit**2
        #
        tether_stiffness = 80.0 * energy_unit
        tether_repulsive_onset = 0.8
        tether_repulsive_singularity = 0.6
        tether_attractive_onset = 1.2
        tether_attractive_singularity = 1.4
        #
        drag_coefficient = 0.4 * energy_unit * time_unit / length_unit**2
        #
        flipping_frequency = 1e6 / time_unit
        flipping_probability = 0.3
        #

        brane_kwargs = {
            "spontaneous_curvature": spontaneous_curvature,
            "bending_modulus": bending_modulus,
            "splay_modulus": splay_modulus,
            #
            "volume_reg_stiffness": volume_reg_stiffness,
            "area_reg_stiffness": area_reg_stiffness,
            #
            "tether_stiffness": tether_stiffness,
            "tether_repulsive_onset": tether_repulsive_onset,
            "tether_repulsive_singularity": tether_repulsive_singularity,
            "tether_attractive_onset": tether_attractive_onset,
            "tether_attractive_singularity": tether_attractive_singularity,
            #
            "drag_coefficient": drag_coefficient,
            #
            "flipping_frequency": flipping_frequency,
            "flipping_probability": flipping_probability,
        }
        brane_kwargs = {key: float(val) for key, val in brane_kwargs.items()}
        return brane_kwargs

    @classmethod
    def default_sphere(cls):
        ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
        self = cls.from_he_ply(ply_path, **cls.default_params())
        return self

    @staticmethod
    def vutukuri_params(num_faces=5120, radius=1.0):
        Nf = num_faces
        # vesicle radius at equilibrium
        R = radius
        # thermal energy unit
        kBT = 0.2
        # time scale
        tau = 1.28e5

        # friction coefficient
        drag_coefficient = 0.4 * kBT * tau / R**2

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
        preferred_face_area = A / Nf
        # local area stiffness
        area_reg_stiffness = 6.43e6 * kBT / A

        ##########################################
        # Volume constraint/penalty parameters
        # desired vesicle volume
        preferred_volume = 4 * np.pi * R**3 / 3
        # volume stiffness
        volume_reg_stiffness = 1.6e7 * kBT / R**3

        ################################################
        # Edge length and tethering potential parameters
        # bond stiffness
        tether_stiffness = 80 * kBT
        # average bond length
        preferred_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))
        # minimum bond length
        min_edge_length = 0.6 * preferred_edge_length
        # potential cutoff lengths
        tether_cutoff_length1 = 0.8 * preferred_edge_length  # onset of repulsion
        tether_cutoff_length0 = 1.2 * preferred_edge_length  # onset of attraction
        # maximum bond length
        max_edge_length = 1.4 * preferred_edge_length

        Lscale = 1.0  # length scale
        Dl = 0.5 * (max_edge_length - min_edge_length)  # = 0.4 * preferred_edge_length
        mu = (preferred_edge_length - tether_cutoff_length1) / Dl  # = .5
        lam = Lscale / Dl
        nu = Lscale / Dl

        brane_kwargs = {
            "tether_stiffness": tether_stiffness,
            "area_reg_stiffness": area_reg_stiffness,
            "volume_reg_stiffness": volume_reg_stiffness,
            "bending_modulus": bending_modulus,
            "splay_modulus": splay_modulus,
            "spontaneous_curvature": spontaneous_curvature,
            "drag_coefficient": drag_coefficient,
            "preferred_edge_length": preferred_edge_length,
            "preferred_face_area": preferred_face_area,
            "preferred_volume": preferred_volume,
            "tether_Dl": Dl,
            "tether_mu": mu,
            "tether_lam": lam,
            "tether_nu": nu,
        }
        brane_kwargs = {key: float(val) for key, val in brane_kwargs.items()}
        return brane_kwargs

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
        drag_coefficient = 0.4 * kBT * tau / R**2

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
        preferred_face_area = A / Nf
        # local area stiffness
        area_reg_stiffness = 6.43e6 * kBT / A

        ##########################################
        # Volume constraint/penalty parameters
        # desired vesicle volume
        preferred_volume = 4 * np.pi * R**3 / 3
        # volume stiffness
        volume_reg_stiffness = 1.6e7 * kBT / R**3

        ################################################
        # Edge length and tethering potential parameters
        # bond stiffness
        tether_stiffness = 80 * kBT
        # average bond length
        preferred_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))
        # minimum bond length
        min_edge_length = 0.6 * preferred_edge_length
        # potential cutoff lengths
        tether_cutoff_length1 = 0.8 * preferred_edge_length  # onset of repulsion
        tether_cutoff_length0 = 1.2 * preferred_edge_length  # onset of attraction
        # maximum bond length
        max_edge_length = 1.4 * preferred_edge_length

        Lscale = 1.0  # length scale
        Dl = 0.5 * (max_edge_length - min_edge_length)  # = 0.4 * preferred_edge_length
        mu = (preferred_edge_length - tether_cutoff_length1) / Dl  # = .5
        lam = Lscale / Dl
        nu = Lscale / Dl

        brane_kwargs = {
            "tether_stiffness": tether_stiffness,
            "area_reg_stiffness": area_reg_stiffness,
            "volume_reg_stiffness": volume_reg_stiffness,
            "bending_modulus": bending_modulus,
            "splay_modulus": splay_modulus,
            "spontaneous_curvature": spontaneous_curvature,
            "drag_coefficient": drag_coefficient,
            "preferred_edge_length": preferred_edge_length,
            "preferred_face_area": preferred_face_area,
            "preferred_volume": preferred_volume,
            "tether_Dl": Dl,
            "tether_mu": mu,
            "tether_lam": lam,
            "tether_nu": nu,
        }

        m = cls.from_he_ply(ply_path, **brane_kwargs)

        return m

    @classmethod
    def unit_vutukuri_vesicle(cls):

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

        # Nv = 5120
        # Ne = 3 * Nv - 6
        # Nf = 2 * Nv - 4
        Nf = 5120
        ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"

        # vesicle radius at equilibrium
        R = 1.0
        # thermal energy unit
        kBT = 0.2
        # time scale
        tau = 1.28e5

        # friction coefficient
        drag_coefficient = 0.4 * kBT * tau / R**2

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
        preferred_face_area = A / Nf
        # local area stiffness
        area_reg_stiffness = 6.43e6 * kBT / A

        ##########################################
        # Volume constraint/penalty parameters
        # desired vesicle volume
        preferred_volume = 4 * np.pi * R**3 / 3
        # volume stiffness
        volume_reg_stiffness = 1.6e7 * kBT / R**3

        ################################################
        # Edge length and tethering potential parameters
        # bond stiffness
        tether_stiffness = 80 * kBT
        # average bond length
        preferred_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))
        # minimum bond length
        min_edge_length = 0.6 * preferred_edge_length
        # potential cutoff lengths
        tether_cutoff_length1 = 0.8 * preferred_edge_length  # onset of repulsion
        tether_cutoff_length0 = 1.2 * preferred_edge_length  # onset of attraction
        # maximum bond length
        max_edge_length = 1.4 * preferred_edge_length

        Lscale = 1.0  # length scale
        Dl = 0.5 * (max_edge_length - min_edge_length)  # = 0.4 * preferred_edge_length
        mu = (preferred_edge_length - tether_cutoff_length1) / Dl  # = .5
        lam = Lscale / Dl
        nu = Lscale / Dl

        brane_kwargs = {
            "tether_stiffness": tether_stiffness,
            "area_reg_stiffness": area_reg_stiffness,
            "volume_reg_stiffness": volume_reg_stiffness,
            "bending_modulus": bending_modulus,
            "splay_modulus": splay_modulus,
            "spontaneous_curvature": spontaneous_curvature,
            "drag_coefficient": drag_coefficient,
            "preferred_edge_length": preferred_edge_length,
            "preferred_face_area": preferred_face_area,
            "preferred_volume": preferred_volume,
            "tether_Dl": Dl,
            "tether_mu": mu,
            "tether_lam": lam,
            "tether_nu": nu,
        }

        m = cls.from_he_ply(ply_path, **brane_kwargs)

        return m

    ######################
    def pressure_soft_penalty(self):
        """
        Compute the effective pressure
        """
        Kvol = self.volume_reg_stiffness
        V0 = self.preferred_volume
        V = self.total_volume()
        return Kvol * (V - V0) / V0

    def surface_tension_soft_penalty_f(self, f):
        """
        Compute the effective surface tension
        """
        Karea = self.area_reg_stiffness
        A0 = self.preferred_area
        A = self.area_f(f)
        return Karea * (A - A0) / A0

    def surface_tension_soft_penalty_F(self):
        """
        Compute the effective surface tension
        """
        Karea = self.area_reg_stiffness
        A0 = self.preferred_area
        A = self.area_F()
        return Karea * (A - A0) / A0

    ######################################################
    # experimental stuff
    ######################

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
        n = np.zeros((self.num_vertices, 3), dtype=FLOAT_TYPE)
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
        n = np.zeros((self.num_vertices, 3), dtype=FLOAT_TYPE)
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
        A0 = self.preferred_face_area
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
        V0 = self.preferred_volume
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

            Fl += (
                4 * np.dot(sum_xi_xj_Tij, sum_xi_xj_dTij_dxl_Tij_dil_dij) / sum_Lij2_Tij
            )
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
                    sum_Tij_dLij2_dxl_Lij2_dTij_dxl += (
                        Tij * dLij2_dxl + Lij**2 * dTij_dxl
                    )

                Fl += (
                    4
                    * np.dot(sum_xi_xj_Tij, sum_xi_xj_dTij_dxl_Tij_dil_dij)
                    / sum_Lij2_Tij
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

    #####################################
    # tethering

    def tethering_potential_vutukuri_vectorized(self, s):
        L0 = self.preferred_edge_length
        Lmin = tether_repulsive_singularity * L0
        Lmax = tether_attractive_singularity * L0
        Lrep = tether_repulsive_onset * L0
        Latt = tether_attractive_onset * L0
        k = self.tether_stiffness

        U = np.zeros_like(s)
        Irep = s < Lrep
        srep = s[Irep]
        U[Irep] = np.exp(1 / (srep - Lrep)) / (srep - Lmin)
        Iatt = s > Latt
        satt = s[Iatt]
        U[Iatt] = np.exp(1 / (Latt - satt)) / (Lmax - satt)
        return U

    def tethering_potential(
        self,
        s,
        preferred_length,
        repulsive_onset=0.8,
        repulsive_singularity=0.6,
        attractive_onset=1.2,
        attractive_singularity=1.4,
        length_unit=1.0,
    ):
        L0 = preferred_length
        Lmin = repulsive_singularity * preferred_length
        Lmax = attractive_singularity * preferred_length
        Lrep = repulsive_onset * preferred_length
        Latt = attractive_onset * preferred_length

        U = np.zeros_like(s)
        Irep = s < Lrep
        srep = s[Irep]
        U[Irep] = np.exp(1 / (srep - Lrep)) / (srep - Lmin)
        Iatt = s > Latt
        satt = s[Iatt]
        U[Iatt] = np.exp(1 / (Latt - satt)) / (Lmax - satt)
        return U

    def Ftether(self):
        success = True
        kb = self.tether_stiffness
        # tether_repulsive_onset
        # tether_repulsive_singularity
        # tether_attractive_onset
        # tether_attractive_singularity
        L0 = self.preferred_edge_length
        Lmin = self.tether_repulsive_singularity * L0
        Lmax = self.tether_attractive_onset * L0
        Lrep = self.tether_repulsive_onset * L0
        Latt = self.tether_attractive_onset * L0
        num_vertices = self.num_vertices
        F = np.zeros((num_vertices, 3))
        for p in range(num_vertices):
            xp = self.xyz_coord_v(p)
            for h in self.generate_H_out_v_clockwise(p):
                q = self.v_head_h(h)
                xq = self.xyz_coord_v(q)
                r = xp - xq
                s = np.linalg.norm(r)
                if s < Lrep and s > Lmin:
                    U = kb * np.exp(1 / (s - Lrep)) / (s - Lmin)
                    F[p] += -(-1 / (s - Lmin) - 1 / (s - Lrep) ** 2) * U * r / s
                elif s > Latt and s < Lmax:
                    U = kb * np.exp(1 / (Latt - s)) / (Lmax - s)
                    F[p] += -(1 / (Lmax - s) + 1 / (Latt - s) ** 2) * U * r / s
                elif s <= Lmin:
                    self.edge_length_smaller_than_min = True
                    success = False
                elif s >= Lmax:
                    self.edge_length_bigger_than_max = True
                    success = False
        return F, success

    def _Utether_OG(self, s, _a=None):
        l0 = self.preferred_edge_length
        if _a is None:
            a = l0
        else:
            a = _a
        Kl = self.tether_stiffness
        Dl = 0.8 * l0
        normDs = np.abs(s - l0)
        if normDs > Dl / 4 and normDs < Dl / 2:
            # U = a * Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
            U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
        else:
            U = 0.0
        return U

    def _Utether(self, s, _a=None):
        l0 = self.preferred_edge_length
        if _a is None:
            a = l0
        else:
            a = _a
        Kl = self.tether_stiffness
        Dl = 0.8 * l0
        normDs = np.abs(s - l0)
        if normDs <= Dl / 4:
            U = 0.0
        elif normDs < Dl / 2:
            U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
        else:
            U = np.inf
        return U

    def _Ftether_OG(self, _a=None):
        Nv = self.num_vertices
        Fl = np.zeros((Nv, 3))
        l0 = self.preferred_edge_length
        Kl = self.tether_stiffness
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

    def _Ftether(self, _a=None, cutoff=0.99):
        Nv = self.num_vertices
        Fl = np.zeros((Nv, 3))
        l0 = self.preferred_edge_length
        Kl = self.tether_stiffness
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

    ######################################################
    # To be deprecated
