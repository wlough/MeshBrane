#pragma once

/**
 * @file envelope_test_sim.hpp
 * @brief Defines EnvelopeTestSim class and associated classes
 */

#include "meshbrane/kmc.hpp"
#include "meshbrane/matrix_mesh.hpp"
#include "meshbrane/membrane.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/simulation_base.hpp"
#include "meshbrane/viewer.hpp"
#include <cassert>
#include <cstdint>
#include <filesystem>

namespace meshbrane {

class EnvelopeTestSim : public SimulationBase {
public:
  // core
  kmc::RandomNumberGenerator rng_;
  double kBT_;
  double bulk_viscosity_;
  double dt_save_{1.0};
  Viewer viewer_;

  bool save_sim_data_{true};

  Membrane envelope_;

  double dt_mean_{0.0};

  bool no_graph_{false};

  EnvelopeTestSim(const std::filesystem::path &path_to_parameters);

  double dt_max();
  void evolve_until_next_frame();
  void evolve_until(double t_end);
  void draw_scene();
  void run(int argc, char *argv[]);
  void run_no_graph();

  void save_frame();
  void write_outputs();
  void save_enevelope_state();
  std::filesystem::path get_envelope_ply_path();

  //
  void print_info();
  void clear_interactions();
  void update_cached_data();
  void apply_pair_interactions();
  void apply_internal_interactions();
  void apply_thermal_fluctuations(double dt);
  void update_state_variables(double dt);

  void reinit_envelope() {
    printf("EnvelopeTestSim::reinit_envelope()\n");

    envelope_.init_core_data_from_minimal_half_edge_data();
    envelope_.init_cache_from_core_data();
    envelope_.initial_area_ =
        envelope_.average_face_area_ * envelope_.get_num_faces();
    envelope_.initial_volume_ = envelope_.total_volume_;
    envelope_.target_face_area_ = envelope_.average_face_area_;
    envelope_.target_volume_ = envelope_.total_volume_;
    envelope_.update_geotargets();

    envelope_.mcvec_V_.resize(envelope_.get_num_vertices(), 3);
    envelope_.mcvec_V_.setZero();

    envelope_.force_V_.resize(envelope_.get_num_vertices(), 3);
    envelope_.force_V_.setZero();

    envelope_.update_surface_tension_and_pressure();

    if (envelope_.enable_flipping_) {
      envelope_.t_flip_ = envelope_.dt_flip_; // time to next flip
    } else {
      envelope_.t_flip_ = std::numeric_limits<double>::infinity();
    }
  }

  void init_concentric_spheres0() {
    printf("EnvelopeTestSim::init_concentric_spheres()\n");

    // envelope_.refine_icososphere();
    // envelope_.refine_icososphere();
    // std::cout << "envelope_.V_cycle_F = " << envelope_.V_cycle_F_ << '\n';
    //
    //
    //
    //
    //
    //
    Samples3d xyz_coord_V0 = envelope_.xyz_coord_V_;
    Samplesi h_out_V0 = envelope_.h_out_V_;

    Samples2i V_cycle_E0 = envelope_.V_cycle_E_;
    Samplesi h_directed_E0 = envelope_.h_directed_E_;

    Samples3i V_cycle_F0 = envelope_.V_cycle_F_;
    Samplesi h_right_F0 = envelope_.h_right_F_;

    Samplesi h_negative_B0 = envelope_.h_negative_B_;

    Samplesi v_origin_H0 = envelope_.v_origin_H_;
    Samplesi e_undirected_H0 = envelope_.e_undirected_H_;
    Samplesi f_left_H0 = envelope_.f_left_H_;
    Samplesi h_next_H0 = envelope_.h_next_H_;
    Samplesi h_twin_H0 = envelope_.h_twin_H_;

    int Nv0 = h_out_V0.size();
    int Ne0 = h_directed_E0.size();
    int Nf0 = h_right_F0.size();
    int Nb0 = h_negative_B0.size();
    int Nh0 = v_origin_H0.size();

    int Nv = 2 * Nv0;
    int Ne = 2 * Ne0;
    int Nf = 2 * Nf0;
    int Nb = 2 * Nb0;
    int Nh = 2 * Nh0;

    Samples3d xyz_coord_V1 = xyz_coord_V0;
    Samplesi h_out_V1 = h_out_V0;
    Samples2i V_cycle_E1 = V_cycle_E0;
    Samplesi h_directed_E1 = h_directed_E0;
    Samples3i V_cycle_F1 = V_cycle_F0;
    Samplesi h_right_F1 = h_right_F0;
    Samplesi h_negative_B1 = h_negative_B0;
    Samplesi v_origin_H1 = v_origin_H0;
    Samplesi e_undirected_H1 = e_undirected_H0;
    Samplesi f_left_H1 = f_left_H0;
    Samplesi h_next_H1 = h_next_H0;
    Samplesi h_twin_H1 = h_twin_H0;

    Samples3d xyz_coord_V(Nv, 3);
    Samplesi h_out_V(Nv);
    Samples2i V_cycle_E(Ne, 2);
    Samplesi h_directed_E(Ne);
    Samples3i V_cycle_F(Nf, 3);
    Samplesi h_right_F(Nf);
    Samplesi h_negative_B(Nb);
    Samplesi v_origin_H(Nh);
    Samplesi e_undirected_H(Nh);
    Samplesi f_left_H(Nh);
    Samplesi h_next_H(Nh);
    Samplesi h_twin_H(Nh);

    Eigen::Matrix<double, 1, 3> xyz_translation{0.0, 0.0, 2.25};
    double inner_membrane_scale = 0.8;

    xyz_coord_V.topRows(Nv0) = xyz_coord_V0;
    xyz_coord_V.bottomRows(Nv0) = inner_membrane_scale * xyz_coord_V0;
    h_out_V.head(Nv0) = h_out_V0;
    h_out_V.tail(Nv0) = h_out_V0.array() + Nh0;

    V_cycle_E.topRows(Ne0) = V_cycle_E0;
    V_cycle_E.bottomRows(Ne0) = V_cycle_E0.array() + Nv0;
    h_directed_E.head(Ne0) = h_directed_E0;
    h_directed_E.tail(Ne0) = h_directed_E0.array() + Nh0;

    V_cycle_F.topRows(Nf0) = V_cycle_F0;
    V_cycle_F.bottomRows(Nf0) = V_cycle_F0.array() + Nv0;
    h_right_F.head(Nf0) = h_right_F0;
    h_right_F.tail(Nf0) = h_right_F0.array() + Nh0;

    h_negative_B.head(Nb0) = h_negative_B0;
    h_negative_B.tail(Nb0) = h_negative_B0.array() + Nh0;

    v_origin_H.head(Nh0) = v_origin_H0;
    v_origin_H.tail(Nh0) = v_origin_H0.array() + Nv0;
    e_undirected_H.head(Nh0) = e_undirected_H0;
    e_undirected_H.tail(Nh0) = e_undirected_H0.array() + Ne0;
    f_left_H.head(Nh0) = f_left_H0;
    f_left_H.tail(Nh0) = f_left_H0.array() + Nf0;
    h_next_H.head(Nh0) = h_next_H0;
    h_next_H.tail(Nh0) = h_next_H0.array() + Nh0;
    h_twin_H.head(Nh0) = h_twin_H0;
    h_twin_H.tail(Nh0) = h_twin_H0.array() + Nh0;

    envelope_.xyz_coord_V_ = xyz_coord_V;
    // envelope_.h_out_V_ = h_out_V;
    // envelope_.V_cycle_E_ = V_cycle_E;
    // envelope_.h_directed_E_ = h_directed_E;
    envelope_.V_cycle_F_ = V_cycle_F;
    // envelope_.h_right_F_ = h_right_F;
    // envelope_.h_negative_B_ = h_negative_B;
    // envelope_.v_origin_H_ = v_origin_H;
    // envelope_.e_undirected_H_ = e_undirected_H;
    // envelope_.f_left_H_ = f_left_H;
    // envelope_.h_next_H_ = h_next_H;
    // envelope_.h_twin_H_ = h_twin_H;
    //
    // std::cout << "V_cycle_F = " << V_cycle_F.transpose() << '\n';
    // std::cout << "V_cycle_F = " << V_cycle_F << '\n';
    printf("EnvelopeTestSim::init_concentric_spheres() - 0\n");
    //
    //
    int v_hole0 = 0;
    int v_hole1 = v_hole0 + Nv0;

    int valance_v_hole0 = envelope_.get_valence_v(v_hole0);
    //
    //
    printf("EnvelopeTestSim::init_concentric_spheres() - 1\n");
    envelope_.init_core_data_from_V_cycle_F();
    printf("EnvelopeTestSim::init_concentric_spheres() - 2\n");
    //
    //
    //
    //
    //
    //
    reinit_envelope();
    printf("EnvelopeTestSim::init_concentric_spheres() - 3\n");
  }

  void init_concentric_spheres() {
    printf("EnvelopeTestSim::init_concentric_spheres()\n");
    MatrixMesh m = MatrixMesh::spherical_shell(3, 0.05);
    // MatrixMesh m = MatrixMesh::icososphere(2);
    // m.reverse_faces();
    // MatrixMesh m = MatrixMesh::icosohedron();
    // m.reverse_orientation();

    printf("EnvelopeTestSim::init_concentric_spheres() - 0\n");
    envelope_.copy_mesh_data(m);
    // int v0 = 0;
    // double r_max = 0.5;
    // int v1 = m.find_opposite_vertex(v0, r_max);
    printf("EnvelopeTestSim::init_concentric_spheres() - 1\n");
    envelope_.copy_mesh_cache(m);
    // m.remove_v_neighborhood(v0);
    printf("EnvelopeTestSim::init_concentric_spheres() - 2\n");

    reinit_envelope();
    // envelope_.reverse_orientation();
    // m.check_he_matrices();
    printf("EnvelopeTestSim::init_concentric_spheres() - 3\n");

    // envelope_.xyz_coord_V_ = m.xyz_coord_V_;
    // envelope_.V_cycle_F_ = m.V_cycle_F_;

    // envelope_.xyz_coord_V_ = m.xyz_coord_V_;
    // envelope_.h_out_V_ = m.h_out_V_;
    // envelope_.V_cycle_E_ = m.V_cycle_E_;
    // envelope_.h_directed_E_ = m.h_directed_E_;
    // envelope_.V_cycle_F_ = m.V_cycle_F_;
    // envelope_.h_right_F_ = m.h_right_F_;
    // envelope_.h_negative_B_ = m.h_negative_B_;
    // envelope_.v_origin_H_ = m.v_origin_H_;
    // envelope_.e_undirected_H_ = m.e_undirected_H_;
    // envelope_.f_left_H_ = m.f_left_H_;
    // envelope_.h_next_H_ = m.h_next_H_;
    // envelope_.h_twin_H_ = m.h_twin_H_;
    // envelope_.copy_mesh_data(m);
    // envelope_.copy_mesh_cache(m);
    // reinit_envelope();
    // envelope_.copy_mesh_data(m);
    // envelope_.check_he_matrices();
    // reinit_envelope();
    // envelope_.reverse_faces();
    // envelope_.update_shifted_half_edge_arrows();
  }

  MatrixMesh build_cylinder(int Nphi, int Nz, double r, double z_min,
                            double z_max) {
    if (Nphi < 3) {
      throw std::runtime_error("build_cylinder requires Nphi >= 3");
    }
    if (Nz < 2) {
      throw std::runtime_error("build_cylinder requires Nz >= 2");
    }
    if (r <= 0) {
      throw std::runtime_error("build_cylinder requires r > 0");
    }
    if (z_min >= z_max) {
      throw std::runtime_error("build_cylinder requires z_min < z_max");
    }

    int Nv = Nphi * Nz;
    int Nf = 2 * Nphi * (Nz - 1);
    Samples3d xyz_coord_V(Nv, 3);
    Samples3i V_cycle_F(Nf, 3);
    double Delta_z = z_max - z_min;

    for (int nz = 0; nz < Nz; nz++) {
      double z = z_min + nz * Delta_z / (Nz - 1);
      for (int nphi = 0; nphi < Nphi; nphi++) {
        int v = nphi + nz * Nphi;
        double phi = nphi * 2 * M_PI / Nphi;
        double x = r * std::cos(phi);
        double y = r * std::sin(phi);
        xyz_coord_V.row(v) << x, y, z;
      }
    }

    for (int nz = 0; nz < Nz - 1; nz++) {
      int nz0 = nz;
      int nz1 = nz;
      int nz2 = nz + 1;
      int nz3 = nz + 1;
      for (int nphi = 0; nphi < Nphi; nphi++) {
        int nphi0 = nphi;
        int nphi1 = (nphi + 1) % Nphi;
        int nphi2 = (nphi + 1) % Nphi;
        int nphi3 = nphi;
        int v0 = nphi0 + nz0 * Nphi;
        int v1 = nphi1 + nz1 * Nphi;
        int v2 = nphi2 + nz2 * Nphi;
        int v3 = nphi3 + nz3 * Nphi;
        int f0 = 2 * v0;
        int f1 = 2 * v0 + 1;
        V_cycle_F.row(f0) << v0, v1, v2;
        V_cycle_F.row(f1) << v2, v3, v0;
      }
    }

    MatrixMesh m(xyz_coord_V, V_cycle_F);
    return m;
  }

  void init_with_npcs() {
    printf("EnvelopeTestSim::init_with_npcs()\n");
    envelope_.check_he_matrices();
    double inner_membrane_scale = 0.8;

    int Nv0 = envelope_.get_num_vertices();
    int Nf0 = envelope_.get_num_faces();

    printf("EnvelopeTestSim::init_with_npcs() - 0\n");
    //
    //
    int v_hole0 = 0;
    int valance_v_hole0 = envelope_.get_valence_v(v_hole0);
    while (valance_v_hole0 != 6) {
      v_hole0++;
      valance_v_hole0 = envelope_.get_valence_v(v_hole0);
    }

    // envelope_.swap_vertex_indices(v_hole0, Nv0 - 1);
    // envelope_.swap_vertex_indices(0, 1);
    // envelope_.swap_vertex_indices(0, 1);
    // envelope_.swap_vertex_indices(2, 6);
    //
    // envelope_.swap_edge_indices(0, 1);
    // envelope_.swap_edge_indices(0, 1);
    // envelope_.swap_edge_indices(2, 6);
    //
    // envelope_.swap_face_indices(0, 1);
    // envelope_.swap_face_indices(0, 1);
    // envelope_.swap_face_indices(2, 6);
    //
    // envelope_.swap_halfedge_indices(0, 1);
    // envelope_.swap_halfedge_indices(0, 1);
    // envelope_.swap_halfedge_indices(2, 6);

    printf("EnvelopeTestSim::init_with_npcs() - 0.1\n");

    Samples3d xyz_coord_V0 = envelope_.xyz_coord_V_;
    Samples3i V_cycle_F0 = envelope_.V_cycle_F_;

    int Nv = 2 * Nv0;
    int Nf = 2 * Nf0;

    printf("EnvelopeTestSim::init_with_npcs() - 1\n");
    Samples3d xyz_coord_V1 = inner_membrane_scale * xyz_coord_V0;
    Samples3i V_cycle_F1 = V_cycle_F0.array() + Nv0;
    V_cycle_F1.col(1).swap(V_cycle_F1.col(2));

    Samples3d xyz_coord_V(Nv, 3);
    Samples3i V_cycle_F(Nf, 3);

    xyz_coord_V.topRows(Nv0) = xyz_coord_V0;
    xyz_coord_V.bottomRows(Nv0) = xyz_coord_V1;

    V_cycle_F.topRows(Nf0) = V_cycle_F0;
    V_cycle_F.bottomRows(Nf0) = V_cycle_F1;

    //
    // std::cout << "V_cycle_F = " << V_cycle_F.transpose() << '\n';
    // std::cout << "V_cycle_F = " << V_cycle_F << '\n';

    //
    //
    printf("EnvelopeTestSim::init_with_npcs() - 2\n");
    // envelope_.xyz_coord_V_ = xyz_coord_V;
    // envelope_.V_cycle_F_ = V_cycle_F;

    // envelope_.remove_f(0);
    // MatrixMesh m0 = build_cylinder(3, 2, 1.0, -1.0, 0.0);
    // MatrixMesh m1 = build_cylinder(3, 2, 1.0, 0.0, 1.0);
    // MatrixMesh m0 = build_cylinder(13, 20, 1.0, -1.0, 0.0);
    // MatrixMesh m1 = build_cylinder(13, 11, 1.0, 0.0, 1.0);

    Samples3d xyz_loop0{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    Samples3d xyz_loop1{{0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
    Samples3d xyz_loop2{{0, 0, 2}, {1, 0, 2}, {0, 1, 2}};
    MatrixMesh m0 = MatrixMesh::tube_between_loops(xyz_loop0, xyz_loop1, 3);
    MatrixMesh m1 = MatrixMesh::tube_between_loops(xyz_loop1, xyz_loop2, 3);

    m0.print_simplex_mats();
    m1.print_simplex_mats();
    m0.check_he_matrices();
    m1.check_he_matrices();

    int b0 = 0;
    int b1 = 1;
    int h0 = m0.h_negative_b(b0);
    int h1 = m1.h_twin_h(m1.h_negative_b(b1));
    int v0 = m0.v_origin_h(h0);
    int v1 = m1.v_origin_h(h1);
    double x0 = m0.xyz_coord_V_(v0, 0);
    double y0 = m0.xyz_coord_V_(v0, 1);
    double z0 = m0.xyz_coord_V_(v0, 2);
    double x1 = m1.xyz_coord_V_(v1, 0);
    double y1 = m1.xyz_coord_V_(v1, 1);
    double z1 = m1.xyz_coord_V_(v1, 2);

    printf("x0, y0, z0 = %f, %f, %f\n", x0, y0, z0);
    printf("x1, y1, z1 = %f, %f, %f\n", x1, y1, z1);

    m0.normal_v(0);

    m0.glue_boundaries(m1, b0, b1);

    envelope_.xyz_coord_V_ = m0.xyz_coord_V_;
    envelope_.V_cycle_F_ = m0.V_cycle_F_;
    envelope_.init_core_data_from_V_cycle_F();

    printf("EnvelopeTestSim::init_with_npcs() - 3\n");
    //
    //
    //
    //
    // envelope_.swap_vertex_indices(0, 1);
    // envelope_.swap_vertex_indices(0, 1);
    // envelope_.swap_vertex_indices(2, 6);
    //
    // envelope_.swap_edge_indices(0, 1);
    // envelope_.swap_edge_indices(0, 1);
    // envelope_.swap_edge_indices(2, 6);
    //
    // envelope_.swap_face_indices(0, 1);
    // envelope_.swap_face_indices(0, 1);
    // envelope_.swap_face_indices(2, 6);

    // envelope_.swap_halfedge_indices(0, 1);
    // envelope_.swap_halfedge_indices(0, 1);
    // envelope_.swap_halfedge_indices(2, 6);
    //
    //
    printf("EnvelopeTestSim::init_with_npcs() - 4\n");
    reinit_envelope();

    printf("EnvelopeTestSim::init_with_npcs() - 5\n");
    envelope_.check_he_matrices();
  }
};

} // namespace meshbrane
