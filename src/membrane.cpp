/**
 * @file membrane.cpp
 */
#include "meshbrane/membrane.hpp"
#include "meshbrane/kmc.hpp"
#include "meshbrane/tethering_force.hpp"
#include <cmath> // For std::acos and std::sqrt
#include <filesystem>

namespace fs = std::filesystem;

namespace meshbrane {

////////////////////////////////////////
// stuff that actually gets used
void Membrane::apply_internal_interactions() {
  apply_tether_force_V();
  apply_bending_force_V();
  apply_area_force_V();
  apply_volume_force_V();
}

void Membrane::clear_interactions() {
  force_V_.resize(get_num_vertices(), 3);
  force_V_.setZero();
}

void Membrane::apply_thermal_fluctuations(double dt, double kBT,
                                          kmc::RandomNumberGenerator &rng) {
  if (!enable_fluctuations_) {
    return;
  }
  size_t Nv = get_num_vertices();
  for (int v = 0; v < Nv; v++) {
    // Vec3d force;
    for (int i = 0; i < 3; i++) {
      force_V_(v, i) +=
          std::sqrt(2 * kBT * area_V_[v] * local_drag_coefficient_ / dt) *
          rng.standard_normal();
    }
  }
}

void Membrane::update_cached_data() {
  update_mesh_geometric_data();
  update_geotargets();

  update_laplacian_matrix();
  update_mean_curvature(); // add_vertex in patch.cpp causes a problem here
  update_gaussian_curvature();
  update_surface_tension_and_pressure();
  // update_internal_forces();
}

void Membrane::update_state_variables(double dt) {

  if (use_local_drag_coefficient_) {
    for (int v = 0; v < get_num_vertices(); v++) {
      double drag_coefficient = local_drag_coefficient_ * area_V_[v];
      xyz_coord_V_.row(v) += dt * force_V_.row(v) / drag_coefficient;
    }
  } else {
    for (int v = 0; v < get_num_vertices(); v++) {
      xyz_coord_V_.row(v) += dt * force_V_.row(v) / node_drag_coefficient_;
    }
  }

  // apply_fluctuations_V(dt);
  num_flips_ = 0;
  if (t_flip_ <= t_) {
    // num_flips_ = monte_flip_sweep();
    update_mesh_geometric_data();
    update_geotargets();
    num_flips_ = flip_sweep();
    // printf("num_flips = %d\n", num_flips_);
    t_flip_ = t_flip_ + dt_flip_;
  }

  t_ += dt;
  dt_ = dt;
}

double Membrane::dt_max() const {
  if (use_local_drag_coefficient_) {
    double dx_max = std::min(0.1 * target_edge_length_, dx_max_);
    double fmax = 0.0;
    for (int v = 0; v < get_num_vertices(); v++) {
      fmax = std::max(fmax, math::L2norm(force_V_.row(v) / area_V_[v]));
    }
    return dx_max * local_drag_coefficient_ / fmax;
  }
  double dx_max = std::min(0.1 * target_edge_length_, dx_max_);
  double Fmax = 0.0;
  for (int v = 0; v < get_num_vertices(); v++) {
    Fmax = std::max(Fmax, math::L2norm(force_V_.row(v)));
  }
  return dx_max * node_drag_coefficient_ / Fmax;
}

void Membrane::print_info() {
  double A = area_F_.sum();
  double A0 = initial_area_;
  double V = total_volume_;
  double V0 = initial_volume_;
  double P = pressure_;
  std::cout << "  " << name_ << std::endl;
  printf("    (A-A0)/A0=%.10f\n", (A - A0) / A0);
  printf("    (V-V0)/V0=%.10f\n", (V - V0) / V0);
  printf("    pressure=%.10f\n", pressure_);
  printf("    surface tension mean=%.10f\n", surface_tension_F_.mean());
  printf("    surface tension max=%.10f\n", surface_tension_F_.maxCoeff());
  printf("    surface tension min=%.10f\n", surface_tension_F_.minCoeff());
  // printf("    num_flips: %d\n", num_flips_);
  printf("    num_flips: %d\n", total_edge_flips_);
  // printf("    num_flips/num_edges: %.10f\n",
  //        static_cast<double>(num_flips_) /
  //            static_cast<double>(get_num_edges()));
  // printf("  dt=%.20f\n", dt_);
}

void Membrane::update_membrane_visuals() {
  // printf("Membrane::update_membrane_visuals\n");
  update_mesh_visuals();
  update_force_arrows();
  update_mcvec_arrows();
  if (show_contact_patches_) {
    // spb_patch_minus_.uncolor_faces();
    // spb_patch_plus_.uncolor_faces();
    spb_patch_minus_.color_faces();
    spb_patch_plus_.color_faces();
    if (draw_wireframe_) {
      spb_patch_plus_.color_edges();
      spb_patch_minus_.color_edges();
    }
  }
}

///////////////////////////////////////////////////////
// initialization /////////////////////////////////////
///////////////////////////////////////////////////////
void Membrane::set_attributes_from_parameters() {
  MatrixMesh::set_attributes_from_parameters();
  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  if (parameters_["timestep_type"]) {
    timestep_type_ = parameters_["timestep_type"].as<std::string>();
  }
  // Bending force
  if (parameters_["bending_modulus"]) {
    bending_modulus_ = parameters_["bending_modulus"].as<double>();
  }
  if (parameters_["spontaneous_curvature"]) {
    spontaneous_curvature_ = parameters_["spontaneous_curvature"].as<double>();
  }
  if (parameters_["splay_modulus"]) {
    splay_modulus_ = parameters_["splay_modulus"].as<double>();
  }
  ///////////////////////////////////////////////////////
  // Tether force
  if (parameters_["dimensionless_tether_repulsive_singularity"]) {
    dimensionless_tether_repulsive_singularity_ =
        parameters_["dimensionless_tether_repulsive_singularity"].as<double>();
  }
  if (parameters_["dimensionless_tether_repulsive_onset"]) {
    dimensionless_tether_repulsive_onset_ =
        parameters_["dimensionless_tether_repulsive_onset"].as<double>();
  }
  if (parameters_["dimensionless_tether_attractive_onset"]) {
    dimensionless_tether_attractive_onset_ =
        parameters_["dimensionless_tether_attractive_onset"].as<double>();
  }
  if (parameters_["dimensionless_tether_attractive_singularity"]) {
    dimensionless_tether_attractive_singularity_ =
        parameters_["dimensionless_tether_attractive_singularity"].as<double>();
  }
  if (parameters_["tether_stiffness"]) {
    tether_stiffness_ = parameters_["tether_stiffness"].as<double>();
  }
  // Area conservation force
  if (parameters_["area_stiffness"]) {
    area_stiffness_ = parameters_["area_stiffness"].as<double>();
  }
  if (parameters_["fix_target_face_area"]) {
    fix_target_face_area_ = parameters_["fix_target_face_area"].as<bool>();
  }
  // Volume conservation force
  if (parameters_["fix_target_volume"]) {
    fix_target_volume_ = parameters_["fix_target_volume"].as<bool>();
  }
  if (parameters_["volume_stiffness"]) {
    volume_stiffness_ = parameters_["volume_stiffness"].as<double>();
  }
  if (parameters_["node_drag_coefficient"]) {
    node_drag_coefficient_ = parameters_["node_drag_coefficient"].as<double>();
  }
  ///////////////////////////////////////////////////////
  if (parameters_["vector_field_scale"]) {
    vector_field_scale_ = parameters_["vector_field_scale"].as<double>();
  }
  if (parameters_["show_force_field"]) {
    show_force_field_ = parameters_["show_force_field"].as<bool>();
  }
  if (parameters_["show_mcvec_field"]) {
    show_mcvec_field_ = parameters_["show_mcvec_field"].as<bool>();
  }
  if (parameters_["enable_flipping"]) {
    enable_flipping_ = parameters_["enable_flipping"].as<bool>();
  }
  if (parameters_["enable_fluctuations"]) {
    enable_fluctuations_ = parameters_["enable_fluctuations"].as<bool>();
  }
  if (parameters_["dt_flip"]) {
    dt_flip_ = parameters_["dt_flip"].as<double>();
  }
  if (parameters_["flipping_probability"]) {
    flipping_probability_ = parameters_["flipping_probability"].as<double>();
  }
  ///////////////////////////////////////////////////////
  if (parameters_["show_contact_patches"]) {
    show_contact_patches_ = parameters_["show_contact_patches"].as<bool>();
  }
  if (parameters_["pressure_type"]) {
    pressure_type_ = parameters_["pressure_type"].as<std::string>();
  }
  if (parameters_["use_surface_tension_constant"]) {
    use_surface_tension_constant_ =
        parameters_["use_surface_tension_constant"].as<bool>();
  }
  if (parameters_["use_surface_tension_penalty_local"]) {
    use_surface_tension_penalty_local_ =
        parameters_["use_surface_tension_penalty_local"].as<bool>();
  }
  if (parameters_["surface_tension_constant"]) {
    surface_tension_constant_ =
        parameters_["surface_tension_constant"].as<double>();
  } else if (use_surface_tension_constant_) {
    printf(
        "Warning: use_surface_tension_constant_ = true but no constant value "
        "found\n");
  }
  if (parameters_["wca_sigma"]) {
    wca_sigma_ = parameters_["wca_sigma"].as<double>();
    // 0.1*r_cutoff=0.1*2^(1/6)*sigma
    dx_max_ = 0.1122462048309373 * wca_sigma_;
  }
  if (parameters_["wca_epsilon"]) {
    wca_epsilon_total_ = parameters_["wca_epsilon"].as<double>();
  }
  // set global parameters
  if ((*sim_parameters_)["kBT"]) {
    kBT_ = (*sim_parameters_)["kBT"].as<double>();
  }
  if ((*sim_parameters_)["dt"]) {
    dt0_ = (*sim_parameters_)["dt"].as<double>();
  }
  if ((*sim_parameters_)["bulk_viscosity"]) {
    bulk_viscosity_ = (*sim_parameters_)["bulk_viscosity"].as<double>();
  }
  if (parameters_["use_local_drag_coefficient"]) {
    use_local_drag_coefficient_ =
        parameters_["use_local_drag_coefficient"].as<bool>();
    // local_drag_coefficient_ = 4 * M_PI * bulk_viscosity_;
    local_drag_coefficient_ = 1.5 * bulk_viscosity_; // * 1/R but R=1
  }
}

// void Membrane::set_attributes_from_yaml_node(const YAML::Node &node) {
//   MatrixMesh::set_attributes_from_yaml_node(node);

//   if (node["timestep_type"]) {
//     timestep_type_ = node["timestep_type"].as<std::string>();
//   }
//   // Bending force
//   if (node["bending_modulus"]) {
//     bending_modulus_ = node["bending_modulus"].as<double>();
//   }
//   if (node["spontaneous_curvature"]) {
//     spontaneous_curvature_ = node["spontaneous_curvature"].as<double>();
//   }
//   if (node["splay_modulus"]) {
//     splay_modulus_ = node["splay_modulus"].as<double>();
//   }
//   ///////////////////////////////////////////////////////
//   // Tether force
//   if (node["dimensionless_tether_repulsive_singularity"]) {
//     dimensionless_tether_repulsive_singularity_ =
//         node["dimensionless_tether_repulsive_singularity"].as<double>();
//   }
//   if (node["dimensionless_tether_repulsive_onset"]) {
//     dimensionless_tether_repulsive_onset_ =
//         node["dimensionless_tether_repulsive_onset"].as<double>();
//   }
//   if (node["dimensionless_tether_attractive_onset"]) {
//     dimensionless_tether_attractive_onset_ =
//         node["dimensionless_tether_attractive_onset"].as<double>();
//   }
//   if (node["dimensionless_tether_attractive_singularity"]) {
//     dimensionless_tether_attractive_singularity_ =
//         node["dimensionless_tether_attractive_singularity"].as<double>();
//   }
//   if (node["tether_stiffness"]) {
//     tether_stiffness_ = node["tether_stiffness"].as<double>();
//   }
//   // Area conservation force
//   if (node["area_stiffness"]) {
//     area_stiffness_ = node["area_stiffness"].as<double>();
//   }
//   if (node["fix_target_face_area"]) {
//     fix_target_face_area_ = node["fix_target_face_area"].as<bool>();
//   }
//   // Volume conservation force
//   if (node["fix_target_volume"]) {
//     fix_target_volume_ = node["fix_target_volume"].as<bool>();
//   }
//   if (node["volume_stiffness"]) {
//     volume_stiffness_ = node["volume_stiffness"].as<double>();
//   }
//   if (node["node_drag_coefficient"]) {
//     node_drag_coefficient_ = node["node_drag_coefficient"].as<double>();
//   }
//   ///////////////////////////////////////////////////////
//   if (node["vector_field_scale"]) {
//     vector_field_scale_ = node["vector_field_scale"].as<double>();
//   }
//   if (node["show_force_field"]) {
//     show_force_field_ = node["show_force_field"].as<bool>();
//   }
//   if (node["show_mcvec_field"]) {
//     show_mcvec_field_ = node["show_mcvec_field"].as<bool>();
//   }
//   if (node["enable_flipping"]) {
//     enable_flipping_ = node["enable_flipping"].as<bool>();
//   }
//   if (node["enable_fluctuations"]) {
//     enable_fluctuations_ = node["enable_fluctuations"].as<bool>();
//   }
//   if (node["dt_flip"]) {
//     dt_flip_ = node["dt_flip"].as<double>();
//   }
//   if (node["flipping_probability"]) {
//     flipping_probability_ = node["flipping_probability"].as<double>();
//   }
//   ///////////////////////////////////////////////////////
//   if (node["show_contact_patches"]) {
//     show_contact_patches_ = node["show_contact_patches"].as<bool>();
//   }
//   if (node["pressure_type"]) {
//     pressure_type_ = node["pressure_type"].as<std::string>();
//   }
//   if (node["use_surface_tension_constant"]) {
//     use_surface_tension_constant_ =
//         node["use_surface_tension_constant"].as<bool>();
//   }
//   if (node["use_surface_tension_penalty_local"]) {
//     use_surface_tension_penalty_local_ =
//         node["use_surface_tension_penalty_local"].as<bool>();
//   }
//   if (node["surface_tension_constant"]) {
//     surface_tension_constant_ =
//     node["surface_tension_constant"].as<double>();
//   } else if (use_surface_tension_constant_) {
//     printf(
//         "Warning: use_surface_tension_constant_ = true but no constant value
//         " "found\n");
//   }
//   if (node["wca_sigma"]) {
//     wca_sigma_ = node["wca_sigma"].as<double>();
//     // 0.1*r_cutoff=0.1*2^(1/6)*sigma
//     dx_max_ = 0.1122462048309373 * wca_sigma_;
//   }
//   if (node["wca_epsilon"]) {
//     wca_epsilon_total_ = node["wca_epsilon"].as<double>();
//   }
//   // set global parameters
//   if ((*sim_parameters_)["kBT"]) {
//     kBT_ = (*sim_parameters_)["kBT"].as<double>();
//   }
//   if ((*sim_parameters_)["dt"]) {
//     dt0_ = (*sim_parameters_)["dt"].as<double>();
//   }
//   if ((*sim_parameters_)["bulk_viscosity"]) {
//     bulk_viscosity_ = (*sim_parameters_)["bulk_viscosity"].as<double>();
//   }
//   if (node["use_local_drag_coefficient"]) {
//     use_local_drag_coefficient_ =
//     node["use_local_drag_coefficient"].as<bool>();
//     // local_drag_coefficient_ = 4 * M_PI * bulk_viscosity_;
//     local_drag_coefficient_ = 1.5 * bulk_viscosity_; // * 1/R but R=1
//   }
// }

void Membrane::init_membrane_from_attributes() {

  initial_area_ = average_face_area_ * get_num_faces();
  initial_volume_ = total_volume_;
  target_face_area_ = average_face_area_;
  target_volume_ = total_volume_;
  update_geotargets();

  mcvec_V_.resize(get_num_vertices(), 3);
  mcvec_V_.setZero();

  force_V_.resize(get_num_vertices(), 3);
  force_V_.setZero();

  // contact_force_V_.resize(get_num_vertices(), 3);
  // contact_force_V_.setZero();

  // external_force_V_.resize(get_num_vertices(), 3);
  // external_force_V_.setZero();

  // internal_force_V_.resize(get_num_vertices(), 3);
  // internal_force_V_.setZero();

  // update_pressure_soft_penalty();
  // update_surface_tension_soft_penalty();
  update_surface_tension_and_pressure();

  // // randng_
  // force_V_.resize(get_num_vertices(), 3);
  // force_V_.setZero();
  if (enable_flipping_) {
    t_flip_ = dt_flip_; // time to next flip
  } else {
    t_flip_ = std::numeric_limits<double>::infinity();
  }

  // integration patch initialization
  integration_patch_.supermesh_ = this;
  integration_patch_.rgba_face_ = RGBA_DICT.at("meshbrane_orange");

  spb_patch_plus_.supermesh_ = this;
  spb_patch_minus_.supermesh_ = this;
}

// void Membrane::init(const YAML::Node &node) {
//   set_attributes_from_yaml_node(node);
//   init_matrixmesh_from_attributes();
//   init_membrane_from_attributes();
// }

void Membrane::update_geotargets() {
  target_edge_length_ = average_edge_length_;
  if (!fix_target_face_area_) {
    target_face_area_ = average_face_area_;
  }
  if (!fix_target_volume_) {
    target_volume_ = total_volume_;
  }
}

void Membrane::init_from_ply() {
  MatrixMesh::init_from_ply();
  initial_area_ = average_face_area_ * get_num_faces();
  initial_volume_ = total_volume_;
  target_face_area_ = average_face_area_;
  target_volume_ = total_volume_;
  update_geotargets();

  mcvec_V_.resize(get_num_vertices(), 3);
  mcvec_V_.setZero();

  force_V_.resize(get_num_vertices(), 3);
  force_V_.setZero();

  update_surface_tension_and_pressure();

  // // randng_
  // force_V_.resize(get_num_vertices(), 3);
  // force_V_.setZero();
  if (enable_flipping_) {
    t_flip_ = dt_flip_; // time to next flip
  } else {
    t_flip_ = std::numeric_limits<double>::infinity();
  }

  // integration patch initialization
  integration_patch_.supermesh_ = this;
  integration_patch_.rgba_face_ = RGBA_DICT.at("meshbrane_orange");

  // set_node_drag_coefficient_from_bulk_viscosity();
  // printf("*****************************************************************"
  //        "*******************************************node_drag_coefficient_"
  //        " = %.10f\n",
  //        node_drag_coefficient_);
}

///////////////////////////////////////////////////////
// Membrane physics ///////////////////////////////////
///////////////////////////////////////////////////////
void Membrane::update_pressure_soft_penalty() {
  pressure_ =
      volume_stiffness_ * (total_volume_ - target_volume_) / target_volume_;
}

void Membrane::update_surface_tension_soft_penalty() {
  // surface_tension_F_.resize(get_num_faces());
  for (int f = 0; f < get_num_faces(); f++) {
    surface_tension_F_(f) +=
        area_stiffness_ * (area_F_[f] - target_face_area_) / target_face_area_;
  }
}

void Membrane::update_surface_tension_constant() {
  // surface_tension_F_.resize(get_num_faces());
  for (int f = 0; f < get_num_faces(); f++) {
    surface_tension_F_(f) += surface_tension_constant_;
  }
}

void Membrane::update_surface_tension_and_pressure() {

  // set surface tension to zero
  surface_tension_F_.resize(get_num_faces());
  surface_tension_F_.setZero();
  // if (surface_tension_type_ == "constant") {
  //   update_surface_tension_constant();
  // }

  // if (surface_tension_type_ == "penalty_local") {
  //   update_surface_tension_soft_penalty();
  // }
  if (use_surface_tension_constant_) {
    update_surface_tension_constant();
  }

  if (use_surface_tension_penalty_local_) {
    update_surface_tension_soft_penalty();
  }

  if (pressure_type_ == "penalty") {
    update_pressure_soft_penalty();
  }
}

Vec3d Membrane::get_volume_force_v(int v) const {
  Vec3d F = Vec3d::Zero();
  for (auto f : generate_F_incident_v(v)) {
    F += -pressure_ * area_F_[f] * normal_F_.row(f) / 3.0;
  }
  return F;
}

void Membrane::apply_volume_force_V() {
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    force_V_.row(v) += get_volume_force_v(v);
  }
}

Vec3d Membrane::get_tether_force_v(int v) const {
  Vec3d F = Vec3d::Zero();
  for (auto h : generate_H_out_v_clockwise(v)) {
    // int e = e_undirected_H_[h];
    Vec3d x = xyz_coord_v(v), xp = xyz_coord_v(v_head_h(h));

    //     target_edge_length_
    // dimensionless_tether_repulsive_onset_
    // dimensionless_tether_repulsive_singularity_
    // dimensionless_tether_attractive_onset_
    // dimensionless_tether_attractive_singularity_
    // tether_stiffness_
    F += tether_force(x, xp, target_edge_length_, tether_stiffness_,
                      dimensionless_tether_repulsive_singularity_,
                      dimensionless_tether_repulsive_onset_,
                      dimensionless_tether_attractive_onset_,
                      dimensionless_tether_attractive_singularity_, 1.0, 1.0);
    ;
  }
  return F;
}

void Membrane::apply_tether_force_V() {
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    force_V_.row(v) += get_tether_force_v(v);
  }
}

Vec3d Membrane::get_area_force_v(int v) const {
  Vec3d F = Vec3d::Zero();

  for (auto h : generate_H_out_v_clockwise(v)) {
    if (some_negative_boundary_contains_h(h)) {
      continue;
    }
    int f = f_left_h(h);
    double gamma = surface_tension_F_[f];
    int h_next = h_next_h(h);
    int v1 = v_origin_h(h_next);
    int v2 = v_head_h(h_next);
    Vec3d dr = xyz_coord_v(v2) - xyz_coord_v(v1);
    Vec3d n = normal_F_.row(f);
    Vec3d x = xyz_coord_v(v);
    Vec3d dr_perp = math::cross(dr, n);
    F += gamma * dr_perp / 2;
  }
  return F;
}

void Membrane::apply_area_force_V() {
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    force_V_.row(v) += get_area_force_v(v);
  }
}

Vec3d Membrane::get_bending_force_v(int v) const {
  double B = bending_modulus_;
  double H = mean_curvature_V_[v];
  double K = gaussian_curvature_V_[v];
  double lapH = lap_mean_curvature_V_[v];
  double H0 = spontaneous_curvature_;
  double Fdensity = -2 * B * (lapH + 2 * (H - H0) * (H * H + H0 * H - K));
  double A = area_V_[v];
  Vec3d n = normal_V_.row(v);
  // Vec3d n = normal_F_.row(f_left_h(h_out_v(v)));
  return Fdensity * A * n;
  // return n;
}

void Membrane::apply_bending_force_V() {
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    force_V_.row(v) += get_bending_force_v(v);
  }
}

void Membrane::update_force_arrows() {
  Eigen::Vector3d rgb = force_arrows_.rgb_;
  double scale = vector_field_scale_;
  force_arrows_.update(xyz_coord_V_, force_V_, scale, rgb);
}

void Membrane::update_mcvec_arrows() {
  Eigen::Vector3d rgb = mcvec_arrows_.rgb_;
  double scale = vector_field_scale_;
  mcvec_arrows_.update(xyz_coord_V_, mcvec_V_, scale, rgb);
}

double Membrane::monte_flip_probability(int e) const {
  // requires cached data to be up to date
  // target_edge_length_
  // length_E_
  // area_F_
  // target_face_area_
  int h0 = h_directed_E_(e);
  int h1 = h_twin_h(h0);
  int h2 = h_next_h(h0);
  int h3 = h_next_h(h2);
  int h4 = h_next_h(h1);
  int h5 = h_next_h(h4);
  int v0 = v_origin_h(h1);
  int v1 = v_origin_h(h3);
  int v2 = v_origin_h(h0);
  int v3 = v_origin_h(h5);
  int f0 = f_left_h(h0);
  int f1 = f_left_h(h1);

  Vec3d x;
  Vec3d xp;
  double norm_dx_pre, norm_dx_post;
  double zpre;
  double zpost;
  double Upre = 0;
  double Upost = 0;

  //   x = xyz_coord_v(v0);
  //   xp = xyz_coord_v(v2);
  //   norm_dx = math::L2norm(x - xp);
  norm_dx_pre = length_E_[e];               // ***
  zpre = norm_dx_pre / target_edge_length_; // ***

  double Utether_pre =
      Utether(zpre, 1.0, 1.0, dimensionless_tether_repulsive_singularity_,
              dimensionless_tether_repulsive_onset_,
              dimensionless_tether_attractive_onset_,
              dimensionless_tether_attractive_singularity_);

  x = xyz_coord_v(v3);
  xp = xyz_coord_v(v1);
  norm_dx_post = math::L2norm(x - xp);
  zpost = norm_dx_post / target_edge_length_;

  double Utether_post =
      Utether(zpost, 1.0, 1.0, dimensionless_tether_repulsive_singularity_,
              dimensionless_tether_repulsive_onset_,
              dimensionless_tether_attractive_onset_,
              dimensionless_tether_attractive_singularity_);

  // check if they are infinite
  if (std::isinf(Utether_post)) {
    // printf("Utether_post is infinite\n");
    return 0.0;
  }
  if (std::isinf(Utether_pre)) {
    printf("Utether_pre is infinite\n");
    return 1.0;
  }

  //   double L2 = math::L2norm(xyz_coord_v(v0) - xyz_coord_v(v1));
  //   double L3 = math::L2norm(xyz_coord_v(v1) - xyz_coord_v(v2));
  //   double L4 = math::L2norm(xyz_coord_v(v2) - xyz_coord_v(v3));
  //   double L5 = math::L2norm(xyz_coord_v(v3) - xyz_coord_v(v0));
  double L2 = length_E_[e_undirected_h(h2)];
  double L3 = length_E_[e_undirected_h(h3)];
  double L4 = length_E_[e_undirected_h(h4)];
  double L5 = length_E_[e_undirected_h(h5)];

  double A201 = area_F_[f0]; // math::heron_area(norm_dx_pre, L2, L3);
  double A023 = area_F_[f1]; // math::heron_area(norm_dx_pre, L4, L5);

  double A312 = math::heron_area(norm_dx_post, L3, L4);
  double A130 = math::heron_area(norm_dx_post, L5, L2);

  double Uarea_pre =
      math::POW2(A201 - target_face_area_) / (2 * target_face_area_) +
      math::POW2(A023 - target_face_area_) / (2 * target_face_area_);

  double Uarea_post =
      math::POW2(A312 - target_face_area_) / (2 * target_face_area_) +
      math::POW2(A130 - target_face_area_) / (2 * target_face_area_);

  double dEtot = tether_stiffness_ * (Utether_post - Utether_pre) +
                 area_stiffness_ * (Uarea_post - Uarea_pre);
  return std::exp(-dEtot / kBT_);
}

int Membrane::monte_flip_sweep() {
  if (!enable_flipping_) {
    return 0;
  }
  int flip_count = 0;
  int num_edges = get_num_edges();
  std::vector<int> Eperm = randng_.random_permutation(num_edges);
  for (int _e = 0; _e < num_edges; _e++) {
    int e = Eperm[_e];
    if (!h_is_flippable(h_directed_E_(e))) {
      continue;
    }
    // Edges are flipped with probability of flipping_probability_
    double _r = randng_.standard_uniform();
    if (_r > flipping_probability_) {
      continue;
    }
    // Flips are accepted with a probability monte_flip_probability(e).
    // Note: if the flip results in an energy decrease, then
    // monte_flip_probability > 1 and flip is always accepted
    _r = randng_.standard_uniform();
    if (_r < monte_flip_probability(e)) {
      flip_edge(e);
      flip_count++;
    }
  }
  return flip_count;
}

int Membrane::flip_sweep() {
  if (!enable_flipping_) {
    return 0;
  }
  int num_flips{0};
  num_flips = monte_flip_sweep();
  return num_flips;
}

} // namespace meshbrane
