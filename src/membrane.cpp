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

///////////////////////////////////////////////////////
// initialization /////////////////////////////////////
///////////////////////////////////////////////////////

void Membrane::update_geotargets() {
  // printf("*********************Setting target_edge_length_ = %.10f\n",
  //        average_edge_length_);
  target_edge_length_ = average_edge_length_;
  if (!fix_target_face_area_) {
    // printf("Setting target_face_area_ = %.10f\n", average_face_area_);
    target_face_area_ = average_face_area_;
  }
  if (!fix_target_volume_) {
    target_volume_ = total_volume_;
  }
}

void Membrane::init() {
  // if (sim_parameters_ == nullptr) {
  //   printf("No parameters found for Membrane\n");
  // } else {
  //   printf("Initializing Membrane with parameters:\n");
  //   // std::cout << (*sim_parameters_)[name_] << std::endl;
  //   std::cout << parameters_ << std::endl;
  // }

  init_mesh();
  initial_area_ = average_face_area_ * get_num_faces();
  initial_volume_ = total_volume_;
  target_face_area_ = average_face_area_;
  target_volume_ = total_volume_;
  update_geotargets();

  mcvec_V_.resize(get_num_vertices(), 3);
  mcvec_V_.setZero();

  force_V_.resize(get_num_vertices(), 3);
  force_V_.setZero();

  contact_force_V_.resize(get_num_vertices(), 3);
  contact_force_V_.setZero();

  external_force_V_.resize(get_num_vertices(), 3);
  external_force_V_.setZero();

  internal_force_V_.resize(get_num_vertices(), 3);
  internal_force_V_.setZero();

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

  heat_dt_ = initial_area_ / get_num_vertices();
  // belkin_dt_ = heat_dt_;
}

void Membrane::init_from_ply() {
  // if (sim_parameters_ == nullptr) {
  //   printf("No parameters found for Membrane\n");
  // } else {
  //   printf("Initializing Membrane with parameters:\n");
  //   // std::cout << (*sim_parameters_)[name_] << std::endl;
  //   std::cout << parameters_ << std::endl;
  // }

  // init_mesh();
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

  contact_force_V_.resize(get_num_vertices(), 3);
  contact_force_V_.setZero();

  external_force_V_.resize(get_num_vertices(), 3);
  external_force_V_.setZero();

  internal_force_V_.resize(get_num_vertices(), 3);
  internal_force_V_.setZero();

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

  heat_dt_ = initial_area_ / get_num_vertices();
  // belkin_dt_ = heat_dt_;

  // set_node_drag_coefficient_from_bulk_viscosity();
  // printf("*****************************************************************"
  //        "*******************************************node_drag_coefficient_"
  //        " = %.10f\n",
  //        node_drag_coefficient_);
}

void Membrane::update_internal_forces() {
  int num_vertices = get_num_vertices();
  internal_force_V_.resize(num_vertices, 3);
  internal_force_V_.setZero();
  for (int v = 0; v < num_vertices; v++) {
    internal_force_V_.row(v) += get_volume_force_v(v);
    internal_force_V_.row(v) += get_tether_force_v(v);
    internal_force_V_.row(v) += get_area_force_v(v);
    internal_force_V_.row(v) += get_bending_force_v(v);
  }
}

void Membrane::update_cached_data() {
  check_he_matrices();
  printf("Membrane::update_cached_data\n");
  update_mesh_geometric_data();
  check_he_matrices();
  printf("Membrane::update_cached_data - update_mesh_geometric_data\n");
  update_geotargets();
  check_he_matrices();
  printf("Membrane::update_cached_data - update_geotargets\n");

  update_laplacian_matrix();
  check_he_matrices(); // this is fine up to here
  printf("Membrane::update_cached_data - update_laplacian_matrix\n");
  update_mean_curvature(); // add_vertex in patch.cpp causes a problem here
  check_he_matrices();
  printf("Membrane::update_cached_data - update_mean_curvature\n");
  update_gaussian_curvature();
  check_he_matrices();
  printf("Membrane::update_cached_data - update_gaussian_curvature\n");
  update_surface_tension_and_pressure();
  printf(
      "Membrane::update_cached_data - update_surface_tension_and_pressure\n");
  // update_internal_forces();
}

void Membrane::update_membrane() {
  // printf("***************************Updating membrane\n");
  update_mesh_geometric_data();
  update_geotargets();

  update_laplacian_matrix();
  update_mean_curvature();
  update_gaussian_curvature();
  update_surface_tension_and_pressure();
  apply_forces();
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

Vec3d Membrane::get_fluctuations_v(int v, double dt) {

  if (use_local_drag_coefficient_) {
    Vec3d dx = Vec3d::Zero();
    for (int i = 0; i < 3; i++) {
      dx[i] = std::sqrt(2 * kBT_ * dt * area_V_[v] / local_drag_coefficient_) *
              randng_.standard_normal();
    }
    return dx;
  }
  Vec3d dx = Vec3d::Zero();
  for (int i = 0; i < 3; i++) {
    dx[i] = std::sqrt(2 * kBT_ * dt / node_drag_coefficient_) *
            randng_.standard_normal();
  }
  // printf("dx_noise: %.10f %.10f %.10f\n", dx(0), dx(1), dx(2));
  return dx;
}

void Membrane::apply_fluctuations_V(double dt) {
  if (!enable_fluctuations_) {
    return;
  }
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    xyz_coord_V_.row(v) += get_fluctuations_v(v, dt);
  }
}

void Membrane::apply_contact_force_V() {
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    force_V_.row(v) += contact_force_V_.row(v);
  }
}

void Membrane::apply_external_force_V() {
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    force_V_.row(v) += external_force_V_.row(v);
  }
}

void Membrane::update_force_arrows() {
  // size_t num_vertices = get_num_vertices();
  // for (int i = 0; i < 3; i++) {
  //   force_arrows_[i].resize(num_vertices, 3);
  // }
  // double shaft_len = 0.9;
  // double tip_len = 0.1;
  // double force_scale = vector_field_scale_;
  // for (int v = 0; v < num_vertices; v++) {
  //   Vec3d p1 = xyz_coord_V_.row(v);
  //   Vec3d u12 = force_scale * force_V_.row(v);
  //   Vec3d p2 = p1 + u12;
  //   Vec3d n = normal_V_.row(v);
  //   Vec3d u12_perp;
  //   math::cross_inplace(n, u12, u12_perp);

  //   Vec3d p3 = p2 - tip_len * u12 + tip_len * u12_perp;

  //   force_arrows_[0].row(v) = p1;
  //   force_arrows_[1].row(v) = p2;
  //   force_arrows_[2].row(v) = p3;
  // }
  Eigen::Vector3d rgb = force_arrows_.rgb_;
  double scale = vector_field_scale_;
  force_arrows_.update(xyz_coord_V_, force_V_, scale, rgb);
}

void Membrane::update_mcvec_arrows() {
  // size_t num_vertices = get_num_vertices();
  // for (int i = 0; i < 3; i++) {
  //   mcvec_arrows_[i].resize(num_vertices, 3);
  // }
  // double shaft_len = 0.9;
  // double tip_len = 0.1;
  // double scale = vector_field_scale_;
  // for (int v = 0; v < num_vertices; v++) {
  //   Vec3d p1 = xyz_coord_V_.row(v);
  //   Vec3d u12 = scale * mcvec_V_.row(v);
  //   Vec3d p2 = p1 + u12;
  //   Vec3d n = normal_V_.row(v);
  //   Vec3d u12_perp;
  //   math::cross_inplace(n, u12, u12_perp);

  //   Vec3d p3 = p2 - tip_len * u12 + tip_len * u12_perp;

  //   mcvec_arrows_[0].row(v) = p1;
  //   mcvec_arrows_[1].row(v) = p2;
  //   mcvec_arrows_[2].row(v) = p3;
  Eigen::Vector3d rgb = mcvec_arrows_.rgb_;
  double scale = vector_field_scale_;
  mcvec_arrows_.update(xyz_coord_V_, mcvec_V_, scale, rgb);
}

double Membrane::monte_flip_probability(int e) const {
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
  norm_dx_pre = length_E_[e];
  zpre = norm_dx_pre / target_edge_length_;

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

void Membrane::euler_step(double dt) {

  if (use_local_drag_coefficient_) {
    for (int v = 0; v < get_num_vertices(); v++) {
      double drag_coefficient = local_drag_coefficient_ * area_V_[v];
      xyz_coord_V_.row(v) += dt * force_V_.row(v) / drag_coefficient;
    }

    t_ += dt;
    return;
  }

  for (int v = 0; v < get_num_vertices(); v++) {
    xyz_coord_V_.row(v) += dt * force_V_.row(v) / node_drag_coefficient_;
  }

  t_ += dt;
}

void Membrane::time_step(double dt) {

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

  apply_fluctuations_V(dt);
  num_flips_ = 0;
  if (t_flip_ <= t_) {
    // num_flips_ = monte_flip_sweep();
    num_flips_ = flip_sweep();
    // printf("num_flips = %d\n", num_flips_);
    t_flip_ = t_flip_ + dt_flip_;
  }

  t_ += dt;
  dt_ = dt;
}

void Membrane::evolve_until(double t_end) {
  while (t_ < t_end) {
    update_membrane();
    dt_ = std::min(dt0_, dt_max());
    double dt_end = t_end - t_;
    dt_ = std::min(dt_, dt_end);
    // if (dt != dt0_) {
    //   printf("dt=dtmax, t = %.20f, %.20f\n", dt, t_);
    // } else {
    //   printf("dt=dt0, t = %.20f, %.20f\n", dt, t_);
    // }
    euler_step(dt_);
    apply_fluctuations_V(dt_);
    num_flips_ = 0;
    if (t_flip_ <= t_) {
      // num_flips_ = monte_flip_sweep();
      num_flips_ = flip_sweep();
      // printf("num_flips = %d\n", num_flips_);
      t_flip_ = t_flip_ + dt_flip_;
    }
    // update_membrane();
  }
}

void Membrane::evolve_until(double t_end, double dt0) {
  while (t_ < t_end) {
    dt_ = std::min(dt0, dt_max());
    double dt_end = t_end - t_;
    dt_ = std::min(dt_, dt_end);
    euler_step(dt_);
    apply_fluctuations_V(dt_);
    num_flips_ = 0;
    if (t_flip_ <= t_) {
      num_flips_ = flip_sweep();
      t_flip_ = t_flip_ + dt_flip_;
    }
    update_membrane();
  }
}

void Membrane::save_state(const fs::path &filename) const {
  //
  //
  //
}

//////////////////////
// to be deprecated //
//////////////////////
void Membrane::update_curvature_data() {
  int num_vertices = get_num_vertices();
  mean_curvature_V_.resize(num_vertices);
  lap_mean_curvature_V_.resize(num_vertices);
  gaussian_curvature_V_.resize(num_vertices);
  mcvec_V_ = laplacian(xyz_coord_V_);
  for (int v = 0; v < num_vertices; v++) {
    // Vec3d n = normal_V_.row(v);
    Vec3d mcvec = mcvec_V_.row(v);
    int f = f_left_h(h_out_v(v));
    Vec3d n = normal_F_.row(f);
    double mcvec_sign = math::sign(math::dot(mcvec, n));
    mean_curvature_V_[v] = mcvec_sign * math::L2norm(mcvec) / 2;
    gaussian_curvature_V_[v] = get_gaussian_curvature_angle_defect_v(v);
  }
  lap_mean_curvature_V_ = laplacian(mean_curvature_V_);
  // lap_mean_curvature_V_ =
  // adaptive_belkin_laplacian(mean_curvature_V_);
}

bool Membrane::tether_wants_flip(int e) const {
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

  Vec3d x;
  Vec3d xp;
  double norm_dx;
  double zpre;
  double zpost;
  double Upre = 0;
  double Upost = 0;

  x = xyz_coord_v(v0);
  xp = xyz_coord_v(v2);
  norm_dx = math::L2norm(x - xp);
  zpre = norm_dx / target_edge_length_;

  Upre = Utether(zpre, 1.0, 1.0, dimensionless_tether_repulsive_singularity_,
                 dimensionless_tether_repulsive_onset_,
                 dimensionless_tether_attractive_onset_,
                 dimensionless_tether_attractive_singularity_);

  x = xyz_coord_v(v3);
  xp = xyz_coord_v(v1);
  norm_dx = math::L2norm(x - xp);
  zpost = norm_dx / target_edge_length_;

  Upost = Utether(zpost, 1.0, 1.0, dimensionless_tether_repulsive_singularity_,
                  dimensionless_tether_repulsive_onset_,
                  dimensionless_tether_attractive_onset_,
                  dimensionless_tether_attractive_singularity_);

  // check if they are infinite
  if (std::isinf(Upre) && std::isinf(Upost)) {
    return std::abs(zpost - 1) < std::abs(zpre - 1);
  }

  return Upost < Upre;
}

int Membrane::tether_flip_sweep() {
  int flip_count = 0;
  int num_edges = get_num_edges();
  for (int e = 0; e < num_edges; e++) {
    if (!h_is_flippable(h_directed_E_(e))) {
      continue;
    }
    // int v0 = v_origin_h(h_directed_E_(e));
    // int v1 = v_origin_h(h_twin_h(h_directed_E_(e)));
    // int valence0 = get_valence_v(v0);
    // int valence1 = get_valence_v(v1);
    // if (valence0 <= 5 || valence1 <= 5) {
    //   continue;
    // }
    if (tether_wants_flip(e)) {
      flip_edge(e);
      flip_count++;
      continue;
    }
    double _r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    if (_r < 0.01) {
      flip_edge(e);
      flip_count++;
    }
  }
  return flip_count;
}

} // namespace meshbrane
