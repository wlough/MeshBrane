/**
 * @file viewer.cpp
 */
#include "meshbrane/viewer.hpp"

namespace meshbrane {

void Viewer::print_info() {
  eye_ = iglviewer_.core().camera_eye;
  center_ = iglviewer_.core().camera_center;
  up_ = iglviewer_.core().camera_up;
  dnear_ = iglviewer_.core().camera_dnear; // Near clipping plane
  dfar_ = iglviewer_.core().camera_dfar;   // Far clipping plane
  printf("Viewer info\n");

  printf("  eye: %f, %f, %f\n", eye_[0], eye_[1], eye_[2]);
  printf("  center: %f, %f, %f\n", center_[0], center_[1], center_[2]);
  printf("  up: %f, %f, %f\n", up_[0], up_[1], up_[2]);
  printf("  dfar: %f\n", dfar_);
  printf("  dnear: %f\n", dnear_);
}

///////////////////////////////
// Initialization /////////////
///////////////////////////////

void Viewer::sync_parameters() {
  if (sim_parameters_ == nullptr) {
    printf("sim_parameters_ is nullptr\n");
    return;
  }
  if ((*sim_parameters_)["viewer"]["width"]) {
    width_ = (*sim_parameters_)["viewer"]["width"].as<int>();
  }
  if ((*sim_parameters_)["viewer"]["height"]) {
    height_ = (*sim_parameters_)["viewer"]["height"].as<int>();
  }
  if ((*sim_parameters_)["viewer"]["eye"]) {
    eye_ = Eigen::Map<Eigen::Vector3f>(
        (*sim_parameters_)["viewer"]["eye"].as<std::vector<float>>().data());
  }
  if ((*sim_parameters_)["viewer"]["center"]) {
    center_ = Eigen::Map<Eigen::Vector3f>(
        (*sim_parameters_)["viewer"]["center"].as<std::vector<float>>().data());
  }
  if ((*sim_parameters_)["viewer"]["up"]) {
    up_ = Eigen::Map<Eigen::Vector3f>(
        (*sim_parameters_)["viewer"]["up"].as<std::vector<float>>().data());
  }
  if ((*sim_parameters_)["viewer"]["dfar"]) {
    dfar_ = (*sim_parameters_)["viewer"]["dfar"].as<float>();
  }
  if ((*sim_parameters_)["viewer"]["dnear"]) {
    dnear_ = (*sim_parameters_)["viewer"]["dnear"].as<float>();
  }
  if ((*sim_parameters_)["viewer"]["background_color"]) {
    background_color_ = Eigen::Map<Eigen::Vector4f>(
        (*sim_parameters_)["viewer"]["background_color"]
            .as<std::vector<float>>()
            .data());
  }
}

void Viewer::init() {
  // iglviewer_.core().background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
  // Eigen::Vector4f background_color = iglviewer_.core().background_color;
  // printf("*********************************************************************"
  //        "*********************************************************************"
  //        "***background color: %f, %f, %f, %f\n",
  //        background_color[0], background_color[1], background_color[2],
  //        background_color[3]);
  iglviewer_.core().background_color = background_color_;
  iglviewer_.core().lighting_factor = 0.0;
  iglviewer_.resize(width_, height_);
  iglviewer_.core().camera_eye = eye_;
  iglviewer_.core().camera_center = center_;
  iglviewer_.core().camera_up = up_;
  iglviewer_.core().camera_dnear = dnear_; // Near clipping plane
  iglviewer_.core().camera_dfar = dfar_;   // Far clipping plane
  iglviewer_.core().set_rotation_type(
      igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
}

///////////////////////////////
// Drawing ////////////////////
///////////////////////////////

void Viewer::draw_simple_vector_field(SimpleVectorField &vfield) {
  iglviewer_.data().add_edges(vfield.arrows_[0], vfield.arrows_[1],
                              vfield.rgb_);
  iglviewer_.data().add_edges(vfield.arrows_[1], vfield.arrows_[2],
                              vfield.rgb_);
};

// void Viewer::draw_mesh(MatrixMesh &m) {
//   auto data = iglviewer_.data();
//   iglviewer_.data().set_mesh(m.xyz_coord_V_, m.V_cycle_F_);
//   iglviewer_.data().set_face_based(true);
//   iglviewer_.data().set_colors(m.rgba_F);
//   if (m.show_vertices_) {
//     iglviewer_.data().point_size = m.radius_vertex_;
//     Eigen::Matrix<double, Eigen::Dynamic, 3> rgb_V =
//         m.rgba_V.block(0, 0, m.rgba_V.rows(), 3);
//     iglviewer_.data().add_points(m.get_xyz_coord_V(), rgb_V);
//   }

//   if (m.show_half_edges_) {
//     Eigen::Matrix<double, Eigen::Dynamic, 3> rgb_H =
//         m.rgba_H.block(0, 0, m.rgba_H.rows(), 3);
//     iglviewer_.data().add_edges(m.shifted_half_edge_arrows_[0],
//                                 m.shifted_half_edge_arrows_[1], rgb_H);
//     iglviewer_.data().add_edges(m.shifted_half_edge_arrows_[1],
//                                 m.shifted_half_edge_arrows_[2], rgb_H);
//   }

//   if (m.show_edges_) {
//     iglviewer_.data().show_lines = true;
//     iglviewer_.data().line_width = 2.0;
//   } else {
//     iglviewer_.data().show_lines = false;
//   }
// };

void Viewer::draw_mesh(MatrixMesh &m) {

  auto *data = &iglviewer_.data();
  data->set_mesh(m.xyz_coord_V_, m.V_cycle_F_);
  data->set_face_based(true);
  data->set_colors(m.rgba_F);
  if (m.show_vertices_) {
    data->point_size = m.radius_vertex_;
    Eigen::Matrix<double, Eigen::Dynamic, 3> rgb_V =
        m.rgba_V.block(0, 0, m.rgba_V.rows(), 3);
    data->add_points(m.get_xyz_coord_V(), rgb_V);
  }

  if (m.show_half_edges_) {
    Eigen::Matrix<double, Eigen::Dynamic, 3> rgb_H =
        m.rgba_H.block(0, 0, m.rgba_H.rows(), 3);
    data->add_edges(m.shifted_half_edge_arrows_[0],
                    m.shifted_half_edge_arrows_[1], rgb_H);
    data->add_edges(m.shifted_half_edge_arrows_[1],
                    m.shifted_half_edge_arrows_[2], rgb_H);
  }

  if (m.show_edges_) {
    data->show_lines = true;
    data->line_width = 2.0;
  } else {
    data->show_lines = false;
  }
};

void Viewer::draw_wireframe(MatrixMesh &m) {

  auto *data = &iglviewer_.data();
  if (m.show_vertices_) {
    data->point_size = m.radius_vertex_;
    Eigen::Matrix<double, Eigen::Dynamic, 3> rgb_V =
        m.rgba_V.block(0, 0, m.rgba_V.rows(), 3);
    data->add_points(m.get_xyz_coord_V(), rgb_V);
  }

  int num_edges = m.get_num_edges();
  Samples3d P0{num_edges, 3};
  Samples3d P1{num_edges, 3};
  for (int e = 0; e < num_edges; e++) {
    int v0 = m.V_cycle_E_(e, 0);
    int v1 = m.V_cycle_E_(e, 1);
    P0.row(e) = m.xyz_coord_v(v0);
    P1.row(e) = m.xyz_coord_v(v1);
  }
  Eigen::Matrix<double, Eigen::Dynamic, 3> rgb_E =
      m.rgba_E_.block(0, 0, m.rgba_E_.rows(), 3);
  data->add_edges(P0, P1, rgb_E);

  // if (m.show_edges_) {
  //   data->show_lines = true;
  data->line_width = 2.0;
  // } else {
  //   data->show_lines = false;
  // }
};

void Viewer::draw_tube(Vec3d &p1, Vec3d &p2, double radius,
                       Eigen::Vector3d color, int Nphi) {
  // printf("Drawing tube\n");
  // printf("  p1: %f, %f, %f\n", p1[0], p1[1], p1[2]);
  // printf("  p2: %f, %f, %f\n", p2[0], p2[1], p2[2]);
  // printf("  radius: %f\n", radius);
  // printf("  color: %f, %f, %f\n", color[0], color[1], color[2]);
  // int Nphi = 10;
  Vec3d ez = (p2 - p1).normalized();
  Vec3d ey = ez.cross(Eigen::Vector3d(0, 0, 1));
  if (ey.norm() < 1e-6) {
    ey = ez.cross(Eigen::Vector3d(0, 1, 0));
  }
  ey.normalize();
  Vec3d ex = ey.cross(ez);

  double dphi = 2 * M_PI / Nphi;
  Samples3d Pa{Nphi, 3};
  Samples3d Pb{Nphi, 3};
  Samples3d Color{Nphi, 3};
  for (int i = 0; i < Nphi; i++) {
    double phi = i * dphi;
    Vec3d r = radius * (cos(phi) * ex + sin(phi) * ey);
    Pa.row(i) = p1 + r;
    Pb.row(i) = p2 + r;
    Color.row(i) = color;
  }
  iglviewer_.data().add_edges(Pa, Pb, Color);
};

void Viewer::draw_meshes(std::vector<MatrixMesh *> &meshes) {
  std::vector<int> Nv;
  int Nv_total = 0;
  int Nf_total = 0;
  for (auto &m : meshes) {
    Nv_total += m->get_num_vertices();
    Nf_total += m->get_num_faces();
    Nv.push_back(m->get_num_vertices());
  }
  std::vector<int> cum_Nv;
  cum_Nv.push_back(0);
  for (int i = 0; i < meshes.size() - 1; i++) {
    cum_Nv.push_back(cum_Nv[i] + Nv[i]);
  }

  // std::vector<Eigen::MatrixXd> vector_V;
  // std::vector<Eigen::MatrixXi> vector_F;
  std::vector<Eigen::RowVector3d> vector_V;
  std::vector<Eigen::RowVector3i> vector_F;
  std::vector<Eigen::RowVector4d> vector_F_rgba;
  vector_V.reserve(Nv_total);
  vector_F.reserve(Nf_total);
  vector_F_rgba.reserve(Nf_total);

  for (int i = 0; i < meshes.size(); i++) {
    auto &m = meshes[i];
    int Nv_shift = cum_Nv[i];
    Samples3i F_shifted = m->V_cycle_F_;

    for (int j = 0; j < F_shifted.rows(); j++) {
      F_shifted(j, 0) += Nv_shift;
      F_shifted(j, 1) += Nv_shift;
      F_shifted(j, 2) += Nv_shift;
    }

    // vector_V.push_back(m.get_xyz_coord_V());
    // vector_F.push_back(m.V_cycle_F_);
    for (int j = 0; j < m->get_num_vertices(); j++) {
      Eigen::RowVector3d xyz = m->xyz_coord_V_.row(j);
      vector_V.push_back(xyz);
    }
    for (int j = 0; j < m->get_num_faces(); j++) {
      Eigen::RowVector3i F_shifted_row = F_shifted.row(j);
      Eigen::RowVector4d rgba_F_row = m->rgba_F.row(j);
      vector_F.push_back(F_shifted_row);
      vector_F_rgba.push_back(rgba_F_row);
    }
  }
  Eigen::MatrixXd V_total(Nv_total, 3);
  Eigen::MatrixXi F_total(Nf_total, 3);
  Eigen::MatrixXd rgba_F_total(Nf_total, 4);

  for (int i = 0; i < Nv_total; i++) {
    V_total.row(i) = vector_V[i];
  }
  for (int i = 0; i < Nf_total; i++) {
    F_total.row(i) = vector_F[i];
    rgba_F_total.row(i) = vector_F_rgba[i];
  }

  iglviewer_.data().set_mesh(V_total, F_total);
  iglviewer_.data().set_face_based(true);
  iglviewer_.data().set_colors(rgba_F_total);
  iglviewer_.data().show_lines = true;
  iglviewer_.data().line_width = 2.0;
}

// void Viewer::draw_patch(Patch &p) {
//   // auto [xyz_coord_V, V_cycle_F] = p.get_vf_tuple();
//   // iglviewer_.data().set_mesh(xyz_coord_V, V_cycle_F);
//   // iglviewer_.data().set_face_based(true);
//   // int num_faces = p.F_.size();
//   // Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_F(num_faces, 4);
//   // for (int f : p.F_) {
//   //   rgba_F.row(f) = p.rgba_face_;
//   // }
//   // iglviewer_.data().set_colors(rgba_F);
//   // p.uncolor_faces();
//   p.color_faces();
// };

void Viewer::draw_membrane(Membrane &envelope) {
  draw_mesh(envelope);
  // forces
  if (envelope.show_force_field_) {
    //   envelope.force_arrows_.draw(iglviewer_);
    draw_simple_vector_field(envelope.force_arrows_);
  }
  // mcvec
  if (envelope.show_mcvec_field_) {
    //   envelope.mcvec_arrows_.draw(iglviewer_);
    draw_simple_vector_field(envelope.mcvec_arrows_);
  }
}

void Viewer::draw_rigid_mt_bundle(RigidMTBundle &mt_bundle) {
  Vec3d rgb1 = mt_bundle.rgb_mt1_;
  Vec3d rgb2 = mt_bundle.rgb_mt2_;
  int num_mts = mt_bundle.num_mts_;
  double dphi = 2 * M_PI / num_mts;
  Samples3d Pa{2 * num_mts, 3};
  Samples3d Pb{2 * num_mts, 3};
  Samples3d Color{2 * num_mts, 3};
  double radius = mt_bundle.radius_;
  Vec3d ex = mt_bundle.rotation_matrix_center_.col(0);
  Vec3d ey = mt_bundle.rotation_matrix_center_.col(1);
  Vec3d ez = mt_bundle.rotation_matrix_center_.col(2);
  Vec3d c = mt_bundle.xyz_center_;

  Vec3d pa = c + 0.5 * mt_bundle.length_ * ez;
  Vec3d pb = c - 0.5 * mt_bundle.overlap_length_ * ez;
  for (int i = 0; i < num_mts; i++) {
    double phi = i * dphi;
    Vec3d r = radius * (cos(phi) * ex + sin(phi) * ey);
    Pa.row(i) = pa + r;
    Pb.row(i) = pb + r;
    Color.row(i) = rgb1;
  }

  pa = c - 0.5 * mt_bundle.length_ * ez;
  pb = c + 0.5 * mt_bundle.overlap_length_ * ez;
  for (int i = num_mts; i < 2 * num_mts; i++) {
    double phi = i * dphi + 0.5 * dphi;
    Vec3d r = radius * (cos(phi) * ex + sin(phi) * ey);
    Pa.row(i) = pa + r;
    Pb.row(i) = pb + r;
    Color.row(i) = rgb2;
  }

  iglviewer_.data().add_edges(Pa, Pb, Color);
}

void Viewer::draw_mt_bundle_spb_axes(RigidMTBundle &mt_bundle,
                                     SphericalSPB &spb1, SphericalSPB &spb2) {
  Vec3d ez_mt = mt_bundle.rotation_matrix_center_.col(2);
  Vec3d ez_spb1 = spb1.rotation_matrix_frame_.col(2);
  Vec3d ez_spb2 = spb2.rotation_matrix_frame_.col(2);
  Vec3d rgb_spb = {1.0, 0.0, 0.0};
  Vec3d rgb_mt1 = mt_bundle.rgb_mt1_;
  Vec3d rgb_mt2 = mt_bundle.rgb_mt2_;

  Vec3d o1 = mt_bundle.get_xyz1();
  Vec3d o2 = mt_bundle.get_xyz2();

  double length = 2.0 * spb1.contact_radius_;
  Vec3d u_mt1 = length * ez_mt;
  Vec3d p_mt1 = o1 + u_mt1;
  Vec3d u_mt2 = -length * ez_mt;
  Vec3d p_mt2 = o2 + u_mt2;
  Vec3d u_spb1 = length * ez_spb1;
  Vec3d p_spb1 = o1 + u_spb1;
  Vec3d u_spb2 = -length * ez_spb2;
  Vec3d p_spb2 = o2 + u_spb2;

  Samples3d Pa{4, 3};
  Samples3d Pb{4, 3};
  Samples3d ColorE{4, 3};

  Pa.row(0) = o1;
  Pb.row(0) = p_mt1;
  ColorE.row(0) = rgb_mt1;
  Pa.row(1) = o1;
  Pb.row(1) = p_spb1;
  ColorE.row(1) = rgb_spb;

  Pa.row(2) = o2;
  Pb.row(2) = p_mt2;
  ColorE.row(2) = rgb_mt2;
  Pa.row(3) = o2;
  Pb.row(3) = p_spb2;
  ColorE.row(3) = rgb_spb;

  iglviewer_.data().add_edges(Pa, Pb, ColorE);

  Samples3d V{6, 3};
  Samples3d ColorV{6, 3};

  V.row(0) = o1;
  ColorV.row(0) = rgb_mt1;
  V.row(1) = p_mt1;
  ColorV.row(1) = rgb_mt1;
  V.row(2) = p_spb1;
  ColorV.row(2) = rgb_spb;
  V.row(3) = o2;
  ColorV.row(3) = rgb_mt2;
  V.row(4) = p_mt2;
  ColorV.row(4) = rgb_mt2;
  V.row(5) = p_spb2;
  ColorV.row(5) = rgb_spb;
  iglviewer_.data().point_size = 15.0;
  iglviewer_.data().add_points(V, ColorV);
}
///////////////////////////////
// Output /////////////////////
///////////////////////////////

void Viewer::save_frame(std::string &frame_path) {
  // Allocate temporary buffers
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width_,
                                                                 height_);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width_,
                                                                 height_);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width_,
                                                                 height_);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width_,
                                                                 height_);

  // Draw the scene in the buffers
  iglviewer_.core().draw_buffer(iglviewer_.data(), false, R, G, B, A);
  // // Save it to a PNG
  igl::stb::write_image(frame_path, R, G, B, A);
}

// void Viewer::make_a_movie(std::string image_dir, std::string image_prefix,
//                           int index_length, std::string image_format,
//                           std::string movie_dir, std::string movie_name,
//                           std::string movie_format, int frame_rate,
//                           std::string video_codec, int video_quality,
//                           std::string pixel_format) {

//   //   std::string
//   //   image_filename="${image_prefix}_%0${index_length}d.${image_format}";
//   std::string image_filename =
//       image_prefix + "_%0" + std::to_string(index_length) + "d." +
//       image_format;
//   std::string movie_filename = movie_name + "." + movie_format;

//   std::string run_command = "ffmpeg ";
//   // # overwrite output file without asking if it already exists
//   run_command += "-y ";
//   // # frame rate (Hz)
//   run_command += "-r " + std::to_string(frame_rate) + " ";
//   // # frame width x height (pixels)
//   run_command += "-s " + std::to_string(width_) + "x" +
//                  std::to_string(height_) + " ";
//   // # input files path and format
//   run_command += "-i " + image_filename + " ";
//   // # video codec
//   run_command += "-vcodec " + video_codec + " ";
//   // # video quality, lower means better
//   run_command += "-crf " + std::to_string(video_quality) + " ";
//   // # pixel format
//   run_command += "-pix_fmt " + pixel_format + " ";
//   // # output file
//   run_command += movie_filename;
//   printf("Running command: %s\n", run_command.c_str());
//   int result = std::system(run_command.c_str());
// }
} // namespace meshbrane
