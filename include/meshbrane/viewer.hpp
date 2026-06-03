#pragma once

/**
 * @file viewer.hpp
 * @brief 3D viewer for simulations
 */

#include "igl/opengl/glfw/Viewer.h" // igl::opengl::glfw::Viewer
#include "meshbrane/matrix_mesh.hpp"
#include "meshbrane/membrane.hpp"
#include "meshbrane/rigid_spindle.hpp"
#include "meshbrane/simple_vector_field.hpp"
#include <Eigen/Dense>
#include <igl/stb/write_image.h>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

/**
 * @brief Simple vector field for visualization.
 *
 */
struct Viewer {
  //   int width_{1280};
  //   int height_{800};

  YAML::Node *sim_parameters_{nullptr};

  int width_{1920};
  int height_{1080};
  Eigen::Vector3f eye_;
  Eigen::Vector3f center_;
  Eigen::Vector3f up_;
  float dfar_{1000.0f};
  float dnear_{0.1f};
  // double view_angle{};
  Eigen::Vector4f background_color_{1.0, 1.0, 1.0, 1.0};

  Viewer() = default;
  ~Viewer() = default;
  igl::opengl::glfw::Viewer iglviewer_;

  ///////////////////////////////
  // Initialization /////////////
  ///////////////////////////////

  void sync_parameters();

  void init();

  ///////////////////////////////
  // Drawing ////////////////////
  ///////////////////////////////
  void draw_simple_vector_field(SimpleVectorField &vfield);

  void draw_mesh(MatrixMesh &m);

  void draw_membrane(Membrane &envelope);

  void draw_meshes(std::vector<MatrixMesh *> &meshes);

  void draw_wireframe(MatrixMesh &m);

  void draw_tube(Vec3d &p1, Vec3d &p2, double radius, Eigen::Vector3d color,
                 int Nphi = 20);

  void draw_rigid_mt_bundle(RigidMTBundle &mt_bundle);
  void draw_mt_bundle_spb_axes(RigidMTBundle &mt_bundle, SphericalSPB &spb1,
                               SphericalSPB &spb2);
  // void draw_patch(Patch &patch);

  ///////////////////////////////
  // Output /////////////////////
  ///////////////////////////////

  void save_frame(std::string &frame_path);

  //   void make_a_movie(std::string image_dir, std::string image_prefix,
  //                     int index_length, std::string image_format,
  //                     std::string movie_dir, std::string movie_name,
  //                     std::string movie_format, int frame_rate,
  //                     std::string video_codec, int video_quality,
  //                     std::string pixel_format);

  void print_info();
};

} // namespace meshbrane
