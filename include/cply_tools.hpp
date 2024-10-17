/**
 * @file cply_tools.hpp
 */
#ifndef CPLY_TOOLS_HPP
#define CPLY_TOOLS_HPP
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include <data_types.hpp>

class MeshConverter {
public:
  // Constructors
  MeshConverter();
  static MeshConverter from_vf_ply(const std::string &ply_path,
                                   bool compute_he_stuff = true);
  static MeshConverter from_vf_samples(const Samples3d &xyz_coord_V,
                                       const Samples3i &V_of_F,
                                       bool compute_he_stuff = true);
  static MeshConverter from_he_ply(const std::string &ply_path,
                                   bool compute_vf_stuff = true);
  static MeshConverter
  from_he_samples(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                  const Samplesi &v_origin_H, const Samplesi &h_next_H,
                  const Samplesi &h_twin_H, const Samplesi &f_left_H,
                  const Samplesi &h_bound_F, const Samplesi &h_right_B,
                  bool compute_vf_stuff = true);

  // Methods
  HalfEdgeSamples vf_samples_to_he_samples();
  VertexFaceSamples vf_ply_data_to_samples();
  tinyply::PlyData vf_samples_to_ply_data(bool use_binary = true);
  std::tuple<Samples3d, Samples3i> he_samples_to_vf_samples();
  std::tuple<Samples3d, Samplesi, Samplesi, Samplesi, Samplesi, Samplesi,
             Samplesi, Samplesi>
  he_ply_data_to_samples();
  tinyply::PlyData he_samples_to_ply_data(bool use_binary = true);
  void write_vf_ply(const std::string &ply_path, bool use_binary = true);
  void write_he_ply(const std::string &ply_path, bool use_binary = true);
  void write_he_samples(const std::string &path, bool compressed = false,
                        bool chunk = false, bool remove_unchunked = false);

private:
  std::string vf_ply_path;
  tinyply::PlyData vf_ply_data;
  std::tuple<Samples3d, Samples3i> vf_samples;

  std::string he_ply_path;
  tinyply::PlyData he_ply_data;
  std::tuple<Samples3d, Samplesi, Samplesi, Samplesi, Samplesi, Samplesi,
             Samplesi, Samplesi>
      he_samples;

  // Helper methods for dealing with old data that doesn't include h_right_B
  std::tuple<Samples3d, Samplesi, Samplesi, Samplesi, Samplesi, Samplesi,
             Samplesi, Samplesi>
  no_boundary_he_ply_data_to_samples();
  static MeshConverter from_no_boundary_he_ply(const std::string &ply_path,
                                               bool compute_vf_stuff = true);
  static void update_no_boundary_he_plys();
};

#endif /* CPLY_TOOLS_HPP */