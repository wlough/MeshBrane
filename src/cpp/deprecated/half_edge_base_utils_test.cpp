// half_edge_base_utils.cpp

#include "half_edge_base_utils.hpp"
#include <iostream>

int main() {
  // Example usage
  MatrixXd xyz_coord_V(5, 3);
  xyz_coord_V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0; // Duplicate point for demonstration

  MatrixXi vvv_of_F(4, 3);
  vvv_of_F << 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3;

  auto result = vf_samples_to_he_samples(xyz_coord_V, vvv_of_F);

  // Print results
  std::cout << "xyz_coord_V:\n" << std::get<0>(result) << "\n";
  std::cout << "h_out_V:\n" << std::get<1>(result).transpose() << "\n";
  std::cout << "v_origin_H:\n" << std::get<2>(result).transpose() << "\n";
  std::cout << "h_next_H:\n" << std::get<3>(result).transpose() << "\n";
  std::cout << "h_twin_H:\n" << std::get<4>(result).transpose() << "\n";
  std::cout << "f_left_H:\n" << std::get<5>(result).transpose() << "\n";
  std::cout << "h_bound_F:\n" << std::get<6>(result).transpose() << "\n";
  std::cout << "h_right_B:\n" << std::get<7>(result).transpose() << "\n";

  return 0;
}