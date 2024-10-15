// half_edge_base_utils.hpp

#ifndef HALF_EDGE_BASE_UTILS_HPP
#define HALF_EDGE_BASE_UTILS_HPP

#include <Eigen/Dense> // for Eigen::Vector3d
#include <vector>      // for std::vector

using samples_i = std::vector<int32_t>;
using samples_i2 = std::vector<std::array<int32_t, 2>>;
using samples_i3 = std::vector<std::array<int32_t, 3>>;
using samples_d3 = std::vector<Eigen::Vector3d>;
//////////////////////////////////////////////////////////

#include <Eigen/Dense>
#include <algorithm>
#include <set>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using std::set;
using std::vector;
// Function declarations
int find_halfedge_index_of_twin(const MatrixXi &H, int h);

MatrixXi find_V_of_F(const MatrixXd &xyz_coord_V, const VectorXi &h_out_V,
                     const VectorXi &v_origin_H, const VectorXi &h_next_H,
                     const VectorXi &h_twin_H, const VectorXi &f_left_H,
                     const VectorXi &h_bound_F, const VectorXi &h_right_B);

VectorXi find_h_right_B(const MatrixXd &xyz_coord_V, const VectorXi &h_out_V,
                        VectorXi &v_origin_H, const VectorXi &h_next_H,
                        const VectorXi &h_twin_H, VectorXi &f_left_H,
                        const VectorXi &h_bound_F);

std::tuple<MatrixXd, VectorXi, VectorXi, VectorXi, VectorXi, VectorXi, VectorXi,
           VectorXi>
vf_samples_to_he_samples(const MatrixXd &xyz_coord_V, const MatrixXi &vvv_of_F);

#endif // PY_TO_CPP_HPP