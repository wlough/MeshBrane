#pragma once

/**
 * @file geometric_predicates.hpp
 */

#include "meshbrane/math_utils.hpp"
#include "meshbrane/meshbrane_data_types.hpp"

namespace meshbrane {

/**
 * @brief Check if a point is in a cylinder.
 *
 * @param p (Vec3d): point to check
 * @param p0 (Vec3d): point on cylinder axis
 * @param ez (Vec3d): unit vector along axis of the cylinder
 * @param r_max (double): radius of the cylinder
 * @param z_min (double): minimum z-coordinate of the cylinder
 * @param z_max (double): maximum z-coordinate of the cylinder
 * @return true if the point is in the cylinder, false otherwise
 */
inline bool
point_is_in_cylinder(Vec3d p, Vec3d p0, Vec3d ez, double r_max,
                     double z_min = 0,
                     double z_max = std::numeric_limits<double>::max()) {
  Vec3d p0_p = p - p0;
  double r = math::L2norm(math::cross(ez, p0_p));
  double z = math::dot(ez, p0_p);
  return r <= r_max && z >= z_min && z <= z_max;
};

// /**
//  * @brief Check if a point is in a cylinder.
//  *
//  * @param p (Vec3d): point to check
//  * @param p0 (Vec3d): point on cylinder axis
//  * @param p1 (Vec3d): point on cylinder axis
//  * @param r_max (double): radius of the cylinder
//  * @return true if the point is in the cylinder, false otherwise
//  */
// inline bool point_is_in_cylinder(Vec3d p, Vec3d p0, Vec3d p1, double r_max) {
//   Vec3d p0_p1 = p1 - p0;
//   double z_min = 0;
//   double z_max = math::L2norm(p0_p1);
//   Vec3d ez = p0_p1 / z_max;
//   return point_is_in_cylinder(p, p0, ez, r_max, z_min, z_max);
// };

/**
 * @brief Check if a point is in a closed ball.
 *
 * @param p (Vec3d): point to check
 * @param p0 (Vec3d): center of the ball
 * @param r_max (double): radius of the ball
 * @return true if the point is in the ball, false otherwise
 */
inline bool point_is_in_ball(Vec3d p, Vec3d p0, double r_max) {
  // Vec3d p0_p = p - p0;
  // double r = math::L2norm(p0_p);
  // return r <= r_max;
  return math::L2norm(p - p0) <= r_max;
};
} // namespace meshbrane
