#pragma once

/**
 * @file math_utils.hpp
 * @brief Numerical functions and constants.
 */

#include <Eigen/Dense>
#include <cmath> // For M_PI and std::acos
#include <cstdint>

namespace meshbrane {

namespace math {
// Constants
constexpr double small = 1e-6;
constexpr double very_small = 1e-9;
constexpr double tol = 1e-12;
constexpr double PI = 3.14159265358979323846;

// template <typename T> inline T ABS(T x) { return (x < 0) ? -x : x; }

template <typename T> inline bool IS_SMALL(T x) { return std::abs(x) < small; }

template <typename T> inline T POW2(T x) { return x * x; }

template <typename T> inline T POW3(T x) { return x * x * x; }

template <typename T> inline T POW4(T x) { return x * x * x * x; }

template <typename T> inline T max(T x, T y) { return (x > y) ? x : y; }

template <typename T> inline T min(T x, T y) { return (x < y) ? x : y; }

inline int sign(double x) { return (x >= 0.0) ? 1 : -1; }

template <typename T> inline T clip(T x, T min, T max) {
  return (x < min) ? min : (x > max) ? max : x;
}

constexpr uint64_t factorial_lookup_table[21] = {1,
                                                 1,
                                                 2,
                                                 6,
                                                 24,
                                                 120,
                                                 720,
                                                 5040,
                                                 40320,
                                                 362880,
                                                 3628800,
                                                 39916800,
                                                 479001600,
                                                 6227020800ULL,
                                                 87178291200ULL,
                                                 1307674368000ULL,
                                                 20922789888000ULL,
                                                 355687428096000ULL,
                                                 6402373705728000ULL,
                                                 121645100408832000ULL,
                                                 2432902008176640000ULL};

template <typename T> inline T factorial(T n) {
  return (n < 21) ? factorial_lookup_table[n] : n * factorial<T>(n - 1);
}

/**
 * @brief Compute the 2nd order accurate extrapolation of Y at t=0 from samples
 * at t=-h1 and t=-h2.
 *
 * @tparam T
 * @param Ytm0 The value to be extrapolated t=0
 * @param Ytm1 The value at t=-h1
 * @param Ytm2 The value at t=-h2
 * @param h1 The first time step
 * @param h2 The second time step
 * @return T
 */
template <typename T>
inline void second_order_extrap(T &Ytm0, const T &Ytm1, const T &Ytm2,
                                double h1, double h2) {
  for (int i = 0; i < Ytm0.size(); i++) {
    Ytm0[i] = (h2 * Ytm1[i] - h1 * Ytm2[i]) / (h2 - h1);
  }
}

/**
 * @brief Numerical integration of Y with respect to x using the trapezoidal
 * rule.
 *
 * @tparam T
 * @param Y
 * @param x
 * @param integral_Y_dx
 */
template <typename T>
inline void trapint(T &Y, const T &x, double &integral_Y_dx) {
  size_t n = Y.size();
  integral_Y_dx = 0.5 * (x[1] - x[0]) * Y[0] + 0.5 * (x[n] - x[n - 1]) * Y[n];
  for (int j = 1; j < n - 1; j++) {
    integral_Y_dx += 0.5 * (x[j + 1] - x[j - 1]) * Y[j];
  }
}

template <typename T>
inline void cross_inplace(const T &U, const T &V, T &UxV) {
  UxV[0] = U[1] * V[2] - U[2] * V[1];
  UxV[1] = U[2] * V[0] - U[0] * V[2];
  UxV[2] = U[0] * V[1] - U[1] * V[0];
}

template <typename T>
inline void dot_inplace(const T &U, const T &V, double &UdotV) {
  UdotV = 0.0;
  for (int i = 0; i < 3; i++) {
    UdotV += U[i] * V[i];
  }
}

template <typename T> inline T cross(const T &U, const T &V) {
  T UxV;
  UxV[0] = U[1] * V[2] - U[2] * V[1];
  UxV[1] = U[2] * V[0] - U[0] * V[2];
  UxV[2] = U[0] * V[1] - U[1] * V[0];
  return UxV;
}

template <typename T> inline double dot(const T &U, const T &V) {
  double UdotV = 0.0;
  for (int i = 0; i < 3; i++) {
    UdotV += U[i] * V[i];
  }
  return UdotV;
}

template <typename T> inline double L2norm(const T &U) {
  double normU = 0.0;
  for (int i = 0; i < 3; i++) {
    normU += U[i] * U[i];
  }
  return std::sqrt(normU);
}

/**
 * @brief Compute triangle area from edge lengths using numerically stable
 * variant of Heron's formula.
 * @param a
 * @param b
 * @param c
 * @return double
 *
 * See https://en.wikipedia.org/wiki/Heron%27s_formula#Numerical_stability
 */
double inline heron_area(const double &L1, const double &L2, const double &L3) {
  double a = L1, b = L2, c = L3;
  if (b > a) {
    std::swap(a, b);
  }
  if (c > a) {
    std::swap(a, c);
  }
  if (c > b) {
    std::swap(b, c);
  }
  return std::sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) *
                   (a + (b - c))) /
         4;
}

/**
 * @brief Compute interior angle opposite edge c from edge lengths
 * @param a
 * @param b
 * @param c
 * @return double

 */
double inline heron_angle(const double &L1, const double &L2,
                          const double &L3) {
  double a = L1, b = L2, c = L3;
  // printf("a = %f, b = %f, c = %f\n", a, b, c);
  double mu;
  if (b > a) {
    std::swap(a, b);
  }
  // printf("a = %f, b = %f, c = %f\n", a, b, c);
  if (b >= c) {
    mu = c - (a - b);
  } else {
    mu = b - (a - c);
  }
  // printf("mu = %f\n", mu);
  double Y = std::sqrt(((a - b) + c) * mu);
  double X = std::sqrt(((a + (b + c)) * ((a - c) + b)));
  // printf("X = %f, Y = %f\n", X, Y);
  return 2 * std::atan(Y / X);
  // double C = 2 * std::atan(std::sqrt(((a - b) + c) * mu /
  //                                    ((a + (b + c)) * ((a - c) + b))));
  // return 2 * std::atan2(std::sqrt(((a - b) + c) * mu),
  //                       std::sqrt(((a + (b + c)) * ((a - c) + b))));
}

template <typename T> void normalize3vec(T &vec) {
  double norm_vec = L2norm(vec);
  for (int i = 0; i < 3; i++) {
    vec[i] /= norm_vec;
  }
}

} // namespace math

namespace lie {

inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d &angle_vec) {
  double th1 = angle_vec(0);
  double th2 = angle_vec(1);
  double th3 = angle_vec(2);
  double th11 = th1 * th1;
  double th22 = th2 * th2;
  double th33 = th3 * th3;
  double th12 = th1 * th2;
  double th13 = th1 * th3;
  double th23 = th2 * th3;
  // double theta = angle_vec.norm();
  double theta = std::sqrt(th11 + th22 + th33);
  double A;
  double B;
  if (math::IS_SMALL(theta)) {
    double theta_sqr = theta * theta;
    double theta_fourth = theta_sqr * theta_sqr;
    A = 1.0 - theta_sqr / 6.0 + theta_fourth / 120.0;
    B = 1.0 / 2.0 - theta_sqr / 24.0 + theta_fourth / 720.0;
  } else {
    A = std::sin(theta) / theta;
    B = (1.0 - std::cos(theta)) / (theta * theta);
  }
  // Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  // R(0,1) = -A * th3;
  // R(0,2) = A * th2;
  // R(1,0) = A * th3;
  // R(1,2) = -A * th1;
  // R(2,0) = -A * th2;
  // R(2,1) = A * th1;
  // R(0,0) += B * (-(th22 + th33));
  // R(0,1) += B * (th12);
  // R(0,2) += B * (th13);
  // R(1,0) += B * (th12);
  // R(1,1) += B * (-(th11 + th33));
  // R(1,2) += B * (th23);
  // R(2,0) += B * (th13);
  // R(2,1) += B * (th23);
  // R(2,2) += B * (-(th11 + th22));
  // Eigen::Matrix3d R;
  // R(0, 0) = 1.0 + B * (-(th22 + th33));
  // R(0, 1) = -A * th3 + B * (th12);
  // R(0, 2) = A * th2 + B * (th13);
  // R(1, 0) = A * th3 + B * (th12);
  // R(1, 1) = 1.0 + B * (-(th11 + th33));
  // R(1, 2) = -A * th1 + B * (th23);
  // R(2, 0) = -A * th2 + B * (th13);
  // R(2, 1) = A * th1 + B * (th23);
  // R(2, 2) = 1.0 + B * (-(th11 + th22));
  Eigen::Matrix3d R{
      {1.0 + B * (-(th22 + th33)), -A * th3 + B * (th12), A * th2 + B * (th13)},
      {A * th3 + B * (th12), 1.0 + B * (-(th11 + th33)), -A * th1 + B * (th23)},
      {-A * th2 + B * (th13), A * th1 + B * (th23),
       1.0 + B * (-(th11 + th22))}};
  return R;
}

/**
 * @brief Returns the components of log(R) in the canonical basis for Lie
 * algebra so(3)
 *
 */
inline Eigen::Vector3d log_so3(const Eigen::Matrix3d &R) {
  double cos_theta = 0.5 * (R(0, 0) + R(1, 1) + R(2, 2) - 1.0);
  double theta = std::acos(math::clip(cos_theta, -1.0, 1.0));
  double theta_over_two_sin_theta;
  if (math::IS_SMALL(theta)) {
    double theta_sqr = theta * theta;
    double theta_fourth = theta_sqr * theta_sqr;
    theta_over_two_sin_theta =
        0.5 + theta_sqr / 12.0 + 7.0 * theta_fourth / 720.0;
  } else {
    theta_over_two_sin_theta = 0.5 * theta / std::sin(theta);
  }
  double thetax = (R(2, 1) - R(1, 2)) * theta_over_two_sin_theta;
  double thetay = (R(0, 2) - R(2, 0)) * theta_over_two_sin_theta;
  double thetaz = (R(1, 0) - R(0, 1)) * theta_over_two_sin_theta;
  return Eigen::Vector3d(thetax, thetay, thetaz);
}

/**
 * @brief Convert an orthogonal matrix to a unit quaternion.
 *
 * @param Q The orthogonal matrix
 * @param q The quaternion
 */
inline Eigen::Vector4d orthogonal_to_quaternion(const Eigen::Matrix3d &Q) {
  double diagQ[3] = {Q(0, 0), Q(1, 1), Q(2, 2)};
  double trQ = Q(0, 0) + Q(1, 1) + Q(2, 2);
  double diagQmax = *std::max_element(diagQ, diagQ + 3);

  if (diagQmax > trQ) {
    int i = std::distance(diagQ, std::max_element(diagQ, diagQ + 3));
    int j = (i + 1) % 3;
    int k = (j + 1) % 3;

    double qi = std::sqrt(1 + 2 * diagQmax - trQ) / 2;
    double Qkj_Qjk = Q(k, j) - Q(j, k);
    if (Qkj_Qjk < 0) {
      qi *= -1;
    }
    double qj = (Q(i, j) + Q(j, i)) / (4 * qi);
    double qk = (Q(i, k) + Q(k, i)) / (4 * qi);
    double qs = Qkj_Qjk / (4 * qi);

    Eigen::Vector4d q;
    q[0] = qs;
    q[i + 1] = qi;
    q[j + 1] = qj;
    q[k + 1] = qk;
    return q;
  } else {
    int i = 0, j = 1, k = 2;
    double qs = std::sqrt(1 + trQ) / 2;
    double qi = (Q(2, 1) - Q(1, 2)) / (4 * qs);
    double qj = (Q(0, 2) - Q(2, 0)) / (4 * qs);
    double qk = (Q(1, 0) - Q(0, 1)) / (4 * qs);

    Eigen::Vector4d q;
    q[0] = qs;
    q[i + 1] = qi;
    q[j + 1] = qj;
    q[k + 1] = qk;
    return q;
  }
}

/**
 * @brief applies rotation representated by unit quaternion q
    to vector r
 *
 * @param q Unit quaternion
 * @param r The vector to be rotated
 * @param rot_r The rotated vector
 */
inline Eigen::Vector3d rotate_by_quaternion(const Eigen::Vector4d &q,
                                            const Eigen::Vector3d &r) {
  double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  double x = r[0], y = r[1], z = r[2];
  double qrw = -qx * x - qy * y - qz * z;
  double qrx = qw * x + qy * z - qz * y;
  double qry = qw * y - qx * z + qz * x;
  double qrz = qw * z + qx * y - qy * x;
  Eigen::Vector3d rot_r;
  rot_r[0] = -qrw * qx + qrx * qw - qry * qz + qrz * qy;
  rot_r[1] = -qrw * qy + qrx * qz + qry * qw - qrz * qx;
  rot_r[2] = -qrw * qz - qrx * qy + qry * qx + qrz * qw;
  return rot_r;
}

/**
 * @brief Exponential map from so(3) to SO(3)~(unit quaternions).
 *
 * @param angle_vec
 * @param q
 */
inline Eigen::Vector4d exp_so3_quaternion(const Eigen::Vector3d &angle_vec) {
  double ax = angle_vec[0], ay = angle_vec[1], az = angle_vec[2];
  double a_sqr = ax * ax + ay * ay + az * az;
  double a = std::sqrt(a_sqr);
  Eigen::Vector4d q;
  if (meshbrane::math::IS_SMALL(a)) {
    // double a_fourth = a * a;
    double a_fourth = a_sqr * a_sqr;
    q[0] = 1 - a_sqr / 8 + a_fourth / 384;
    double D = 1.0 / 2.0 - a_sqr / 48.0 + a_fourth / 3840.0;
    q[1] = D * ax;
    q[2] = D * ay;
    q[3] = D * az;
    return q;
  } else {
    q[0] = std::cos(a / 2);
    double D = std::sin(a / 2) / a;
    q[1] = D * ax;
    q[2] = D * ay;
    q[3] = D * az;
    return q;
  }
}

/**
 * @brief Rigid transformation of a vector
 *
 * @param translation
 * @param angle_vec
 * @param X
 *
 */
inline Eigen::Vector3d rigid_transform(const Eigen::Vector3d &translation,
                                       const Eigen::Vector3d &angle_vec,
                                       const Eigen::Vector3d &X) {
  return rotate_by_quaternion(exp_so3_quaternion(angle_vec), X) + translation;
}
} // namespace lie
} // namespace meshbrane
