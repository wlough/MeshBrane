/**
 * @file math_utils.cpp
 */

#include "meshbrane/math_utils.hpp"

namespace meshbrane {
namespace math {

// Explicit template instantiation
template bool IS_SMALL<double>(double);
template double POW2<double>(double);
template int POW2<int>(int);
template size_t POW2<size_t>(size_t);
template double POW3<double>(double);
template int POW3<int>(int);
template size_t POW3<size_t>(size_t);
template double POW4<double>(double);
template int POW4<int>(int);
template size_t POW4<size_t>(size_t);

// template double ABS<double>(double);
// template int ABS<int>(int);

template double max<double>(double, double);
template int max<int>(int, int);
template size_t max<size_t>(size_t, size_t);

template double min<double>(double, double);
template int min<int>(int, int);
template size_t min<size_t>(size_t, size_t);

template double clip<double>(double, double, double);
template int factorial<int>(int);
template size_t factorial<size_t>(size_t);
template void second_order_extrap<Eigen::VectorXd>(Eigen::VectorXd &,
                                                   const Eigen::VectorXd &,
                                                   const Eigen::VectorXd &,
                                                   double, double);
template void trapint<Eigen::VectorXd>(Eigen::VectorXd &,
                                       const Eigen::VectorXd &, double &);
template void cross_inplace<Eigen::Vector3d>(const Eigen::Vector3d &,
                                             const Eigen::Vector3d &,
                                             Eigen::Vector3d &);
template void dot_inplace<Eigen::Vector3d>(const Eigen::Vector3d &,
                                           const Eigen::Vector3d &, double &);
template double L2norm<Eigen::Vector3d>(const Eigen::Vector3d &);
template double L2norm<Eigen::VectorXd>(const Eigen::VectorXd &);
template double L2norm<std::vector<double>>(const std::vector<double> &);
template double L2norm<std::vector<int>>(const std::vector<int> &);
} // namespace math
} // namespace meshbrane
