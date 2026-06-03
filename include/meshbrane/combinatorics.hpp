#pragma once

/**
 * @file combinatorics.hpp
 */

#include <Eigen/Dense>
#include <vector>

namespace meshbrane {

namespace math {

/**
 * @brief Returns permutation of the indices of X that sorts X.
 *
 * @param X Vector of elements to be sorted.
 * @param reverse If true, sort in descending order.
 * @return std::vector<int> List of indices that sorts X.
 */
template <typename T>
std::vector<int> argsort(const std::vector<T> &X, bool reverse = false);

} // namespace math

} // namespace meshbrane
