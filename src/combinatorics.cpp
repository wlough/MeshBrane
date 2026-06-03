/**
 * @file combinatorics.cpp
 */

#include "meshbrane/combinatorics.hpp"

namespace meshbrane {
namespace math {
template <typename T>
std::vector<int> argsort(const std::vector<T> &X, bool reverse) {
  std::vector<int> indices(X.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }
  if (reverse) {
    std::sort(indices.begin(), indices.end(),
              [&X](int a, int b) { return X[a] > X[b]; });
  } else {
    std::sort(indices.begin(), indices.end(),
              [&X](int a, int b) { return X[a] < X[b]; });
  }

  return indices;
}
// Explicit template instantiation
template std::vector<int> argsort<int>(const std::vector<int> &, bool);
template std::vector<int> argsort<double>(const std::vector<double> &, bool);

} // namespace math
} // namespace meshbrane
