#pragma once

/**
 * @file meshbrane_object.hpp
 */

#include <cstddef> // size_t
#include <limits>

namespace meshbrane {

/**
 * @brief Base class for meshbrane objects.
 */
struct MeshBraneObject {
  size_t index_{std::numeric_limits<std::size_t>::max()};
  virtual ~MeshBraneObject() = default;
  void set_index(size_t index) { index_ = index; }
  size_t get_index() const { return index_; }
};
} // namespace meshbrane
