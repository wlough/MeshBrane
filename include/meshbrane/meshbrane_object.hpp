#pragma once

/**
 * @file meshbrane_object.hpp
 */

#include <cstddef>

namespace meshbrane {
///////////////////////////////////////////////////////////////////////
// Base classes inherit directly from MeshBraneObject /////////////////
///////////////////////////////////////////////////////////////////////

/**
 * @brief Base class for meshbrane objects.
 */
struct MeshBraneObject {
  size_t index_{0};
  size_t flags_{0};
  enum Flags {
    DIRTY = 1 << 0,         // 0001
    INDEX_WAS_SET = 1 << 1, // 0010
    GHOST = 1 << 2          // 0100
  };
  MeshBraneObject() = default;
  MeshBraneObject(size_t index) : index_(index), flags_(INDEX_WAS_SET) {}
  virtual ~MeshBraneObject() = default;
  MeshBraneObject(const MeshBraneObject &mbo)
      : index_(mbo.index_), flags_(mbo.flags_) {}
  void set_index(size_t index) {
    index_ = index;
    // flags_ |= INDEX_WAS_SET;
    // if (index_was_set()) {
    //   throw std::invalid_argument("Index was already set.");
    // }
    set_flag(INDEX_WAS_SET);
  }
  // Methods to manage flags
  void set_flag(Flags flag) { flags_ |= flag; }
  void clear_flag(Flags flag) { flags_ &= ~flag; }
  bool is_flag_set(Flags flag) const { return flags_ & flag; }

  void unset_index() { flags_ &= ~INDEX_WAS_SET; }
  bool index_was_set() const { return flags_ & INDEX_WAS_SET; }

  void set_ghost() { flags_ |= GHOST; }
  bool is_ghost() const { return flags_ & GHOST; }

  size_t index() const { return index_; }
};
} // namespace meshbrane
