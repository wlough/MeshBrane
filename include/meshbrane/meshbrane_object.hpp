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
  int index_{std::numeric_limits<int>::max()};

  virtual ~MeshBraneObject() = default;

  // virtual void set_attributes_from_yaml_node(const YAML::Node &node) = 0;
  // virtual void initialize() = 0;

  void set_index(int index) { index_ = index; }
  int get_index() const { return index_; }
};

// template <typename T>
// std::unique_ptr<T>
// make_meshbrane_object_from_yaml_node(const YAML::Node &node) {
//   auto obj = std::make_unique<T>();

//   obj->set_attributes_from_yaml_node(node);
//   obj->init();

//   return obj;
// }
// auto d = make_meshbrane_object_from_yaml_node<Derived>(node);
} // namespace meshbrane
