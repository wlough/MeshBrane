#pragma once

/**
 * @file meshbrane_object.hpp
 */

// #include "meshbrane/pair_interaction.hpp"
#include <cstddef> // size_t
#include <limits>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

class RandomNumberGenerator;

/**
 * @brief Base class for simulated objects.
 *
 * @details Each simulated object...
 */
// struct MeshBraneObject {
//   int index_{std::numeric_limits<int>::max()};
//   MeshBraneObject *supercomponent_{nullptr};
//   std::vector<std::unique_ptr<MeshBraneObject>> subcomponents_{};
//   std::vector<std::unique_ptr<PairInteraction>> subcomponent_interactions_{};

//   bool is_composite() { return subcomponents_.size() != 0; }

//   virtual ~MeshBraneObject() = default;
//   virtual void set_attributes_from_yaml_node(const YAML::Node &node) = 0;

//   virtual void init(const YAML::Node &node) = 0;

//   void set_index(int index) { index_ = index; }
//   int get_index() const { return index_; }

//   // some objects do not have subcomponents, but still have
//   // internal interactions. For example, Membrane bending forces.
//   // If numerical state vars are all owned by subcomponents, this does
//   nothing virtual void apply_self_interactions() {};
//   // Accumulate forces/torques from internal interactions
//   void apply_internal_interactions() {
//     apply_self_interactions();
//     for (auto &interaction : subcomponent_interactions_) {
//       interaction->interact();
//     }
//   }
//   // Accumulate fluctuation forces for own state vars.
//   // If numerical state vars are all owned by subcomponents, this does
//   nothing virtual void apply_thermal_fluctuations_to_self(double dt, double
//   kBT,
//                                                   RandomNumberGenerator *rng)
//                                                   {
//   };
//   // Accumulate thermal fluctuation forces for self and all subcomponents.
//   void apply_thermal_fluctuations(double dt, double kBT,
//                                   RandomNumberGenerator *rng) {
//     apply_thermal_fluctuations_to_self(dt, kBT, rng);
//     for (auto &component : subcomponents_) {
//       component->apply_thermal_fluctuations(dt, kBT, rng);
//     }
//   };
// };

struct MeshBraneObject {
  int index_{std::numeric_limits<int>::max()};
  // MeshBraneObject *supercomponent_{nullptr};
  // std::vector<std::unique_ptr<MeshBraneObject>> subcomponents_{};
  // std::vector<std::unique_ptr<PairInteraction>> subcomponent_interactions_{};

  virtual ~MeshBraneObject() = default;
  // virtual void set_attributes_from_yaml_node(const YAML::Node &node) = 0;

  // virtual void init(const YAML::Node &node) = 0;

  void set_index(int index) { index_ = index; }
  int get_index() const { return index_; }
};

template <typename T>
std::unique_ptr<T>
make_meshbrane_object_from_yaml_node(const YAML::Node &node) {
  static_assert(std::is_base_of_v<MeshBraneObject, T>,
                "T must derive from MeshBraneObject");

  auto obj = std::make_unique<T>();
  obj->init(node);
  return obj;
}

template <typename T>
std::unique_ptr<T>
make_meshbrane_object_from_sim_yaml_node(const YAML::Node &sim_node,
                                         const std::string &key) {
  static_assert(std::is_base_of_v<MeshBraneObject, T>,
                "T must derive from MeshBraneObject");

  const YAML::Node object_node = sim_node[key];

  if (!object_node) {
    throw std::runtime_error("Key '" + key +
                             "' not found in simulation parameters");
  }

  return make_meshbrane_object_from_yaml_node<T>(object_node);
}

/**
 * @example meshbrane_object.hpp
 * Create a derived class of MeshBraneObject and initialize it from a YAML node:
 * @code
 * struct MyObject : public MeshBraneObject {
 *   std::string name_;
 *   int value_;
 *  void set_attributes_from_yaml_node(const YAML::Node &node) override {
 *    if (node["name"]) {
 *     name_ = node["name"].as<std::string>();
 *   }
 *   if (node["value"]) {
 *    value_ = node["value"].as<int>();
 *  }
 * }
 * void init(const YAML::Node &node) override {
 *   set_attributes_from_yaml_node(node);
 *   // Additional initialization stuff can go here...
 * }
 * };
 * Then create an instance of MyObject from a YAML node:
 * @code
 * YAML::Node node = YAML::Load("{name: example, value: 42}");
 * auto obj = make_meshbrane_object_from_yaml_node<MyObject>(node);
 * std::cout << "Name: " << obj->name_ << ", Value: " << obj->value_ <<
 * std::endl;
 * @endcode
 * Create a vector of MeshBraneObjects from a YAML sequence:
 * @code
 * YAML::Node nodes = YAML::Load("[{name: obj1, value: 1}, {name: obj2, value:
 * 2}]"); std::vector<std::unique_ptr<MeshBraneObject>> objects;
 * for (const auto
 * &node : nodes) {
 *   auto obj = make_meshbrane_object_from_yaml_node<MyObject>(node);
 *   objects.push_back(std::move(obj));
 *   // or in one line with
 *   // objects.push_back(make_meshbrane_object_from_yaml_node<MyObject>(node));
 * }
 * for (const auto &obj : objects) {
 *   auto my_obj = dynamic_cast<MyObject *>(obj.get());
 *   if (my_obj) {
 *     std::cout << "Name: " << my_obj->name_ << ", Value: " << my_obj->value_
 * << std::endl;
 *   }
 * }
 * @endcode
 */

} // namespace meshbrane
