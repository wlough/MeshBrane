#pragma once

/**
 * @file meshbrane_object.hpp
 */

#include "meshbrane/kmc.hpp"
#include "meshbrane/pair_interaction.hpp"
#include <cstddef> // size_t
#include <limits>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

/**
 * @brief Base class for simulated objects.
 *
 * @details Each simulated object...
 */
struct MeshBraneObject {
  MeshBraneObject *supercomponent_{nullptr};

  std::vector<std::unique_ptr<MeshBraneObject>> subcomponents_{};
  std::vector<std::unique_ptr<PairInteraction>> subcomponent_interactions_{};

  MeshBraneObject() = default;

  MeshBraneObject(const MeshBraneObject &) = delete;
  MeshBraneObject &operator=(const MeshBraneObject &) = delete;

  MeshBraneObject(MeshBraneObject &&) = delete;
  MeshBraneObject &operator=(MeshBraneObject &&) = delete;

  // virtual methods
  virtual ~MeshBraneObject() = default;
  // set global sim paramaters (e.g. temperature, bulk viscosity, etc...)
  // set object specific parameters
  virtual void set_own_parameters(const YAML::Node &sim_node,
                                  const YAML::Node &own_node) {}
  // initialize subcomponents
  virtual void init_subcomponents(const YAML::Node &sim_node,
                                  const YAML::Node &own_node) {}
  virtual void set_initial_conditions(const YAML::Node &sim_node,
                                      const YAML::Node &own_node) {}
  // initialize interactions between subcomponents
  virtual void init_interactions(const YAML::Node &sim_node,
                                 const YAML::Node &own_node) {}
  // misc object-specific init stuff
  virtual void after_init() {}
  // some objects do not have subcomponents, but still have
  // internal interactions. For example, Membrane bending forces.
  // If numerical state vars are all owned by subcomponents, this does
  // nothing
  virtual void apply_self_interactions() {};
  // Accumulate fluctuation forces for own state vars.
  // If numerical state vars are all owned by subcomponents, this does
  // nothing
  virtual void
  apply_thermal_fluctuations_to_self(double dt, double kBT,
                                     kmc::RandomNumberGenerator &rng) {}
  virtual void clear_own_forces() {}

  virtual void update_own_cached_data() {}

  virtual void update_own_state_variables(double dt) {}

  // other methods
  bool is_composite() const { return !subcomponents_.empty(); }
  // Accumulate forces/torques from internal interactions
  void apply_internal_interactions() {
    apply_self_interactions();
    for (auto &component : subcomponents_) {
      component->apply_internal_interactions();
    }
    for (auto &interaction : subcomponent_interactions_) {
      interaction->interact();
    }
  }
  // Accumulate thermal fluctuation forces for self and all subcomponents.
  void apply_thermal_fluctuations(double dt, double kBT,
                                  kmc::RandomNumberGenerator &rng) {
    apply_thermal_fluctuations_to_self(dt, kBT, rng);

    for (auto &component : subcomponents_) {
      component->apply_thermal_fluctuations(dt, kBT, rng);
    }
  }

  void clear_forces() {
    clear_own_forces();

    for (auto &component : subcomponents_) {
      component->clear_forces();
    }
  }

  void update_cached_data() {
    update_own_cached_data();
    for (auto &component : subcomponents_) {
      component->update_cached_data();
    }
  }

  void update_state_variables(double dt) {
    update_own_state_variables(dt);
    for (auto &component : subcomponents_) {
      component->update_state_variables(dt);
    }
  }

  template <typename T>
  T &init_subcomponent(const YAML::Node &sim_node, const YAML::Node &own_node,
                       const std::string &key) {
    static_assert(std::is_base_of_v<MeshBraneObject, T>,
                  "T must derive from MeshBraneObject");

    if (!own_node[key]) {
      throw std::runtime_error("Missing subcomponent key: " + key);
    }

    auto obj = std::make_unique<T>();
    T &ref = *obj;

    obj->init(sim_node, own_node[key], this);
    subcomponents_.push_back(std::move(obj));

    return ref;
  }

  template <typename Tinteraction, typename Tobj1, typename Tobj2>
  Tinteraction &init_interaction(Tobj1 &obj1, Tobj2 &obj2,
                                 const YAML::Node &sim_node,
                                 const YAML::Node &own_node) {
    static_assert(std::is_base_of_v<MeshBraneObject, Tobj1>,
                  "Tobj1 must derive from MeshBraneObject");

    static_assert(std::is_base_of_v<MeshBraneObject, Tobj2>,
                  "Tobj2 must derive from MeshBraneObject");

    static_assert(std::is_base_of_v<PairInteraction, Tinteraction>,
                  "Tinteraction must derive from PairInteraction");

    static_assert(std::is_constructible_v<Tinteraction, Tobj1 &, Tobj2 &>,
                  "Tinteraction must be constructible from Tobj1& and Tobj2&");

    auto interaction = std::make_unique<Tinteraction>(obj1, obj2);
    Tinteraction &ref = *interaction;
    ref.init(sim_node, own_node);
    subcomponent_interactions_.push_back(std::move(interaction));
    return ref;
  }

  void init(const YAML::Node &sim_node, const YAML::Node &own_node,
            MeshBraneObject *supercomponent = nullptr) {
    supercomponent_ = supercomponent;

    set_own_parameters(sim_node, own_node);
    init_subcomponents(sim_node, own_node);
    init_interactions(sim_node, own_node);
    after_init();
  }
};

// class Obj1 : public MeshBraneObject {
// public:
//   double some_data1_;
// };

// class Obj2 : public MeshBraneObject {
// public:
//   double some_data2_;
// };

// class Obj0 : public MeshBraneObject {
// public:
//   // Non-owning aliases. Ownership is held by subcomponents_.
//   Obj1 *obj1_{nullptr};
//   Obj2 *obj2_{nullptr};
//   //
// private:
//   class Obj1Obj2Interaction : public TypedPairInteraction<Obj1, Obj2> {
//   public:
//     using Base = TypedPairInteraction<Obj1, Obj2>;
//     Obj1Obj2Interaction(Obj1 &obj1, Obj2 &obj2) : Base(obj1, obj2) {}

//     void interact() override {
//       // use obj1_ and obj2_
//     }
//   };
//   //
// public:
//   void init_subcomponents(const YAML::Node &sim_node,
//                           const YAML::Node &own_node) override {
//     obj1_ = &init_subcomponent<Obj1>(sim_node, own_node, "obj1");
//     obj2_ = &init_subcomponent<Obj2>(sim_node, own_node, "obj2");
//   }
//   void init_interactions(const YAML::Node &sim_node,
//                          const YAML::Node &own_node) override {
//     init_interaction<Obj1Obj2Interaction>(*obj1_, *obj2_, sim_node,
//     own_node);
//   }
// };

// template <typename T>
// std::unique_ptr<T>
// make_meshbrane_object_from_yaml_node(const YAML::Node &node) {
//   static_assert(std::is_base_of_v<MeshBraneObject, T>,
//                 "T must derive from MeshBraneObject");

//   auto obj = std::make_unique<T>();
//   // obj->init(node);
//   return obj;
// }

// template <typename T>
// std::unique_ptr<T>
// make_meshbrane_object_from_sim_yaml_node(const YAML::Node &sim_node,
//                                          const std::string &key) {
//   static_assert(std::is_base_of_v<MeshBraneObject, T>,
//                 "T must derive from MeshBraneObject");

//   const YAML::Node object_node = sim_node[key];

//   if (!object_node) {
//     throw std::runtime_error("Key '" + key +
//                              "' not found in simulation parameters");
//   }

//   return make_meshbrane_object_from_yaml_node<T>(object_node);
// }

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
