#pragma once

/**
 * @file pair_interaction.hpp
 * @brief Base class for pairwise interactions
 */

#include "meshbrane/meshbrane_object.hpp"
#include <stdexcept>
#include <type_traits>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

class PairInteraction {
public:
  virtual ~PairInteraction() = default;
  virtual void interact() = 0;
  virtual void init(const YAML::Node &sim_node,
                    const YAML::Node &composite_node) {};
};

template <typename Tobj1, typename Tobj2>
class TypedPairInteraction : public PairInteraction {
protected:
  Tobj1 &obj1_;
  Tobj2 &obj2_;

public:
  TypedPairInteraction(Tobj1 &obj1, Tobj2 &obj2) : obj1_(obj1), obj2_(obj2) {}
};

// template <typename Obj1T, typename Obj2T>
// class TypedPairInteraction : public PairInteraction {
// public:
//   TypedPairInteraction(Obj1T *obj1, Obj2T *obj2) : obj1_(obj1), obj2_(obj2) {
//     static_assert(std::is_base_of_v<MeshBraneObject, Obj1T>);
//     static_assert(std::is_base_of_v<MeshBraneObject, Obj2T>);

//     if (!obj1_ || !obj2_) {
//       throw std::invalid_argument("TypedPairInteraction received null
//       object");
//     }
//   }

// protected:
//   Obj1T *obj1_{nullptr}; // non-owning, typed
//   Obj2T *obj2_{nullptr}; // non-owning, typed
// };

// class MatrixMesh; // forward declaration
// class MeshMeshVertexInteractionWCA : public TypedPairInteraction<MatrixMesh,
// MatrixMesh> {

//   // ....
// }

} // namespace meshbrane
