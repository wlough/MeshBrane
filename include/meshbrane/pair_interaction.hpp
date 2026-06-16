#pragma once

/**
 * @file pair_interaction.hpp
 * @brief Base class for pairwise interactions
 */

#include "meshbrane/meshbrane_object.hpp"
#include <stdexcept>
#include <type_traits>

namespace meshbrane {

class PairInteraction {
public:
  virtual ~PairInteraction() = default;
  virtual void interact() = 0;
};

template <typename Obj1T, typename Obj2T>
class TypedPairInteraction : public PairInteraction {
public:
  TypedPairInteraction(Obj1T *obj1, Obj2T *obj2) : obj1_(obj1), obj2_(obj2) {
    static_assert(std::is_base_of_v<MeshBraneObject, Obj1T>);
    static_assert(std::is_base_of_v<MeshBraneObject, Obj2T>);

    if (!obj1_ || !obj2_) {
      throw std::invalid_argument("TypedPairInteraction received null object");
    }
  }

protected:
  Obj1T *obj1_{nullptr}; // non-owning, typed
  Obj2T *obj2_{nullptr}; // non-owning, typed
};

// class MatrixMesh; // forward declaration
// class MeshMeshVertexInteractionWCA : public TypedPairInteraction<MatrixMesh,
// MatrixMesh> {

//   // ....
// }

} // namespace meshbrane
