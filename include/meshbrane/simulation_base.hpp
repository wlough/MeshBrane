#pragma once

/**
 * @file simulation_base.hpp
 * @brief Base class for simulations
 */

#include "meshbrane/meshbrane_object.hpp"
#include "meshbrane/pair_interaction.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

/**
 * @brief Base class for simulations. Makes output directory and loads parameter
 * file.
 *
 */
class SimulationBase {
public:
  //
  //
  std::vector<std::unique_ptr<MeshBraneObject>> objects_{};
  std::vector<std::unique_ptr<PairInteraction>> interactions_{};
  template <typename T>
  T &init_object(const YAML::Node &sim_node, const std::string &key) {
    static_assert(std::is_base_of_v<MeshBraneObject, T>,
                  "T must derive from MeshBraneObject");

    if (!sim_node[key]) {
      throw std::runtime_error("Missing subcomponent key: " + key);
    }

    auto obj = std::make_unique<T>();
    T &ref = *obj;

    obj->init(sim_node, sim_node[key], nullptr);
    objects_.push_back(std::move(obj));

    return ref;
  }
  template <typename Tinteraction, typename Tobj1, typename Tobj2>
  Tinteraction &init_interaction(Tobj1 &obj1, Tobj2 &obj2,
                                 const YAML::Node &sim_node) {
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
    ref.init(sim_node, sim_node);
    interactions_.push_back(std::move(interaction));
    return ref;
  }
  virtual void set_sim_parameters(const YAML::Node &sim_node) {}
  virtual void init_objects(const YAML::Node &sim_node) {}
  virtual void init_interactions(const YAML::Node &sim_node) {}
  virtual void update_own_cached_data() {}
  void clear_forces() {
    for (auto &obj : objects_) {
      obj->clear_forces();
    }
  }
  // void update_cached_data() {
  //   for (auto &obj : objects_) {
  //     obj->update_cached_data();
  //   }
  //   update_own_cached_data();
  // }
  void apply_interactions() {
    for (auto &obj : objects_) {
      obj->apply_internal_interactions();
    }
    for (auto &interaction : interactions_) {
      interaction->interact();
    }
  }
  // void apply_thermal_fluctuations(double dt) {
  //   for (auto &obj : objects_) {
  //     obj->apply_thermal_fluctuations(dt, kBT_, rng_);
  //   }
  // }
  // void update_state_variables(double dt) {
  //   for (auto &obj : objects_) {
  //     obj->update_state_variables(dt);
  //   }
  // }
  virtual void after_init() {}
  void init(const YAML::Node &sim_node) {
    init_objects(sim_node);
    init_interactions(sim_node);
    after_init();
  }
  //
  //
  //
  //
  YAML::Node parameters_;
  std::string run_name_;
  std::filesystem::path output_dir_;
  double dt_max_;
  double dt_frame_;
  double T_run_;

  std::filesystem::path logs_dir_;
  std::filesystem::path raw_data_dir_;
  std::filesystem::path visualizations_dir_;
  std::filesystem::path temp_images_dir_;
  std::filesystem::path log_path_;
  int frame_count_ = 0;
  int frame_index_length_ = 6;
  std::string frame_prefix_ = "frame";
  double t_ = 0.0;

  // std::vector<std::unique_ptr<MeshBraneObject>> objects_;
  // std::vector<std::unique_ptr<PairInteraction>> pair_interactions_;

  SimulationBase(const std::filesystem::path &path_to_parameters);

  /**
   * @brief Make output directories for logs/checkpoints/temp_images/etc...
   *
   * @param overwrite
   */
  void make_output_directory(bool overwrite = false);

  /**
   * @brief Make a movie from images in temp_images_dir_
   *
   */
  void make_a_movie();

  // /**
  //  * @brief Initialize objects in the simulation
  //  *
  //  */
  // virtual void initialize_sim() = 0;

  // virtual void timestep() = 0;

  // virtual void run_sim() = 0;

  void configure_logging();

  std::filesystem::path get_frame_path();
};

} // namespace meshbrane
