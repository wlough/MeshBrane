/**
 * @file pretty_pictures.cpp
 */

#include <nlohmann/json.hpp> // nlohmann::json
#include <pretty_pictures.hpp>

void create_pyply_plot_json(const Params_create_pyply_plot_json &params) {
  nlohmann::json j;
  j["show"] = params.show;
  j["plot_vertices"] = params.plot_vertices;
  j["plot_edges"] = params.plot_edges;
  j["plot_faces"] = params.plot_faces;
  j["save"] = params.save;
  j["fig_path"] = params.fig_path;
}