/**
 * @file pretty_pictures.hpp
 */

#ifndef pretty_pictures_hpp
#define pretty_pictures_hpp

#include <string> // for std::string

////////////////////////////////////////////
// pretty_pictures /////////////////////////
////////////////////////////////////////////

/**
 * @struct Params_create_pyply_plot_json
 * @brief Input parameters for create_pyply_plot_json function.
 */
struct Params_create_pyply_plot_json {
  std::string ply_path;
  bool show = true;
  bool plot_vertices = true;
  bool plot_edges = true;
  bool plot_faces = true;
  bool save = false;
  std::string fig_path;
};

/**
 * @brief Create a JSON file for plotting a .ply file with ply_plot.py.
 */
void create_pyply_plot_json(const Params_create_pyply_plot_json &params);

void call_pyply_plot();

#endif // pretty_pictures_hpp