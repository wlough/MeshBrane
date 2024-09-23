# from src.python.half_edge_base_brane import Brane
# from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim  # , Spindle, SPB, Envelope, ParamManager

# from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from src.python.pretty_pictures import _RGBA_DICT_

yaml_path = "./data/parameters.yaml"

sim = StretchSim.from_parameters_file(yaml_path, overwrite_output_dir=True)
sim.run()
# sim.update(patch=True, force=True, pretty=True)
# sim.plot(save=False, show=True, title="")
# sim.evolve_for_DT(1e-2, 1e-3)
# %%
pm = ParamManager.from_yaml(yaml_path)
p = pm.parameters
envelope = Envelope.from_he_ply(**p["envelope"])
spindle = Spindle(envelope, **p["spindle"])


p["envelope"]
p["spindle"]
output_dir = "./output/stretch_test"
# image_dir = f"{output_dir}/temp_images"
envelope_params = {
    "ply_path": "./data/half_edge_base/ply/unit_sphere_005120_he.ply",
    "rgba_surface": _RGBA_DICT_["green50"],
    "rgba_edge": _RGBA_DICT_["blue80"],
}
spb_params = {
    "axis_origin": (0, 0, 0),
    "force_total": 320,
    "force_profile": "bump",
    "visual_length": 0.1,
    "visual_force_scale": 10.0,
    "find_contact_data": True,
}
spb1_params = spb_params.copy()
spb1_params.update(
    {
        "contact_radius": 0.3,
        "axis_vec": (0, 0, 1),
        "contact_rgba": _RGBA_DICT_["purple50"],
        "force_rgba": _RGBA_DICT_["black"],
    }
)
spb2_params = spb_params.copy()
spb2_params.update(
    {
        "contact_radius": 0.6,
        "axis_vec": (0, 0, -1),
        "contact_rgba": _RGBA_DICT_["orange50"],
        "force_rgba": _RGBA_DICT_["white"],
    }
)
viewer_params = {
    "image_dir": f"{output_dir}/temp_images",
    "show_face_colored_surface": False,
    "show_vertex_colored_surface": True,
    "show_wireframe_surface": True,
    "rgba_vertex": _RGBA_DICT_["green50"],
    "rgba_edge": _RGBA_DICT_["blue80"],
    "show_plot_axes": True,
    "figsize": (2180, 2180),
}
m = Brane.from_he_ply(**envelope_params)
spb1 = SPB(m, **spb1_params)
spb2 = SPB(m, **spb2_params)
mv = MeshViewer(m, **viewer_params)
# %%
for spb in [spb1, spb2]:
    mv.add_vector_field(**spb.viewer_kwargs_add_vector_field())
    mv.update_rgba_V(**spb.viewer_kwargs_update_rgba_V())

mv.plot()
# %%
# StretchSim.make_output_dir(output_dir=output_dir, overwrite=True)
# parameters_path = "./output/stretch_test.yaml"
# parameters_path = "./data/shane.yaml"
sim = StretchSim.from_parameters_file(parameters_path, output_dir)

# sim.run()
# %%
# sim.time_step()
# sim.mesh_viewer.plot()
R = 1
A = 4 * np.pi * R**2

# sim.time_step()
# %%

m = sim.envelope
spb = sim.spb_force
radius = spb.radius
center_plus, center_minus = spb.find_center_vertices(m)
xyz_plus = m.xyz_coord_v(center_plus)
xyz_minus = m.xyz_coord_v(center_minus)
patch_plus = HalfEdgePatch.from_seed_vertex(center_plus, m)
patch_minus = HalfEdgePatch.from_seed_vertex(center_minus, m)
patch_plus.expand_to_radius(xyz_center, radius)
