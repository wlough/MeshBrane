import numpy as np
from src.model import Brane
from src.utils import load_mesh_from_ply
import os
from src.pretty_pictures import mayavi_plots as mp
import dill
from copy import deepcopy

# from src.numdiff import (quaternion_to_matrix,matrix_to_quaternion,jitdot,jitnorm, jitcross)


def initialize_sim(
    vertices,
    faces,
    Tplot,
    Tsave,
    length_reg_stiffness,
    area_reg_stiffness,
    conformal_reg_stiffness,
    bending_modulus,
    splay_modulus,
    linear_drag_coeff,
    dt,
    output_directory,
):
    init_data = {
        "vertices": vertices,
        "faces": faces,
        "Tplot": 5e-2,
        "Tsave": 5e-1,
        "length_reg_stiffness": 1.0,
        "area_reg_stiffness": 1.0,
        "conformal_reg_stiffness": 1.0,
        "bending_modulus": 1.0,
        "splay_modulus": 1.0,
        "linear_drag_coeff": 1.0,
        "dt": 1e-2,
        "output_directory": "./output/reg_sim",
    }
    os.system(f"rm -r {output_directory}")
    os.system(f"mkdir {output_directory}")
    os.system(f"mkdir {output_directory}/temp_images")
    os.system(f"mkdir {output_directory}/states")
    with open(f"{output_directory}/init_data.txt", "w") as _f:
        _f.write(str(init_data))
    with open(f"{output_directory}/init_data.pickle", "wb") as _f:
        dill.dump(init_data, _f)

    params = np.array(
        [
            length_reg_stiffness,
            area_reg_stiffness,
            conformal_reg_stiffness,
            bending_modulus,
            splay_modulus,
            linear_drag_coeff,
            dt,
        ]
    )
    b = Brane(vertices, faces, params)
    brane_init_data = {
        "vertices": vertices,
        "faces": faces,
        "params": params,
    }

    b = Brane(**brane_init_data)

    sim_state = {
        "b": b,
        "success": True,
        "t": 0.0,
        "dt": dt,
        "t2plot": Tplot,
        "t2save": Tsave,
        "tstop": Tplot,
        "image_count": 0,
        "Tplot": Tplot,
        "Tsave": Tsave,
        "output_directory": output_directory,
    }

    return sim_state


def run_sim(sim_state, Trun):
    return sim_state


def save_fig_data(sim_state):
    b = sim_state["b"]
    output_directory = sim_state["output_directory"]
    image_count = sim_state["image_count"]
    data_path = f"{output_directory}/temp_images/fig_{image_count:0>4}.pickle"
    fig_path = f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    plot_args = b.get_plot_data()
    (
        V_pq,
        faces,
        V_rgb,
        V_radius,
        H_rgb,
        F_rgb,
        F_opacity,
        H_opacity,
        V_opacity,
        V_normal_rgb,
        V_frames,
    ) = plot_args
    fig_kwargs = {
        "V_pq": V_pq,
        "faces": faces,
        "V_rgb": V_rgb,
        "V_radius": V_radius,
        "H_rgb": H_rgb,
        "F_rgb": F_rgb,
        "F_opacity": F_opacity,
        "H_opacity": H_opacity,
        "V_opacity": V_opacity,
        "V_normal_rgb": V_normal_rgb,
        "V_frames": V_frames,
        "show": False,
        "save": True,
        "show_surface": True,
        "show_edges": True,
        "show_vertices": False,
        "show_normals": False,
        "show_tangent1": False,
        "show_tangent2": False,
        "fig_path": fig_path,
    }
    with open(data_path, "wb") as _f:
        dill.dump(fig_kwargs, _f)
    return fig_kwargs


def load_fig_data(output_directory, image_count):
    data_path = f"{output_directory}/temp_images/fig_{image_count:0>4}.pickle"

    # fig_kwargs = {
    #     "V_pq": V_pq,
    #     "faces": faces,
    #     "V_rgb": V_rgb,
    #     "V_radius": V_radius,
    #     "H_rgb": H_rgb,
    #     "F_rgb": F_rgb,
    #     "F_opacity": F_opacity,
    #     "H_opacity": H_opacity,
    #     "V_opacity": V_opacity,
    #     "V_normal_rgb": V_normal_rgb,
    #     "V_frames": V_frames,
    #     "show": False,
    #     "save": True,
    #     "show_surface": True,
    #     "show_edges": True,
    #     "show_vertices": False,
    #     "show_normals": False,
    #     "show_tangent1": False,
    #     "show_tangent2": False,
    #     "fig_path": fig_path,
    # }
    with open(data_path, "rb") as _f:
        fig_kwargs = dill.load(_f)
    return fig_kwargs


def save_state_data(sim_state):
    output_directory = sim_state["output_directory"]
    image_count = sim_state["image_count"]
    data_path = f"{output_directory}/states/state_{image_count:0>4}.pickle"
    sim_data = {
        "success": sim_state["success"],
        "t": sim_state["t"],
        "t2plot": sim_state["t2plot"],
        "t2save": sim_state["t2save"],
        "tstop": sim_state["tstop"],
        "image_count": sim_state["image_count"],
        "Tplot": sim_state["Tplot"],
        "Tsave": sim_state["Tsave"],
        "output_directory": sim_state["output_directory"],
    }
    b = sim_state["b"]

    brane_data = b.get_state_data()
    (
        sample_times,
        V_pq_samples,
        faces,
        halfedges,
        V_label,
        V_hedge,
        H_label,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        H_isboundary,
        F_label,
        F_hedge,
    ) = brane_data

    sim_data |= {
        "sample_times": sample_times,
        "V_pq_samples": V_pq_samples,
        "faces": faces,
        "halfedges": halfedges,
        "V_label": V_label,
        "V_hedge": V_hedge,
        "H_label": H_label,
        "H_vertex": H_vertex,
        "H_face": H_face,
        "H_next": H_next,
        "H_prev": H_prev,
        "H_twin": H_twin,
        "H_isboundary": H_isboundary,
        "F_label": F_label,
        "F_hedge": F_hedge,
    }
    with open(data_path, "wb") as _f:
        dill.dump(sim_data, _f)


def load_state_data(output_directory, image_count):
    data_path = f"{output_directory}/states/state_{image_count:0>4}.pickle"

    with open(data_path, "rb") as _f:
        sim_data = dill.load(_f)
    return sim_data


def run(sim_state, Trun, make_plots=True):
    b = sim_state["b"]
    success = sim_state["success"]
    t = sim_state["t"]
    dt = sim_state["dt"]
    # t2plot = sim_state["t2plot"]
    t2save = sim_state["t2save"]
    tstop = sim_state["tstop"]
    image_count = sim_state["image_count"]
    Tplot = sim_state["Tplot"]
    Tsave = sim_state["Tsave"]
    output_directory = sim_state["output_directory"]

    tt = np.round(t, int(-np.floor(np.log10(dt))))
    print(f"T={Trun}", end="\n")
    print(f"t={tt}              ", end="\r")

    fig_kwargs = save_fig_data(sim_state)

    if make_plots:
        mp.plot_from_data(**fig_kwargs)
    image_count += 1

    while Trun - t > 0.5 * dt and success:
        # while t < tstop and success:
        while tstop - t > 0.5 * dt and success:
            b.forward_euler_reg_step(dt)
            if success:
                t += dt
            else:
                print("oh no")

        if success:
            tstop = np.min([t + Tplot, Trun])
            t2save -= Tplot

            tt = np.round(t, int(-np.floor(np.log10(dt))))
            print(f"t={tt}              ", end="\r")

            sim_state["success"] = success
            sim_state["t"] = t
            # sim_state["t2plot"] = t2plot
            sim_state["t2save"] = t2save
            sim_state["tstop"] = tstop
            sim_state["image_count"] = image_count

            fig_kwargs = save_fig_data(sim_state)

            if make_plots:
                mp.plot_from_data(**fig_kwargs)
            image_count += 1

            if t2save <= 0:
                t2save = Tsave
                sim_state["success"] = success
                sim_state["t"] = t
                # sim_state["t2plot"] = t2plot
                sim_state["t2save"] = t2save
                sim_state["tstop"] = tstop
                sim_state["image_count"] = image_count
                save_state_data(sim_state)

    return sim_state


# %%
ply_path = "./data/ply_files/pyramid3.ply"
vertices, faces = load_mesh_from_ply(ply_path)
init_data = {
    "vertices": vertices,
    "faces": faces,
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "length_reg_stiffness": 1.0,
    "area_reg_stiffness": 1.0,
    "conformal_reg_stiffness": 1.0,
    "bending_modulus": 1.0,
    "splay_modulus": 1.0,
    "linear_drag_coeff": 1.0,
    "dt": 1e-3,
    "output_directory": "./output/reg_sim",
}
Trun = 0.5
sim_state = initialize_sim(**init_data)
# sim_state0 = initialize_sim(**init_data)
sim_state = run(sim_state, Trun)
movie_dir = sim_state["output_directory"] + "/temp_images"
mp.movie(movie_dir)


# %%
ply_path = "./data/ply_files/oblate_coarse.ply"
vertices, faces = load_mesh_from_ply(ply_path)
init_data = {
    "vertices": vertices,
    "faces": faces,
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "length_reg_stiffness": 1.0,
    "area_reg_stiffness": 1.0,
    "conformal_reg_stiffness": 1.0,
    "bending_modulus": 1.0,
    "splay_modulus": 1.0,
    "linear_drag_coeff": 1.0,
    "dt": 1e-3,
    "output_directory": "./output/reg_sim",
}
Trun = 0.5
sim_state = initialize_sim(**init_data)
b = sim_state["b"]
Ac = 0
for v in b.V_label:
    Ac += b.cell_area(v)

Af = 0.0
N = len(b.F_label)
for f in b.F_label:
    Avec = b.face_area_vector(f)
    Af += np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2)

(Af - Ac) / Ac
