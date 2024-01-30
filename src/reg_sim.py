from numba import njit
import numpy as np
import src.model as m
import src.model2 as m2
from src.utils import load_mesh_from_ply
import os
from src.pretty_pictures import mayavi_plots as mp
from src.numdiff import jitnorm, jitcross, jitdot
import dill
from copy import deepcopy
from matplotlib import colormaps as plt_cmap
import matplotlib.pyplot as plt


# from src.numdiff import (quaternion_to_matrix,matrix_to_quaternion,jitdot,jitnorm, jitcross)
@njit
def get_new_xyz(vertices, r_com, rad):
    Nv = len(vertices)
    r_new = np.zeros_like(vertices)
    for i in range(Nv):
        r = vertices[i]
        r_rel = r - r_com
        r_unit = r_rel / jitnorm(r_rel)
        r_new[i] = r_com + rad * r_unit
    return r_new


def get_crange(samps, Nstd=2):
    c0 = np.mean(samps)
    sig = np.std(samps)
    cmin = c0 - Nstd * sig
    cmax = c0 + Nstd * sig
    samps_clipped = np.clip(samps, cmin, cmax)
    return samps_clipped, cmin, cmax


def get_cmap(cmin=0.0, cmax=1.0, name="hsv"):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
    'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
    'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
    'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
    'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
    'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
    'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
    'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
    'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
    'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
    'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
    'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
    'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
    'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot',
    'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma',
    'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',
    'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
    'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
    'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
    'winter_r'

    """
    cnum = lambda x: (x - cmin) / (cmax - cmin)
    cmap01 = plt_cmap[name]
    my_cmap = lambda x: cmap01(cnum(float(x)))
    return my_cmap


def scalars_to_rgbs(samples, cmin=None, cmax=None, name="coolwarm"):
    if cmin is None:
        cmin = np.min(samples)
    if cmax is None:
        cmax = np.max(samples)
    # Nsamps = len(samples)
    cmap = get_cmap(cmin=cmin, cmax=cmax, name=name)
    rgbs = np.array([cmap(_)[:-1] for _ in samples])
    return rgbs


def rgb_float_to_int(rgb_float):
    """converts normalized rgb 0<r,g,b<1 to 0<r,g,b<255
    rgb_float=[r,g,b]
    rgb_float=[r,g,b,alpha]
    rgb_float=[...,[r,g,b],...]
    rgb_float=[...,[r,g,b,alpha],...]"""
    rgb_int = np.round(np.array([_ for _ in rgb_float]) * 255).astype(int)
    return rgb_int


def rgb_int_to_float(rgb_int):
    """converts normalized rgb 0<r,g,b<255 to 0<r,g,b<1"""
    rgb_float = np.array([_ for _ in rgb_int], dtype=np.float64) / 255
    return rgb_float


def save_fig_data(sim_state):
    b = sim_state["b"]
    mesh_data = b.pack_mesh_data()
    vis_data = b.pack_visual_data()
    output_directory = sim_state["output_directory"]
    image_count = sim_state["image_count"]
    mesh_data_file = f"{output_directory}/temp_images/mesh_data_{image_count:0>4}.npy"
    vis_data_file = f"{output_directory}/temp_images/vis_data_{image_count:0>4}.npy"
    np.save(mesh_data_file, mesh_data)
    np.save(vis_data_file, vis_data)
    fig_path = f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    fig_kwargs = {
        "V_pq": b.V_pq,
        "faces": b.faces,
        "V_rgb": b.V_rgb,
        "V_radius": b.V_radius,
        "H_rgb": b.H_rgb,
        "F_rgb": b.F_rgb,
        "F_opacity": b.F_opacity,
        "H_opacity": b.H_opacity,
        "V_opacity": b.V_opacity,
        "V_normal_rgb": b.V_normal_rgb,
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
    return fig_kwargs


def initialize_sim2(
    Tplot,
    Tsave,
    dt,
    vertices,
    faces,
    length_reg_stiffness,
    area_reg_stiffness,
    volume_reg_stiffness,
    bending_modulus,
    splay_modulus,
    spontaneous_curvature,
    linear_drag_coeff,
    output_directory,
):
    init_data = {
        "Tplot": Tplot,
        "Tsave": Tsave,
        "dt": dt,
        "vertices": vertices,
        "faces": faces,
        "length_reg_stiffness": length_reg_stiffness,
        "area_reg_stiffness": area_reg_stiffness,
        "volume_reg_stiffness": volume_reg_stiffness,
        "bending_modulus": bending_modulus,
        "splay_modulus": splay_modulus,
        "spontaneous_curvature": spontaneous_curvature,
        "linear_drag_coeff": linear_drag_coeff,
        "output_directory": output_directory,
    }
    os.system(f"rm -r {output_directory}")
    os.system(f"mkdir {output_directory}")
    os.system(f"mkdir {output_directory}/temp_images")
    os.system(f"mkdir {output_directory}/states")
    os.system(f"mkdir {output_directory}/init_data")
    np.save(f"{output_directory}/init_data/vertices.npy", vertices)
    np.save(f"{output_directory}/init_data/faces.npy", faces)
    with open(f"{output_directory}/init_data/init_data.txt", "w") as _f:
        init_str = ""
        for key, val in init_data.items():
            if not key in ["vertices", "faces"]:
                init_str += f"{key} = {val}\n"
        _f.write(init_str)
    # with open(f"{output_directory}/init_data.pickle", "wb") as _f:
    #     dill.dump(init_data, _f)

    brane_init_data = {
        "vertices": vertices,
        "faces": faces,
        "length_reg_stiffness": length_reg_stiffness,
        "area_reg_stiffness": area_reg_stiffness,
        "volume_reg_stiffness": volume_reg_stiffness,
        "bending_modulus": bending_modulus,
        "splay_modulus": splay_modulus,
        "linear_drag_coeff": linear_drag_coeff,
        "spontaneous_curvature": spontaneous_curvature,
    }

    b = m2.Brane(**brane_init_data)

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


def initialize_sim(
    Tplot,
    Tsave,
    dt,
    vertices,
    faces,
    length_reg_stiffness,
    area_reg_stiffness,
    volume_reg_stiffness,
    bending_modulus,
    splay_modulus,
    spontaneous_curvature,
    linear_drag_coeff,
    output_directory,
):
    init_data = {
        "Tplot": Tplot,
        "Tsave": Tsave,
        "dt": dt,
        "vertices": vertices,
        "faces": faces,
        "length_reg_stiffness": length_reg_stiffness,
        "area_reg_stiffness": area_reg_stiffness,
        "volume_reg_stiffness": volume_reg_stiffness,
        "bending_modulus": bending_modulus,
        "splay_modulus": splay_modulus,
        "spontaneous_curvature": spontaneous_curvature,
        "linear_drag_coeff": linear_drag_coeff,
        "output_directory": output_directory,
    }
    os.system(f"rm -r {output_directory}")
    os.system(f"mkdir {output_directory}")
    os.system(f"mkdir {output_directory}/temp_images")
    os.system(f"mkdir {output_directory}/states")
    os.system(f"mkdir {output_directory}/init_data")
    np.save(f"{output_directory}/init_data/vertices.npy", vertices)
    np.save(f"{output_directory}/init_data/faces.npy", faces)
    with open(f"{output_directory}/init_data/init_data.txt", "w") as _f:
        init_str = ""
        for key, val in init_data.items():
            if not key in ["vertices", "faces"]:
                init_str += f"{key} = {val}\n"
        _f.write(init_str)
    # with open(f"{output_directory}/init_data.pickle", "wb") as _f:
    #     dill.dump(init_data, _f)

    brane_init_data = {
        "vertices": vertices,
        "faces": faces,
        "length_reg_stiffness": length_reg_stiffness,
        "area_reg_stiffness": area_reg_stiffness,
        "volume_reg_stiffness": volume_reg_stiffness,
        "bending_modulus": bending_modulus,
        "splay_modulus": splay_modulus,
        "linear_drag_coeff": linear_drag_coeff,
        "spontaneous_curvature": spontaneous_curvature,
    }

    b = m.Brane(**brane_init_data)

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


def regularize_run(sim_state, make_plots=True, iters=20, weight=0.2):
    b = sim_state["b"]
    image_count = sim_state["image_count"]
    vertices = b.V_pq[:, :3]
    Nvertices = len(b.V_pq)
    T = 5e-2
    dt = 1e-3
    Nt = int(T / dt)
    vals = np.array([b.valence(v) for v in range(Nvertices)])
    val_min = min(vals)
    val_max = max(vals)
    iter = -1
    Nflips = -1
    print(
        f"iter={iter} of {iters}, Nflips={Nflips}, val_min={val_min}, val_max={val_max}            ",
        end="\n",
    )

    if make_plots:
        fig_kwargs = save_fig_data(sim_state)
        mp.plot_from_data(**fig_kwargs)
        image_count += 1
    for iter in range(iters):
        Nflips = b.regularize_by_flips()
        vals = np.array([b.valence(v) for v in range(Nvertices)])
        val_min = min(vals)
        val_max = max(vals)
        b.regularize_by_shifts(weight)
        for _ in range(Nt):
            b.forward_euler_reg_step(dt)
        # b.V_pq[:, :3] = get_new_xyz(b.V_pq[:, :3], r_com, rad)
        print(
            f"iter={iter} of {iters}, Nflips={Nflips}, val_min={val_min}, val_max={val_max}, L/L0={b.average_hedge_length()/b.preferred_edge_length}            ",
            end="\n",
        )
        # print(
        #     f"iter={iter} of {iters}, V={b.volume}, V0={b.volume0}          ",
        #     end="\n",
        # )
        # print(f"iter={iter} of {iters}, Nflips={0}            ", end="\n")

        if make_plots:
            sim_state["image_count"] = image_count
            fig_kwargs = save_fig_data(sim_state)
            mp.plot_from_data(**fig_kwargs)
            image_count += 1

    return sim_state


def regularize_sphere_run(sim_state, make_plots=True, iters=20, weight=0.2):
    b = sim_state["b"]
    image_count = sim_state["image_count"]
    vertices = b.V_pq[:, :3]
    Nvertices = len(b.V_pq)
    r_com = np.einsum("vi->i", vertices) / Nvertices
    XYZ = np.array([xyz - r_com for xyz in vertices])
    rad = np.mean(np.linalg.norm(XYZ, axis=1))
    T = 5e-2
    dt = 1e-3
    Nt = int(T / dt)
    vals = np.array([b.valence(v) for v in range(Nvertices)])
    val_min = min(vals)
    val_max = max(vals)
    iter = -1
    Nflips = -1
    print(
        f"iter={iter} of {iters}, Nflips={Nflips}, val_min={val_min}, val_max={val_max}            ",
        end="\n",
    )

    if make_plots:
        fig_kwargs = save_fig_data(sim_state)
        mp.plot_from_data(**fig_kwargs)
        image_count += 1
    for iter in range(iters):
        Nflips = b.regularize_by_flips()
        vals = np.array([b.valence(v) for v in range(Nvertices)])
        val_min = min(vals)
        val_max = max(vals)
        b.regularize_by_shifts(weight)
        for _ in range(Nt):
            b.forward_euler_reg_step(dt)
        b.V_pq[:, :3] = get_new_xyz(b.V_pq[:, :3], r_com, rad)
        print(
            f"iter={iter} of {iters}, Nflips={Nflips}, val_min={val_min}, val_max={val_max}, L/L0={b.average_hedge_length()/b.preferred_edge_length}            ",
            end="\n",
        )
        # print(
        #     f"iter={iter} of {iters}, V={b.volume}, V0={b.volume0}          ",
        #     end="\n",
        # )
        # print(f"iter={iter} of {iters}, Nflips={0}            ", end="\n")

        if make_plots:
            sim_state["image_count"] = image_count
            fig_kwargs = save_fig_data(sim_state)
            mp.plot_from_data(**fig_kwargs)
            image_count += 1

    return sim_state


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
        # b.regularize_by_shifts(0.2)
        while tstop - t > 0.5 * dt and success:
            # b.forward_euler_step(dt)
            vertices, success = b.get_new_euler_state(dt)
            # success=True
            if success:
                b.V_pq[:, :3] = vertices
                # b.vertices
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
                # save_state_data(sim_state)

    return sim_state


# %%

ply_path = "./data/ply_files/sphere0.ply"
vertices, faces = load_mesh_from_ply(ply_path)

init_data = {
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "dt": 1e-3,
    "vertices": vertices,
    "faces": faces,
    "length_reg_stiffness": 0 * 1e-1,
    "area_reg_stiffness": 0 * 1e-2,
    "volume_reg_stiffness": 0 * 1e-1,
    "bending_modulus": 1.0,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1.0,
    "output_directory": "./output/euler_sim",
}

sim_state = initialize_sim(**init_data)
Trun = 0.2
# b = sim_state["b"]
# b.regularize_by_shifts(0.001)
# b.regularize_by_flips()
# sim_state = run(sim_state, Trun, make_plots=True)
# regularize_run(sim_state, make_plots=True, iters=5, weight=0.15)
regularize_sphere_run(sim_state, make_plots=True, iters=5, weight=0.15)
# %%
# def get_mean_curvature(self):
#     """ """
#     Nv = self.V_pq.shape[0]
#     H = np.zeros(Nv)
#
#     for v in range(Nv):
#         Atot = 0.0
#         r = self.V_pq[v, :3]
#         r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
#         Hvec = np.zeros(3)
#         n = np.zeros(3)
#         h_start = self.V_hedge[v]
#         h = h_start
#         while True:
#             v1 = self.H_vertex[h]
#             r1 = self.V_pq[v1, :3]
#             h = self.H_twin[self.H_prev[h]]
#             v2 = self.H_vertex[h]
#             r2 = self.V_pq[v2, :3]
#
#             r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
#             r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
#             r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
#             r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
#             r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
#
#             normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
#             normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
#             normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
#             cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
#             cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
#             cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
#             cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
#
#             print(f"Hvec={Hvec} Atot={Atot} n={n}")
#
#             Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
#             Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
#             n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)
#
#             if h == h_start:
#                 break
#
#         Hvec /= 2 * Atot
#         n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
#         H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
#     return H
# def get_mean_curvature_vector(self):
#     """ """
#     Nv = self.V_pq.shape[0]
#     Hvec = np.zeros((Nv, 3))
#
#     for v in range(Nv):
#         Atot = 0.0
#         r = self.V_pq[v, :3]
#         r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
#         # Hvec = np.zeros(3)
#         # n = np.zeros(3)
#         h_start = self.V_hedge[v]
#         h = h_start
#         while True:
#             v1 = self.H_vertex[h]
#             r1 = self.V_pq[v1, :3]
#             h = self.H_twin[self.H_prev[h]]
#             v2 = self.H_vertex[h]
#             r2 = self.V_pq[v2, :3]
#
#             r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
#             r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
#             r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
#             r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
#             r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
#
#             normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
#             normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
#             normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
#             cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
#             cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
#             cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
#             cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
#
#             Hvec[v] += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
#             Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
#             # n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)
#
#             if h == h_start:
#                 break
#
#         Hvec /= 2 * Atot
#         # n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
#         # H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
#     return Hvec
#
# get_mean_curvature(b)
# %%
dt = 1e-3  # sim_state["dt"]
# for iter in range(100):
#     Fb = b.Fbend2()
#
#     b.V_pq[:, :3] += dt * Fb

Fb = b.Fbend2()
fb = np.linalg.norm(Fb, axis=1)
fb_rgb = scalars_to_rgbs(fb)

Fl = b.Flength()
fl = np.linalg.norm(Fl, axis=1)
fl_rgb = scalars_to_rgbs(fl)

Fa = b.Farea()
fa = np.linalg.norm(Fa, axis=1)
fa_rgb = scalars_to_rgbs(fa)

Fv = b.Fvolume()
fv = np.linalg.norm(Fv, axis=1)
fv_rgb = scalars_to_rgbs(fv)

F = Fb + Fl + Fa + Fv
f = np.linalg.norm(F, axis=1)
f_rgb = scalars_to_rgbs(f)

# b.F_opacity = 1.0
H, K = b.get_curvatures()
Hvec = b.get_mean_curvature_vector()
n = np.array([b.area_weighted_vertex_normal(v) for v, _ in enumerate(b.V_pq)])
Hvec0 = np.einsum("v, vi->vi", H, n)
lapH = b.cotan_laplacian(H)
H_rgb = scalars_to_rgbs(H)
K_rgb = scalars_to_rgbs(K)
lapH_rgb = scalars_to_rgbs(lapH)
# Hvec-Hvec0
# b.V_vector_data = -Hvec0 * 1e-1
b.V_vector_data = 0.5 * Fb  # / np.max(np.linalg.norm(Fb, axis=1, ord=np.inf))

# b.V_vector_data = Fl
# b.V_vector_data = Fa*1e2
# b.V_vector_data = Fv*1e4
b.V_rgb = lapH_rgb
mp.brane_plot(
    b,
    color_by_V_rgb=True,
    show_halfedges=True,
    show_normals=False,
    show_V_vector_data=False,
)
# brane_plot(b, color_by_F_scalar=True, show_halfedges=True, show_normals=False)
# %%
lapH = b.cotan_laplacian(H)
sum(lapH)


# %%
##############################
def _initialize_sim(
    vertices,
    faces,
    Tplot,
    Tsave,
    length_reg_stiffness,
    area_reg_stiffness,
    volume_reg_stiffness,
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
        "length_reg_stiffness": length_reg_stiffness,
        "area_reg_stiffness": area_reg_stiffness,
        "bending_modulus": bending_modulus,
        "splay_modulus": splay_modulus,
        "linear_drag_coeff": linear_drag_coeff,
        "dt": dt,
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

    brane_init_data = {
        "vertices": vertices,
        "faces": faces,
        "length_reg_stiffness": length_reg_stiffness,
        "area_reg_stiffness": area_reg_stiffness,
        "volume_reg_stiffness": volume_reg_stiffness,
        "bending_modulus": bending_modulus,
        "splay_modulus": splay_modulus,
        "linear_drag_coeff": linear_drag_coeff,
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


# def run(sim_state, Trun, make_plots=True):
#     b = sim_state["b"]
#     success = sim_state["success"]
#     t = sim_state["t"]
#     dt = sim_state["dt"]
#     # t2plot = sim_state["t2plot"]
#     t2save = sim_state["t2save"]
#     tstop = sim_state["tstop"]
#     image_count = sim_state["image_count"]
#     Tplot = sim_state["Tplot"]
#     Tsave = sim_state["Tsave"]
#     output_directory = sim_state["output_directory"]
#
#     tt = np.round(t, int(-np.floor(np.log10(dt))))
#     print(f"T={Trun}", end="\n")
#     print(f"t={tt}              ", end="\r")
#
#     fig_kwargs = save_fig_data(sim_state)
#
#     if make_plots:
#         mp.plot_from_data(**fig_kwargs)
#     image_count += 1
#
#     while Trun - t > 0.5 * dt and success:
#         # while t < tstop and success:
#         # b.regularize_by_shifts(0.2)
#         while tstop - t > 0.5 * dt and success:
#             b.forward_euler_reg_step(dt)
#             if success:
#                 t += dt
#             else:
#                 print("oh no")
#
#         if success:
#             tstop = np.min([t + Tplot, Trun])
#             t2save -= Tplot
#
#             tt = np.round(t, int(-np.floor(np.log10(dt))))
#             print(f"t={tt}              ", end="\r")
#
#             sim_state["success"] = success
#             sim_state["t"] = t
#             # sim_state["t2plot"] = t2plot
#             sim_state["t2save"] = t2save
#             sim_state["tstop"] = tstop
#             sim_state["image_count"] = image_count
#
#             fig_kwargs = save_fig_data(sim_state)
#
#             if make_plots:
#                 mp.plot_from_data(**fig_kwargs)
#             image_count += 1
#
#             if t2save <= 0:
#                 t2save = Tsave
#                 sim_state["success"] = success
#                 sim_state["t"] = t
#                 # sim_state["t2plot"] = t2plot
#                 sim_state["t2save"] = t2save
#                 sim_state["tstop"] = tstop
#                 sim_state["image_count"] = image_count
#                 save_state_data(sim_state)
#
#     return sim_state


def regularize_by_shifts_run(sim_state, make_plots=True, weight=5e-2, iters=20):
    b = sim_state["b"]
    image_count = sim_state["image_count"]

    if make_plots:
        fig_kwargs = save_fig_data(sim_state)
        mp.plot_from_data(**fig_kwargs)
        image_count += 1
    for iter in range(iters):
        for v in b.V_label:
            b.shift_vertex_towards_barycenter(v, weight)

        if make_plots:
            sim_state["image_count"] = image_count
            fig_kwargs = save_fig_data(sim_state)
            mp.plot_from_data(**fig_kwargs)
            image_count += 1

    return sim_state


def regularize_by_flips_run(sim_state, make_plots=True, iters=20):
    b = sim_state["b"]
    image_count = sim_state["image_count"]

    if make_plots:
        fig_kwargs = save_fig_data(sim_state)
        mp.plot_from_data(**fig_kwargs)
        image_count += 1
    for iter in range(iters):
        Nflips = b.regularize_by_flips()
        print(f"Nflips={Nflips}            ", end="\r")

        if make_plots:
            sim_state["image_count"] = image_count
            fig_kwargs = save_fig_data(sim_state)
            mp.plot_from_data(**fig_kwargs)
            image_count += 1

    return sim_state


def regularize_by_random_run(sim_state, make_plots=True, weight=5e-2, iters=20):
    b = sim_state["b"]
    H_rgb = b.H_rgb.copy()
    F_rgb = b.F_rgb.copy()
    V_rgb = b.V_rgb.copy()
    V_radius = b.V_radius.copy()
    image_count = sim_state["image_count"]
    fig_kwargs = save_fig_data(sim_state)
    if make_plots:
        mp.plot_from_data(**fig_kwargs)
    image_count += 1
    h = 0
    v = 0
    Nh = len(b.H_label)
    Nv = len(b.V_label)
    Hgo = True
    Vgo = True
    f1, f2 = 0, 0
    # for iter in range(iters):
    while Hgo or Vgo:
        print(f"v/Nv={v/Nv}, h/Nh={h/Nh}")

        x = np.random.rand(1)[0]
        if x <= 0.5:
            b.shift_vertex_towards_barycenter(v, weight)
            b.V_rgb[v] = np.array([1.0, 0.0, 0.0])
            b.V_radius[v] = 0.1
            v += 1
        if v >= Nv:
            v = 0
            Vgo = False

        if x > 0.5:
            flip_it = False
            while True:
                flip_it = b.flip_helps_valence(h)
                if flip_it:
                    b.edge_flip(h)
                    f1 = b.f_of_h(h)
                    f2 = b.f_of_h(b.twin(h))
                    b.F_rgb[f1] = np.array([1.0, 0.0, 0.0])
                    b.F_rgb[f2] = np.array([0.0, 0.0, 1.0])
                    h += 1
                    break
                else:
                    h += 1
                if h >= Nh:
                    h = 0
                    Hgo = False
                    break

        sim_state["image_count"] = image_count
        fig_kwargs = save_fig_data(sim_state)
        fig_kwargs = save_fig_data(sim_state)

        fig_kwargs["show_halfedges"] = True
        fig_kwargs["show_vertices"] = True

        if make_plots:
            mp.plot_from_data(**fig_kwargs)
            image_count += 1
            b.V_rgb[v - 1 % Nv] = V_rgb[v - 1 % Nv]
            b.V_radius[v - 1 % Nv] = V_radius[v - 1 % Nv]
            b.F_rgb[f1] = F_rgb[f1]
            b.F_rgb[f2] = F_rgb[f2]

    return sim_state


def reg_sim_run(sim_state, make_plots=True, iters=20, weight=0.1):
    b = sim_state["b"]
    image_count = sim_state["image_count"]

    if make_plots:
        fig_kwargs = save_fig_data(sim_state)
        mp.plot_from_data(**fig_kwargs)
        image_count += 1
    for iter in range(iters):
        Nflips = b.regularize_by_flips()
        b.regularize_by_shifts(weight)
        print(f"iter={iter} of {iters}, Nflips={Nflips}            ", end="\n")

        if make_plots:
            sim_state["image_count"] = image_count
            fig_kwargs = save_fig_data(sim_state)
            mp.plot_from_data(**fig_kwargs)
            image_count += 1

    return sim_state


def sphere_reg_sim_run(sim_state, make_plots=True, iters=20, weight=0.1):
    b = sim_state["b"]
    image_count = sim_state["image_count"]
    vertices = b.V_pq[:, :3]
    Nverts = len(b.V_pq)
    r_com = np.einsum("si->i", vertices) / Nverts
    rad = np.sqrt(np.einsum("si,si->", vertices, vertices) / Nverts)
    T = 5e-2
    dt = 1e-3
    Nt = int(T / dt)
    Kl, Ka, Kv = b.Klength, b.Karea, b.Kvolume
    vals = np.array([b.valence(v) for v in b.V_label])
    val_min = min(vals)
    val_max = max(vals)
    print(
        f"iter={-1} of {iters}, Nflips={0}, val_min={val_min}, val_max={val_max}            ",
        end="\n",
    )

    if make_plots:
        fig_kwargs = save_fig_data(sim_state)
        mp.plot_from_data(**fig_kwargs)
        image_count += 1
    for iter in range(iters):
        # Nflips = b.regularize_by_flips()
        Nflips = b.regularize_by_flips_min()
        vals = np.array([b.valence(v) for v in b.V_label])
        val_min = min(vals)
        val_max = max(vals)
        # b.regularize_by_shifts(weight)
        # for _ in range(Nt):
        #     b.forward_euler_reg_step(dt)
        # b.V_pq[:, :3] = get_new_xyz(b.V_pq[:, :3], r_com, rad)
        print(
            f"iter={iter} of {iters}, Nflips={Nflips}, val_min={val_min}, val_max={val_max}            ",
            end="\n",
        )
        # print(
        #     f"iter={iter} of {iters}, V={b.volume}, V0={b.volume0}          ",
        #     end="\n",
        # )
        # print(f"iter={iter} of {iters}, Nflips={0}            ", end="\n")

        if make_plots:
            sim_state["image_count"] = image_count
            fig_kwargs = save_fig_data(sim_state)
            mp.plot_from_data(**fig_kwargs)
            image_count += 1

    return sim_state


@njit
def get_new_xyz(vertices, r_com, rad):
    Nv = len(vertices)
    r_new = np.zeros_like(vertices)
    for i in range(Nv):
        r = vertices[i]
        r_rel = r - r_com
        r_unit = r_rel / jitnorm(r_rel)
        r_new[i] = r_com + rad * r_unit
    return r_new


# b.V_pq[:, :3] = get_new_xyz(b.V_pq[:, :3], r_com, rad)

#
ply_path = "./data/ply_files/sphere.ply"
vertices, faces = load_mesh_from_ply(ply_path)
brane_init_data = {
    "vertices": vertices,
    "faces": faces,
    "length_reg_stiffness": 1e-1,
    "area_reg_stiffness": 1e-2,
    "volume_reg_stiffness": 1.0,
    "bending_modulus": 1.0,
    "splay_modulus": 1.0,
    "linear_drag_coeff": 1.0,
    "spontaneous_curvature": 0.0,
}

b = Brane(**brane_init_data)
b.forward_euler_reg_step(1e-3)
output_directory = "./output/sphere_reg_sim2"

b.length_reg_force(13)
b.area_reg_force(13)
b.volume_reg_force(13)
Vol = b.volume_of_mesh()
rad = np.sqrt(np.einsum("si,si->", b.V_pq[:, :3], b.V_pq[:, :3]) / len(b.V_pq))
Krad = 1 / rad**2
Volrad = 4 * np.pi * rad**3 / 3
K = b.get_Gaussian_curvature()
Kp = np.array([_ for _ in K if _ >= 0])
Km = np.array([_ for _ in K if _ < 0])
Kave = sum(K) / len(K)
Kvar = sum((K - Kave) ** 2) / len(K)
Kstd = np.sqrt(Kvar)

cmin = Kave - 2 * Kstd
cmax = Kave + 2 * Kstd
Kclose = np.array([_ for _ in K if _ >= cmin and _ <= cmax])

Krgb = scalars_to_rgbs(K, cmin=cmin, cmax=cmax)
b.V_rgb = Krgb

mp.brane_plot(b, color_by_V_rgb=True, show_halfedges=True, show_normals=True)

# %%
ply_path = "./data/ply_files/sphere_coarse_backup.ply"
vertices, faces = load_mesh_from_ply(ply_path)

init_data = {
    "vertices": vertices,
    "faces": faces,
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "length_reg_stiffness": 1e-1,
    "area_reg_stiffness": 1e-2,
    "volume_reg_stiffness": 1e4,
    "bending_modulus": 1.0,
    "splay_modulus": 1.0,
    "linear_drag_coeff": 1.0,
    "dt": 1e-3,
    "output_directory": "./output/sphere_reg_sim2",
}
Trun = 15
sim_state = initialize_sim(**init_data)
# sim_state = run(sim_state, Trun)
# sim_state = reg_sim_run(sim_state, make_plots=True, iters=100, weight=0.1)
sim_state = sphere_reg_sim_run(sim_state, make_plots=True, iters=10, weight=0.5)
movie_dir = sim_state["output_directory"] + "/temp_images"
# mp.movie(movie_dir)
b = sim_state["b"]
arr = b.V_pq

# %%


# %%
# b = sim_state["b"]
# b.forward_euler_reg_step(1e-3)
b._average_hedge_length(b.V_pq[:, :3], b.H_label, b.H_twin, b.H_vertex)
b.average_hedge_length()
mp.brane_plot(b)
for v in b.V_label:
    F1 = b.length_reg_force(v)
    F2 = b.area_reg_force(v)
    print(f"{F1@F1}, {F2@F2}")
# %%
for _ in range(100):
    b.forward_euler_reg_step(1e-3)
    print(_)

# %%
import matplotlib.pyplot as plt

ply_path = "./data/ply_files/torus_coarse.ply"
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
Trun = 0.1
sim_state = initialize_sim(**init_data)
sim_state = run(sim_state, Trun)
b = sim_state["b"]
Ac = 0
K = np.zeros(len(b.V_label))
for v in b.V_label:
    # Ac += b.cell_area(v)
    K[v] = b.gaussian_curvature(v)

# %%
Kmin = min(K)
Kmax = max(K) * 0 + 100.0
cmap = get_cmap(cmin=Kmin, cmax=Kmax, name="coolwarm")

for v in b.V_label:
    Kv = K[v]
    b.V_rgb[v] = cmap(Kv)[:-1]
plt.plot(K)
for h in b.H_label:
    b.flip_helps_valence(h)
b.flip_bad_edges
