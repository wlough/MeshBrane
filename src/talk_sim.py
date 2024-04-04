from time import time
import mayavi.mlab as mlab
from numba import njit
import numpy as np

import src.model as m
from src.utils import load_mesh_from_ply, load_halfedge_mesh_data
import os
from src.pretty_pictures import mayavi_plots as mp

from src.numdiff import jitnorm, jitcross, jitdot, quaternion_to_matrix
import dill
from scipy.linalg import expm, logm, inv

# from copy import deepcopy
from matplotlib import colormaps as plt_cmap
import matplotlib.pyplot as plt
import multiprocessing as mu


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
    if cmax > cmin:
        cnum = lambda x: (x - cmin) / (cmax - cmin)
    else:
        cnum = lambda x: 0 * x
        print("bad min-max range")
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


def initialize_sim(
    Tplot,
    Tsave,
    dt,
    output_directory,
    vertices,
    faces,
    length_reg_stiffness,
    area_reg_stiffness,
    volume_reg_stiffness,
    bending_modulus,
    splay_modulus,
    spontaneous_curvature,
    linear_drag_coeff,
    V_hedge=None,
    halfedges=None,
    H_vertex=None,
    H_face=None,
    H_next=None,
    H_prev=None,
    H_twin=None,
    F_hedge=None,
):
    init_data = {
        "Tplot": Tplot,
        "Tsave": Tsave,
        "dt": dt,
        "output_directory": output_directory,
        "vertices": vertices,
        "faces": faces,
        "length_reg_stiffness": length_reg_stiffness,
        "area_reg_stiffness": area_reg_stiffness,
        "volume_reg_stiffness": volume_reg_stiffness,
        "bending_modulus": bending_modulus,
        "splay_modulus": splay_modulus,
        "spontaneous_curvature": spontaneous_curvature,
        "linear_drag_coeff": linear_drag_coeff,
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
        "V_hedge": V_hedge,
        "halfedges": halfedges,
        "H_vertex": H_vertex,
        "H_face": H_face,
        "H_next": H_next,
        "H_prev": H_prev,
        "H_twin": H_twin,
        "F_hedge": F_hedge,
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


def run(sim_state, Trun, make_plots=True):
    Dtdict = {"Fbend_mixed": 0, "Ftether": 0, "Fa_Fv": 0, "brane_plot": 0}
    view = {
        "azimuth": 0,
        "elevation": 55,
        "distance": 4,
        "focalpoint": (0, 0, 0),
    }
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

    # fig_kwargs = save_fig_data(sim_state)
    fig_path = f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
    if make_plots:
        Dt = time()
        Fb = b.Fbend_mixed()
        Dtdict["Fbend_mixed"] += time() - Dt

        Dt = time()
        Fl = b.Ftether()
        Dtdict["Ftether"] += time() - Dt

        Dt = time()
        Fa, Fv = b.Fa_Fv()
        Dtdict["Fa_Fv"] += time() - Dt

        F = Fb + Fl + Fa + Fv
        b.F_opacity = 0.8
        Fplot = 0.1 * F
        Fmax = np.max(np.linalg.norm(Fplot, np.inf, axis=0))
        if Fmax > 0.25:
            Fplot *= 0.25 / Fmax
        b.V_vector_data = Fplot

        Dt = time()
        mp.brane_plot(
            b,
            # color_by_V_scalar=False,
            # color_by_V_rgb=True,
            show_halfedges=True,
            show_normals=False,
            show_V_vector_data=True,
            show_tangent1=False,
            show_tangent2=False,
            show=False,
            save=True,
            fig_path=fig_path,
            view=view,
        )
        Dtdict["brane_plot"] += time() - Dt
    image_count += 1
    while Trun - t > 0.5 * dt and success:
        while tstop - t > 0.5 * dt and success:
            try:
                # b.delaunay_regularize_by_flips()
                # Nflips = b.do_the_monte_flips()
                Nflips = 111
                # Fl = b.Ftether()
                # Fa, Fv = b.Fa_Fv()
                # Fb = b.Fbend_mixed()
                Dt = time()
                Fb = b.Fbend_mixed()
                Dtdict["Fbend_mixed"] += time() - Dt

                Dt = time()
                Fl = b.Ftether()
                Dtdict["Ftether"] += time() - Dt

                Dt = time()
                Fa, Fv = b.Fa_Fv()
                Dtdict["Fa_Fv"] += time() - Dt

                F = Fb + Fl + Fa + Fv

                Vold = b.V_pq[:, :3].copy()
                b.V_pq[:, :3] += dt * F / b.linear_drag_coeff

                Dt = time()
                Fb = b.Fbend_mixed()
                Dtdict["Fbend_mixed"] += time() - Dt

                Dt = time()
                Fl = b.Ftether()
                Dtdict["Ftether"] += time() - Dt

                Dt = time()
                Fa, Fv = b.Fa_Fv()
                Dtdict["Fa_Fv"] += time() - Dt

                F = Fb + Fl + Fa + Fv
                b.V_pq[:, :3] = Vold + dt * F / b.linear_drag_coeff

                success = True
            except Exception:
                success = False
            if success:
                # Vold = b.V_pq[:, :3].copy()
                # b.V_pq[:, :3] += dt * F / b.linear_drag_coeff
                # Fl = b.Ftether()
                # Fa, Fv = b.Fa_Fv()
                # Fb = b.Fbend_mixed()
                # F = Fb + Fl + Fa + Fv
                # b.V_pq[:, :3] = Vold + dt * F / b.linear_drag_coeff
                #
                #
                #
                # Vold = b.V_pq[:, :3].copy()
                # b.V_pq[:, :3] += b.weighted_drag_coeffs_step(dt)
                # b.V_pq[:, :3] = Vold + b.weighted_drag_coeffs_step(dt)

                t += dt

            else:
                print("oh no")

        if success:
            tstop = np.min([t + Tplot, Trun])
            t2save -= Tplot

            tt = np.round(t, int(-np.floor(np.log10(dt))))
            Fbmax = np.round(np.max(np.linalg.norm(Fb, np.inf, axis=0)), 3)
            Flmax = np.round(np.max(np.linalg.norm(Fl, np.inf, axis=0)), 3)
            Famax = np.round(np.max(np.linalg.norm(Fa, np.inf, axis=0)), 3)
            Fvmax = np.round(np.max(np.linalg.norm(Fv, np.inf, axis=0)), 3)
            # print(
            #     f"t={tt}, Fb={Fbmax},Fl={Flmax},Fa={Famax},Fv={Fvmax},Nflips={Nflips}              ",
            #     end="\r",
            # )
            Ttot = Dtdict["Fbend_mixed"] + Dtdict["Ftether"] + Dtdict["Fa_Fv"] + Dtdict["brane_plot"]
            T_Fbend_mixed = np.round(np.max(Dtdict["Fbend_mixed"] / Ttot), 3)
            T_Ftether = np.round(np.max(Dtdict["Ftether"] / Ttot), 3)
            T_Fa_Fv = np.round(np.max(Dtdict["Fa_Fv"] / Ttot), 3)
            T_brane_plot = np.round(np.max(Dtdict["brane_plot"] / Ttot), 3)
            log_str = f"t={tt}, Fb={Fbmax},Fl={Flmax},Fa={Famax},Fv={Fvmax},T_Fbend_mixed={T_Fbend_mixed},T_Ftether={T_Ftether},T_Fa_Fv={T_Fa_Fv},T_brane_plot={T_brane_plot}             "
            print(log_str, end="\r")

            sim_state["success"] = success
            sim_state["t"] = t
            sim_state["t2save"] = t2save
            sim_state["tstop"] = tstop
            sim_state["image_count"] = image_count

            # fig_kwargs = save_fig_data(sim_state)
            fig_path = f"{output_directory}/temp_images/fig_{image_count:0>4}.png"
            if make_plots:
                b.F_opacity = 0.8
                Fplot = 0.2 * F
                Fmax = np.max(np.linalg.norm(Fplot, np.inf, axis=0))
                # FFmax = .25
                if Fmax > 0.25:
                    print("\nFmax\n")
                    Fplot *= 0.25 / Fmax
                b.V_vector_data = Fplot
                Dt = time()
                mp.brane_plot(
                    b,
                    color_by_V_scalar=False,
                    # color_by_V_rgb=True,
                    show_halfedges=True,
                    show_normals=False,
                    show_V_vector_data=True,
                    show_tangent1=False,
                    show_tangent2=False,
                    show=False,
                    save=True,
                    fig_path=fig_path,
                    view=view,
                )
                Dtdict["brane_plot"] += time() - Dt
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


def round_to(x, n=3):
    if x == 0:
        return 0.0
    else:
        sgn_x = np.sign(x)
        abs_x = abs(x)
        return round(x, -int(np.floor(np.log10(abs_x))) + (n - 1))


def round_sci(x, n=3):
    return np.format_float_scientific(x, precision=n)


##########################################################
##########################################################
# Disretization fig
# %%
mesh_directory = "./data/halfedge_meshes/dumbbell_coarse"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    "length_reg_stiffness": 1e-9,
    "area_reg_stiffness": 1e-3,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e-1,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1e0,
} | mesh_data

b = m.Brane(**brane_kwargs)
# %%
_cm = get_cmap(name="nipy_spectral")
cm = lambda x: _cm(x)[:3]
black = cm(0)
purple = cm(0.1)
darkblue = cm(0.2)
lightblue = cm(0.3125)
green = cm(0.45)
yellow = cm(0.7)
orange = cm(0.8)
red = cm(0.9)
grey = cm(1)
white = (1.0, 1.0, 1.0)

b.V_radius = 0 * b.V_radius + 0.0125
b.F_alpha = 0 * b.F_alpha + 1.0

view = {
    "azimuth": 0,
    "elevation": 55,
    "distance": 3,
    "focalpoint": (0, 0, 0),
}
#
fig_path = "../helfrich_talk/images/dumbbell_smooth.png"
mp.brane_plot(
    b,
    show=False,
    save=True,
    fig_path=fig_path,
    figsize=(720, 720),
    show_surface=True,
    show_halfedges=False,
    show_edges=False,
    show_vertices=False,
    show_normals=False,
    show_tangent1=False,
    show_tangent2=False,
    show_plot_axes=False,
    color_by_V_rgb=False,
    color_by_V_scalar=False,
    color_by_F_scalar=False,
    show_V_vector_data=False,
    frame_scale=0.07,
    view=view,
)
#
fig_path = "../helfrich_talk/images/dumbbell_edges.png"
mp.brane_plot(
    b,
    show=False,
    save=True,
    fig_path=fig_path,
    figsize=(720, 720),
    show_surface=True,
    show_halfedges=False,
    show_edges=False,
    show_vertices=True,
    show_normals=False,
    show_tangent1=False,
    show_tangent2=False,
    show_plot_axes=False,
    color_by_V_rgb=False,
    color_by_V_scalar=False,
    color_by_F_scalar=False,
    show_V_vector_data=False,
    frame_scale=0.07,
    view=view,
)
#
fig_path = "../helfrich_talk/images/dumbbell_halfedges.png"
mp.brane_plot(
    b,
    show=False,
    save=True,
    fig_path=fig_path,
    figsize=(2180, 2180),
    show_surface=True,
    show_halfedges=True,
    show_edges=False,
    show_vertices=True,
    show_normals=False,
    show_tangent1=False,
    show_tangent2=False,
    show_plot_axes=False,
    color_by_V_rgb=False,
    color_by_V_scalar=False,
    color_by_F_scalar=False,
    show_V_vector_data=False,
    frame_scale=0.07,
    view=view,
)
##########################################################
##########################################################
# %%
# ply_path = "./data/ply_files/oblate.ply"
# vertices, faces = load_mesh_from_ply(ply_path)
# mesh_directory = "./data/halfedge_meshes/dumbbell"
mesh_directory = "./data/halfedge_meshes/dumbbell_coarse"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    "length_reg_stiffness": 1e-9,
    "area_reg_stiffness": 1e-3,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e-1,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1e0,
} | mesh_data
sim_kwargs = {
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "dt": 1e-3,
    "output_directory": "./output/monte_output",
} | brane_kwargs
sim_state = initialize_sim(**sim_kwargs)
# %%
Trun = 1
sim_state = run(
    sim_state,
    Trun,
    make_plots=True,
)
# %%
# movie_dir = sim_state["output_directory"] + "/temp_images"
movie_dir = sim_kwargs["output_directory"] + "/temp_images"
mp.movie(movie_dir)
# %%

b = sim_state["b"]
mp.brane_plot(
    b,
    # color_by_V_scalar=False,
    # color_by_V_rgb=True,
    show_halfedges=True,
    # show_normals=False,
    # show_V_vector_data=True,
)
# %%
mesh_directory = "./data/halfedge_meshes/dumbbell_coarse"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    "length_reg_stiffness": 1e-9,
    "area_reg_stiffness": 1e-3,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e-1,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1e0,
} | mesh_data

b = m.Brane(**brane_kwargs)


mp.brane_plot(
    b,
    show=True,
    save=False,
    fig_path=None,
    figsize=(2180, 2180),
    show_surface=True,
    show_halfedges=False,
    show_edges=False,
    show_vertices=False,
    show_normals=False,
    show_tangent1=False,
    show_tangent2=False,
    show_plot_axes=False,
    color_by_V_rgb=False,
    color_by_V_scalar=False,
    color_by_F_scalar=False,
    show_V_vector_data=False,
    frame_scale=0.07,
    view=None,
)
