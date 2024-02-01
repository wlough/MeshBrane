from numba import njit
import numpy as np
import src.model as m
from src.utils import load_mesh_from_ply, load_halfedge_mesh_data
import os
from src.pretty_pictures import mayavi_plots as mp

# from src.numdiff import jitnorm, jitcross, jitdot
import dill

# from copy import deepcopy
from matplotlib import colormaps as plt_cmap
import matplotlib.pyplot as plt


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


def smooth_Fbend_run(sim_state, make_plots=True, iters=20, weight=0.2):
    b = sim_state["b"]
    image_count = sim_state["image_count"]

    Nvertices = len(b.V_pq)

    vals = np.array([b.valence(v) for v in range(Nvertices)])
    valmin = min(vals)
    valmax = max(vals)
    vvalmin = list(vals).index(valmin)
    vvalmax = list(vals).index(valmax)
    valrgb = scalars_to_rgbs(vals)

    F = b.Fbend_dg()
    normF = np.linalg.norm(F, axis=1)
    Fmax = np.max(normF)
    vFmax = list(normF).index(Fmax)
    Frgb = scalars_to_rgbs(normF)

    H = b.get_mean_curvature_dg()
    Hmin = min(H)
    Hmax = max(H)
    vHmin = list(H).index(Hmin)
    vHmax = list(H).index(Hmax)
    Hrgb = scalars_to_rgbs(H)
    #
    b.V_rgb = valrgb.astype(np.float64)
    b.V_vector_data = F

    print(
        f"iter={-1} of {iters}, Fmax={Fmax}, Hmin={Hmin}, Hmax={Hmax}, valmin={valmin}, valmax={valmax}, L/L0={b.average_hedge_length()/b.preferred_edge_length}            ",
        end="\n",
    )

    if make_plots:
        fig_kwargs = save_fig_data(sim_state)
        # [show=True, save=False, fig_path=None, figsize=(2180, 2180), show_surface=True, show_halfedges=False, show_edges=False, show_vertices=False, show_normals=False, show_tangent1=False, show_tangent2=False, show_plot_axes=False, color_by_V_rgb=False, color_by_V_scalar=False, color_by_F_scalar=False, show_V_vector_data=False, frame_scale=0.07]
        # print(fig_kwargs.keys())
        # bfig_kwargs = {"show": False, "save": True, "fig_path": fig_kwargs["fig_path"]}
        # mp.plot_from_data(**fig_kwargs)
        mp.brane_plot(
            b,
            color_by_V_rgb=True,
            show_halfedges=True,
            show_V_vector_data=True,
            show=False,
            save=True,
            fig_path=fig_kwargs["fig_path"],
        )
        image_count += 1
    for iter in range(iters):
        F = b.smooth_samples(F, 0.1, 1)
        normF = np.linalg.norm(F, axis=1)
        Fmax = np.max(normF)
        vFmax = list(normF).index(Fmax)
        Frgb = scalars_to_rgbs(normF)

        H = b.smooth_samples(H, 0.1, 1)
        Hmin = min(H)
        Hmax = max(H)
        vHmin = list(H).index(Hmin)
        vHmax = list(H).index(Hmax)
        Hrgb = scalars_to_rgbs(H)

        b.V_rgb = valrgb
        b.V_vector_data = F

        print(
            f"iter={iter} of {iters}, Fmax={Fmax}, Hmin={Hmin}, Hmax={Hmax}            ",
            end="\n",
        )

        if make_plots:
            sim_state["image_count"] = image_count
            fig_kwargs = save_fig_data(sim_state)
            # mp.plot_from_data(**fig_kwargs)
            mp.brane_plot(
                b,
                color_by_V_rgb=True,
                show_halfedges=True,
                show_V_vector_data=True,
                show=False,
                save=True,
                fig_path=fig_kwargs["fig_path"],
            )
            image_count += 1

    return sim_state


# %%

# ply_path = "./data/ply_files/oblate.ply"
# vertices, faces = load_mesh_from_ply(ply_path)
# mesh_directory = "./data/halfedge_meshes/dumbbell"
mesh_directory = "./data/halfedge_meshes/dumbbell_fine"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    "length_reg_stiffness": 0 * 1e-1,
    "area_reg_stiffness": 0 * 1e-2,
    "volume_reg_stiffness": 0 * 1e-1,
    "bending_modulus": 1.0,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1.0,
} | mesh_data
sim_kwargs = {
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "dt": 1e-3,
    "output_directory": "./output/smoothing",
} | brane_kwargs
sim_state = initialize_sim(**sim_kwargs)
Trun = 1
# sim_state = smooth_Fbend_run(sim_state, make_plots=True, iters=2, weight=0.2)
# b = sim_state["b"]


# sim_state = run(sim_state, Trun, make_plots=True)
# sim_state = smooth_Fbend_run(sim_state, make_plots=True, iters=2, weight=0.2)
# b = sim_state["b"]
# movie_dir = sim_state["output_directory"] + "/temp_images"
# mp.movie(movie_dir)
# view=(45.0, 54.735610317245346, 4.472045526822957, array([ 3.45557928e-05, -2.15917826e-05,  1.52885914e-05]))


b = m.Brane(**brane_kwargs)
# %%

for azim in range(0, 360, 30):
    for elev in range(0, 90, 10):
        view = {
            "azimuth": azim,
            "elevation": elev,
            "distance": 3,
            "focalpoint": (0, 0, 0),
        }
        fig_path = f"./output/smoothing/{azim:0>4}_{elev:0>4}.png"
        mp.brane_plot(
            b,
            color_by_V_rgb=True,
            show_halfedges=False,
            show_V_vector_data=False,
            show=False,
            save=True,
            view=view,
            fig_path=fig_path,
        )

# %%
dt = 1e-3  # sim_state["dt"]
for iter in range(20):
    Fb = b.Fbend_dg()
    Fb = b.smooth_samples(Fb, 0.15, 20)
    b.V_pq[:, :3] = b.V_pq[:, :3] + dt * Fb
    b.vertices = b.V_pq[:, :3]
b.F_opacity = 1
fb = np.linalg.norm(Fb, axis=1)

b.V_rgb = scalars_to_rgbs(fb)
b.V_vector_data = Fb
mp.brane_plot(
    b,
    color_by_V_scalar=False,
    color_by_V_rgb=True,
    show_halfedges=True,
    show_normals=False,
    show_V_vector_data=True,
)
b.vertices - b1.vertices
# %%
dt = 1e-4  # sim_state["dt"]
for iter in range(1):
    Fb = b.Fbend_dg()
    b.smooth_samples(Fb, 0.25, 20)
    b.V_pq[:, :3] += dt * Fb

Fb = b.Fbend_dg()
fb = np.linalg.norm(Fb, axis=1)
fb_rgb = scalars_to_rgbs(fb)

# Fb2 = b.Fbend2()
# fb2 = np.linalg.norm(Fb2, axis=1)
# fb_rgb2 = scalars_to_rgbs(fb2)

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
Hdg = b.get_mean_curvature_dg()
# Hdg_smooth = Hdg.copy()
Hdg_smooth = b.smooth_samples(Hdg, 1, 10)


Hdg_rgb = scalars_to_rgbs(Hdg)
H, K = b.get_curvatures()

Hvec = b.get_mean_curvature_vector()
n = np.array([b.area_weighted_vertex_normal(v) for v, _ in enumerate(b.V_pq)])
Hdgvec = np.einsum("v,vi->vi", Hdg, n)
Hvec0 = np.einsum("v, vi->vi", H, n)
lapH = b.cotan_laplacian(H)
H_rgb = scalars_to_rgbs(H)
K_rgb = scalars_to_rgbs(K)
lapH_rgb = scalars_to_rgbs(lapH)

# b.V_vector_data = -Hvec0 * 1e-1
b.V_vector_data = Hdgvec

b.V_vector_data = b.smooth_samples(Fb, 0.1, 20)
# b.V_vector_data = Fa*1e2
# b.V_vector_data = Fv*1e4
# b.V_vector_data = Hvec0
b.V_rgb = fb_rgb
V_scalar = (K - min(K)) / (max(K) - min(K))

# V_scalar = b.smooth_samples(V_scalar, .25, 20)
b.V_scalar = V_scalar
b.F_opacity = 1

mp.brane_plot(
    b,
    color_by_V_scalar=False,
    color_by_V_rgb=True,
    show_halfedges=True,
    show_normals=False,
    show_V_vector_data=True,
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
