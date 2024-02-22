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


def save_brane_state(brane, output_directory):
    vertices = brane.V_pq[:, :3]
    faces = brane.faces
    length_reg_stiffness = brane.length_reg_stiffness
    area_reg_stiffness = brane.area_reg_stiffness
    volume_reg_stiffness = brane.volume_reg_stiffness
    bending_modulus = brane.bending_modulus
    splay_modulus = brane.splay_modulus
    spontaneous_curvature = brane.spontaneous_curvature
    linear_drag_coeff = brane.linear_drag_coeff
    V_hedge = brane.V_hedge
    halfedges = brane.halfedges
    H_vertex = brane.H_vertex
    H_face = brane.H_face
    H_next = brane.H_next
    H_prev = brane.H_prev
    H_twin = brane.H_twin
    F_hedge = brane.F_hedge


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


def run(
    sim_state,
    Trun,
    make_plots=True,
    Ncurvaturesmooth=50,
    Nbendingforcesmooth=50,
    Nvertexsmooth=1,
    smooth_length=None,
):
    view = {
        "azimuth": 0,
        "elevation": 55,
        "distance": 4,
        "focalpoint": (0, 0, 0),
    }
    b = sim_state["b"]
    if smooth_length is None:
        smooth_length = b.preferred_edge_length

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
        b.V_pq[:, :3] = b.gaussian_smooth_samples(
            b.V_pq[:, :3], Nvertexsmooth, smooth_length
        )
        H, K = b.get_angle_weighted_arc_curvatures()
        K = b.gaussian_smooth_samples(K, Ncurvaturesmooth, smooth_length)
        H = b.gaussian_smooth_samples(H, Ncurvaturesmooth, smooth_length)
        lapH = b.cotan_laplacian(H)
        Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
        Fn = b.gaussian_smooth_samples(Fn, Nbendingforcesmooth, smooth_length)
        Fb = np.array(
            [
                fn * b.other_weighted_vertex_normal(v) * b.vorcell_area(v)
                for v, fn in enumerate(Fn)
            ]
        )
        Fl = b.Flength()
        Fa = b.Farea()
        Fv = b.Fvolume()
        F = Fb + Fl + Fa + Fv
        # mp.plot_from_data(**fig_kwargs)
        b.V_rgb = scalars_to_rgbs(Fn)
        b.F_opacity = 0.8
        Fplot = 0.1 * F
        Fmax = np.max(np.linalg.norm(Fplot, np.inf, axis=0))
        # FFmax = .25
        if Fmax > 0.25:
            Fplot *= 0.25 / Fmax
        b.V_vector_data = Fplot
        mp.brane_plot(
            b,
            color_by_V_scalar=False,
            color_by_V_rgb=True,
            show_halfedges=True,
            show_normals=False,
            show_V_vector_data=True,
            show_tangent1=False,
            show_tangent2=False,
            show=False,
            save=True,
            fig_path=fig_kwargs["fig_path"],
            view=view,
        )
    image_count += 1

    while Trun - t > 0.5 * dt and success:
        # while t < tstop and success:
        # b.regularize_by_shifts(0.2)
        while tstop - t > 0.5 * dt and success:
            # b.forward_euler_step(dt)
            # vertices, success = b.get_new_euler_state(dt)

            success = True
            b.preferred_edge_length = b._average_hedge_length(
                b.V_pq[:, :3], b.H_twin, b.H_vertex
            )

            # b.total_volume = b.volume_of_mesh()
            # b.total_area = b.get_total_area()
            b.V_pq[:, :3] = b.gaussian_smooth_samples(
                b.V_pq[:, :3], Nvertexsmooth, smooth_length
            )
            H, K = b.get_angle_weighted_arc_curvatures()
            K = b.gaussian_smooth_samples(K, Ncurvaturesmooth, smooth_length)
            H = b.gaussian_smooth_samples(H, Ncurvaturesmooth, smooth_length)
            lapH = b.cotan_laplacian(H)
            Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
            Fn = b.gaussian_smooth_samples(Fn, Nbendingforcesmooth, smooth_length)
            n = np.array([b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)])
            Fb = np.array(
                [
                    fn * b.other_weighted_vertex_normal(v) * b.vorcell_area(v)
                    for v, fn in enumerate(Fn)
                ]
            )
            Fl = b.Flength()
            Fa = b.Farea()
            Fv = b.Fvolume()
            F = Fl + Fa
            F = np.array([fi - (fi @ ni) * ni for fi, ni in zip(F, n)])
            # F = np.array([fi - (fi @ ni) * ni for fi, ni in zip(F, n)])
            # Fv = np.array([(fi @ ni) * ni for fi, ni in zip(Fv, n)])
            F += Fv
            F += Fb
            if success:
                # b.V_pq[:, :3] = vertices
                b.V_pq[:, :3] += dt * F / b.linear_drag_coeff
                # b.vertices
                t += dt

            else:
                print("oh no")
        # b.V_pq[:, :3] = b.gaussian_smooth_samples(b.V_pq[:, :3], 1, smooth_length)
        try:
            Nflips = b.delaunay_regularize_by_flips()
            # Nflips = b.regularize_by_flips()
        except ZeroDivisionError:
            success = False
            print(f"\nZeroDivisionError at t={t}\n")

        b.preferred_cell_area = b._average_cell_area(
            b.V_pq[:, :3],
            b.V_hedge,
            b.H_vertex,
            b.H_prev,
            b.H_twin,
        )
        # Nflips = 0
        # if Nflips != 0:
        #     print(f"\n Nflips={Nflips} \n")
        if success:
            tstop = np.min([t + Tplot, Trun])
            t2save -= Tplot

            tt = np.round(t, int(-np.floor(np.log10(dt))))
            Fbmax = np.round(np.max(np.linalg.norm(Fb, np.inf, axis=0)), 3)
            Flmax = np.round(np.max(np.linalg.norm(Fl, np.inf, axis=0)), 3)
            Famax = np.round(np.max(np.linalg.norm(Fa, np.inf, axis=0)), 3)
            Fvmax = np.round(np.max(np.linalg.norm(Fv, np.inf, axis=0)), 3)
            print(
                f"t={tt}, Fb={Fbmax},Fl={Flmax},Fa={Famax},Fv={Fvmax}              ",
                end="\r",
            )

            sim_state["success"] = success
            sim_state["t"] = t
            # sim_state["t2plot"] = t2plot
            sim_state["t2save"] = t2save
            sim_state["tstop"] = tstop
            sim_state["image_count"] = image_count

            fig_kwargs = save_fig_data(sim_state)
            if make_plots:
                # mp.plot_from_data(**fig_kwargs)
                # fig_kwargs = {
                #     "V_pq": b.V_pq,
                #     "faces": b.faces,
                #     "V_rgb": b.V_rgb,
                #     "V_radius": b.V_radius,
                #     "H_rgb": b.H_rgb,
                #     "F_rgb": b.F_rgb,
                #     "F_opacity": b.F_opacity,
                #     "H_opacity": b.H_opacity,
                #     "V_opacity": b.V_opacity,
                #     "V_normal_rgb": b.V_normal_rgb,
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
                b.V_rgb = scalars_to_rgbs(Fn)
                b.F_opacity = 0.8
                Fplot = 0.2 * F
                Fmax = np.max(np.linalg.norm(Fplot, np.inf, axis=0))
                # FFmax = .25
                if Fmax > 0.25:
                    print("\nFmax\n")
                    Fplot *= 0.25 / Fmax
                b.V_vector_data = Fplot
                mp.brane_plot(
                    b,
                    color_by_V_scalar=False,
                    color_by_V_rgb=True,
                    show_halfedges=True,
                    show_normals=False,
                    show_V_vector_data=True,
                    show_tangent1=False,
                    show_tangent2=False,
                    show=False,
                    save=True,
                    fig_path=fig_kwargs["fig_path"],
                    view=view,
                )
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


def smooth_Fbend_run(sim_state, make_plots=True, iters=20, weight=0.2, Dazim=5):
    b = sim_state["b"]
    theta = np.pi / 2
    e0 = np.array([1, 0, 0, 0])
    u = np.array([0, 1, 0, 0])
    q = np.cos(theta / 2) * e0 + np.sin(theta / 2) * u
    p = np.array([0, 0, 0])
    pq = np.array([*p, *q])
    b.rigid_transform(pq)
    view = {
        "azimuth": 0,
        "elevation": 55,
        "distance": 4,
        "focalpoint": (0, 0, 0),
    }
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
        f"iter={-1} of {iters}, Fmax={Fmax}, Hmin={Hmin}, Hmax={Hmax}, valmin={valmin}, valmax={valmax}          ",
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
            view=view,
        )
        image_count += 1
    for iter in range(iters):
        F = b.smooth_samples(F, 0.1, 1)
        normF = np.linalg.norm(F, axis=1)
        Fmax = np.max(normF)
        vFmax = list(normF).index(Fmax)
        Frgb = scalars_to_rgbs(normF)

        H = b.smooth_samples(H, weight, 1)
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
        view = {
            "azimuth": iter * Dazim,
            "elevation": 55,
            "distance": 4,
            "focalpoint": (0, 0, 0),
        }

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
                view=view,
            )
            image_count += 1

    return sim_state


# %%
# b.preferred_cell_area = b._average_cell_area(
#     b.V_pq[:, :3],
#     b.V_hedge,
#     b.H_vertex,
#     b.H_prev,
#     b.H_twin,
# )
# sim_state["b"].Karea = 1e-2
# b.total_area = b.get_total_area()
# b.total_volume = b.volume_of_mesh()
# ply_path = "./data/ply_files/oblate.ply"
# vertices, faces = load_mesh_from_ply(ply_path)
# mesh_directory = "./data/halfedge_meshes/dumbbell"
mesh_directory = "./data/halfedge_meshes/oblate"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    "length_reg_stiffness": 1e-2,
    "area_reg_stiffness": 1e-3,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e1,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 5.0,
} | mesh_data
sim_kwargs = {
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "dt": 1e-3,
    "output_directory": "./output/ultrafine_output",
} | brane_kwargs
# sim_state = initialize_sim(**sim_kwargs)
# Trun = 15
# sim_state = run(
#     sim_state,
#     Trun,
#     make_plots=True,
#     Ncurvaturesmooth=40,
#     Nbendingforcesmooth=40,
#     Nvertexsmooth=0,
#     smooth_length=None,
# )  # smooth_Fbend_run(sim_state, make_plots=True, iters=20, weight=0.2)
# b = sim_state["b"]
# os.system("mkdir ./output")
# sim_state = run(sim_state, Trun, make_plots=True)
# sim_state = smooth_Fbend_run(sim_state, make_plots=True, iters=2, weight=0.2)
# b = sim_state["b"]
# movie_dir = sim_state["output_directory"] + "/temp_images"
# movie_dir = "./output/sim_output/temp_images"
# mp.movie(movie_dir)
# view=(45.0, 54.735610317245346, 4.472045526822957, array([ 3.45557928e-05, -2.15917826e-05,  1.52885914e-05]))


b = m.Brane(**brane_kwargs)
#
F = len(b.faces)
E = int(len(b.halfedges) / 2)
V = len(b.V_pq)
3 * F / 2 - E

# %%

# mp.brane_plot(
#     b,
#     # color_by_V_scalar=False,
#     # color_by_V_rgb=True,
#     show_halfedges=True,
#     # show_normals=False,
#     # show_V_vector_data=True,
# )


# %%
b = sim_state["b"]


def test_mesh(self):
    Nv = len(self.V_pq)
    for v in range(Nv):
        h_start = self.V_hedge[v]
        h = h_start
        v1 = self.H_vertex[h]
        h = self.H_twin[self.H_prev[h]]
        v2 = self.H_vertex[h]
        while True:
            v1 = self.H_vertex[h]
            h = self.H_twin[self.H_prev[h]]
            v2 = self.H_vertex[h]

            if v == v1 or v == v2 or v1 == v2:
                print(f"(v,v1,v2)={(v,v1,v2)} normu1=0")

            if h == h_start:
                break


# b.delaunay_regularize_by_flips()
test_mesh(b)
Nh = len(b.halfedges)
Nv = len(b.V_pq)
vals = [b.valence(v) for v in range(Nv)]
max(vals)
min(vals)
flips = [b.is_flippable(h) for h in range(Nh)]

for h in range(Nh):
    is_del = b.is_delaunay(h)
    is_flip = b.is_flippable(h)
    if not is_del:
        print(f"h={h} not delaunay        ", end="\r")
        if is_flip:
            b._edge_flip(h)
        else:
            print(f"h={h} not delaunay but not flippable        ", end="\n")
# H, K = b.get_angle_weighted_arc_curvatures()
b.valence(1086)


def get_curvatures(self):
    """ """
    Nv = self.V_pq.shape[0]
    H = np.zeros(Nv)
    K = np.zeros(Nv)

    for v in range(Nv):
        print(f"vertex {v}              ", end="\r")
        Atot = 0.0
        r = self.V_pq[v, :3]
        r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
        Hvec = np.zeros(3)
        n = np.zeros(3)
        defect = 2 * np.pi
        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_twin[self.H_prev[h]]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]

            r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
            r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
            r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
            r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
            r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

            normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
            normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
            normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
            if normu1 == 0:
                print(f"(v,v1,v2)={(v,v1,v2)} normu1=0")
            if normu2 == 0:
                print(f"(v,v1,v2)={(v,v1,v2)} normu2=0")
            if normu3 == 0:
                print(f"(v,v1,v2)={(v,v1,v2)} normu3=0")
            cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
            cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
            cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
            cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
            cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)

            defect -= np.arccos(cos_gamma)
            Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
            Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
            n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)

            if h == h_start:
                break

        Hvec /= 2 * Atot
        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
        K[v] = defect / Atot
    return H, K


H, K = get_curvatures(b)
plt.plot(K)
lapH = b.cotan_laplacian(H)
Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
Fb = np.array([fn * b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)])
Fl = b.Flength()
Fa = b.Farea()
Fv = b.Fvolume()
v = 912
h_start = b.V_hedge[v]
h = h_start
v1 = b.H_vertex[h]
h = b.H_twin[b.H_prev[h]]
v2 = b.H_vertex[h]
while True:
    v1 = b.H_vertex[h]
    r1 = b.V_pq[v1, :3]
    h = b.H_twin[b.H_prev[h]]
    v2 = b.H_vertex[h]
    r2 = b.V_pq[v2, :3]

    r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

    normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
    normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
    normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
    if normu1 == 0:
        print(f"(v,v1,v2)={(v,v1,v2)} normu1=0")
    if normu2 == 0:
        print(f"(v,v1,v2)={(v,v1,v2)} normu2=0")
    if normu3 == 0:
        print(f"(v,v1,v2)={(v,v1,v2)} normu3=0")
    cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
    cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)

    # defect -= np.arccos(cos_gamma)
    # Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
    # Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    # n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)

    if h == h_start:
        break


# %%
def is_delaunay(self, h):
    r"""
      v2
      /|\
    v3 | v1
      \|/
       v4
    """
    vi = self.H_vertex[self.H_next[self.H_twin[h]]]
    vj = self.H_vertex[h]
    vk = self.H_vertex[self.H_next[h]]
    vl = self.H_vertex[self.H_prev[h]]

    pij = self.V_pq[vj, :3] - self.V_pq[vi, :3]
    pil = self.V_pq[vl, :3] - self.V_pq[vi, :3]
    pkj = self.V_pq[vj, :3] - self.V_pq[vk, :3]
    pkl = self.V_pq[vl, :3] - self.V_pq[vk, :3]

    pij_pil = pij[0] * pil[0] + pij[1] * pil[1] + pij[2] * pil[2]
    pkl_pkj = pkl[0] * pkj[0] + pkl[1] * pkj[1] + pkl[2] * pkj[2]
    normpij = np.sqrt(pij[0] ** 2 + pij[1] ** 2 + pij[2] ** 2)
    normpil = np.sqrt(pil[0] ** 2 + pil[1] ** 2 + pil[2] ** 2)
    normpkj = np.sqrt(pkj[0] ** 2 + pkj[1] ** 2 + pkj[2] ** 2)
    normpkl = np.sqrt(pkl[0] ** 2 + pkl[1] ** 2 + pkl[2] ** 2)
    # print()

    alphai = np.arccos(pij_pil / (normpij * normpil))
    alphak = np.arccos(pkl_pkj / (normpkl * normpkj))

    return alphai + alphak <= np.pi


def delaunay_regularize_by_flips(self):
    Nh = len(self.halfedges)
    Nflips = 0
    for h in range(Nh):
        # print("is_del                  ", end="\r")
        is_del = is_delaunay(b, h)
        if not is_del:
            # print("eflip             ", end="\r")
            self.edge_flip(h)
            Nflips += 1
    return Nflips


def cotan_laplacian(self, Y):
    """computes the laplacian of Y at each vertex"""
    Nv = self.V_pq.shape[0]
    lapY = np.zeros_like(Y)
    for vi in range(Nv):
        Atot = 0.0
        ri = self.V_pq[vi, :3]
        yi = Y[vi]
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        h_start = self.V_hedge[vi]
        hij = h_start
        while True:
            hijm1 = self.H_next[self.H_twin[hij]]
            hijp1 = self.H_twin[self.H_prev[hij]]
            vjm1 = self.H_vertex[hijm1]
            vj = self.H_vertex[hij]
            vjp1 = self.H_vertex[hijp1]

            yj = Y[vj]

            rjm1 = self.V_pq[vjm1, :3]
            rj = self.V_pq[vj, :3]
            rjp1 = self.V_pq[vjp1, :3]

            rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
            rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
            rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
            ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
            ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
            rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
            ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
            rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

            Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
            Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
            Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
            Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
            Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

            cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

            cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

            cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
            cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)
            if cot_thetam is np.nan:
                print("cot_thetam")
            if cot_thetap is np.nan:
                print("cot_thetap")

            Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
            lapY[vi] += (cot_thetam + cot_thetap) * (yj - yi) / 2

            hij = hijp1

            if hij == h_start:
                break
        lapY[vi] /= Atot

    return lapY


b = m.Brane(**brane_kwargs)
smooth_length = b.preferred_edge_length
Ncurvaturesmooth = 10
Nbendingforcesmooth = 10
weight = 0.2
dt = 1e-2
T = 1e-2
t = 0
while t <= T:
    print(f"t={t/T}    ", end="\r")
    # b.V_pq[:, :3] = b.gaussian_smooth_samples(b.V_pq[:,:3], 1, smooth_length)
    b.preferred_edge_length = b._average_hedge_length(
        b.V_pq[:, :3], b.H_twin, b.H_vertex
    )

    b.total_volume = b.volume_of_mesh()
    b.total_area = b.get_total_area()
    H, K = b.get_angle_weighted_arc_curvatures()
    K = b.gaussian_smooth_samples(K, Ncurvaturesmooth, smooth_length)
    H = b.gaussian_smooth_samples(H, Ncurvaturesmooth, smooth_length)
    lapH = b.cotan_laplacian(H)
    Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
    Fn = b.gaussian_smooth_samples(Fn, Nbendingforcesmooth, smooth_length)
    Fb = np.array([fn * b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)])
    Fl = b.Flength()
    Fa = b.Farea()
    Fv = b.Fvolume()

    # Fl =  b.gaussian_smooth_samples(Fl, 0, smooth_length)
    F = Fb + Fl + Fa + Fv
    b.V_pq[:, :3] += dt * F / b.linear_drag_coeff
    t += dt
    # b.delaunay_regularize_by_flips()
    # delaunay_regularize_by_flips(b)
#
# for _ in range(1):
# print(f"iter={_}    ", end="\r")
# # b.V_pq[:, :3] = b.gaussian_smooth_samples(b.V_pq[:,:3], 1, smooth_length)
# b.preferred_edge_length = b._average_hedge_length(
#     b.V_pq[:, :3], b.H_twin, b.H_vertex
# )
#
# b.total_volume = b.volume_of_mesh()
# b.total_area = b.get_total_area()
# H, K = b.get_angle_weighted_arc_curvatures()
# K = b.gaussian_smooth_samples(K, Ncurvaturesmooth, smooth_length)
# H = b.gaussian_smooth_samples(H, Ncurvaturesmooth, smooth_length)
# lapH = cotan_laplacian(b,H)
# Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
# Fn = b.gaussian_smooth_samples(Fn, Nbendingforcesmooth, smooth_length)
# Fb = np.array([fn * b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)])
# Fl = b.Flength()
# Fa = b.Farea()
# Fv = b.Fvolume()

# Fl =  b.gaussian_smooth_samples(Fl, 0, smooth_length)
# F = Fb + Fl + Fa + Fv
# b.V_pq[:, :3] += dt * F / b.linear_drag_coeff
# b.delaunay_regularize_by_flips()
# delaunay_regularize_by_flips(b)
#
#
b.V_rgb = scalars_to_rgbs(Fn)
b.F_opacity = 0.8
Fplot = F
Fmax = np.max(np.linalg.norm(Fplot, np.inf, axis=0))
b.V_vector_data = 0.1 * Fplot / Fmax
mp.brane_plot(
    b,
    color_by_V_scalar=False,
    color_by_V_rgb=True,
    show_halfedges=False,
    show_normals=False,
    show_V_vector_data=True,
    show_tangent1=False,
    show_tangent2=False,
)


# %%
def runrunrun():
    b = m.Brane(**brane_kwargs)
    smooth_length = b.preferred_edge_length
    Ncurvaturesmooth = 40
    Nbendingforcesmooth = 40
    weight = 0.2
    dt = 1e-2
    for _ in range(20):
        H, K = b.get_angle_weighted_arc_curvatures()
        H = b.smooth_samples(H, weight, Ncurvaturesmooth)
        K = b.smooth_samples(K, weight, Ncurvaturesmooth)
        lapH = b.cotan_laplacian(H)
        # vecH = b.cotan_laplacian(b.V_pq[:, :3])
        # Hn = np.array([0.5 * hn @ b.quat_normal_vector(v) for v, hn in enumerate(vecH)])
        Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
        Fn = b.smooth_samples(Fn, 0.2, Nbendingforcesmooth)
        Fbend = np.array(
            [fn * b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)]
        )
        b.V_pq[:, :3] += dt * Fbend
        b.delaunay_regularize_by_flips()

    b.V_rgb = scalars_to_rgbs(Fn)
    b.F_opacity = 0.8
    b.V_vector_data = 0.1 * Fbend
    # b.V_vector_data = b.smooth_samples(b.V_vector_data, 0.2, 200)
    mp.brane_plot(
        b,
        color_by_V_scalar=False,
        color_by_V_rgb=True,
        show_halfedges=True,
        show_normals=False,
        show_V_vector_data=True,
        show_tangent1=False,
        show_tangent2=False,
    )
