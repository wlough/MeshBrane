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


def run0(
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
            Nflips = b.delaunay_regularize_by_flips()
            success = True
            b.preferred_edge_length = b._average_hedge_length(
                b.V_pq[:, :3], b.H_twin, b.H_vertex
            )

            # b.total_volume = b.volume_of_mesh()
            # b.total_area = b.get_total_area()
            # b.V_pq[:, :3] = b.gaussian_smooth_samples(
            #     b.V_pq[:, :3], Nvertexsmooth, smooth_length
            # )
            H, K = b.get_angle_weighted_arc_curvatures()
            K = b.gaussian_smooth_samples(K, Ncurvaturesmooth, smooth_length)
            H = b.gaussian_smooth_samples(H, Ncurvaturesmooth, smooth_length)
            lapH = b.cotan_laplacian(H)
            Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
            Fn = b.gaussian_smooth_samples(Fn, Nbendingforcesmooth, smooth_length)
            n = np.array([b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)])
            Fb = np.array(
                [fn * b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)]
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
        # try:
        # Nflips = b.delaunay_regularize_by_flips()
        #     # Nflips = b.regularize_by_flips()
        # except ZeroDivisionError:
        #     success = False
        #     print(f"\nZeroDivisionError at t={t}\n")

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
                f"t={tt}, Fb={Fbmax},Fl={Flmax},Fa={Famax},Fv={Fvmax},Nflips={Nflips}             ",
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


def run(sim_state, Trun, make_plots=True):
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
        Fb = b.Fbend_mixed()
        # Fl = b.Flength()
        Fl = b.Ftether()
        Fa, Fv = b.Fa_Fv()
        F = Fb + Fl + Fa + Fv
        b.F_opacity = 0.8
        Fplot = 0.1 * F
        Fmax = np.max(np.linalg.norm(Fplot, np.inf, axis=0))
        if Fmax > 0.25:
            Fplot *= 0.25 / Fmax
        b.V_vector_data = Fplot
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
    image_count += 1

    while Trun - t > 0.5 * dt and success:
        while tstop - t > 0.5 * dt and success:
            try:
                b.delaunay_regularize_by_flips()
                # Fl = b.Flength()
                Fl = b.Ftether()
                Fa, Fv = b.Fa_Fv()
                # Fb = b.Fbend_mixed()
                #############################
                #############################
                smooth_length = b.preferred_edge_length
                Nvertexsmooth = 1
                Ncurvaturesmooth = 150
                Nbendingforcesmooth = 50
                b.V_pq[:, :3] = b.gaussian_smooth_samples(
                    b.V_pq[:, :3], Nvertexsmooth, smooth_length
                )
                H, K = b.get_angle_weighted_arc_curvatures()
                K = b.gaussian_smooth_samples(K, Ncurvaturesmooth, smooth_length)
                H = b.gaussian_smooth_samples(H, Ncurvaturesmooth, smooth_length)
                lapH = b.cotan_laplacian(H)
                Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
                Fn = b.gaussian_smooth_samples(Fn, Nbendingforcesmooth, smooth_length)
                # n = np.array([b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)])
                Fb = np.array(
                    [fn * b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)]
                )
                #############################
                #############################
                F = Fb + Fl + Fa + Fv
                success = True
            except Exception:
                success = False
            if success:
                b.V_pq[:, :3] += dt * F / b.linear_drag_coeff
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
            print(
                f"t={tt}, Fb={Fbmax},Fl={Flmax},Fa={Famax},Fv={Fvmax}              ",
                end="\r",
            )

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


def color_by_tether(b, _a=None):
    Nh = len(b.halfedges)
    U = np.zeros(Nh)
    for h in range(Nh):
        i, j = b.halfedges[h]
        ri, rj = b.V_pq[i, :3], b.V_pq[j, :3]
        Drij = rj - ri
        s = np.sqrt(Drij @ Drij)
        U[h] = b.Utether(s, _a)
    return U


#
# l0 = b.preferred_edge_length
#
# s = np.linspace(.6*l0, 1.4*l0, 5000)[1:-1]
# a = np.array([l0*10**(p) for p in [-3,-2,-1, 0, 1]])
# # a = np.linspace(l0*10**-1, l0, 3)
# # U = np.array([[b.Utether(si, _a) for si in s] for _a in a])
# for p in range(len(a)):
#     ap = a[p]
#     pp = np.log10(ap/l0)
#     Up = np.array([b.Utether(si, ap) for si in s])
#     plt.plot(s/l0, Up, "--", label=f"a/l0=1e{pp}")
# plt.legend()
# plt.ylim(-.01, .1)
# plt.show()
# plt.close()
# %%
# ply_path = "./data/ply_files/oblate.ply"
# vertices, faces = load_mesh_from_ply(ply_path)
# mesh_directory = "./data/halfedge_meshes/dumbbell"
mesh_directory = "./data/halfedge_meshes/dumbbell_coarse"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    # "length_reg_stiffness": 1e-6,
    # "area_reg_stiffness": 1e-3,
    # "volume_reg_stiffness": 1e1,
    # "bending_modulus": 1e1,
    # "splay_modulus": 1.0,
    # "spontaneous_curvature": 0.0,
    # "linear_drag_coeff": 1e1,
    "length_reg_stiffness": 1e-1,
    "area_reg_stiffness": 1e-1,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e0,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1e1,
} | mesh_data
sim_kwargs = {
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "dt": 1e-2,
    "output_directory": "./output/sim_output",
} | brane_kwargs
sim_state = initialize_sim(**sim_kwargs)
Trun = 0.5
# sim_state = run0(
#     sim_state,
#     Trun,
#     make_plots=True,
# )
# b = sim_state["b"]
# os.system("mkdir ./output")
# sim_state = run(sim_state, Trun, make_plots=True)
# sim_state = smooth_Fbend_run(sim_state, make_plots=True, iters=2, weight=0.2)
b = sim_state["b"]

# movie_dir = sim_state["output_directory"] + "/temp_images"
# # movie_dir = "./output/sim_output/temp_images"
# mp.movie(movie_dir)
# view=(45.0, 54.735610317245346, 4.472045526822957, array([ 3.45557928e-05, -2.15917826e-05,  1.52885914e-05]))

#
b = m.Brane(**brane_kwargs)

mp.brane_plot(
    b,
    # color_by_V_scalar=False,
    # color_by_V_rgb=True,
    show_halfedges=True,
    # show_normals=False,
    # show_V_vector_data=True,
)


# %%
def broke_del(self, h):
    r"""
    checks if edge is locally delaunay
      vj
      /|\
    vk | vi
      \|/
       vl
    """
    vi = self.H_vertex[self.H_next[self.H_twin[h]]]
    vj = self.H_vertex[h]
    vk = self.H_vertex[self.H_next[h]]
    vl = self.H_vertex[self.H_prev[h]]

    # pij = self.V_pq[vj, :3] - self.V_pq[vi, :3]
    # pil = self.V_pq[vl, :3] - self.V_pq[vi, :3]
    # pkj = self.V_pq[vj, :3] - self.V_pq[vk, :3]
    # pkl = self.V_pq[vl, :3] - self.V_pq[vk, :3]
    #
    # # pij_pil = pij[0] * pil[0] + pij[1] * pil[1] + pij[2] * pil[2]
    # # pkl_pkj = pkl[0] * pkj[0] + pkl[1] * pkj[1] + pkl[2] * pkj[2]
    # # normpij = np.sqrt(pij[0] ** 2 + pij[1] ** 2 + pij[2] ** 2)
    # # normpil = np.sqrt(pil[0] ** 2 + pil[1] ** 2 + pil[2] ** 2)
    # # normpkj = np.sqrt(pkj[0] ** 2 + pkj[1] ** 2 + pkj[2] ** 2)
    # # normpkl = np.sqrt(pkl[0] ** 2 + pkl[1] ** 2 + pkl[2] ** 2)
    if vi == vj:
        print(f"oh no... h={h}, vi=vj={vi}")
        bad = True
    if vi == vl:
        print(f"oh no... h={h}, vi=vl={vi}")
        bad = True
    if vk == vj:
        print(f"oh no... h={h}, vk=vj={vk}")
        bad = True
    if vk == vl:
        print(f"oh no... h={h}, vk=vl={vk}")
        bad = True
    if all([vi != vj, vi != vl, vk != vj, vk != vl]):
        bad = False

    return bad


def make_bs():
    mesh_directory = "./data/halfedge_meshes/dumbbell_coarse"
    mesh_data = load_halfedge_mesh_data(mesh_directory)
    brane_kwargs = {
        "length_reg_stiffness": 1e-1,
        "area_reg_stiffness": 1e-1,
        "volume_reg_stiffness": 1e1,
        "bending_modulus": 1e0,
        "splay_modulus": 1.0,
        "spontaneous_curvature": 0.0,
        "linear_drag_coeff": 1e1,
    } | mesh_data
    b0 = m.Brane(**brane_kwargs)
    b1 = m.Brane(**brane_kwargs)
    b2 = m.Brane(**brane_kwargs)

    Vflip = [0, 1, 2, 3]
    max_iters = 2
    weight = 0.5
    Nreg = 3
    iters = 0
    go = True
    while go and (iters <= max_iters):
        Nflips = 0
        for vm in Vflip:
            hm = b0.V_hedge[vm]
            if b1.is_flippable(hm) and go:
                b0.edge_flip(hm)
                b1.edge_flip(hm)
                b2.edge_flip(hm)
                for ii in range(Nreg):
                    b0.regularize_by_shifts(weight)
                    b1.regularize_by_shifts(weight)
                    b2.regularize_by_shifts(weight)
                Nflips += 1
                for h, _ in enumerate(b1.halfedges):
                    bad = broke_del(b1, h)
                    if bad:
                        go = False

        iters += 1
    Vflip = [1, 2]
    for vm in Vflip:
        hm = b1.V_hedge[vm]
        if b1.is_flippable(hm) and go:
            v01, v02 = b1.halfedges[hm]
            b1.edge_flip(hm)
            b2.edge_flip(hm)
            v11, v12 = b1.halfedges[hm]
            print(f"b0->b1  ---  flip h={hm}, e={(v01,v02)}-->e={(v11,v12)}")
            for ii in range(Nreg):
                b0.regularize_by_shifts(weight)
                b1.regularize_by_shifts(weight)
                b2.regularize_by_shifts(weight)
            Nflips += 1
            for h, _ in enumerate(b1.halfedges):
                bad = broke_del(b1, h)
                if bad:
                    go = False

    Vflip = [2]
    for vm in Vflip:
        hm = b2.V_hedge[vm]
        if b2.is_flippable(hm) and go:
            v01, v02 = b2.halfedges[hm]
            b2.edge_flip(hm)
            v11, v12 = b2.halfedges[hm]
            print(f"b1->b2  ---  flip h={hm}, e={(v01,v02)}-->e={(v11,v12)}")
            for ii in range(Nreg):
                b0.regularize_by_shifts(weight)
                b1.regularize_by_shifts(weight)
                b2.regularize_by_shifts(weight)
            Nflips += 1
            # print(f"iters={iters}, Nflips={Nflips}, hm={hm}, vm={vm}")
            for h, _ in enumerate(b2.halfedges):
                bad = broke_del(b2, h)
                if bad:
                    go = False
                    print(f"broke on flip vm={vm}, hm={hm}")
        return b0, b1, b2  # ,V_pq_OG = b0.V_pq.copy()


b0, b1, b2 = make_bs()
V_pq_OG = b0.V_pq.copy()


# %%
def make_bs0():
    mesh_directory = "./data/halfedge_meshes/dumbbell_coarse"
    mesh_data = load_halfedge_mesh_data(mesh_directory)
    brane_kwargs = {
        "length_reg_stiffness": 1e-1,
        "area_reg_stiffness": 1e-1,
        "volume_reg_stiffness": 1e1,
        "bending_modulus": 1e0,
        "splay_modulus": 1.0,
        "spontaneous_curvature": 0.0,
        "linear_drag_coeff": 1e1,
    } | mesh_data
    b0 = m.Brane(**brane_kwargs)
    b1 = m.Brane(**brane_kwargs)
    b2 = m.Brane(**brane_kwargs)

    Vflip = [0, 1, 2, 3]
    max_iters = 2
    weight = 0.5
    Nreg = 3
    iters = 0
    go = True
    while go and (iters <= max_iters):
        Nflips = 0
        for vm in Vflip:
            hm = b0.V_hedge[vm]
            if b1.is_flippable(hm) and go:
                b0.edge_flip(hm)
                b1.edge_flip(hm)
                b2.edge_flip(hm)
                for ii in range(Nreg):
                    b0.regularize_by_shifts(weight)
                    b1.regularize_by_shifts(weight)
                    b2.regularize_by_shifts(weight)
                Nflips += 1
                # for h, _ in enumerate(b1.halfedges):
                #     bad = broke_del(b1, h)
                #     if bad:
                #         go = False

        iters += 1
    Vflip = [1, 2]
    for vm in Vflip:
        hm = b1.V_hedge[vm]
        if b1.is_flippable(hm) and go:
            v01, v02 = b1.halfedges[hm]
            b1.edge_flip(hm)
            b2.edge_flip(hm)
            v11, v12 = b1.halfedges[hm]
            print(f"b0->b1  ---  flip h={hm}, e={(v01,v02)}-->e={(v11,v12)}")
            for ii in range(Nreg):
                b0.regularize_by_shifts(weight)
                b1.regularize_by_shifts(weight)
                b2.regularize_by_shifts(weight)
            Nflips += 1
            # for h, _ in enumerate(b1.halfedges):
            #     bad = broke_del(b1, h)
            #     if bad:
            #         go = False

    Vflip = [2]
    for vm in Vflip:
        hm = b2.V_hedge[vm]
        if b2.is_flippable(hm) and go:
            v01, v02 = b2.halfedges[hm]
            b2.edge_flip(hm)
            v11, v12 = b2.halfedges[hm]
            print(f"b1->b2  ---  flip h={hm}, e={(v01,v02)}-->e={(v11,v12)}")
            for ii in range(Nreg):
                b0.regularize_by_shifts(weight)
                b1.regularize_by_shifts(weight)
                b2.regularize_by_shifts(weight)
            Nflips += 1
            # print(f"iters={iters}, Nflips={Nflips}, hm={hm}, vm={vm}")
            # for h, _ in enumerate(b2.halfedges):
            #     bad = broke_del(b2, h)
            #     if bad:
            #         go = False
            #         print(f"broke on flip vm={vm}, hm={hm}")
        return b0, b1, b2  # ,V_pq_OG = b0.V_pq.copy()


b0, b1, b2 = make_bs0()
V_pq_OG = b0.V_pq.copy()
# %%
# black = plt_cmap["nipy_spectral"]
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


# colors = [black, purple, darkblue, lightblue, green, yellow, orange, red, grey, white]
# names = [
#     "black",
#     "purple",
#     "darkblue",
#     "lightblue",
#     "green",
#     "yellow",
#     "orange",
#     "red",
#     "grey",
#     "white",
# ]
#
# for i in range(len(colors)):
#     ci = colors[i]
#     ni = names[i]
#     plt.plot([i], [0], "o", color=ci, markersize=10, label=ni)
# plt.legend()
# plt.show()
# plt.close()
# plt_cmap["nipy_spectral"]
# %%
def show_before():
    b0, b1, b2 = make_bs0()
    V_pq_OG = b0.V_pq.copy()
    b = b0
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

    V_radius = np.zeros(len(b.V_pq))
    F_rgb = np.ones((len(b.faces), 3))
    H_rgb = np.ones((len(b.halfedges), 3))
    V_rgb = np.ones((len(b.V_pq), 3))
    F_alpha = 0.05 * np.ones(len(b.faces))
    H_alpha = 0.05 * np.ones(len(b.halfedges))
    V_alpha = 0.05 * np.ones(len(b.V_pq))
    hm1, hm2, hm3 = 290, 297, 34
    hms = [hm1, hm2, hm3]
    vm1, vm2, vm3, vm4, vm5 = 56, 16, 1, 2, 15
    vms = [vm1, vm2, vm3, vm4, vm5]
    htm1, htm2, htm3 = [b.H_twin[hh] for hh in hms]

    h = hm1
    ht = b.H_twin[h]
    h1 = b.H_next[h]
    h2 = b.H_prev[h]
    h3 = b.H_next[ht]
    h4 = b.H_prev[ht]
    f1 = b.H_face[h]
    f2 = b.H_face[ht]
    v1 = b.H_vertex[h4]
    v2 = b.H_vertex[h1]
    v3 = b.H_vertex[h2]
    v4 = b.H_vertex[h3]
    ##
    ht1 = b.H_twin[h1]
    ht2 = b.H_twin[h2]
    ht3 = b.H_twin[h3]
    ht4 = b.H_twin[h4]
    v12 = b.H_vertex[b.H_next[ht1]]
    v23 = b.H_vertex[b.H_next[ht2]]
    v34 = b.H_vertex[b.H_next[ht3]]
    v41 = b.H_vertex[b.H_next[ht4]]
    print(f"{(v12,v23, v34,v41)}")
    f12 = b.H_face[ht1]
    f23 = b.H_face[ht2]
    f34 = b.H_face[ht3]
    f41 = b.H_face[ht4]

    for vv in vms:
        V_radius[vv] = 0.0025
        V_alpha[vv] = 1
    V_rgb[vm1], V_rgb[vm3] = black, purple
    V_rgb[vm2], V_rgb[vm4] = darkblue, lightblue
    V_rgb[vm5] = grey
    H_alpha[hm1] = 0.8
    H_rgb[hm1] = red

    F_alpha[f1] = 0.8
    F_rgb[f1] = green
    F_alpha[f2] = 0.8
    F_rgb[f2] = green

    F_alpha[f1] = 0.8
    F_rgb[f1] = green
    F_alpha[f2] = 0.8
    F_rgb[f2] = green

    # b.V_pq[vm2, :3] = V_pq_OG[vm2, :3] + .5*(V_pq_OG[vm1, :3]-V_pq_OG[vm2, :3])
    r13 = (b.V_pq[vm1, :3] + b.V_pq[vm3, :3]) / 2
    view = {
        "azimuth": 180,
        "elevation": 90,
        "distance": 0.25,
        "focalpoint": r13,
    }

    b.V_radius = V_radius
    b.F_rgb = F_rgb
    b.H_rgb = H_rgb
    b.V_rgb = V_rgb
    b.F_alpha = F_alpha
    b.H_alpha = H_alpha
    b.V_alpha = V_alpha
    mp.brane_plot(
        b,
        # color_by_V_scalar=False,
        # color_by_V_rgb=True,
        show_halfedges=True,
        show_vertices=True,
        # show_normals=False,
        # show_V_vector_data=True,
        view=view,
    )


def show_after1():
    b0, b1, b2 = make_bs0()
    V_pq_OG = b0.V_pq.copy()
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
    b = b0
    V_radius = np.zeros(len(b.V_pq))
    F_rgb = np.ones((len(b.faces), 3))
    H_rgb = np.ones((len(b.halfedges), 3))
    V_rgb = np.ones((len(b.V_pq), 3))
    F_alpha = 0.05 * np.ones(len(b.faces))
    H_alpha = 0.05 * np.ones(len(b.halfedges))
    V_alpha = 0.05 * np.ones(len(b.V_pq))
    hm1, hm2, hm3 = 290, 297, 34
    vm1, vm2, vm3, vm4, vm5 = 56, 16, 1, 2, 15
    vms = [vm1, vm2, vm3, vm4, vm5]
    for vv in vms:
        V_radius[vv] = 0.0025
        V_alpha[vv] = 1
    V_rgb[vm1], V_rgb[vm3] = black, black
    V_rgb[vm2], V_rgb[vm4] = purple, purple
    V_rgb[vm5] = grey
    H_alpha[hm1] = 0.8
    H_rgb[hm1] = red
    r13 = (b.V_pq[vm1, :3] + b.V_pq[vm3, :3]) / 2
    view = {
        "azimuth": 180,
        "elevation": 90,
        "distance": 0.25,
        "focalpoint": r13,
    }

    b.V_radius = V_radius
    b.F_rgb = F_rgb
    b.H_rgb = H_rgb
    b.V_rgb = V_rgb
    b.F_alpha = F_alpha
    b.H_alpha = H_alpha
    b.V_alpha = V_alpha
    mp.brane_plot(
        b,
        # color_by_V_scalar=False,
        # color_by_V_rgb=True,
        show_halfedges=True,
        show_vertices=True,
        # show_normals=False,
        # show_V_vector_data=True,
        view=view,
    )


show_before()
# %%
b0, b1, b2 = make_bs0()
V_pq_OG = b0.V_pq.copy()
b = b0
# b.edge_flip(h)
h = 290
ht = b.H_twin[h]
h1 = b.H_next[h]
h2 = b.H_prev[h]
h3 = b.H_next[ht]
h4 = b.H_prev[ht]
f1 = b.H_face[h]
f2 = b.H_face[ht]
v1 = b.H_vertex[h4]
v2 = b.H_vertex[h1]
v3 = b.H_vertex[h2]
v4 = b.H_vertex[h3]
##
ht1 = b.H_twin[h1]
ht2 = b.H_twin[h2]
ht3 = b.H_twin[h3]
ht4 = b.H_twin[h4]
f12 = b.H_face[ht1]
f23 = b.H_face[ht2]
f34 = b.H_face[ht3]
f41 = b.H_face[ht4]

for vv in [v1, v2, v3, v4]:
    Vradius[vv] = 0.0025
    V_alpha[vv] = 1
for hh in [h1, h2, h3, h4]:
    H_rgb[hh] = np.array([0, 0, 1])
for hh in [h, ht, h1, h2, h3, h4]:
    H_alpha[hh] = 0.8
for ff in [f1, f2, f12, f23, f34, f41]:
    F_alpha[ff] = 0.8
    F_rgb[ff] = np.array([0, 1, 0])
V_rgb[v1] = np.array([1, 0, 0])
V_rgb[v3] = np.array([1, 0.5, 0])
V_rgb[v2] = np.array([0, 0, 1])
V_rgb[v4] = np.array([0.5, 0, 1])
F_rgb[f1] = np.array([1, 0, 0])
F_rgb[f2] = np.array([0, 0, 1])
H_rgb[h] = np.array([1, 0, 0])
H_rgb[ht] = np.array([1, 0, 0])

# b.H_rgb = scalars_to_rgbs(Hscalar)
# b.V_rgb = scalars_to_rgbs(Vscalar)
b.F_rgb = F_rgb
b.H_rgb = H_rgb
b.V_rgb = V_rgb
b.V_radius = Vradius
b.F_opacity = 0.5
b.F_alpha = F_alpha
b.H_alpha = H_alpha
b.V_alpha = V_alpha

# for iters in range(100):
#     b.regularize_by_shifts(.5)
# b.V_pq[v4,:3] = b.V_pq[v4,:3]+.5*(b.V_pq[v1,:3]-b.V_pq[v3,:3])
# V_pq_OG = b.V_pq.copy()
b.V_pq = V_pq_OG.copy()
# b.V_pq[v4,:3] = b.V_pq[v4,:3]+(b.V_pq[v4,:3]-b.V_pq[v3,:3])/1
# b.V_pq[v3,:3] = b.V_pq[v3,:3]+(b.V_pq[v4,:3]-b.V_pq[v3,:3])/2
# b.V_pq[v4,:3] = b.V_pq[v4,:3]+(b.V_pq[v1,:3]-b.V_pq[v4,:3])/2.5
# b.V_pq[v3,:3] = b.V_pq[v3,:3]+(b.V_pq[v1,:3]-b.V_pq[v3,:3])/5
r13 = (b.V_pq[v1, :3] + b.V_pq[v3, :3]) / 2
view = {
    "azimuth": 180,
    "elevation": 90,
    "distance": 0.25,
    "focalpoint": r13,
}

mp.brane_plot(
    b,
    # color_by_V_scalar=False,
    # color_by_V_rgb=True,
    show_halfedges=True,
    show_vertices=True,
    # show_normals=False,
    # show_V_vector_data=True,
    view=view,
)
# %%
b = b0
Hscalar = np.zeros(len(b.halfedges))
Vscalar = np.zeros(len(b.V_pq))
Vradius = np.zeros(len(b.V_pq))
F_rgb = np.ones((len(b.faces), 3))
H_rgb = np.ones((len(b.halfedges), 3))
V_rgb = np.ones((len(b.V_pq), 3))
F_alpha = 0.05 * np.ones(len(b.faces))
H_alpha = 0.05 * np.ones(len(b.halfedges))
V_alpha = 0.05 * np.ones(len(b.V_pq))
h = 290
# b.edge_flip(h)
# h=290
ht = b.H_twin[h]
h1 = b.H_next[h]
h2 = b.H_prev[h]
h3 = b.H_next[ht]
h4 = b.H_prev[ht]
f1 = b.H_face[h]
f2 = b.H_face[ht]
v1 = b.H_vertex[h4]
v2 = b.H_vertex[h1]
v3 = b.H_vertex[h2]
v4 = b.H_vertex[h3]
##
ht1 = b.H_twin[h1]
ht2 = b.H_twin[h2]
ht3 = b.H_twin[h3]
ht4 = b.H_twin[h4]
f12 = b.H_face[ht1]
f23 = b.H_face[ht2]
f34 = b.H_face[ht3]
f41 = b.H_face[ht4]

for vv in [v1, v2, v3, v4]:
    Vradius[vv] = 0.0025
    V_alpha[vv] = 1
for hh in [h1, h2, h3, h4]:
    H_rgb[hh] = np.array([0, 0, 1])
for hh in [h, ht, h1, h2, h3, h4]:
    H_alpha[hh] = 0.8
for ff in [f1, f2, f12, f23, f34, f41]:
    F_alpha[ff] = 0.8
    F_rgb[ff] = np.array([0, 1, 0])
V_rgb[v1] = np.array([1, 0, 0])
V_rgb[v3] = np.array([1, 0.5, 0])
V_rgb[v2] = np.array([0, 0, 1])
V_rgb[v4] = np.array([0.5, 0, 1])
F_rgb[f1] = np.array([1, 0, 0])
F_rgb[f2] = np.array([0, 0, 1])
H_rgb[h] = np.array([1, 0, 0])
H_rgb[ht] = np.array([1, 0, 0])

# b.H_rgb = scalars_to_rgbs(Hscalar)
# b.V_rgb = scalars_to_rgbs(Vscalar)
b.F_rgb = F_rgb
b.H_rgb = H_rgb
b.V_rgb = V_rgb
b.V_radius = Vradius
b.F_opacity = 0.5
b.F_alpha = F_alpha
b.H_alpha = H_alpha
b.V_alpha = V_alpha

# for iters in range(100):
#     b.regularize_by_shifts(.5)
# b.V_pq[v4,:3] = b.V_pq[v4,:3]+.5*(b.V_pq[v1,:3]-b.V_pq[v3,:3])
# V_pq_OG = b.V_pq.copy()
b.V_pq = V_pq_OG.copy()
# b.V_pq[v4,:3] = b.V_pq[v4,:3]+(b.V_pq[v4,:3]-b.V_pq[v3,:3])/1
# b.V_pq[v3,:3] = b.V_pq[v3,:3]+(b.V_pq[v4,:3]-b.V_pq[v3,:3])/2
# b.V_pq[v4,:3] = b.V_pq[v4,:3]+(b.V_pq[v1,:3]-b.V_pq[v4,:3])/2.5
# b.V_pq[v3,:3] = b.V_pq[v3,:3]+(b.V_pq[v1,:3]-b.V_pq[v3,:3])/5
r13 = (b.V_pq[v1, :3] + b.V_pq[v3, :3]) / 2
view = {
    "azimuth": 180,
    "elevation": 90,
    "distance": 0.25,
    "focalpoint": r13,
}

mp.brane_plot(
    b,
    # color_by_V_scalar=False,
    # color_by_V_rgb=True,
    show_halfedges=True,
    show_vertices=True,
    # show_normals=False,
    # show_V_vector_data=True,
    view=view,
)
# %%

b.faces[f1] = np.array([v2, v3, v4], dtype=np.int32)
b.faces[f2] = np.array([v4, v1, v2])
b.F_hedge[f1] = h2
b.F_hedge[f2] = h4
b.halfedges[h] = np.array([v4, v2], dtype=np.int32)
b.halfedges[ht] = np.array([v2, v4], dtype=np.int32)
b.H_next[h] = h2
b.H_prev[h2] = h
b.H_next[h2] = h3
b.H_prev[h3] = h2
b.H_next[h3] = h
b.H_prev[h] = h3  #
b.H_next[ht] = h4
b.H_prev[h4] = ht
b.H_next[h4] = h1
b.H_prev[h1] = h4
b.H_next[h1] = ht
b.H_prev[ht] = h1
b.H_face[h3] = f1
b.H_face[h1] = f2
b.H_vertex[h] = v2
b.H_vertex[ht] = v4
b.V_hedge[v3] = h3
b.V_hedge[v1] = h1
b.V_hedge[v2] = h2
b.V_hedge[v4] = h4


# %%
# b = sim_state["b"]
# is_flippable = np.array([b.is_flippable(h) for h, _ in enumerate(b.halfedges)])
# unflippable = np.array([not b.is_flippable(h) for h, _ in enumerate(b.halfedges)])
# any(unflippable)
# # is_delaunay = np.array([is_delaunay(b, h) for h, _ in enumerate(b.halfedges)])

# flip_helps_valence = np.array(
#     [b.flip_helps_valence(h) for h, _ in enumerate(b.halfedges)]
# )
# flip_it = np.array(
#     [
#         b.flip_helps_valence(h) and not is_delaunay(b, h) and b.is_flippable(h)
#         for h, _ in enumerate(b.halfedges)
#     ]
# )
valence = np.array([b1.valence(v) for v, _ in enumerate(b1.V_pq)])
vals = list(valence.copy())


#################################
max_iters = 308
iters = 0
go = True
while go and (iters <= max_iters):
    print(f"iters={iters}", end="\r")
    vm = vals.index(min(vals))
    hm = b1.V_hedge[vm]
    if b1.is_flippable(hm):
        b0.edge_flip(hm)
        b1.edge_flip(hm)
        b2.edge_flip(hm)
    else:
        vals[vm] = 33
    for h, _ in enumerate(b1.halfedges):
        bad = broke_del(b1, h)
        if bad:
            go = False
            print(f"iters={iters}, vm={vm}, hm={hm}")
            break
    iters += 1
#########################
# %%
vm = vals.index(min(vals))
hm = b1.V_hedge[vm]
if b1.is_flippable(hm):
    b1.edge_flip(hm)
    b2.edge_flip(hm)
else:
    vals[vm] = 33
for h, _ in enumerate(b1.halfedges):
    bad = broke_del(b1, h)
    if bad:
        go = False
        print(f"iters={iters}, vm={vm}, hm={hm}")
        break
#######################
vm = vals.index(min(vals))
hm = b2.V_hedge[vm]
if b2.is_flippable(hm):
    b2.edge_flip(hm)
else:
    vals[vm] = 33
for h, _ in enumerate(b2.halfedges):
    bad = broke_del(b2, h)
    if bad:
        go = False
        print(f"iters={iters}, vm={vm}, hm={hm}")
        break
# %%


# %%
b = b2
hlj = 1
hjk = b.H_next[hlj]
hkl = b.H_next[hjk]
ffljk = [b.H_face[h] for h in [hlj, hjk, hkl]]

hjl = b.H_twin[hlj]
hli = b.H_next[hjl]
hij = b.H_next[hli]
fflij = [b.H_face[h] for h in [hjl, hli, hij]]

hh = [hlj, hjk, hkl, hjl, hli, hij]

vi = b.H_vertex[hli]
vj = b.H_vertex[hlj]
vk = b.H_vertex[hjk]
vl = b.H_vertex[hkl]


hkj = b.H_twin[hjk]
fkj = b.H_face[hkj]
hji = b.H_twin[hij]
fji = b.H_face[hji]

hlk = b.H_twin[hkl]
flk = b.H_face[hlk]
hil = b.H_twin[hli]
fil = b.H_face[hil]

flippable = (fkj != fji) and (flk != fil)
vals1 = [b.valence(v) for v in [vi, vj, vk, vl]]
vals2 = [b.valence(v) for v in [vi, vj, vk, vl]]

# %%
# b = b0
# Hscalar = np.zeros(len(b.halfedges))
# Vscalar = np.zeros(len(b.V_pq))
# Vradius = np.zeros(len(b.V_pq))
# h = 1
# ht = b.H_twin[h]
# h1 = b.H_next[h]
# h2 = b.H_prev[h]
# h3 = b.H_next[ht]
# h4 = b.H_prev[ht]
# f1 = b.H_face[h]
# f2 = b.H_face[ht]
# v1 = b.H_vertex[h4]
# v2 = b.H_vertex[h1]
# v3 = b.H_vertex[h2]
# v4 = b.H_vertex[h3]
# for v in [v1, v2, v3, v4]:
#     Vscalar[v] = 1.0
#     Vradius[v] = 0.025
# for hh in [h1, h2, h3, h4]:
#     Hscalar[hh] = 1.0
# Hscalar[h] = 0.5
#
# b.H_rgb = scalars_to_rgbs(Hscalar)
# b.V_rgb = scalars_to_rgbs(Vscalar)
# b.V_radius = Vradius
# b.F_opacity = 0
# mp.brane_plot(
#     b,
#     # color_by_V_scalar=False,
#     # color_by_V_rgb=True,
#     show_halfedges=True,
#     show_vertices=True,
#     # show_normals=False,
#     # show_V_vector_data=True,
# )
# # %%
#
# b.faces[f1] = np.array([v2, v3, v4], dtype=np.int32)
# b.faces[f2] = np.array([v4, v1, v2])
# b.F_hedge[f1] = h2
# b.F_hedge[f2] = h4
# b.halfedges[h] = np.array([v4, v2], dtype=np.int32)
# b.halfedges[ht] = np.array([v2, v4], dtype=np.int32)
# b.H_next[h] = h2
# b.H_prev[h2] = h
# b.H_next[h2] = h3
# b.H_prev[h3] = h2
# b.H_next[h3] = h
# b.H_prev[h] = h3  #
# b.H_next[ht] = h4
# b.H_prev[h4] = ht
# b.H_next[h4] = h1
# b.H_prev[h1] = h4
# b.H_next[h1] = ht
# b.H_prev[ht] = h1
# b.H_face[h3] = f1
# b.H_face[h1] = f2
# b.H_vertex[h] = v2
# b.H_vertex[ht] = v4
# b.V_hedge[v3] = h3
# b.V_hedge[v1] = h1
# b.V_hedge[v2] = h2
# b.V_hedge[v4] = h4
# %%
b = sim_state["b"]

U = color_by_tether(b)
b.H_rgb = scalars_to_rgbs(U)
Fb_mixed = b.Fbend_mixed()
Ftether = b.Ftether()

b.V_vector_data = Ftether


mp.brane_plot(
    b,
    # color_by_V_scalar=False,
    # color_by_V_rgb=True,
    show_halfedges=True,
    # show_normals=False,
    show_V_vector_data=True,
)
# %%


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
