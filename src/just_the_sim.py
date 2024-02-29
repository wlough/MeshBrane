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
                # b.delaunay_regularize_by_flips()
                Nflips = b.do_the_monte_flips()
                # Fl = b.Flength()
                Fl = b.Ftether()
                Fa, Fv = b.Fa_Fv()
                Fb = b.Fbend_mixed()
                # #############################
                # #############################
                # smooth_length = b.preferred_edge_length
                # Nvertexsmooth = 1
                # Ncurvaturesmooth = 150
                # Nbendingforcesmooth = 50
                # b.V_pq[:, :3] = b.gaussian_smooth_samples(
                #     b.V_pq[:, :3], Nvertexsmooth, smooth_length
                # )
                # H, K = b.get_angle_weighted_arc_curvatures()
                # K = b.gaussian_smooth_samples(K, Ncurvaturesmooth, smooth_length)
                # H = b.gaussian_smooth_samples(H, Ncurvaturesmooth, smooth_length)
                # lapH = b.cotan_laplacian(H)
                # Fn = -2 * b.bending_modulus * (lapH + 2 * H * (H**2 - K))
                # Fn = b.gaussian_smooth_samples(Fn, Nbendingforcesmooth, smooth_length)
                # # n = np.array([b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)])
                # Fb = np.array(
                #     [fn * b.other_weighted_vertex_normal(v) for v, fn in enumerate(Fn)]
                # )
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
                f"t={tt}, Fb={Fbmax},Fl={Flmax},Fa={Famax},Fv={Fvmax},Nflips={Nflips}              ",
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


def color_by_tether(b, _a=None):
    Nh = len(b.halfedges)
    U = np.zeros(Nh)
    for h in range(Nh):
        i, j = b.halfedges[h]
        ri, rj = b.V_pq[i, :3], b.V_pq[j, :3]
        Drij = rj - ri
        s = np.sqrt(Drij @ Drij)
        U[h] = b.Utether(s, 1.0)
    return U


# %%
# ply_path = "./data/ply_files/oblate.ply"
# vertices, faces = load_mesh_from_ply(ply_path)
# mesh_directory = "./data/halfedge_meshes/dumbbell"
mesh_directory = "./data/halfedge_meshes/dumbbell_coarse"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    "length_reg_stiffness": 1e-6,
    "area_reg_stiffness": 1e-3,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e0,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1e1,
} | mesh_data
sim_kwargs = {
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "dt": 1e-4,
    "output_directory": "./output/monte_output",
} | brane_kwargs
sim_state = initialize_sim(**sim_kwargs)

Trun = 50
# sim_state = run(
#     sim_state,
#     Trun,
#     make_plots=True,
# )
#
# movie_dir = sim_state["output_directory"] + "/temp_images"
# mp.movie(movie_dir)

b = sim_state["b"]
U = color_by_tether(b)
b.H_rgb = scalars_to_rgbs(U)
sim_state1 = initialize_sim(**sim_kwargs)
b1 = sim_state1["b"]
# %%
b1.do_the_monte_flips()
U1 = color_by_tether(b1)
b1.H_rgb = scalars_to_rgbs(U1)
# max(U1)
# plt.plot(U)
# # U = color_by_tether(b)
# # max(U)
# l0=b.preferred_edge_length
# s0 = np.linspace(b.preferred_edge_length*.6,b.preferred_edge_length*1.4, 500)
# s = np.linspace(b.preferred_edge_length*.4,b.preferred_edge_length*1.6, 500)
# U1 = np.array([b.Utether1(si, 1.0*l0) for si in s[1:-1]])
# U_OG = np.array([b.Utether_OG(si, 1.0*l0) for si in s[1:-1]])
# U = np.array([b.Utether(si, 1.0*l0) for si in s[1:-1]])
# plt.plot(U_OG)
# plt.plot(np.abs(U1))
# plt.plot(U)
# s[140:150]
mp.brane_plot(
    b1,
    # color_by_V_scalar=False,
    # color_by_V_rgb=True,
    show_halfedges=True,
    # show_normals=False,
    # show_V_vector_data=True,
)


# %%
def monte_wants_flip(self, h):
    Ka = self.Karea
    A0 = self.preferred_face_area
    ht = self.H_twin[h]
    h1 = self.H_next[h]
    h2 = self.H_prev[h]
    h3 = self.H_next[ht]
    h4 = self.H_prev[ht]

    v1 = self.H_vertex[h4]
    v2 = self.H_vertex[h1]
    v3 = self.H_vertex[h2]
    v4 = self.H_vertex[h3]
    r1 = self.V_pq[v1, :3]
    r2 = self.V_pq[v2, :3]
    r3 = self.V_pq[v3, :3]
    r4 = self.V_pq[v4, :3]
    vecA123 = (jitcross(r1, r2) + jitcross(r2, r3) + jitcross(r3, r1)) / 2
    vecA134 = (jitcross(r1, r3) + jitcross(r3, r4) + jitcross(r4, r1)) / 2
    vecA234 = (jitcross(r2, r3) + jitcross(r3, r4) + jitcross(r4, r2)) / 2
    vecA124 = (jitcross(r1, r2) + jitcross(r2, r4) + jitcross(r4, r1)) / 2
    A123 = np.sqrt(vecA123[0] ** 2 + vecA123[1] ** 2 + vecA123[2] ** 2)
    A134 = np.sqrt(vecA134[0] ** 2 + vecA134[1] ** 2 + vecA134[2] ** 2)
    A234 = np.sqrt(vecA234[0] ** 2 + vecA234[1] ** 2 + vecA234[2] ** 2)
    A124 = np.sqrt(vecA124[0] ** 2 + vecA124[1] ** 2 + vecA124[2] ** 2)

    Upre = (Ka / (2 * A0)) * ((A123 - A0) ** 2 + (A134 - A0) ** 2)
    Upos = (Ka / (2 * A0)) * ((A234 - A0) ** 2 + (A124 - A0) ** 2)

    Dr13 = self.V_pq[v1, :3] - self.V_pq[v3, :3]
    Dr24 = self.V_pq[v2, :3] - self.V_pq[v4, :3]
    L13 = np.sqrt(Dr13[0] ** 2 + Dr13[1] ** 2 + Dr13[2] ** 2)
    L24 = np.sqrt(Dr24[0] ** 2 + Dr24[1] ** 2 + Dr24[2] ** 2)
    Upre += self.Utether(L13)
    Upos += self.Utether(L24)
    flip_it = Upos < Upre
    return flip_it, Upre, Upos


for h in range(len(b.halfedges)):
    flip_it, Upre, Upos = monte_wants_flip(b, h)
    if flip_it:
        print(f"h={h}, Upre={Upre}, Upos={Upos}")
