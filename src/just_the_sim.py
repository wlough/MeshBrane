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
from scipy.sparse import csr_matrix

x = np.linspace(0, 1, 50)


def slow_smoothed_laplacian(self, Y):
    """computes the laplacian of Y at each vertex"""

    Nv = len(self.V_pq)
    Nf = len(self.faces)
    lapY = np.zeros_like(Y)
    for vi in range(Nv):
        ri = self.V_pq[vi, :3]
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        Ai = self.meyercell_area(vi)
        for f in range(Nf):
            vj, vk, vl = self.faces[f]
            rj = self.V_pq[vj, :3]
            rk = self.V_pq[vk, :3]
            rl = self.V_pq[vl, :3]
            vecAf = (jitcross(rj, rk) + jitcross(rk, rl) + jitcross(rl, rj)) / 2
            Af = np.sqrt(vecAf[0] ** 2 + vecAf[1] ** 2 + vecAf[2] ** 2)
            for vm in self.faces[f]:
                rm = self.V_pq[vm, :3]
                rm_rm = rm[0] ** 2 + rm[1] ** 2 + rm[2] ** 2
                ri_rm = ri[0] * rm[0] + ri[1] * rm[1] + ri[2] * rm[2]
                Drim_sqr = ri_ri - 2 * ri_rm + rm_rm
                lapY[vi] += (Af / (12 * np.pi * Ai**2)) * np.exp(-Drim_sqr / (4 * Ai)) * (Y[vm] - Y[vi])

    return lapY


# @njit
def get_halfedges_from_triangle(face):
    return np.array([[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]], dtype=np.int32)


def get_halfedges_from_triangles_parallel(F):
    N_cpu = mu.cpu_count()
    with mu.Pool(processes=N_cpu) as p:
        H = np.concatenate(p.map(get_halfedges_from_triangle, F))
    H_label = np.array([h for h, _ in enumerate(H)], dtype=np.int32)
    return H, H_label


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
    # for iii in range(10):
    #     Nflips = b.regularize_by_flips()
    while Trun - t > 0.5 * dt and success:
        while tstop - t > 0.5 * dt and success:
            try:
                # b.delaunay_regularize_by_flips()
                # Nflips = b.do_the_monte_flips()
                Nflips = 111
                # Fl = b.Flength()
                Fl = b.Ftether()
                Fa, Fv = b.Fa_Fv()
                Fb = b.Fbend_mixed()
                F = Fb + Fl + Fa + Fv
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
                Vold = b.V_pq[:, :3].copy()
                b.V_pq[:, :3] += b.weighted_drag_coeffs_step(dt)
                b.V_pq[:, :3] = Vold + b.weighted_drag_coeffs_step(dt)

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


def log_log_fit(X, Y, Xlabel="X", Ylabel="Y", title=""):
    x, y = np.log(X), np.log(Y)
    a11 = x @ x
    a12 = sum(x)
    a21 = a12
    a22 = len(x)
    u1 = x @ y
    u2 = sum(y)
    u = np.array([u1, u2])
    detA = a11 * a22 - a12 * a21
    Ainv = np.array([[a22, -a12], [-a21, a11]]) / detA
    p, c = Ainv @ u
    f = p * x + c
    fit_label = f"${Ylabel}={round_to(c,n=3)}{Xlabel}" "^{" + f"{round_to(p,n=3)}" + "}$"
    plt.plot(
        x,
        f,
        # label=f"log({Ylabel})={round_to(p,n=3)}log({Xlabel})+{round_to(c,n=3)}",
        label=fit_label,
        linewidth=2,
    )
    plt.plot(x, y, "*")
    plt.title(title, fontsize=16)
    plt.xlabel(f"log({Xlabel})", fontsize=16)
    plt.ylabel(f"log({Ylabel})", fontsize=16)
    plt.legend()
    plt.show()
    plt.close()


def round_to(x, n=3):
    if x == 0:
        return 0.0
    else:
        sgn_x = np.sign(x)
        abs_x = abs(x)
        return round(x, -int(np.floor(np.log10(abs_x))) + (n - 1))


def round_sci(x, n=3):
    return np.format_float_scientific(x, precision=n)


# %%
# ply_path = "./data/ply_files/oblate.ply"
# vertices, faces = load_mesh_from_ply(ply_path)
# mesh_directory = "./data/halfedge_meshes/dumbbell"
mesh_directory = "./data/halfedge_meshes/dumbbell"
mesh_data = load_halfedge_mesh_data(mesh_directory)
brane_kwargs = {
    "length_reg_stiffness": 1e-9,
    "area_reg_stiffness": 1e-3,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e-1,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1e3,
} | mesh_data
sim_kwargs = {
    "Tplot": 5e-2,
    "Tsave": 5e-1,
    "dt": 1e-3,
    "output_directory": "./output/heat_kernel_tests",
} | brane_kwargs
sim_state = initialize_sim(**sim_kwargs)
# %%
Trun = 0.1
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
b = m.Brane(**brane_kwargs)  # uses mean area for h
b.L_elements, b.L_indices = b.get_meyer_weighted_heat_laplacian()
b.P = b.get_TM_projection()
b.N = b.get_heat_unit_normals()
# bN = b.N.copy()
# b.N=N
b.flip_normals()
b.B = b.get_curvature_matrix()
H = 0.5 * np.einsum("xii->x", b.B)

b.V_vector_data = 0.1 * b.N
# b.V_rgb = scalars_to_rgbs(H)
mp.brane_plot(
    b,
    color_by_V_scalar=False,
    color_by_V_rgb=False,
    show_halfedges=True,
    # show_normals=False,
    show_V_vector_data=True,
)
# %%

vals = np.zeros((len(b.V_pq), 3))
vecs = np.zeros((len(b.V_pq), 3, 3))
N = np.zeros((len(b.V_pq), 3))
H1 = np.zeros(len(b.V_pq))

for x in range(len(b.V_pq)):
    B = 0.5 * (b.B[x] + b.B[x].T)
    vals[x], vecs[x] = np.linalg.eigh(B)
    vals_list = list(np.abs(vals[x]))
    x_N = vals_list.index(min(vals_list))
    N[x] = vecs[x, :, x_N]
    H1[x] = (sum(vals[x]) - vals[x, x_N]) / 2


np.min(H1)
