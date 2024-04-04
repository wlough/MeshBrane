from matplotlib.patches import Circle
import matplotlib.patheffects as pe
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as plt_cmap

# from numdiff import *
from src.numdiff import jitnorm, jitcross, jitdot, quaternion_to_matrix, diff

# %%


def make_default_lipid():
    ex, ey = np.eye(2)
    ezhat = np.outer(ey, ex) + np.outer(-ex, ey)
    head_radius = 1.0
    head_position = np.array([0.0, 0.0])
    head_director = ey
    head_tangent = ex

    tail_wave_number = 2.5 * np.pi
    tail_amplitude = 0.08
    tail_separation = 0.7
    tail_length = 4
    tail_samples = 30

    # tail shape
    tail_position = np.zeros((tail_samples, 2))
    tail_position[:, 0] = np.linspace(0, tail_length, tail_samples)
    tail_position[:, 1] = tail_amplitude * np.sin(tail_wave_number * tail_position[:, 0] / tail_length)
    tail_position[:, 0] += head_radius
    # rotate tail to negative y-axis
    tail_rotation = np.outer(-ey, ex) + np.outer(ex, ey)
    tail_position = np.einsum("ij,sj->si", tail_rotation, tail_position)
    # make two tails
    tail_plus = tail_position.copy()
    tail_plus[:, 0] += 0.5 * tail_separation
    tail_minus = tail_position.copy()
    tail_minus[:, 0] *= -1
    tail_minus[:, 0] -= 0.5 * tail_separation

    lipid_default = {
        "r": head_position,
        "R": np.eye(2),
        "tail_plus": tail_plus,
        "tail_minus": tail_minus,
        "radius": head_radius,
    }
    return lipid_default


def make_default_escrt():
    Dx1 = 0.5
    Dx2 = 0.7
    Dy1 = 0.5
    Dy_top = 0.25
    Dy_bot = 0.25

    # verts = np.array([[-Dx1, -Dy_bot], [Dx2, -Dy_bot], [-Dx2, Dy_top], [Dx1, Dy_top]])
    s = np.linspace(-Dx1, Dx2, 50)
    x_helix = 0.25 * np.cos(5 * np.pi * s) + s
    y_helix = 0.5 * np.sin(5 * np.pi * s)
    y_anchor = -np.linspace(1.0, 0.5, 10)
    x_anchor = -0.6 + 0.1 * np.sin(5 * np.pi * y_anchor)
    x = np.array([*x_anchor, *x_helix])
    y = np.array([*y_anchor, *y_helix])
    # x = x_helix  # x_anchor#np.array([*x_anchor, *x_helix])
    # y = y_helix  # y_anchor#np.array([*y_anchor, *y_helix])
    # x = x_anchor  # np.array([*x_anchor, *x_helix])
    # y = y_anchor  # np.array([*y_anchor, *y_helix])
    verts = np.array([x, y]).T
    default_escrt = {"verts": verts}
    return default_escrt


ex, ey = np.eye(2)
ezhat = np.outer(ey, ex) + np.outer(-ex, ey)
red = plt_cmap["Set1"](0)
blue = plt_cmap["Set1"](1)
green = plt_cmap["Greens_r"](0)
purple = plt_cmap["Set1"](3)
orange = plt_cmap["Set1"](4)
yellow = plt_cmap["Set1"](5)
brown = plt_cmap["Set1"](6)
pink = plt_cmap["Set1"](7)
grey = plt_cmap["Set1"](8)
white = plt_cmap["Greys"](0)
black = plt_cmap["Greys_r"](0)
# head_facecolor = green
# head_edgecolor = black
# tail_color = yellow
default_lipid = make_default_lipid()
default_escrt = make_default_escrt()


def lipid_transformation(xy0, R, r, radius):
    xy1 = radius * np.einsum("ij,j", R, xy0) + r
    # xy1 = radius * R @ xy0 + r
    return xy1


def rotate_about(xy0, theta, center):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy1 = np.einsum("ij,j", R, xy0 - center) + center
    return xy1


def make_lipid(
    head_position,
    head_director,
    head_radius=1,
    head_facecolor=green,
    head_edgecolor=black,
    tail_color=green,
    head_alpha=0.6,
):
    head_linewidth = 5 * head_radius
    tail_linewidth = 8 * head_radius

    head_tangent = -ezhat @ head_director  # bilayer tangent
    R = np.outer(head_tangent, ex) + np.outer(head_director, ey)
    r = np.array(head_position)
    _tp = default_lipid["tail_plus"]
    _tm = default_lipid["tail_minus"]

    tp = np.array([lipid_transformation(xy0, R, r, head_radius) for xy0 in _tp])
    tm = np.array([lipid_transformation(xy0, R, r, head_radius) for xy0 in _tm])
    lipid = {
        "r": r,
        "R": R,
        "tail_plus": tp,
        "tail_minus": tm,
        "radius": head_radius,
        "head_facecolor": head_facecolor,
        "head_edgecolor": head_edgecolor,
        "tail_color": tail_color,
        "head_alpha": head_alpha,
        "head_linewidth": head_linewidth,
        "tail_linewidth": tail_linewidth,
    }
    return lipid


def make_escrt(
    subunit_position,
    subunit_tangent,
    subunit_length=2.0,
    subunit_color=blue,
    subunit_alpha=1,
    subunit_linewidth=20,
):
    ex, ey = np.eye(2)
    ezhat = np.outer(ey, ex) + np.outer(-ex, ey)
    t = subunit_tangent
    n = ezhat @ t
    r = np.array(subunit_position)
    R = np.outer(t, ex) + np.outer(n, ey)
    verts = np.array([lipid_transformation(vert, R, r, subunit_length) for vert in default_escrt["verts"]])
    escrt = {
        "r": r,
        "R": R,
        "verts": verts,
        "subunit_color": subunit_color,
        "subunit_alpha": subunit_alpha,
        "subunit_length": subunit_length,
        "subunit_linewidth": subunit_linewidth,
    }
    return escrt


def add_lipid_to_plot(ax, lipid):
    # lipid = {
    #     "r": r,
    #     "R": R,
    #     "tail_plus": tp,
    #     "tail_minus": tm,
    #     "radius": head_radius,
    #     "head_facecolor": head_facecolor,
    #     "head_edgecolor": head_edgecolor,
    #     "tail_color": tail_color,
    #     "head_alpha": head_alpha,
    #     "head_linewidth": head_linewidth,
    #     "tail_linewidth": tail_linewidth,
    # }
    tail_plus, tail_minus = lipid["tail_plus"], lipid["tail_minus"]
    head_radius = lipid["radius"]
    head_position = lipid["r"]
    head_facecolor = lipid["head_facecolor"]
    head_alpha = lipid["head_alpha"]
    head_edgecolor = lipid["head_edgecolor"]
    head_linewidth = lipid["head_linewidth"]
    tail_linewidth = lipid["tail_linewidth"]
    tail_color = lipid["tail_color"]

    head_interior = Circle(
        head_position,
        head_radius,
        facecolor=head_facecolor,
        edgecolor=None,
        alpha=head_alpha,
        zorder=100,
    )
    head_outline = Circle(
        head_position,
        head_radius,
        fill=False,
        edgecolor=head_edgecolor,
        linewidth=head_linewidth,
        zorder=100,
    )
    head_cover = Circle(
        head_position,
        head_radius,
        facecolor=white,
        edgecolor=white,
        alpha=1,
        zorder=99,
    )

    #
    ax.plot(
        *tail_plus.T,
        color=head_edgecolor,
        linewidth=(tail_linewidth + 2 * head_linewidth),
        # zorder=-100,
    )
    ax.plot(
        *tail_plus.T,
        color=white,
        linewidth=tail_linewidth,
        # zorder=-99,
    )
    ax.plot(
        *tail_plus.T,
        color=tail_color,
        linewidth=tail_linewidth,
        # zorder=-1,
        alpha=head_alpha,
    )

    ax.plot(
        *tail_minus.T,
        color=head_edgecolor,
        linewidth=(tail_linewidth + 2 * head_linewidth),
        # zorder=-100,
    )
    ax.plot(
        *tail_minus.T,
        color=white,
        linewidth=tail_linewidth,
        # zorder=-99,
    )
    ax.plot(
        *tail_minus.T,
        color=tail_color,
        linewidth=tail_linewidth,
        # zorder=-1,
        alpha=head_alpha,
    )

    ax.add_patch(head_interior)
    ax.add_patch(head_outline)
    ax.add_patch(head_cover)

    return ax


def make_membrane_lipids(**kwargs):
    # N_lipids = 16
    # head_radius = 0.5
    # membrane_length = 5.0
    # membrane_height = 0.8
    # y_fun = lambda x, L, H: H * np.cos(np.pi * x / L)
    # head_radius = 1
    # head_facecolor = green
    # head_edgecolor = black
    # tail_color = black
    # head_alpha = 0.6
    ###############
    N_lipids = kwargs["N_lipids"]
    head_radius = kwargs["head_radius"]
    membrane_length = kwargs["membrane_length"]
    membrane_height = kwargs["membrane_height"]
    y_fun = kwargs["y_fun"]
    head_radius = kwargs["head_radius"]
    head_facecolor = kwargs["head_facecolor"]
    head_edgecolor = kwargs["head_edgecolor"]
    tail_color = kwargs["tail_color"]
    head_alpha = kwargs["head_alpha"]

    x = np.linspace(-membrane_length, membrane_length, N_lipids)
    head_to_head = x[1] - x[0]
    y = y_fun(x, membrane_length, membrane_height)

    ex, ey = np.eye(2)
    ezhat = np.outer(ey, ex) + np.outer(-ex, ey)
    positions = np.array([x, y]).T
    tangents = diff(positions, head_to_head)
    tangents = np.einsum("si,s->si", tangents, 1 / np.linalg.norm(tangents, axis=1))
    directors = np.einsum("ij,sj->si", ezhat, tangents)

    lipids = [
        make_lipid(
            r,
            n,
            head_radius=head_radius,
            head_facecolor=head_facecolor,
            head_edgecolor=head_edgecolor,
            tail_color=tail_color,
            head_alpha=head_alpha,
        )
        for r, n in zip(positions, directors)
    ]
    return lipids


def flip_lipid(lipid, bilayer_separation):
    R = lipid["R"]
    r = lipid["r"]
    radius = lipid["radius"]
    t, n = R[:, 0], R[:, 1]
    tail_end = 0.5 * (lipid["tail_plus"][-1] + lipid["tail_minus"][-1])
    head_to_tail = tail_end - r
    membrane_mid = tail_end - 0.5 * bilayer_separation * n
    # flipped_tail_end = tail_end - bilayer_separation * n
    # flipped_head_position = flipped_tail_end + head_to_tail
    flipid = lipid.copy()
    fR = R @ np.array([[-1.0, 0.0], [0.0, -1.0]])
    fr = rotate_about(r, np.pi, membrane_mid)
    flipid["r"] = fr
    flipid["R"] = fR
    flipid["tail_plus"] = np.array([rotate_about(xy, np.pi, membrane_mid) for xy in lipid["tail_plus"]])
    flipid["tail_minus"] = np.array([rotate_about(xy, np.pi, membrane_mid) for xy in lipid["tail_minus"]])

    return flipid


def make_bound_escrt(lipid, distance_from_membrane=2, subunit_color=blue, subunit_alpha=0.6):
    lipid_position = lipid["r"]
    lipid_frame = lipid["R"]
    lipid_tangent = lipid_frame[:, 0]
    lipid_normal = lipid_frame[:, 1]
    lipid_radius = lipid["radius"]

    subunit_position = lipid_position + 2 * lipid_radius * lipid_normal
    subunit_tangent = lipid_tangent.copy()
    subunit_length = 2.0 * lipid_radius
    subunit_color = subunit_color
    subunit_alpha = subunit_alpha
    subunit_linewidth = 10 * lipid_radius
    escrt = make_escrt(
        subunit_position,
        subunit_tangent,
        subunit_length=subunit_length,
        subunit_color=subunit_color,
        subunit_alpha=subunit_alpha,
        subunit_linewidth=subunit_linewidth,
    )
    # lipid = {
    #     "r": r,
    #     "R": R,
    #     "tail_plus": tp,
    #     "tail_minus": tm,
    #     "radius": head_radius,
    #     "head_facecolor": head_facecolor,
    #     "head_edgecolor": head_edgecolor,
    #     "tail_color": tail_color,
    #     "head_alpha": head_alpha,
    #     "head_linewidth": head_linewidth,
    #     "tail_linewidth": tail_linewidth,
    # }
    return escrt


def copy_tran_escrt(escrt, theta, tran):
    Rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    r = escrt["r"]
    R = escrt["R"]
    verts = escrt["verts"]
    new_escrt = escrt.copy()
    new_escrt["r"] = r + tran  # lipid_transformation(r, Rot, tran, 1.0)
    new_escrt["R"] = Rot @ R
    new_escrt["verts"] = np.array([lipid_transformation(xy - r, Rot, tran + r, 1.0) for xy in verts])
    return new_escrt


def copy_shift_escrt(escrt, theta, pos):
    Rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    r = escrt["r"]

    R = escrt["R"]
    verts = escrt["verts"]
    new_escrt = escrt.copy()
    new_escrt["r"] = pos  # lipid_transformation(r, Rot, tran, 1.0)
    new_escrt["R"] = Rot
    new_escrt["verts"] = np.array([lipid_transformation(xy - r, Rot @ R.T, pos, 1.0) for xy in verts])
    return new_escrt


def add_escrt_to_plot(ax, escrt):
    # escrt = {
    #     "r": r,
    #     "R": R,
    #     "verts": verts,
    #     "subunit_color": subunit_color,
    #     "subunit_alpha": subunit_alpha,
    #     "subunit_length": subunit_length,
    # subunit_linewidth
    # }
    X, Y = escrt["verts"].T
    ax.plot(
        *escrt["verts"].T,
        color=escrt["subunit_color"],
        linewidth=escrt["subunit_linewidth"],
        # zorder=-1,
        alpha=escrt["subunit_alpha"],
    )

    return ax


def bilayer_plot():
    membrane_input = {}
    membrane_input["N_lipids"] = 12
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.8
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bilayer_separation = 0.3
    skip_top = []
    skip_bottom = [4, 6, 7, 8, 9]
    show_top = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bottom = [0, 1, 2, 3, 5, 3, 3, 3, 8, 9, 10, 11]
    xlim = [-6, 6]
    ylim = [-7, 5]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]
    # lipids = [*lipids_top,*lipids_bottom]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # for _lipid, lipid in enumerate(top_layer):
    #     if not _lipid in skip_top:
    #         add_lipid_to_plot(ax, lipid)
    #
    # for _lipid, lipid in enumerate(bottom_layer):
    #     if not _lipid in skip_bottom:
    #         add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)

    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)

    plt.show()
    plt.close()


def escrt_bilayer_plot():
    membrane_input = {}
    membrane_input["N_lipids"] = 12
    show_top = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bottom = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bound = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_free = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.5
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = red
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 0.8
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            lipid,
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for lipid in top_layer
    ]

    reference_escrt = bound_escrts[0].copy()
    reference_escrt["subunit_color"] = free_escrt_color
    reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    thetas = np.array(
        [
            2.1 + 0.1,
            2.6 + 0.1,
            -0.5 + 0.1,
            -2.8 + 0.1,
            0.1 + 0.1,
            -0.5 + 0.1,
            0.9 + 0.1,
            -0.3 + 0.1,
            1.8 + 0.1,
            1.5 + 0.1,
            -2.9 + 0.1,
            -2.3 + 0.1,
        ]
    )
    trans = np.array(
        [
            [0.0 - 5, 3.6 + 1.4],
            [0.7 - 5, 3.4 - 0.5],
            [1.5 - 5, 3.5 + 1.4],
            [2.4 - 5, 3.5 - 0.5],
            [3.3 - 5, 3.6 + 1.4],
            [4.4 - 5, 3.3 - 0.5],
            [5.5 - 5, 3.4 + 1.4],
            [6.6 - 5, 3.5 - 0.5],
            [7.6 - 5, 3.4 + 1.4],
            [8.5 - 5, 3.4 - 0.5],
            [9.3 - 5, 3.3 + 1.4],
            [9.9 - 5, 3.4 - 0.5],
        ]
    )
    #
    # transx = np.array([escrt["r"][0] for escrt in bound_escrts]) - reference_escrt["r"][0]
    # transy = 0.4 * (np.random.rand(len(bound_escrts)) - 0.5) + 3.5 - reference_escrt["r"][1]
    # trans = np.array([transx, transy]).T

    free_escrts = [copy_shift_escrt(reference_escrt, theta, tran) for theta, tran in zip(thetas, trans)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    for _escrt, escrt in enumerate(free_escrts):
        if _escrt in show_free:
            add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    plt.show()
    plt.close()


####################################
####################################
# %%
def lipid_plot():
    # col = green
    membrane_input = {}
    membrane_input["N_lipids"] = 3
    show_top = [1]
    show_bottom = []

    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.0
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"../helfrich_talk/images/lipid.png"
    fig.savefig(fig_path, transparent=False, dpi=2000)
    fig_path = f"../helfrich_talk/images/lipid_transparent.png"
    fig.savefig(fig_path, transparent=True, dpi=2000)
    plt.show()
    plt.close()


lipid_plot()


# %%


def flat_plot():
    # col = green
    membrane_input = {}
    membrane_input["N_lipids"] = 12
    show_top = [3, 4, 5, 6, 7, 8]
    show_bottom = [3, 4, 5, 6, 7, 8]
    # show_bound = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bound = []
    show_free = []
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.0
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = red
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 1
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            lipid,
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for lipid in top_layer
    ]

    reference_escrt = bound_escrts[0].copy()
    reference_escrt["subunit_color"] = free_escrt_color
    reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    thetas = np.array(
        [
            2.1 + 0.1,
            2.6 + 0.1,
            -0.5 + 0.1,
            -2.8 + 0.1,
            0.1 + 0.1,
            -0.5 + 0.1,
            0.9 + 0.1,
            -0.3 + 0.1,
            1.8 + 0.1,
            1.5 + 0.1,
            -2.9 + 0.1,
            -2.3 + 0.1,
        ]
    )
    trans = np.array(
        [
            [0.0 - 5, 3.6 + 1.4],
            [0.7 - 5, 3.4 - 0.5],
            [1.5 - 5, 3.5 + 1.4],
            [2.4 - 5, 3.5 - 0.5],
            [3.3 - 5, 3.6 + 1.4],
            [4.4 - 5, 3.3 - 0.5],
            [5.5 - 5, 3.4 + 1.4],
            [6.6 - 5, 3.5 - 0.5],
            [7.6 - 5, 3.4 + 1.4],
            [8.5 - 5, 3.4 - 0.5],
            [9.3 - 5, 3.3 + 1.4],
            [9.9 - 5, 3.4 - 0.5],
        ]
    )
    #
    # transx = np.array([escrt["r"][0] for escrt in bound_escrts]) - reference_escrt["r"][0]
    # transy = 0.4 * (np.random.rand(len(bound_escrts)) - 0.5) + 3.5 - reference_escrt["r"][1]
    # trans = np.array([transx, transy]).T

    free_escrts = [copy_shift_escrt(reference_escrt, theta, tran) for theta, tran in zip(thetas, trans)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    for _escrt, escrt in enumerate(free_escrts):
        if _escrt in show_free:
            add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"../helfrich_talk/images/flat_bilayer.png"
    fig.savefig(fig_path, transparent=False, dpi=600)
    fig_path = f"../helfrich_talk/images/flat_bilayer_transparent.png"
    fig.savefig(fig_path, transparent=True, dpi=600)
    plt.show()
    plt.close()


flat_plot()


# %%


def stretch_plot():
    # col = green
    membrane_input = {}
    membrane_input["N_lipids"] = 12
    show_top = [0, 2, 4, 6, 8, 10, 19]
    show_bottom = [0, 2, 4, 6, 8, 10, 19]
    # show_bound = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bound = []
    show_free = []
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.0
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = red
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 1
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            lipid,
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for lipid in top_layer
    ]

    reference_escrt = bound_escrts[0].copy()
    reference_escrt["subunit_color"] = free_escrt_color
    reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    thetas = np.array(
        [
            2.1 + 0.1,
            2.6 + 0.1,
            -0.5 + 0.1,
            -2.8 + 0.1,
            0.1 + 0.1,
            -0.5 + 0.1,
            0.9 + 0.1,
            -0.3 + 0.1,
            1.8 + 0.1,
            1.5 + 0.1,
            -2.9 + 0.1,
            -2.3 + 0.1,
        ]
    )
    trans = np.array(
        [
            [0.0 - 5, 3.6 + 1.4],
            [0.7 - 5, 3.4 - 0.5],
            [1.5 - 5, 3.5 + 1.4],
            [2.4 - 5, 3.5 - 0.5],
            [3.3 - 5, 3.6 + 1.4],
            [4.4 - 5, 3.3 - 0.5],
            [5.5 - 5, 3.4 + 1.4],
            [6.6 - 5, 3.5 - 0.5],
            [7.6 - 5, 3.4 + 1.4],
            [8.5 - 5, 3.4 - 0.5],
            [9.3 - 5, 3.3 + 1.4],
            [9.9 - 5, 3.4 - 0.5],
        ]
    )
    #
    # transx = np.array([escrt["r"][0] for escrt in bound_escrts]) - reference_escrt["r"][0]
    # transy = 0.4 * (np.random.rand(len(bound_escrts)) - 0.5) + 3.5 - reference_escrt["r"][1]
    # trans = np.array([transx, transy]).T

    free_escrts = [copy_shift_escrt(reference_escrt, theta, tran) for theta, tran in zip(thetas, trans)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    for _escrt, escrt in enumerate(free_escrts):
        if _escrt in show_free:
            add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"../helfrich_talk/images/stretched_bilayer.png"
    fig.savefig(fig_path, transparent=False, dpi=600)
    fig_path = f"../helfrich_talk/images/stretched_bilayer_transparent.png"
    fig.savefig(fig_path, transparent=True, dpi=600)
    plt.show()
    plt.close()


stretch_plot()


# %%
def bent_plot():
    # yellowish_green = 0.3 * np.array([*green]) + 0.7 * np.array([*yellow])
    # col = red
    bluish_orange = 0.3 * np.array([*green]) + 0.7 * np.array([*blue])
    membrane_input = {}
    membrane_input["N_lipids"] = 12
    show_top = [3, 4, 5, 6, 7, 8]
    show_bottom = [3, 4, 5, 6, 7, 8]
    show_bound = []
    # show_bound = [4, 5, 6, 7]
    show_free = []
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.9
    membrane_input["y_fun"] = lambda x, L, H: H * np.sqrt(16 - x**2)
    membrane_input["head_facecolor"] = green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = bluish_orange
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 0.8
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            top_layer[_],
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for _ in show_top
    ]

    reference_escrt = bound_escrts[0].copy()
    reference_escrt["subunit_color"] = free_escrt_color
    reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    thetas = np.array(
        [
            2.1 + 0.1,
            2.6 + 0.1,
            -0.5 + 0.1,
            -2.8 + 0.1,
            0.1 + 0.1,
            -0.5 + 0.1,
            0.9 + 0.1,
            -0.3 + 0.1,
            1.8 + 0.1,
            1.5 + 0.1,
            -2.9 + 0.1,
            -2.3 + 0.1,
        ]
    )
    trans = np.array(
        [
            [0.0 - 5, 3.6 + 1.4],
            [0.7 - 5, 3.4 - 0.5],
            [1.5 - 5, 3.5 + 1.4],
            [2.4 - 5, 3.5 - 0.5],
            [3.3 - 5, 3.6 + 1.4],
            [4.4 - 5, 3.3 - 0.5],
            [5.5 - 5, 3.4 + 1.4],
            [6.6 - 5, 3.5 - 0.5],
            [7.6 - 5, 3.4 + 1.4],
            [8.5 - 5, 3.4 - 0.5],
            [9.3 - 5, 3.3 + 1.4],
            [9.9 - 5, 3.4 - 0.5],
        ]
    )
    #
    # transx = np.array([escrt["r"][0] for escrt in bound_escrts]) - reference_escrt["r"][0]
    # transy = 0.4 * (np.random.rand(len(bound_escrts)) - 0.5) + 3.5 - reference_escrt["r"][1]
    # trans = np.array([transx, transy]).T

    free_escrts = [copy_shift_escrt(reference_escrt, theta, tran) for theta, tran in zip(thetas, trans)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    for _escrt, escrt in enumerate(free_escrts):
        if _escrt in show_free:
            add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"../helfrich_talk/images/aad_big_bend.png"
    fig.savefig(fig_path, transparent=False, dpi=600)
    fig_path = f"../helfrich_talk/images/big_bend_transparent.png"
    fig.savefig(fig_path, transparent=True, dpi=600)
    plt.show()
    plt.close()


bent_plot()


# %%
# most in solution
def little_bend_plot():
    greenish_yellow = 0.7 * np.array([*green]) + 0.3 * np.array([*yellow])
    membrane_input = {}
    membrane_input["N_lipids"] = 12
    show_top = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bottom = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # show_bound = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bound = [5, 6]
    show_free = [1, 2, 3, 4, 4, 4, 7, 8, 9, 10]
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.2
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = greenish_yellow  # green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = red
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 1
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            lipid,
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for lipid in top_layer
    ]

    reference_escrt = bound_escrts[0].copy()
    reference_escrt["subunit_color"] = free_escrt_color
    reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    thetas = np.array(
        [
            2.1 + 0.1,
            2.6 + 0.1,
            -0.5 + 0.1,
            -2.8 + 0.1,
            0.1 + 0.1,
            -0.5 + 0.1,
            0.9 + 0.1,
            -0.3 + 0.1,
            1.8 + 0.1,
            1.5 + 0.1,
            -2.9 + 0.1,
            -2.3 + 0.1,
        ]
    )
    trans = np.array(
        [
            [0.0 - 5, 3.6 + 1.4],
            [0.7 - 5, 3.4 - 0.5],
            [1.5 - 5, 3.5 + 1.4],
            [2.4 - 5, 3.5 - 0.5],
            [3.3 - 5, 3.6 + 1.4],
            [4.4 - 5, 3.3 - 0.5],
            [5.5 - 5, 3.4 + 1.4],
            [6.6 - 5, 3.5 - 0.5],
            [7.6 - 5, 3.4 + 1.4],
            [8.5 - 5, 3.4 - 0.5],
            [9.3 - 5, 3.3 + 1.4],
            [9.9 - 5, 3.4 - 0.5],
        ]
    )
    #
    # transx = np.array([escrt["r"][0] for escrt in bound_escrts]) - reference_escrt["r"][0]
    # transy = 0.4 * (np.random.rand(len(bound_escrts)) - 0.5) + 3.5 - reference_escrt["r"][1]
    # trans = np.array([transx, transy]).T

    free_escrts = [copy_shift_escrt(reference_escrt, theta, tran) for theta, tran in zip(thetas, trans)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    for _escrt, escrt in enumerate(free_escrts):
        if _escrt in show_free:
            add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"./temp_images/aab_little_bend.png"
    fig.savefig(fig_path, transparent=False, dpi=600)
    fig_path = f"./temp_images/little_bend_transparent.png"
    fig.savefig(fig_path, transparent=True, dpi=600)
    plt.show()
    plt.close()


#
# some in solution
def medium_bend_plot():
    yellowish_green = 0.3 * np.array([*green]) + 0.7 * np.array([*yellow])
    redish_yellow = 0.3 * np.array([*yellow]) + 0.7 * np.array([*red])

    membrane_input = {}
    membrane_input["N_lipids"] = 24
    show_top = [_ for _ in range(0, 24, 2)]
    # show_bottom = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bottom = [
        0,
        1,
        2,
        # 3,
        4,
        # 5,
        6,
        # 7,
        # 8,
        9,
        # 10,
        # 11,
        # 12,
        13,
        # 14,
        # 15,
        16,
        # 17,
        18,
        # 19,
        20,
        21,
        22,
        # 23,
    ]
    # show_bound = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bound = [4, 5, 6, 7]
    show_free = [1, 2, 3, 3, 3, 3, 3, 8, 9, 10]
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.4
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = yellowish_green  # green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = redish_yellow
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 1
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            top_layer[_],
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for _ in show_top
    ]

    reference_escrt = bound_escrts[0].copy()
    reference_escrt["subunit_color"] = free_escrt_color
    reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    thetas = np.array(
        [
            2.1 + 0.1,
            2.6 + 0.1,
            -0.5 + 0.1,
            -2.8 + 0.1,
            0.1 + 0.1,
            -0.5 + 0.1,
            0.9 + 0.1,
            -0.3 + 0.1,
            1.8 + 0.1,
            1.5 + 0.1,
            -2.9 + 0.1,
            -2.3 + 0.1,
        ]
    )
    trans = np.array(
        [
            [0.0 - 5, 3.6 + 1.4],
            [0.7 - 5, 3.4 - 0.5],
            [1.5 - 5, 3.5 + 1.4],
            [2.4 - 5, 3.5 - 0.5],
            [3.3 - 5, 3.6 + 1.4],
            [4.4 - 5, 3.3 - 0.5],
            [5.5 - 5, 3.4 + 1.4],
            [6.6 - 5, 3.5 - 0.5],
            [7.6 - 5, 3.4 + 1.4],
            [8.5 - 5, 3.4 - 0.5],
            [9.3 - 5, 3.3 + 1.4],
            [9.9 - 5, 3.4 - 0.5],
        ]
    )
    #
    # transx = np.array([escrt["r"][0] for escrt in bound_escrts]) - reference_escrt["r"][0]
    # transy = 0.4 * (np.random.rand(len(bound_escrts)) - 0.5) + 3.5 - reference_escrt["r"][1]
    # trans = np.array([transx, transy]).T

    free_escrts = [copy_shift_escrt(reference_escrt, theta, tran) for theta, tran in zip(thetas, trans)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    for _escrt, escrt in enumerate(free_escrts):
        if _escrt in show_free:
            add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"./temp_images/aac_medium_bend.png"
    fig.savefig(fig_path, transparent=False, dpi=600)
    fig_path = f"./temp_images/medium_bend_transparent.png"
    fig.savefig(fig_path, transparent=True, dpi=600)
    plt.show()
    plt.close()


#
# some in solution
def big_bend_plot():
    # yellowish_green = 0.3 * np.array([*green]) + 0.7 * np.array([*yellow])
    # col = red
    bluish_orange = 0.3 * np.array([*green]) + 0.7 * np.array([*blue])
    membrane_input = {}
    membrane_input["N_lipids"] = 24
    show_top = [_ for _ in range(0, 24, 2)]
    # show_bottom = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    show_bottom = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        # 7,
        # 8,
        # 9,
        10,
        # 11,
        # 12,
        # 13,
        # 14,
        # 15,
        # 16,
        # 17,
        18,
        19,
        20,
        21,
        22,
        # 23,
    ]
    # show_bottom = [
    #     0,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     # 6,
    #     # 7,
    #     # 8,
    #     # 9,
    #     # 10,
    #     11,
    #     # 12,
    #     # 13,
    #     # 14,
    #     # 15,
    #     # 16,
    #     # 17,
    #     18,
    #     19,
    #     20,
    #     21,
    #     22,
    #     # 23,
    # ]
    show_bound = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # show_bound = [4, 5, 6, 7]
    show_free = []
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 5.0
    membrane_input["membrane_height"] = 0.9
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = red
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = bluish_orange
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 0.8
    bilayer_separation = 0.3
    xlim = [-6, 6]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            top_layer[_],
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for _ in show_top
    ]

    reference_escrt = bound_escrts[0].copy()
    reference_escrt["subunit_color"] = free_escrt_color
    reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    thetas = np.array(
        [
            2.1 + 0.1,
            2.6 + 0.1,
            -0.5 + 0.1,
            -2.8 + 0.1,
            0.1 + 0.1,
            -0.5 + 0.1,
            0.9 + 0.1,
            -0.3 + 0.1,
            1.8 + 0.1,
            1.5 + 0.1,
            -2.9 + 0.1,
            -2.3 + 0.1,
        ]
    )
    trans = np.array(
        [
            [0.0 - 5, 3.6 + 1.4],
            [0.7 - 5, 3.4 - 0.5],
            [1.5 - 5, 3.5 + 1.4],
            [2.4 - 5, 3.5 - 0.5],
            [3.3 - 5, 3.6 + 1.4],
            [4.4 - 5, 3.3 - 0.5],
            [5.5 - 5, 3.4 + 1.4],
            [6.6 - 5, 3.5 - 0.5],
            [7.6 - 5, 3.4 + 1.4],
            [8.5 - 5, 3.4 - 0.5],
            [9.3 - 5, 3.3 + 1.4],
            [9.9 - 5, 3.4 - 0.5],
        ]
    )
    #
    # transx = np.array([escrt["r"][0] for escrt in bound_escrts]) - reference_escrt["r"][0]
    # transy = 0.4 * (np.random.rand(len(bound_escrts)) - 0.5) + 3.5 - reference_escrt["r"][1]
    # trans = np.array([transx, transy]).T

    free_escrts = [copy_shift_escrt(reference_escrt, theta, tran) for theta, tran in zip(thetas, trans)]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    for _escrt, escrt in enumerate(free_escrts):
        if _escrt in show_free:
            add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"./temp_images/aad_big_bend.png"
    fig.savefig(fig_path, transparent=False, dpi=600)
    fig_path = f"./temp_images/big_bend_transparent.png"
    fig.savefig(fig_path, transparent=True, dpi=600)
    plt.show()
    plt.close()


# no_bend_plot()
# little_bend_plot()
# medium_bend_plot()
# big_bend_plot()
#
# os.system("rm -r temp_images")
# os.system("mkdir temp_images")


#
def curvature_generation_plot():
    yellowish_green = 0.3 * np.array([*green]) + 0.7 * np.array([*yellow])
    redish_yellow = 0.3 * np.array([*yellow]) + 0.7 * np.array([*red])

    membrane_input = {}
    membrane_input["N_lipids"] = 24
    show_top = [_ for _ in range(0, 24, 2)]
    show_bottom = [_ for _ in range(0, 24, 1)]
    show_bound = [2]  # [_ for _ in range(0, 24, 1)]
    # show_free = [_ for _ in range(0, 24, 2)]
    membrane_input["head_radius"] = 0.4
    membrane_input["membrane_length"] = 3.0
    membrane_input["membrane_height"] = 0.4
    membrane_input["y_fun"] = lambda x, L, H: H * np.cos(np.pi * x / L)
    membrane_input["head_facecolor"] = yellowish_green  # green
    membrane_input["head_edgecolor"] = black
    membrane_input["tail_color"] = orange
    membrane_input["head_alpha"] = 0.6
    bound_escrt_color = redish_yellow
    bound_escrt_alpha = 0.8
    free_escrt_color = blue
    free_escrt_alpha = 1
    bilayer_separation = 0.3
    xlim = [-4, 4]
    ylim = [-6, 6]

    top_layer = make_membrane_lipids(**membrane_input)
    bottom_layer = [flip_lipid(lipid, bilayer_separation) for lipid in top_layer]

    bound_escrts = [
        make_bound_escrt(
            top_layer[_],
            distance_from_membrane=2,
            subunit_color=bound_escrt_color,
            subunit_alpha=bound_escrt_alpha,
        )
        for _ in show_top
        if _ in show_bound
    ]

    # reference_escrt = bound_escrts[0].copy()
    # reference_escrt["subunit_color"] = free_escrt_color
    # reference_escrt["subunit_alpha"] = free_escrt_alpha
    # thetas = 2 * np.pi * (np.random.rand(len(bound_escrts)) - 0.5)
    # thetas = np.array(
    #     [
    #         2.1 + 0.1,
    #         2.6 + 0.1,
    #         -0.5 + 0.1,
    #         -2.8 + 0.1,
    #         0.1 + 0.1,
    #         -0.5 + 0.1,
    #         0.9 + 0.1,
    #         -0.3 + 0.1,
    #         1.8 + 0.1,
    #         1.5 + 0.1,
    #         -2.9 + 0.1,
    #         -2.3 + 0.1,
    #     ]
    # )
    # trans = np.array(
    #     [
    #         [0.0 - 5, 3.6 + 1.4],
    #         [0.7 - 5, 3.4 - 0.5],
    #         [1.5 - 5, 3.5 + 1.4],
    #         [2.4 - 5, 3.5 - 0.5],
    #         [3.3 - 5, 3.6 + 1.4],
    #         [4.4 - 5, 3.3 - 0.5],
    #         [5.5 - 5, 3.4 + 1.4],
    #         [6.6 - 5, 3.5 - 0.5],
    #         [7.6 - 5, 3.4 + 1.4],
    #         [8.5 - 5, 3.4 - 0.5],
    #         [9.3 - 5, 3.3 + 1.4],
    #         [9.9 - 5, 3.4 - 0.5],
    #     ]
    # )
    #

    # free_escrts = [
    #     copy_shift_escrt(reference_escrt, theta, tran)
    #     for theta, tran in zip(thetas, trans)
    # ]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for _lipid, lipid in enumerate(top_layer):
        if _lipid in show_top:
            add_lipid_to_plot(ax, lipid)
    for _lipid, lipid in enumerate(bottom_layer):
        if _lipid in show_bottom:
            add_lipid_to_plot(ax, lipid)

    for _escrt, escrt in enumerate(bound_escrts):
        if _escrt in show_bound:
            add_escrt_to_plot(ax, escrt)

    # for _escrt, escrt in enumerate(free_escrts):
    #     if _escrt in show_free:
    #         add_escrt_to_plot(ax, escrt)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    fig_path = f"./temp_images/generation.png"
    fig.savefig(fig_path, transparent=False, dpi=100)
    # fig_path = f"./temp_images/generation.png"
    # fig.savefig(fig_path, transparent=True, dpi=600)
    plt.show()
    plt.close()


# %%
curvature_generation_plot()
