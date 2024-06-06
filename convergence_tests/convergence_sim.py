import numpy as np
import convergence_tests.convergence_model as m
import convergence_tests.convergence_plots as cplt
from source.ply_utils import TriMeshData, HalfEdgeMeshData

import os
import shutil

# from source.pretty_pictures import ply_plot
from time import time

a = [1, 2].copy()
a
# from src.pretty_pictures import mayavi_plots as mp
# from matplotlib import colormaps as plt_cmap
# import matplotlib.pyplot as plt

# import glob


class LapData:
    def __init__(
        self,
        data,
        row_indices,
        col_indices,
        construct_time,
        cutoff_distance=np.inf,
        name="",
        apply_time=-1.0,
        samples=None,
        get_apply=False,
    ):
        self.name = name
        self.data = np.copy(data)
        self.col_indices = np.copy(col_indices)
        self.row_indices = np.copy(row_indices)
        self.construct_time = construct_time
        self.cutoff_distance = cutoff_distance
        ######################################
        self.Nsamples = max(row_indices)
        self.sparsity = len(self.data) / self.Nsamples**2
        if samples is None:
            self.samples = np.zeros(self.Nsamples)
        else:
            self.samples = np.copy(samples)

        if get_apply:
            self.apply()
        else:
            self.apply_time = apply_time
            self.lap_samples = np.zeros_like(self.samples)

    def apply(self):
        t0 = time()
        self.lap_samples = np.zeros_like(self.samples)
        for i, j, wij in zip(self.row_indices, self.col_indices, self.data):
            self.lap_samples[i] += wij * (self.samples[j] - self.samples[i])
        self.apply_time = time() - t0


class ConvTest:
    def __init__(
        self, brane, get_applys=False, cutoff_distance=1e-2, output_dir=None, regenerate_output_dir=False
    ):

        self.brane = brane
        if output_dir is None:
            self.output_dir = "./convergence_tests/output"
        else:
            self.output_dir = output_dir
        self.check_output_dir(regenerate=regenerate_output_dir)
        #######################################################

        (
            heat_data,
            heat_row_indices,
            heat_col_indices,
            heat_construct_time,
        ) = self.get_heat_laplacian_weights(cutoff_distance)

        self.heatLapData = LapData(
            heat_data,
            heat_row_indices,
            heat_col_indices,
            heat_construct_time,
            cutoff_distance=cutoff_distance,
            name="heat",
            samples=brane.V,
            get_apply=get_applys,
        )
        (
            cotan_data,
            cotan_row_indices,
            cotan_col_indices,
            cotan_construct_time,
        ) = self.get_cotan_laplacian_weights()
        self.cotanLapData = LapData(
            cotan_data,
            cotan_row_indices,
            cotan_col_indices,
            cotan_construct_time,
            name="cotan",
            samples=brane.V,
            get_apply=get_applys,
        )

    @classmethod
    def from_ply(cls, ply_path, **kwargs):
        b = m.Brane.from_half_edge_ply(ply_path)
        return cls(b, **kwargs)

    def check_output_dir(self, regenerate=False):
        """checks/generates the output directory and subdirectories for the simulation.
        Removes existing output_directory and regenerates a new one"""
        subdirs = ["temp_images"]
        output_dir = self.output_dir
        if regenerate and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for subdir in subdirs:
            _subdir = os.path.join(output_dir, subdir)
            os.makedirs(_subdir, exist_ok=True)

    def get_heat_laplacian_weights(self, cutoff_distance):

        t0 = time()
        data, row_indices, col_indices = self.brane.get_heat_laplacian_weights(cutoff_distance=cutoff_distance)
        construct_time = time() - t0

        return data, row_indices, col_indices, construct_time

    def get_cotan_laplacian_weights(self):

        t0 = time()
        data, row_indices, col_indices = self.brane.get_cot_laplacian_weights()
        construct_time = time() - t0
        return data, row_indices, col_indices, construct_time

    def update_heat_cutoff(self, cutoff_distance):
        (
            heat_data,
            heat_row_indices,
            heat_col_indices,
            heat_construct_time,
        ) = self.get_heat_laplacian_weights(cutoff_distance)

        self.heatLapData = LapData(
            heat_data,
            heat_row_indices,
            heat_col_indices,
            heat_construct_time,
            cutoff_distance=cutoff_distance,
            name="heat",
            samples=self.brane.V,
            get_apply=True,
        )


# %%
ct_kwargs = {
    "ply_path": "./convergence_tests/data/he_ply_files/dumbbell_coarse.ply",
    "cutoff_distance": 1e-3,
    "output_dir": "./convergence_tests/output",
    "get_applys": True,
}

ct = ConvTest.from_ply(**ct_kwargs)
ct.update_heat_cutoff(1e-2)


# %%
# vf_ply_path = "./convergence_tests/data/vf_ply_files/dumbbell.ply"
# b = m.Brane.from_vertex_face_ply(vf_ply_path)
he_ply_path = "./convergence_tests/data/he_ply_files/dumbbell.ply"
b = m.Brane.from_half_edge_ply(he_ply_path)


# %%
v = 916  # int(np.random.rand() * 1512)#
N = 3
E1 = b.get_E_one_ring_neighbors(v)
E_neighbors = [E1]
for n in range(2, N + 1):
    E_neighbors.append(b.get_order_n_plus_one_edges(E_neighbors[-1]))

black = np.array([0.0, 0.0, 0.0, 1.0])
red = np.array([1.0, 0.0, 0.0, 1.0])
green = np.array([0.0, 1.0, 0.0, 1.0])
blue = np.array([0.0, 0.0, 1.0, 1.0])
half_orange = np.array([1.0, 0.498, 0.0, 0.5])
b.E_rgba = np.array([half_orange for e in b.E_rgba])

# for n, En in enumerate(E_neighbors[:3]):
#     print(f"n={n+1}")
#     print("--------------------------")
#     for e in En:
#         b.E_rgba[e] = red
#         v = b.v_of_e(e)
#         print(f"valence={b.valence(v)}")
#
# print(f"n={4}")
# print("--------------------------")
# for e in E_neighbors[3]:
#     b.E_rgba[e] = blue
#     v = b.v_of_e(e)
#     print(f"valence={b.valence(v)}")


for n, En in enumerate(E_neighbors):
    for e in En:
        b.E_rgba[e] = red
        v = b.v_of_e(e)

# for n, En in enumerate(E_neighbors):
#     for e in En[:1]:
#         b.E_rgba[e] = black


cplt.brane_plot(
    b,
    show=True,
    save=False,
    fig_path=None,
    figsize=(2180, 2180),
    show_surface=True,
    show_halfedges=True,
    show_edges=False,
    show_vertices=False,
    show_normals=False,
    show_plot_axes=False,
    color_by_V_rgba=False,
    V_vector_data=None,
    V_vector_data_rgba=None,
    show_V_vector_data=False,
    frame_scale=0.07,
    view=None,
)


# %%
