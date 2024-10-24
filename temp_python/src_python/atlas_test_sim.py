# import h5py
import logging
import os
import numpy as np
import pickle
import yaml
from temp_python.src_python.half_edge_base_viewer import MeshViewer
from temp_python.src_python.half_edge_base_patch import HalfEdgePatch
from temp_python.src_python.half_edge_mesh import HalfEdgeMeshBase, HalfEdgeBoundary
from temp_python.src_python.pretty_pictures import RGBA_DICT
from temp_python.src_python.simulation_base import SimulationBase


class AtlasTestSim(SimulationBase):
    def __init__(
        self,
        output_dir="./output/atlas_test_sim",
        run_name="run_0000",
        make_output_dir=True,
        clean_output_dir=False,
        rgba_F_surface=RGBA_DICT["green50"],
        rgba_H_surface=RGBA_DICT["orange80"],
        rgba_H_boundary=RGBA_DICT["blue"],
        rgba_F_interior=RGBA_DICT["purple70"],
        rgba_F_frontier=RGBA_DICT["purple30"],
        ply_path="./data/half_edge_base/ply/neovius_he.ply",
        dF_draw=100,
        make_movie_frames=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            output_dir,
            run_name,
            make_output_dir=make_output_dir,
            clean_output_dir=clean_output_dir,
            *args,
            **kwargs,
        )
        self.rgba_F_surface = np.array(rgba_F_surface)
        self.rgba_H_surface = np.array(rgba_H_surface)
        self.rgba_H_boundary = np.array(rgba_H_boundary)
        self.rgba_F_interior = np.array(rgba_F_interior)
        self.rgba_F_frontier = np.array(rgba_F_frontier)

        self.F_interior = set()
        self.F_frontier = set()
        mv_kwargs = {
            "show_half_edges": True,
            "show_wireframe_surface": False,
            "show_face_colored_surface": True,
            "show_vertex_colored_surface": False,
            "rgba_half_edge": self.rgba_H_surface,
            "rgba_face": self.rgba_F_surface,
            "image_dir": self.temp_images_dir,
            "movie_name": self.run_name,
            "movie_dir": self.visualizations_dir,
        }
        self.ply_path = ply_path
        self.dF_draw = dF_draw
        self.make_movie_frames = make_movie_frames

        self.m = HalfEdgeMeshBase.load(ply_path=self.ply_path)
        self.b = HalfEdgeBoundary.from_mesh(self.m)
        self.mv = MeshViewer(self.m, **mv_kwargs)

    def draw_frame(self):
        if self.make_movie_frames:
            mv = self.mv
            print(f"\nDrawing {mv.get_fig_path()}")
            arrF_interior = np.array(list(self.F_interior), dtype="int32")
            arrF_frontier = np.array(list(self.F_frontier), dtype="int32")
            mv = self.mv
            mv.update_rgba_F(self.rgba_F_interior, arrF_interior)
            mv.update_rgba_F(self.rgba_F_frontier, arrF_frontier)
            title = f"{mv.image_prefix}_{mv.image_count:0{mv.image_index_length}d}.{mv.image_format}"
            mv.plot(save=True, show=False, title=title)

    def run(self):
        m = self.m
        b = self.b
        mv = self.mv
        mv.update_rgba_H(self.rgba_H_boundary, b.arrH)
        f_count = 0
        self.draw_frame()
        dF_draw = self.dF_draw
        for F_interior, F_frontier in b.generate_interior_faces_cumulative():
            f_count += 1
            self.F_interior = F_interior
            self.F_frontier = F_frontier
            print(50 * " ", end="\r")
            print(
                f"num_interior={len(F_interior)}, num_frontier={len(F_frontier)}",
                end="\r",
            )
            if f_count % dF_draw == 0:
                self.draw_frame()
        if f_count % dF_draw != 0:
            self.draw_frame()
        if self.make_movie_frames:
            mv.movie()
