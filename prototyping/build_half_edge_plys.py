import sys

sys.path.insert(0, "/home/wlough/git/HalfEdgePy")
import src.HalfEdgeMeshBuilder as mb

vf_ply_dir = "./ply_files_vf"
he_ply_dir = "./ply_files_he"
skip_strings_with = ["ultrafine"]
mb.generate_half_edge_mesh_plys(vf_ply_dir, he_ply_dir, use_binary=True, skip_strings_with=skip_strings_with)
