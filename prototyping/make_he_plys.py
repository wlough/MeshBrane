# def generate_half_edge_mesh_plys(
#     # vf_ply_dir, he_ply_dir, use_ascii=False, skip_strings_with=[]
# ):
#     """Generates a half edge mesh ply in he_ply_dir for each vertex-face list
#     ply in vf_ply_dir"""
#     from src.python.ply_tools import VertTri2HalfEdgeConverter
#     import glob
#     import os
#     import sys
#
#     sys.path.insert(0, "/home/wlough/git/MeshBrane")
#     vf_ply_dir = "./data/ply/ascii"
#     he_ply_dir = "./data/ply/binary"
#     skip_strings_with = ["ultrafine"]
#     use_ascii = False
#     _vf_ply_files = glob.glob(vf_ply_dir + "/*.ply")
#     vf_ply_files = []
#     for ply in _vf_ply_files:
#         use_ply = True
#         for skip_string in skip_strings_with:
#             if skip_string in ply:
#                 use_ply = False
#         if use_ply:
#             vf_ply_files.append(ply)
#     print("Generating half edge mesh data for:")
#     for vf_ply in vf_ply_files:
#         print(os.path.basename(vf_ply))
#     print("-----------------------------------")
#     Nply = len(vf_ply_files)
#     for _, vf_ply in enumerate(vf_ply_files):
#         n_ply = _ + 1
#         ply_name = os.path.basename(vf_ply)
#         he_ply = f"{he_ply_dir}/{ply_name}"
#         print(f"{ply_name} ({n_ply}/{Nply})")
#         vt2he = VertTri2HalfEdgeConverter.from_source_ply(vf_ply)
#         # mesh = HalfEdgeMeshBuilder.from_vertex_face_ply(vf_ply)
#         # mesh.to_half_edge_ply(he_ply, use_binary=use_binary)
#         vt2he.write_target_ply(target_path=he_ply, use_ascii=use_ascii)
#     print("-----------------------------------")
#
# generate_half_edge_mesh_plys()
import sys

sys.path.insert(0, "/home/wlough/git/MeshBrane")
from src.python.ply_tools import VertTri2HalfEdgeConverter
import glob
import os

# sys.path.insert(0, "/home/wlough/git/MeshBrane/src/python")
vf_ply_dir = "./data/ply/ascii"
he_ply_dir = "./data/ply/binary"
skip_strings_with = ["ultrafine"]
use_ascii = False
_vf_ply_files = glob.glob(vf_ply_dir + "/*.ply")
vf_ply_files = []
for ply in _vf_ply_files:
    use_ply = True
    for skip_string in skip_strings_with:
        if skip_string in ply:
            use_ply = False
    if use_ply:
        vf_ply_files.append(ply)
print("Generating half edge mesh data for:")
for vf_ply in vf_ply_files:
    print(os.path.basename(vf_ply))
print("-----------------------------------")
Nply = len(vf_ply_files)
for _, vf_ply in enumerate(vf_ply_files):
    n_ply = _ + 1
    ply_name = os.path.basename(vf_ply)
    he_ply = f"{he_ply_dir}/{ply_name}"
    print(f"{ply_name} ({n_ply}/{Nply})")
    vt2he = VertTri2HalfEdgeConverter.from_source_ply(vf_ply)
    # mesh = HalfEdgeMeshBuilder.from_vertex_face_ply(vf_ply)
    # mesh.to_half_edge_ply(he_ply, use_binary=use_binary)
    vt2he.write_target_ply(target_path=he_ply, use_ascii=use_ascii)
print("-----------------------------------")
