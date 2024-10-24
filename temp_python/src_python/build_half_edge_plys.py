# this builds half-edge plys from vertex-face plys
# using the face index < 0 convention for boundary half-edges
import sys

sys.path.insert(0, "/home/wlough/git/MeshBrane")
import glob
import os
from temp_python.src_python.ply_tools import VertTri2HalfEdgeConverter

vf_ply_dir = "./data/ply/ascii"
he_ply_dir = "./data/ply/binary"
skip_strings_with = ["fine"]
only_use_strings = ["fine"]


def generate_half_edge_mesh_plys_without_str(
    vf_ply_dir, he_ply_dir, use_binary=True, skip_strings_with=[]
):
    """Generates a half edge mesh ply in he_ply_dir for each vertex-face list
    ply in vf_ply_dir"""
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
        v2h = VertTri2HalfEdgeConverter.from_source_ply(vf_ply)
        v2h.write_target_ply(target_path=he_ply, use_ascii=False)

    print("-----------------------------------")


def generate_half_edge_mesh_plys_with_str(
    vf_ply_dir, he_ply_dir, use_binary=True, only_use_strings=[]
):
    """Generates a half edge mesh ply in he_ply_dir for each vertex-face list
    ply in vf_ply_dir"""
    _vf_ply_files = glob.glob(vf_ply_dir + "/*.ply")
    vf_ply_files = []
    for ply in _vf_ply_files:
        use_ply = True
        for _ in only_use_strings:
            if _ not in ply:
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
        v2h = VertTri2HalfEdgeConverter.from_source_ply(vf_ply)
        v2h.write_target_ply(target_path=he_ply, use_ascii=False)

    print("-----------------------------------")


# generate_half_edge_mesh_plys_with_str(
#     vf_ply_dir, he_ply_dir, use_binary=True, only_use_strings=skip_strings_with
# )
