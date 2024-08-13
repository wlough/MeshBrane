from src.python.utilities import (
    load_pkl,
    load_npz,
    save_pkl,
    save_npz,
    chunk_file_with_split,
    unchunk_file_with_cat,
)

_NUM_VERTS_ = [
    12,
    42,
    162,
    642,
    2562,
    10242,
    40962,
    163842,
    655362,
    2621442,
]
_npzs = [f"./data/half_edge_arrays/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_]
_pkls = [f"./data/half_edge_arrays/unit_sphere_{N:07d}.pickle" for N in _NUM_VERTS_]
npzs = [
    f"./data/compressed_half_edge_arrays/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_
]
pkls = [
    f"./data/compressed_half_edge_arrays/unit_sphere_{N:07d}.pickle"
    for N in _NUM_VERTS_
]
# max_size = 40 * 1024 * 1024
# n = [load_npz(f) for f in _npzs]
# p = [load_pkl(f) for f in _pkls]
# for npz_path, pkl_path, npz_data, pkl_data in zip(npzs, pkls, n, p):
#     print(npz_path)
#     save_npz(npz_data, npz_path, compressed=True)
#     print(pkl_path)
#     save_pkl(pkl_data, pkl_path, compressed=True)
# print("done")

filename = npzs[-1]
chunk_size = "40M"
new_filename = "./data/compressed_half_edge_arrays/new_unit_sphere_2621442.npz"
chunk_file_with_split(filename, chunk_size=chunk_size)
unchunk_file_with_cat(filename, new_filename)
a = load_npz(new_filename)
a
