from src.python.half_edge_brane import BraneBuilder
from src.python.mesh_viewer import MeshViewer, downsample, downsample2
import numpy as np
from src.python.pretty_pictures import scalars_to_rgba

# b = BraneBuilder.load_test_sphere()
b = BraneBuilder.load_oblate_sphere(n_v=2)
V, F = b.xyz_array, b.V_of_F
a, aa, aaa = downsample(V, F)
# H, n = b.mean_curvature_unit_normal_V()
H, K, lapH, n = b.compute_curvature_data()
data_arrays = b.data_arrays


Fn = (
    -2
    # * b.bending_modulus
    * (
        lapH
        + 2 * (H - b.spontaneous_curvature) * (H**2 + b.spontaneous_curvature * H - K)
    )
)
A = b.barcell_area_V()


def surfvec3d(
    b,
    data_arrays,
    vec,
    scale_vec=1.0,
    alpha=0.4,
):

    # local_error = np.linalg.norm(mcvec - mcvec_actual, axis=-1)
    # V_rgba = scalars_to_rgba(local_error)
    Vkeys = sorted(b.xyz_coord_V.keys())
    V_rgba = scalars_to_rgba([-b.valence_v(v) for v in Vkeys])
    V_rgba[:, -1] = alpha
    E_rgba = np.zeros((len(b._v_origin_H), 4))
    vfdata0 = [data_arrays[0], vec]

    mv_kwargs = {
        "vector_field_data": [vfdata0],
        "V_rgba": V_rgba,
        "color_by_V_rgba": True,
        "E_rgba": E_rgba,
    }
    mv = MeshViewer(*data_arrays, **mv_kwargs)
    # mv.plot()
    mv.simple_plot()


# %%
vec = np.einsum("i,ij->ij", lapH, n)
surfvec3d(
    b,
    data_arrays,
    vec,
    scale_vec=1.0,
    alpha=0.4,
)
