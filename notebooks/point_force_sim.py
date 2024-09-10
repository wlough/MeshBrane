from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer
import numpy as np
from src.python.utilities import make_output_dir

output_dir = "./output/point_force_sim"
image_dir = f"{output_dir}/temp_images"
make_output_dir(image_dir, overwrite=True)


m = Brane.unscaled_vutukuri_vesicle()
R = 32
com = np.mean(m.xyz_coord_V, axis=0)
R0 = np.linalg.norm(m.xyz_coord_V - com, axis=-1)
m.xyz_coord_V = (R * (m.xyz_coord_V - com).T / R0).T

# mv = MeshViewer(m)
# mv.plot()
# x = m.xyz_coord_V[:, 0]
# xmin, xmax = np.min(x), np.max(x)
# ip = np.where(x == xmax)[0][0]
# im = np.where(x == xmin)[0][0]
scale = 32
ip = 37
im = 32
Fmag = 32 * 1e1
Fp = np.array([Fmag, 0, 0])
Fm = np.array([-Fmag, 0, 0])

dt = 1e-2
Trun = 1.0


def time_step(m, dt, ip, im, Fp, Fm):

    num_flips = m.flip_non_delaunay()
    xyz_coord_V0 = m.xyz_coord_V.copy()
    Fa = m.Farea_harmonic()
    F = Fa
    Fv = m.Fvolume_harmonic()
    F += Fv
    # print(f"Fa={np.linalg.norm(Fa, np.inf)}")
    # print(f"Fv={np.linalg.norm(Fv, np.inf)}")
    # Ft = m.Ftether()
    # F += Ft
    Fb = m.Fbend_analytic()
    F += Fb
    # print(f"F={np.linalg.norm(F, np.inf)}")
    F[ip] += Fp
    F[im] += Fm
    Dxyz_coord_V = dt * F / m.linear_drag_coeff
    # print(f"Dxyz_coord_V={np.linalg.norm(Dxyz_coord_V, np.inf)}")
    xyz_coord_V = xyz_coord_V0 + Dxyz_coord_V
    # xyz_coord_V = f.project_to_torus(*xyz_coord_V.T)
    m.xyz_coord_V = xyz_coord_V
    # vecs = 0.25 * F / np.linalg.norm(F, np.inf)
    vecs = np.zeros_like(F)
    vecs[ip] += 0.25 * scale * Fp / np.linalg.norm(Fp)
    vecs[im] += 0.25 * scale * Fm / np.linalg.norm(Fm)
    return m.xyz_coord_V, vecs


def run(m, dt, ip, im, Fp, Fm, Trun):
    view = {
        # "azimuth": 45.0,
        "azimuth": 90.0,
        "elevation": 54.7,
        "distance": 216.0,
        "focalpoint": np.array([0.0, 0.0, 0.0]),
    }
    mv = MeshViewer(
        m,
        figsize=(720, 720),
        image_dir=image_dir,
        view=view,
    )
    t = 0
    mv.plot(save=True, show=False, title=f"{t=}")
    while t <= Trun:

        points, vectors = time_step(m, dt, ip, im, Fp, Fm)
        mv.clear_vector_field_data()
        mv.add_vector_field(points, vectors)
        # com = np.sum(m.xyz_coord_V, axis=0) / m.num_vertices
        # self.view["focalpoint"] = com
        t += dt
        print(f"{t=}                ", end="\r")
        mv.plot(save=True, show=False, title=f"{t=}")
    mv.movie()


run(m, dt, ip, im, Fp, Fm, Trun)

output_dir = "./output/point_force_sim2"
image_dir = f"{output_dir}/temp_images"
make_output_dir(image_dir, overwrite=True)


m = Brane.vutukuri_vesicle()


# mv = MeshViewer(m)
# mv.plot()
# x = m.xyz_coord_V[:, 0]
# xmin, xmax = np.min(x), np.max(x)
# ip = np.where(x == xmax)[0][0]
# im = np.where(x == xmin)[0][0]
scale = 32
ip = 37
im = 32
Fmag = 32 * 1e1
Fp = np.array([Fmag, 0, 0])
Fm = np.array([-Fmag, 0, 0])

dt = 1e-2
Trun = 1.0


def time_step(m, dt, ip, im, Fp, Fm):

    num_flips = m.flip_non_delaunay()
    xyz_coord_V0 = m.xyz_coord_V.copy()
    Fa = m.Farea_harmonic()
    F = Fa
    Fv = m.Fvolume_harmonic()
    F += Fv
    # print(f"Fa={np.linalg.norm(Fa, np.inf)}")
    # print(f"Fv={np.linalg.norm(Fv, np.inf)}")
    # Ft = m.Ftether()
    # F += Ft
    Fb = m.Fbend_analytic()
    F += Fb
    # print(f"F={np.linalg.norm(F, np.inf)}")
    F[ip] += Fp
    F[im] += Fm
    Dxyz_coord_V = dt * F / m.linear_drag_coeff
    # print(f"Dxyz_coord_V={np.linalg.norm(Dxyz_coord_V, np.inf)}")
    xyz_coord_V = xyz_coord_V0 + Dxyz_coord_V
    # xyz_coord_V = f.project_to_torus(*xyz_coord_V.T)
    m.xyz_coord_V = xyz_coord_V
    # vecs = 0.25 * F / np.linalg.norm(F, np.inf)
    vecs = np.zeros_like(F)
    vecs[ip] += 0.25 * scale * Fp / np.linalg.norm(Fp)
    vecs[im] += 0.25 * scale * Fm / np.linalg.norm(Fm)
    return m.xyz_coord_V, vecs


def run(m, dt, ip, im, Fp, Fm, Trun):
    view = {
        # "azimuth": 45.0,
        "azimuth": 90.0,
        "elevation": 54.7,
        "distance": 216.0,
        "focalpoint": np.array([0.0, 0.0, 0.0]),
    }
    mv = MeshViewer(
        m,
        figsize=(720, 720),
        image_dir=image_dir,
        view=view,
    )
    t = 0
    mv.plot(save=True, show=False, title=f"{t=}")
    while t <= Trun:

        points, vectors = time_step(m, dt, ip, im, Fp, Fm)
        mv.clear_vector_field_data()
        mv.add_vector_field(points, vectors)
        # com = np.sum(m.xyz_coord_V, axis=0) / m.num_vertices
        # self.view["focalpoint"] = com
        t += dt
        print(f"{t=}                ", end="\r")
        mv.plot(save=True, show=False, title=f"{t=}")
    mv.movie()


run(m, dt, ip, im, Fp, Fm, Trun)
