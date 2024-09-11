from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim, SpbForce
from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

output_dir = "./output/stretch_sim"
# StretchSim.make_output_dir(output_dir=output_dir, overwrite=True)
parameters_path = "./output/test.yaml"
sim = StretchSim.from_parameters_file(parameters_path, output_dir)

# %%

m = sim.envelope
spb = sim.spb_force
ip, im = spb.find_center_vertices(m)
Vp, Hp, Fp = m.closure(*m.star_of_vertex(ip))
Vm, Hm, Fm = m.closure(*m.star_of_vertex(im))
V, H, F = Vp | Vm, Hp | Hm, Fp | Fm
patch = HalfEdgePatch(m, V, H, F)
# 55 in patch.generate_H_next_h(3)
he_samples = patch.he_samples()

m = Brane(*he_samples)
# %%

mv = MeshViewer(m)
# patch = HalfEdgePatch.from_seed_vertex(ip, m)
# patch.h_right_B
patch.expand_by_one_ring()
Fcolor = mv.colors["purple50"]
Findices = np.array(list(patch.F))
mv.update_rgba_F(Fcolor, indices=Findices)
mv.plot()

# %%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
make_output_dir(image_dir, overwrite=True)
Nrun = 0

Nlen = 5
run_name = f"run_{Nrun:0{Nlen}d}"


def get_run_dir(sim_directory="./output/point_force_sim", start=0, Nlen=6):
    # os.path.exists
    # os.path.join
    # os.path.relpath
    # os.getcwd
    # os.chdir
    # os.path.basename
    sub_dirs = ["temp_images", "sim_states"]
    Nrun = start
    run_name = f"run_{Nrun:0{Nlen}d}"
    output_dir = os.path.join(sim_dir, run_name)
    while os.path.exists(output_dir):
        Nrun += 1
        run_name = f"run_{Nrun:0{Nlen}d}"
        output_dir = os.path.join(sim_dir, run_name)
    os.system("")


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
import os

os.ex


def time_step(m, dt, ip, im, Fp, Fm):

    num_flips = m.flip_non_delaunay()
    xyz_coord_V0 = m.xyz_coord_V.copy()
    Fa = m.Farea_harmonic()
    F = Fa
    Fv = m.Fvolume_harmonic()
    F += Fv
    # print(f"Fa={np.linalg.norm(Fa, np.inf)}")
    # print(f"Fv={np.linalg.norm(Fv, np.inf)}")
    Ft = m.Ftether()
    F += Ft
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


# %%
import yaml

# Example configuration data
config_data = {
    "initial_conditions": {"temperature": 300, "pressure": 101.3},
    "simulation_params": {"timestep": 0.01, "duration": 1000},
    "environment_settings": {"gravity": 9.81, "wind_speed": 5.0},
}

# Write YAML file
with open("./output/stretch_sim/config.yaml", "w") as file:
    yaml.dump(config_data, file)


# %%
def vutukuri_vesicle(config_path="./output/stretch_sim/config.yaml"):
    # number of vertices/edges/faces
    # Nf-Ne+Nv=2
    # 2*Ne = 3*Nf
    # => Nf = 2*Nv - 4, Ne = 3*Nv - 6
    # Vutukuri actually uses Nv=30000
    # Nv = 30000
    Nv = 40962
    Ne = 3 * Nv - 6
    Nf = 2 * Nv - 4
    ply_path = "./data/half_edge_base/ply/vutukuri_vesicle_he.ply"

    # vesicle radius at equilibrium
    R = 32
    # thermal energy unit
    kBT = 0.2
    # time scale
    tau = 1.28e5

    # friction coefficient
    linear_drag_coeff = 0.4 * kBT * tau / R**2

    ##########################################
    # KMC parameters
    # flipping frequency
    flip_freq = 6.4e6 / tau
    # flipping probability
    flip_prop = 0.3

    ##########################################
    # Bending force parameters
    # bending rigidity
    bending_modulus = 20 * kBT
    # spontaneous curvature
    spontaneous_curvature = 0.0
    # splay modulus
    splay_modulus = 0.0

    ##########################################
    # Area constraint/penalty parameters
    # desired vesicle area
    A = 4 * np.pi * R**2
    # desired face area
    spontaneous_face_area = A / Nf
    # local area stiffness
    area_reg_stiffness = 6.43e6 * kBT / A

    ##########################################
    # Volume constraint/penalty parameters
    # desired vesicle volume
    spontaneous_volume = 4 * np.pi * R**3 / 3
    # volume stiffness
    volume_reg_stiffness = 1.6e7 * kBT / R**3

    ################################################
    # Edge length and tethering potential parameters
    # bond stiffness
    length_reg_stiffness = 80 * kBT
    # average bond length
    spontaneous_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))
    # minimum bond length
    min_edge_length = 0.6 * spontaneous_edge_length
    # potential cutoff lengths
    tether_cutoff_length1 = 0.8 * spontaneous_edge_length  # onset of repulsion
    tether_cutoff_length0 = 1.2 * spontaneous_edge_length  # onset of attraction
    # maximum bond length
    max_edge_length = 1.4 * spontaneous_edge_length

    Lscale = 1.0  # length scale
    Dl = 0.5 * (max_edge_length - min_edge_length)  # = 0.4 * spontaneous_edge_length
    mu = (spontaneous_edge_length - tether_cutoff_length1) / Dl  # = .5
    lam = Lscale / Dl
    nu = Lscale / Dl

    sim_kwargs = {
        "run_name": "run_0000",
        "T": 3,
        "dt": 1e-2,
        "dt_data": 1e-1,
        "dt_write_data": 5e-1,
        "dt_checkpoint": 1,
    }

    brane_kwargs = {
        "ply_path": ply_path,
        "length_reg_stiffness": float(length_reg_stiffness),
        "area_reg_stiffness": float(area_reg_stiffness),
        "volume_reg_stiffness": float(volume_reg_stiffness),
        "bending_modulus": float(bending_modulus),
        "splay_modulus": float(splay_modulus),
        "spontaneous_curvature": float(spontaneous_curvature),
        "linear_drag_coeff": float(linear_drag_coeff),
        "spontaneous_edge_length": float(spontaneous_edge_length),
        "spontaneous_face_area": float(spontaneous_face_area),
        "spontaneous_volume": float(spontaneous_volume),
        "tether_Dl": float(Dl),
        "tether_mu": float(mu),
        "tether_lam": float(lam),
        "tether_nu": float(nu),
    }

    scale = 32

    F_spb = scale * 1e1
    r_spb = scale * 0.05
    # Fp = np.array([Fmag, 0, 0])
    # Fm = np.array([-Fmag, 0, 0])
    spb_force = {"magnitude": F_spb, "radius": r_spb}

    sim_params = sim_kwargs | {"envelope": brane_kwargs, "spb_force": spb_force}

    # Write YAML file
    # with open(config_path, "w") as file:
    #     yaml.dump(sim_params, file, sort_keys=False)
    return sim_params


float(0.5)
p = vutukuri_vesicle(config_path="./output/stretch_sim/parameters.yaml")
e = p.pop("envelope")
p
