import numpy as np
import sympy as sp

#
# Extended Data Table 1 | Parameters used in 3D simulations
keys_principal_properties = [
    "vesicle_radius_in_equilibrium",
    "thermal_energy_unit",
    "time_scale",
]

keys_vesicle_properties = [
    "number_of_vertices",
    "number_of_edges",
    "number_of_faces",
    "bending_rigidity",
    "average_bond_length",
    "bond_stiffness",
    "minimum_bond_length",
    "potential_cutoff_length0",
    "potential_cutoff_length1",
    "maximum_bond_length",
    "desired_vesicle_area",
    "desired_vesicle_volume",
    "local_area_stiffness",
    "volume_stiffness",
    "friction_coefficient",
    "flipping_frequency",
    "flipping_probability",
]

keys_particle_properties = ["friction_particle", "volume_fraction"]

keys_all = keys_principal_properties + keys_vesicle_properties + keys_particle_properties

tex_math_mode = {
    "vesicle_radius_in_equilibrium": r"R",
    "thermal_energy_unit": r"k_{B}T",
    "time_scale": r"\tau",
    "number_of_vertices": r"N_v",
    "number_of_edges": r"N_e",
    "number_of_faces": r"N_f",
    "bending_rigidity": r"\kappa_c",
    "average_bond_length": r"\ell_b",
    "bond_stiffness": r"k_b",
    "minimum_bond_length": r"\ell_{min}",
    "potential_cutoff_length1": r"\ell_{c1}",
    "potential_cutoff_length0": r"\ell_{c0}",
    "maximum_bond_length": r"\ell_{max}",
    "desired_vesicle_area": r"A",
    "desired_vesicle_volume": r"V_0",
    "local_area_stiffness": r"k_{\ell}",
    "volume_stiffness": r"k_v",
    "friction_coefficient": r"\gamma_m",
    "flipping_frequency": r"\omega",
    "flipping_probability": r"\psi",
    "friction_particle": r"\gamma_p",
}
sym_parameters_principal_properties = {_: sp.symbols(tex_math_mode[_]) for _ in keys_principal_properties}
sym_parameters_vesicle_properties = {_: sp.symbols(tex_math_mode[_]) for _ in keys_vesicle_properties}
sym_parameters = sym_parameters_principal_properties | sym_parameters_vesicle_properties

sym_subs_parameters = {
    sym_parameters["number_of_edges"]: 3 * sym_parameters["number_of_vertices"] - 6,
    sym_parameters["number_of_faces"]: 2 * sym_parameters["number_of_vertices"] - 4,
}


model_units_principal_properties = {
    "vesicle_radius_in_equilibrium": 32.0,
    "thermal_energy_unit": 0.2,
    "time_scale": 1.28e5,
}
model_units_vesicle_properties = {
    "number_of_vertices": 3 * 10**4,
    # "number_of_edges": ,
    # "number_of_faces": ,
    "bending_rigidity": 20 * parameters["thermal_energy_unit"],
    "average_bond_length": r"\ell_b",
    "bond_stiffness": r"k_b",
    "minimum_bond_length": r"\ell_{min}",
    "potential_cutoff_length1": r"\ell_{c1}",
    "potential_cutoff_length0": r"\ell_{c0}",
    "maximum_bond_length": r"\ell_{max}",
    "desired_vesicle_area": r"A",
    "desired_vesicle_volume": r"V_0",
    "local_area_stiffness": r"k_{\ell}",
    "volume_stiffness": r"k_v",
    "friction_coefficient": r"\gamma_m",
    "flipping_frequency": r"\omega",
    "flipping_probability": r"\psi",
}

Nv = parameters["number_of_vertices"]
Ne = parameters["number_of_edges"]
Nf = parameters["number_of_faces"]
Ne = 3 * Nf / 2
b = 0
g = 0
chi = 2 - 2 * g - b


Nf = sp.solve(Nf - Ne + Nv - chi, Nf, list=True)[0]
Ne = 3 * Nf / 2
# %%
