from src.utils import generate_sphere_oblate_torus_dumbbell

# import os
#
# a = os.system("ls")
# print("generating course meshes")
# generate_sphere_oblate_torus_dumbbell(
#     reg=False, coarse=True, fine=False, ultrafine=False
# )
# print("generating fine meshes")
# generate_sphere_oblate_torus_dumbbell(
#     reg=False, coarse=True, fine=True, ultrafine=False
# )

print("generating ultrafine meshes")
generate_sphere_oblate_torus_dumbbell(
    reg=False, coarse=True, fine=False, ultrafine=True
)
