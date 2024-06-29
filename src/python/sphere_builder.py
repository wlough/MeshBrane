from src.python.ply_tools import SphereBuilder


def build_spheres():
    sb = SphereBuilder()
    sb.write_plys()

    # 12
    # 42
    # 162
    # 642
    # 2562
    # 10242

    for iter in range(5):
        sb.divide_faces()
        sb.write_plys()
        print(sb.name+" -done")
