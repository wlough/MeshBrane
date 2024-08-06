import sys

sys.path.append("./")
from src.python.ply_tools import SphereFactory

SphereFactory.build_test_plys(num_refine=8, jit=True, name="unit_sphere")
