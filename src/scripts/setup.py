# perform setup tasks like creating necessary directories, downloading data files, etc.
from os import system
from src.utils import save_all_sample_meshes

system("rm -r ./data ")
system("mkdir ./data")
system("mkdir ./data/ply_files")
save_all_sample_meshes()

system("rm -r ./output ")
system("mkdir ./output")
