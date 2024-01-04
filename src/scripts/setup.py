# perform setup tasks like creating necessary directories, downloading data files, etc.
from os import system

system("rm -r ./data ")
system("mkdir ./data")
system("rm -r ./output ")
system("mkdir ./output")
