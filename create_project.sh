#!/bin/bash

# Define the project name
# project_name="MeshBrane"

# Create the main project directory
# mkdir $project_name

# Navigate into the project directory
# cd $project_name

# python and c++ source files
mkdir -p src/python src/cpp
mkdir -p tests/python tests/cpp
# c++ header files
mkdir include
# external libraries
mkdir lib
# documentation, build files, etc
mkdir docs build bin
# ply,default json, etc.. files
mkdir data
mkdir -p data/ply/ascii data/ply/binary
mkdir data/config
# other stuff 
mkdir scripts
mkdir prototyping
mkdir output
touch README.md

# Navigate back to the original directory
# cd ..