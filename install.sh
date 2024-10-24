#!/bin/bash

clean_cmake_files() {
    rm -rf build/CMake*
    rm -rf build/Makefile
    rm -rf build/cmake_install.cmake
    rm -rf build/src
    rm -rf build/Doxyfile
    rm -rf build/tests
}

find_pip_requirements() {
    local venv_path=$1
}

create_python_venv() {
    local venv_path=$1
    if [ -z "$venv_path" ]; then
        echo "Error: No path provided for Python virtual environment."
        exit 1
    fi

    python3 -m venv "$venv_path"
    source "$venv_path/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
}

show_help() {
    echo "Usage:"
    echo "  $0 [-h] [-c] [-p path_to_venv]"
    echo "OPTIONS:"
    echo "  -h      show this menu"
    echo "  -c      clean build directory"
    echo "  -p      create Python virtual environment, install required packages"
}

while getopts "hcp:" opt; do
    case ${opt} in
        h )
            show_help
            exit 0
            ;;
        c )
            clean_cmake_files
            exit 0
            ;;
        p )
            create_python_venv "$OPTARG"
            exit 0
            ;;
        \? )
            show_help
            exit 1
            ;;
    esac
done

# If no options were provided, show help
if [ $OPTIND -eq 1 ]; then
    show_help
    exit 1
fi