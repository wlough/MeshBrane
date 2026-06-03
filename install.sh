#!/bin/bash

# ---------------------------------------------------------------

# This script is used to build MeshBrane and install PyBrane.
# ---------------------------------------------------------------
tests=false # -t run tests
verbose=false # verbose output
clear=false # -c clear venv and build directory before install
build_docs=false # -d build docs
# --------------------------------------------------------------

show_help() {
    echo "USAGE:"
    echo "  $0 [-hreigntvcpsd]"
    echo "OPTIONS:"
    echo "  -t      run (t)ests"
    echo "  -v      (v)erbose output"
    echo "  -c      (c)lear build files"
    echo "  -d      build (d)ocs"
}


##########################
# Adjust install based on optional arguments to the script
# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.
while getopts "h?tvcd" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    t)
        tests=true
        echo "MESHBRANE_RUN_TESTS=ON" >> env.sh
        ;;
    v)
        verbose=true
        ;;
    c)
        clear=true
        ;;
    d)
        build_docs=true
        ;;
    esac
done


shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

clean_build_files() {
    rm -rf build/CMake*
    rm -rf build/Makefile
    rm -rf build/cmake_install.cmake
    rm -rf build/src
    rm -rf build/Doxyfile
    rm -rf build/tests
}

do_meshbrane_build() {
    echo ""
    echo "Building MeshBrane"
    echo "------------------"
    CMAKE_FLAGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    if [ "$verbose" = true ]; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_VERBOSE_MAKEFILE=ON"
    fi

    if [ "$tests" = true ]; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DTESTS=ON"
    fi

    # mkdir -p build
    # cd build || exit 1
    # cmake ${CMAKE_FLAGS} ..
    # make -j8
    cmake -S . -B build $CMAKE_FLAGS
    cmake --build build --parallel 16
    # cmake --install build
    if [ $? -ne 0 ]; then
        echo "Build failed"
        exit 1
    fi
    # cd ..
    echo "------------------"
    echo "Build complete"
}

main() {
    if [ "$clear" = true ]; then
        clean_build_files
    fi
    # if submodules will be used, update them
    # if [ "$skip_build" != true ] || [ "$install" = true ]; then
    #     git submodule update --init --recursive
    # fi
    # build MeshBrane
    if [ "$skip_build" != true ]; then
        do_meshbrane_build
        if [ $tests = true ]; then
            tests/run_tests.sh
        fi
    fi
    if [ "$build_docs" = true ]; then
        echo "Building documentation"
        doxygen Doxyfile
        if [ $? -ne 0 ]; then
            echo "Documentation build failed"
            exit 1
        fi
        echo "Documentation built"

    fi


}


main
