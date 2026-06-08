#!/usr/bin/env python3

"""
Build script for MeshBrane.

Usage:
    python install.py [-t] [-v] [-c] [-d]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_DIR / "build"


def run_command(command, cwd=PROJECT_DIR):
    print(f"\n$ {' '.join(map(str, command))}")
    subprocess.run(command, cwd=cwd, check=True)


def clean_build_files():
    print("Removing build directory")

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


def build_meshbrane(args):
    print()
    print("Building MeshBrane")
    print("------------------")

    cmake_configure_cmd = [
        "cmake",
        "-S",
        str(PROJECT_DIR),
        "-B",
        str(BUILD_DIR),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]

    if args.verbose:
        cmake_configure_cmd.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

    if args.tests:
        cmake_configure_cmd.append("-DTESTS=ON")

    run_command(cmake_configure_cmd)

    parallel_jobs = str(args.parallel or os.cpu_count() or 1)

    cmake_build_cmd = [
        "cmake",
        "--build",
        str(BUILD_DIR),
        "--parallel",
        parallel_jobs,
    ]

    run_command(cmake_build_cmd)

    print("------------------")
    print("Build complete")


def run_tests():
    # CMake needs to register tests using add_test(...)
    """
    Run tests using CTest.
    """
    print()
    print("Running tests")
    print("-------------")

    run_command(
        [
            "ctest",
            "--test-dir",
            str(BUILD_DIR),
            "--output-on-failure",
        ]
    )

    print("Tests complete")


def build_docs():
    print()
    print("Building documentation")
    print("----------------------")

    run_command(["doxygen", "Doxyfile"])

    print("Documentation built")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MeshBrane.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--tests",
        action="store_true",
        help="Build and run tests.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose CMake output.",
    )

    parser.add_argument(
        "-c",
        "--clear",
        action="store_true",
        help="Clear generated build files before building.",
    )

    parser.add_argument(
        "-d",
        "--docs",
        action="store_true",
        help="Build documentation with Doxygen.",
    )

    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel build jobs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.clear:
            clean_build_files()

        build_meshbrane(args)

        if args.tests:
            run_tests()

        if args.docs:
            build_docs()

    except subprocess.CalledProcessError as exc:
        print()
        print(f"Command failed with exit code {exc.returncode}:")
        print(" ".join(map(str, exc.cmd)))
        return exc.returncode

    except FileNotFoundError as exc:
        print()
        print(f"Required executable not found: {exc.filename}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
