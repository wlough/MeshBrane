#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from PIL import Image

# make_sweep_grid_frame.py 3 4 ./output 250
# make_sweep_grid_frame.py 4 4 output/MID_RES_phase_diagram_sweep_radius_p65_p45_p25_p15_force_5e1_1e2_2e2_4e2 10001 -o output/grid_frame.png


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single frame from a grid of simulation PNGs."
    )

    parser.add_argument("num_rows", type=int, help="Number of simulation rows.")
    parser.add_argument("num_cols", type=int, help="Number of simulation columns.")
    parser.add_argument(
        "output_dir", type=Path, help="Simulation sweep output directory."
    )
    parser.add_argument("frame", type=int, help="Frame number (zero-indexed).")

    parser.add_argument(
        "-o",
        "--output",
        default="grid_frame.png",
        help="Output image filename (default: grid_frame.png).",
    )

    return parser.parse_args()


def get_image_path(output_dir, r, c, frame):
    run_name = f"run_{r:02d}_{c:02d}"

    return output_dir / run_name / "temp_images" / f"frame_{frame:06d}.png"


def make_grid(args):
    num_rows = args.num_rows
    num_cols = args.num_cols
    output_dir = args.output_dir.resolve()

    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("Number of rows and columns must be positive.")

    if args.frame < 0:
        raise ValueError("Frame number must be non-negative.")

    # Get dimensions from the first simulation image.
    first_image_path = get_image_path(output_dir, 0, 0, args.frame)

    if not first_image_path.is_file():
        raise FileNotFoundError(f"Image not found: {first_image_path}")

    with Image.open(first_image_path) as img:
        littleR, littleC = img.size

    # Calculate full grid dimensions.
    bigR = littleR * num_cols
    bigC = littleC * num_rows

    grid_size = (bigR, bigC)

    print(f"Grid: {num_rows} x {num_cols}")
    print(f"Frame: {args.frame}")
    print(f"Cell size: {littleR} x {littleC}")
    print(f"Grid size: {bigR} x {bigC}")

    # Create output grid.
    grid = Image.new("RGB", grid_size)

    # Assemble individual simulation frames.
    for r in range(num_rows):
        for c in range(num_cols):

            image_path = get_image_path(output_dir, r, c, args.frame)

            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            print(f"Loading {image_path}")

            with Image.open(image_path) as img:
                img = img.convert("RGB")

                if img.size != (littleR, littleC):
                    raise ValueError(
                        f"Unexpected image dimensions in {image_path}: "
                        f"{img.size}, expected {(littleR, littleC)}"
                    )

                x = c * littleR
                y = r * littleC

                grid.paste(img, (x, y))

    # Save output.
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid.save(output_path)

    print()
    print(f"Grid frame saved to: {output_path}")


def main():
    args = parse_args()

    try:
        make_grid(args)

    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
