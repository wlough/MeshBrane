#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command, cwd):
    print()
    print("$ " + " ".join(map(str, command)))
    subprocess.run(command, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a movie from a sequence of image frames using ffmpeg."
    )

    parser.add_argument(
        "image_dir",
        nargs="?",
        default=None,
        help="Directory containing image frames. Defaults to the current directory.",
    )

    parser.add_argument(
        "movie_dir",
        nargs="?",
        default=None,
        help="Directory where the movie should be saved. Defaults to image_dir.",
    )

    parser.add_argument(
        "-r",
        "--frame-rate",
        type=int,
        default=30,
        help="Input frame rate.",
    )

    parser.add_argument(
        "-s",
        "--frame-size",
        default="1920x1080",
        help="Frame size, for example 1920x1080.",
    )

    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=15,
        help="CRF video quality. Lower values give better quality and larger files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="movie.mp4",
        help="Output movie filename.",
    )

    return parser.parse_args()


def make_movie(args):
    image_dir = Path(args.image_dir).resolve() if args.image_dir else Path.cwd()
    movie_dir = Path(args.movie_dir).resolve() if args.movie_dir else image_dir

    image_format = "png"
    image_prefix = "frame"
    index_length = 6

    video_codec = "libx264"
    pixel_format = "yuv420p"
    filter_v = "pad=ceil(iw/2)*2:ceil(ih/2)*2,setpts=0.25*PTS"

    image_filename = f"{image_prefix}_%0{index_length}d.{image_format}"
    movie_filename = args.output

    movie_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-r",
        str(args.frame_rate),
        "-s",
        args.frame_size,
        "-i",
        image_filename,
        "-vcodec",
        video_codec,
        "-crf",
        str(args.quality),
        "-pix_fmt",
        pixel_format,
        "-filter:v",
        filter_v,
        movie_filename,
    ]

    run_command(ffmpeg_cmd, cwd=image_dir)

    source_movie = image_dir / movie_filename
    target_movie = movie_dir / movie_filename

    if image_dir != movie_dir:
        shutil.move(str(source_movie), str(target_movie))

    print()
    print(f"Movie saved at {target_movie}")


def main():
    args = parse_args()

    try:
        make_movie(args)

    except subprocess.CalledProcessError as exc:
        print()
        print(f"ffmpeg failed with exit code {exc.returncode}")
        return exc.returncode

    except FileNotFoundError as exc:
        print()
        if exc.filename == "ffmpeg":
            print("ffmpeg was not found. Make sure ffmpeg is installed and available on PATH.")
        else:
            print(f"File not found: {exc.filename}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
