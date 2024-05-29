import os


def make_output_directory(output_directory):
    os.system(f"rm -r {output_directory}")
    os.system(f"mkdir {output_directory}")
    os.system(f"mkdir {output_directory}/ply_files")
    os.system(f"mkdir {output_directory}/temp_images")
    os.system(f"mkdir {output_directory}/logs")
    os.system(f"mkdir {output_directory}/config")
    os.system(f"mkdir {output_directory}/checkpoints")
