# import h5py
import logging
import os
import numpy as np
import pickle
import yaml
from src.python.half_edge_base_brane import Brane


class SpbForce:
    def __init__(self, magnitude, length_scale):
        self.magnitude = magnitude
        self.length_scale = length_scale

    def find_center_vertices(self, m):
        x = m.xyz_coord_V[:, 0]
        xmin, xmax = np.min(x), np.max(x)
        ip = np.where(x == xmax)[0][0]
        im = np.where(x == xmin)[0][0]
        return ip, im


class StretchSim:
    """
    Simulation of membrane being stretched by equal and opposite forces
    """

    def __init__(
        self, output_dir, run_name, T, dt, dt_record_data, dt_write_data, dt_checkpoint
    ):
        self.output_dir = output_dir  # "./output/stretch_sim"
        self.run_name = run_name
        self.T = T
        self.dt = dt
        self.dt_record_data = dt_record_data
        self.dt_write_data = dt_write_data
        self.dt_checkpoint = dt_checkpoint

        self.data_path = os.path.join(self.output_dir, "data", f"{self.run_name}.h5")
        self.checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"{self.run_name}.pkl"
        )
        self.log_path = os.path.join(self.output_dir, "logs", f"{self.run_name}.log")
        self.parameters_path = os.path.join(
            self.output_dir, "input", f"{self.run_name}.yaml"
        )

        self.input_dir = os.path.join(self.output_dir, "input")
        self.raw_dir = os.path.join(self.output_dir, "raw")
        self.processed_dir = os.path.join(self.output_dir, "processed")
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_path), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized simulation with parameters: {self.__dict__}")

    @classmethod
    def from_parameters_file(cls, file_path, output_dir):

        with open(file_path, "r") as f:
            parameters = yaml.safe_load(f)
        envelope_params = parameters.pop("envelope")
        spb_force_params = parameters.pop("spb_force")
        s = cls(output_dir, **parameters)
        os.system(f"cp {file_path} {s.parameters_path}")
        if "ply_path" in envelope_params:
            ply_path = envelope_params.pop("ply_path")
            s.logger.info(f"ply_path found: {ply_path}")
            m = Brane.from_he_ply(ply_path, **envelope_params)
        else:
            # raise ValueError("ply_path must be specified in envelope parameters")
            s.logger.warning("ply_path not found in envelope parameters")
        s.envelope = m
        s.logger.info(f"Initialized envelope with parameters: {s.envelope.__dict__}")
        s.spb_force = SpbForce(**spb_force_params)
        s.logger.info(f"Initialized SPB force with parameters: {s.spb_force.__dict__}")
        return s

    @staticmethod
    def make_output_dir(output_dir="./output/stretch_sim", overwrite=False):
        """
        Create sim directories
        """

        sub_dirs = [
            "input",
            "raw",
            "processed",
            "logs",
            "checkpoints",
            "temp_images",
            "visualizations",
        ]
        if os.path.exists(output_dir) and overwrite:
            os.system(f"rm -r {output_dir}")
        elif not os.path.exists(output_dir):
            pass
        else:
            raise ValueError(
                f"{output_dir} already exists. Choose a different output_dir, or set overwrite=True"
            )
        os.system(f"mkdir -p {output_dir}")
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(output_dir, sub_dir)
            os.system(f"mkdir -p {sub_dir_path}")

    @classmethod
    def load_checkpoint(cls, run_num):
        pass

    def save_checkpoint(self):
        chkpt = os.path.join(self.checkpoints_dir, run_name)
