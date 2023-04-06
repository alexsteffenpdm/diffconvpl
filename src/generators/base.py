import numpy as np
import random
import json
from typing import List
import os
from enum import Enum


JSON_PATH = os.path.join(os.getcwd(), "data\\generated")


class Setting(Enum):
    UNSET = 0
    DISTANCE = 1
    SURFACENORMALS = 2
    GRID_DISTANCE = 3


class SDFGenerator2D:
    def __init__(
        self,
        dist_func: str,
        parameters: List[str],
        num_points: int,
        delta: float,
        distances: np.ndarray,
        setting: Setting,
    ):
        random.seed(1)
        self.setting = setting
        self.dist_func_str = dist_func
        self.parameters = parameters
        self.dist_func = eval(dist_func)
        self.num_points = num_points
        self.data = np.zeros((num_points, 2))
        self.distances = np.zeros((num_points))
        assert len(distances) == 3
        self.color_params = distances
        self.delta = delta

    # needs to be overwritten
    def sdf_value(self, x, y, d):
        raise NotImplementedError

    # needs to be overwritten
    def on_surface_points(self, num_points: int):
        raise NotImplementedError

    # needs to be overwritten
    def generate(self):
        raise NotImplementedError

    def plot_normals(self):
        raise NotImplementedError

    def plot_distances(self):
        raise NotImplementedError

    def plot_points(self):
        raise NotImplementedError

    def plot(self):
        if self.setting == Setting.SURFACENORMALS:
            self.plot_normals()
        elif self.setting == Setting.DISTANCE:
            self.plot_distances()
        elif self.setting == Setting.GRID_DISTANCE:
            self.plot_points()

    # append datapoints
    def as_json(self, filename):
        json_data = {
            "func": self.dist_func_str.split(": ")[1],
            "params": self.parameters,
            "m": self.num_points,
            "entries": self.num_points,
            "epochs": 5000,
            "positive_funcs": 1,
            "negative_funcs": 1,
            "format": "[x,y,d]"
            if self.setting != Setting.SURFACENORMALS
            else "[x,y,n_x,n_y]",
            "data": self.data.tolist(),
            "fullplot": True,
        }

        with open(os.path.join(JSON_PATH, filename), "w") as fp:
            json.dump(json_data, fp, indent=4)

    def color_table(self, distance: float):
        # white on surface
        # red inside
        # blue outside

        if distance == 0.0:
            return f"#{255:02x}{255:02x}{255:02x}"

        elif distance > 0.0:
            percentage = abs((distance - self.color_params[1])) / abs(
                (self.color_params[2] - self.color_params[1])
            )
            rg = int(255 * percentage)

            return f"#{255-rg:02x}{255-rg:02x}{255:02x}"

        elif distance < 0.0:
            percentage = abs((distance - self.color_params[1])) / abs(
                (self.color_params[1] - self.color_params[0])
            )
            gb = int(255 * percentage)
            return f"#{255:02x}{255-gb:02x}{255-gb:02x}"

    def distance_domain(self):
        return [np.min(self.data[:, 2]), 0.0, np.max(self.data[:, 2])]
