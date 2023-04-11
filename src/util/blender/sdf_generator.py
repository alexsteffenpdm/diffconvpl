import os
from scene_handler import BlenderSceneHandler
from typing import Any
import json
import numpy as np
from tqdm import tqdm
import random


JSON_PATH = os.path.join(os.getcwd(), "data\\generated")


class BlenderSDFGenerator:
    def __init__(self, handler_dict: dict[str, Any], json_dict: dict[str, Any]):
        random.seed(1)
        self.handler = BlenderSceneHandler(
            mode=handler_dict["mode"],
            filepath_scene=handler_dict["filepath_scene"],
            noise=handler_dict["noise"],
            z_height=handler_dict["z_height"],
        )
        self.json_values = json_dict

    def datapoints_per_edge(self, edges: np.ndarray) -> np.ndarray:
        lengths = [np.linalg.norm(e[0] - e[1]) for e in edges]
        datapoint_distribution = [
            int(round(self.json_values["entries"] * (l / sum(lengths))))
            for l in lengths
        ]

        while sum(datapoint_distribution) < self.json_values["entries"]:
            index = np.argmax(datapoint_distribution)
            datapoint_distribution[index] += 1

        return datapoint_distribution

    def generate_points(self, vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
        _verts = []

        distributions = self.datapoints_per_edge(edges)
        spacing = [np.linspace(0.0, 1.0, dist) for dist in distributions]

        pbar = tqdm(total=self.json_values["entries"])

        for e, s in enumerate(spacing):
            for a in s:
                edge = edges[e]
                v = (1 - a) * vertices[edge[0]] + a * vertices[edge[1]]
                _verts.append(v)
                pbar.update(1)
        pbar.close()

        return _verts

    def apply_noise(self, datapoints: np.ndarray):
        noise = np.asanyarray(
            [
                random.gauss(0.0, self.handler.noise / 2)
                for _ in range(self.json_values["entries"])
            ]
        )

        return np.asanyarray(
            [[dp[0] * (1 + n), dp[1] * (1 + n)] for dp, n in zip(datapoints, noise)]
        )

    def get_negtiave_z_limit(self, vertices: np.ndarray) -> float:
        center = sum(vertices) / len(vertices)
        return np.min(np.asarray([np.linalg.norm(abs(center - v)) for v in vertices]))

    def make_sdf(self):
        sdf_arr = []
        vertices, edges = self.handler.layercut()
        levels = np.asanyarray(
            [random.gauss(0.0, self.handler.noise / 2) for _ in range(100)]
        )

        meta_points = self.generate_points(vertices, edges)
        datapoints = []

        for l in levels:
            for v in meta_points:
                x, y = v[0] * (1 - l), v[1] * (1 - l)
                z = l * -1
                datapoints.append([x, y, z])

        np.random.shuffle(datapoints)

        for _ in range(self.json_values["entries"]):
            e = random.randint(0, len(datapoints))
            sdf_arr.append(datapoints[e])

        self.json_values["data"] = sdf_arr

        pos_edges = 0
        neg_edges = 0
        for e in edges:
            v1 = vertices[e[0]]
            v2 = vertices[e[1]]
            direction = v2 - v1
            if direction[0] < 0:
                neg_edges += 1
            else:
                pos_edges += 1

        self.json_values["data"] = sdf_arr
        self.json_values["statistics"] = {
            "num_datapoints": len(sdf_arr),
            "num_edges": len(edges),
            "pos_edges": pos_edges,
            "neg_edges": neg_edges,
        }

    def as_json(self, filename):
        with open(os.path.join(JSON_PATH, filename), "w") as fp:
            json.dump(self.json_values, fp, indent=4)


if __name__ == "__main__":
    handler_dict = {
        "z_height": 0.15,
        "filepath_scene": "C:\\Users\\skyfe\\Development\\diffconvpl\\src\\assets\\generator_test_suzie.blend",
        "noise": 0.25,
        "mode": "layercut",
    }

    json_dict = {
        "params": ["x", "y"],
        "m": 1000,
        "entries": 5000,
        "epochs": 5000,
        "positive_funcs": 1,
        "negative_funcs": 1,
        "format": "[x,y,d]",
        "fullplot": True,
        "statistics": {
            "num_datapoints": 0,
            "num_edges": 0,
            "pos_edges": 0,
            "neg_edges": 0,
        },
        "data": None,
    }

    generator = BlenderSDFGenerator(handler_dict=handler_dict, json_dict=json_dict)
    generator.make_sdf()
    generator.as_json(filename="star_test.json")
