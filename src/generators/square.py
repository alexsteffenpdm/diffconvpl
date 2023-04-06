from base import SDFGenerator2D, Setting
from typing import List, Dict
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse


class SDFSquare2D(SDFGenerator2D):
    def __init__(
        self,
        dist_func: str,
        parameters: List[str],
        num_points: int,
        delta: float,
        distances: np.array,
        setting: Setting,
        **kwargs,
    ):
        super().__init__(dist_func, parameters, num_points, delta, distances, setting)

        assert "sidelength" in kwargs
        assert "center" in kwargs
        self.width: float = float(kwargs["sidelength"]["width"])
        self.height: float = float(kwargs["sidelength"]["height"])

        self.center: np.array = kwargs["center"]
        self.data: np.array = self.generate()
        self.color_params = super().distance_domain()

    def corners(self, width=None, height=None):
        width = self.width if width == None else width
        height = self.height if height == None else height
        LE = self.center[0] - width / 2.0
        LO = self.center[1] - height / 2.0
        RE = self.center[0] + width / 2.0
        UP = self.center[1] + height / 2.0

        return np.asanyarray(
            [
                [LE, LO],  # lower left 0
                [LE, UP],  # upper left 1
                [RE, LO],  # lower right 2
                [RE, UP],  # upper right 3
            ]
        )

    def sides(self):
        corners = self.corners()
        return np.asanyarray(
            [
                [corners[0], corners[1]],
                [corners[0], corners[2]],
                [corners[1], corners[3]],
                [corners[2], corners[3]],
            ]
        )

    def side_normals(self):
        sides = self.sides()
        return np.asanyarray(
            [
                (sides[1][0] - sides[1][1] / np.linalg.norm(sides[1])),
                (sides[0][0] - sides[0][1] / np.linalg.norm(sides[0])),
                (sides[2][0] + sides[2][1] / np.linalg.norm(sides[2])),
                (sides[3][0] + sides[3][1] / np.linalg.norm(sides[3])),
            ]
        )

    def sdf_value(self, x, y):
        dx = max(abs(x) - self.width / 2.0, 0)
        dy = max(abs(y) - self.height / 2.0, 0)
        d = np.sqrt(dx**2 + dy**2)

        if abs(x) < self.width / 2.0 and abs(y) < self.height / 2.0:
            d = -d
        return d

    def generate(self):
        self.num_points -= 4
        points = []
        lambdas = np.linspace(start=0, stop=1, num=2 + (self.num_points // 4))[1:-1]
        for i in lambdas:
            for side, normal in zip(self.sides(), self.side_normals()):
                p = ((1.0 - i) * side[0] + i * side[1]) * random.gauss(1.0, self.delta)
                if self.setting == Setting.DISTANCE:
                    d = self.sdf_value(p[0], p[1])
                    points.append([p[0], p[1], d])
                elif self.setting == Setting.SURFACENORMALS:
                    points.append(np.asarray([p[0], p[1], normal[0], normal[1]]))

        for corner in self.corners():
            corner_normal = (corner - self.center) / np.linalg.norm(
                corner - self.center
            )
            if self.setting == Setting.DISTANCE:
                corner *= random.gauss(1.0, self.delta)
                d = self.sdf_value(corner[0], corner[1])
                points.append([corner[0], corner[1], d])
            elif self.setting == Setting.SURFACENORMALS:
                points.append(
                    np.asarray(
                        [corner[0], corner[1], corner_normal[0], corner_normal[1]]
                    )
                )

        self.num_points += 4
        return np.asanyarray(points)

    def plot_normals(self):
        _, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], color="black", s=2)
        corners = np.asanyarray(self.corners())
        ax.scatter(corners[:, 0], corners[:, 1], color="red", s=0.9)

        for d in self.data:
            ax.arrow(
                d[0], d[1], d[2], d[3], length_includes_head=True, head_width=0.025
            )

        ax.axis("equal")
        plt.show()

    def plot_distances(self):
        _, ax = plt.subplots()
        patches = []
        for d in self.data:
            distance = d[2]
            factor = max(
                abs(d[0] - self.center[0]) / self.width,
                abs(d[1] - self.center[1]) / self.height,
            )
            corner = self.corners(
                width=(self.width * factor), height=(self.height * factor)
            )[0]
            patches.append(
                (distance, corner, (self.width * factor), (self.height * factor))
            )
        for p in sorted(patches, key=lambda x: (x[0]))[::-1]:
            rec = plt.Rectangle(
                xy=p[1],
                width=p[2],
                height=p[3],
                color=self.color_table(p[0]),
                fill=True,
            )
            ax.add_patch(rec)
        ax.scatter(self.data[:, 0], self.data[:, 1], color="black", s=2)
        corners = np.asanyarray(self.corners())
        ax.scatter(corners[:, 0], corners[:, 1], color="red", s=0.9)
        ax.axis("equal")
        plt.show()

    def plot(self):
        return super().plot()

    def color_table(self, distance: float):
        return super().color_table(distance)

    def as_json(self, filename):
        return super().as_json(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF Data Generator - Rectangle")
    parser.add_argument(
        "--setting",
        metavar=str,
        default="distance",
        action="store",
        nargs="?",
        help="Compute SDF sample data (distance or normals)",
    )

    parser.add_argument(
        "--width",
        metavar=float,
        default=1.0,
        action="store",
        nargs="?",
        help="Set width of the rectangle.",
    )
    parser.add_argument(
        "--height",
        metavar=float,
        default=1.0,
        action="store",
        nargs="?",
        help="Set height of the rectangle.",
    )

    parser.add_argument(
        "--datapoints",
        metavar=int,
        default=100,
        action="store",
        nargs="?",
        help="Set the amount of datapoints generated.",
    )

    parser.add_argument(
        "--delta",
        metavar=float,
        default=0.0,
        action="store",
        nargs="?",
        help="Set the amount of datapoints generated.",
    )
    args = parser.parse_args()
    _setting: Setting = Setting.UNSET
    width = float(args.width)
    height = float(args.height)

    if args.setting not in ["distance", "normals"]:
        raise ValueError(f"Unkown Setting {args.setting}")
    elif args.setting == "distance":
        _setting = Setting.DISTANCE
    elif args.setting == "normals":
        _setting = Setting.SURFACENORMALS
    assert width > 0.0 and height > 0.0
    assert (
        int(args.datapoints) % 4 == 0 and int(args.datapoints) >= 4
    ), "Cannot build a rectangle with less than 4 points."

    square = SDFSquare2D(
        dist_func=f" lambda x, y: torch.sqrt(torch.pow(torch.max(torch.abs(x) -  {width/2},0).values,2) + torch.pow(torch.max(torch.abs(y) - {height/2},0).values,2))",
        parameters=["x,y"],
        num_points=int(args.datapoints),
        delta=float(args.delta),
        sidelength={"width": width, "height": height},
        center=np.asarray([0.0, 0.0]),
        distances=np.asarray([-0.5, 0.0, 0.5]),
        setting=_setting,
    )
    square.plot()
    square.as_json(filename="2DSDF_Rectangle.json")
