from base import SDFGenerator2D, Setting
from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

class SDFSphere2D(SDFGenerator2D):
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
        super().__init__(dist_func, parameters, num_points, delta, distances,setting)

        assert "radius" in kwargs
        assert "center" in kwargs
        self.radius: float = kwargs["radius"]
        self.center: np.array = kwargs["center"]
        if "griddomain" in kwargs:
            self.griddomain = kwargs["griddomain"]
        self.data = self.generate()

    def sdf_value(self, x, y):
        return np.sqrt(x**2 + y**2) - self.radius

    def on_surface_points(self):
        return [
            (
                np.cos(2 * np.pi / self.num_points * x) * self.radius,
                np.sin(2 * np.pi / self.num_points * x) * self.radius,
            )
            for x in range(0, self.num_points + 1)
        ]

    def plot_normals(self):
        _, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], color="black",s=2)

        for d in self.data:
            ax.arrow(d[0],d[1],d[2],d[3],length_includes_head=True,head_width=0.025)

        ax.axis("equal")
        plt.show()

    def plot_distances(self):
        _, ax = plt.subplots()

        patches = []
        for d in self.data:
            patches.append(
                (
                    np.sqrt(d[0] ** 2 + d[1] ** 2),
                    self.color_table(self.sdf_value(d[0], d[1])),
                )
            )

        # ensure that surface is represented within the generated data
        patches.append((1.0, self.color_table(0.0)))

        for p in sorted(patches, key=lambda x: (x[0]))[::-1]:
            circle = plt.Circle(
                (self.center[0], self.center[1]), p[0], color=p[1], fill=True
            )
            ax.add_patch(circle)

        ax.scatter(self.data[:, 0], self.data[:, 1], color="black", s=0.02)
        plt.show()

    def plot_points(self):
        _, ax = plt.subplots()
        
        ax.scatter(self.data[:, 0], self.data[:, 1], color="black", s=0.02)
        plt.show()

    def plot(self):
        if self.setting == Setting.DISTANCE:
            return self.plot_distances()
        
        return super().plot()
        
    def grid_data(self):
        data = []
        domain = np.linspace(self.griddomain[0],self.griddomain[1],self.num_points +1)
        x,y = np.meshgrid(domain,domain)
        for _ in range(self.num_points):
            xi = x[random.randint(0,len(domain)-1),random.randint(0,len(domain)-1)]
            yi = y[random.randint(0,len(domain)-1),random.randint(0,len(domain)-1)]
            
            di = self.sdf_value(xi,yi)
            data.append([xi,yi,di])
        return np.asanyarray(data)


    def generate(self):
        surface = self.on_surface_points()

        if self.setting == Setting.DISTANCE:
            if self.delta != 0.0:
                deltas = [random.gauss(self.radius, self.delta) for _ in range(self.num_points)]
            else:
                deltas = [1.0 for _ in range(self.num_points)]
                
            surface = [
                (delta_i * surface_i[0], delta_i * surface_i[1])
                for delta_i, surface_i in zip(deltas, surface)
            ]

            distances = [
                self.sdf_value(surface_i[0], surface_i[1]) for surface_i in surface
            ]

            if self.center.all() != 0.0:
                surface = [
                    (surface_i[0] + self.center[0], surface_i[1] + self.center[1])
                    for surface_i in surface
                ]

            return np.asanyarray(
                [
                    (surface_i[0], surface_i[1], distance)
                    for surface_i, distance in zip(surface, distances)
                ]
            )
        elif self.setting == Setting.SURFACENORMALS:
            normals = [point / np.linalg.norm(point) for point in surface]
            return np.asanyarray(
                [
                    (
                        surface_i[0],surface_i[1],normal_i[0],normal_i[1]
                    )
                    for surface_i, normal_i in zip(surface,normals)
                ]
            )

        elif self.setting == Setting.GRID_DISTANCE:
            return self.grid_data()


    def color_table(self, distance: float):
        return super().color_table(distance)

    def as_json(self, filename):
        return super().as_json(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF Data Generator - Sphere")
    parser.add_argument(
        "--setting",
        metavar=str,
        default="distance",
        action="store",
        nargs="?",
        help="Compute SDF sample data (distance or normals)"
        )
    
    parser.add_argument(
        "--radius",
        metavar=float,
        default=None,
        action="store",
        nargs="?",
        help="Set the radius for the generated sphere surface."
    )

    parser.add_argument(
        "--datapoints",
        metavar=int,
        default=100,
        action="store",
        nargs="?",
        help="Set the amount of datapoints generated."
    )

    parser.add_argument(
        "--delta",
        metavar=float,
        default=None,
        action="store",
        nargs="?",
        help="Randomly varies the range of distances for generated points. Only applicable with setting 'distance'."
    )

    parser.add_argument(
        "--gridsize",
        metavar=float,
        default=None,
        action="store",
        nargs="?",
        help="Specifies the factor of the grid created based on the given radius. (i.e. -radius * gridsize <= data <= radius * gridsize)"
    )

    args = parser.parse_args()
    _setting:Setting = Setting.UNSET
    
    if args.setting not in ["distance","normals","grid"]:
        raise ValueError(f"Unkown Setting {args.setting}")
    elif args.setting == "distance":
        _setting = Setting.DISTANCE
    elif args.setting == "normals":
        _setting = Setting.SURFACENORMALS
    elif args.setting == "grid":
        _setting = Setting.GRID_DISTANCE

    if args.delta is not None:
        assert _setting == Setting.DISTANCE or Setting.GRID_DISTANCE,"Delta can only be specified when using the 'distance' or 'grid' setting."
    
    assert float(args.radius) > 0.0
    assert int(args.datapoints) > 0

    gridsize = float(args.gridsize) * float(args.radius) if args.gridsize is not None else 0.0
    

    sphere = SDFSphere2D(
        dist_func=f"lambda x,y: np.sqrt(x**2 + y**2) - {args.radius}",
        parameters=["x", "y"],
        num_points=int(args.datapoints),
        delta=0.0 if args.delta is None else float(args.delta),
        radius=float(args.radius),
        center=np.asarray([0.0, 0.0]),
        distances=np.asarray([-np.Inf, 0.0, np.Inf]),
        setting=_setting,
        griddomain=np.asarray([-gridsize,gridsize])
    )
    sphere.plot()
    sphere.as_json(filename="2DSDF_Circle.json")
