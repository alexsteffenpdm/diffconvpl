from base import SDFGenerator2D
from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt


class SDFSphere2D(SDFGenerator2D):
    def __init__(
        self,
        dist_func: str,
        parameters: List[str],
        num_points: int,
        delta: float,
        distances: np.array,
        **kwargs,
    ):
        super().__init__(dist_func, parameters, num_points, delta, distances)

        assert "radius" in kwargs
        assert "center" in kwargs
        self.radius: float = kwargs["radius"]
        self.center: np.array = kwargs["center"]
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

    def plot(self):
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

    def generate(self):
        surface = self.on_surface_points()
        deltas = [random.gauss(self.radius, self.delta) for _ in range(self.num_points)]
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

    def color_table(self, distance: float):
        return super().color_table(distance)

    def as_json(self, filename):
        return super().as_json(filename)


if __name__ == "__main__":
    radius = 0.25
    sphere = SDFSphere2D(
        dist_func=f"lambda x,y: np.sqrt(x**2 + y**2) - {radius}",
        parameters=["x", "y"],
        num_points=100,
        delta=0.1,
        radius=radius,
        center=np.asarray([0.0, 0.0]),
        distances=np.asarray([-1.0, 0.0, 1.0]),
    )
    # sphere.plot()
    sphere.as_json(filename="2DSDF_Circle.json")
