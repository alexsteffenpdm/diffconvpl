from base import SDFGenerator2D
from typing import List, Dict
import numpy as np
import random
import matplotlib.pyplot as plt


class SDFSquare2D(SDFGenerator2D):
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

        assert "sidelength" in kwargs
        assert "center" in kwargs
        self.sidelength: Dict[str:float] = kwargs["sidelength"]
        self.center: np.array = kwargs["center"]
        self.data = self.generate()
        self.color_params = super().distance_domain()

    def corners(self):
        a = self.sidelength["a"]
        b = self.sidelength["b"]
        return [
            self.center + [-a / 2, b / 2],
            self.center + [a / 2, b / 2],
            self.center + [-a / 2, -b / 2],
            self.center + [a / 2, -b / 2],
        ]

    def sides(self):

        corners = self.corners()
        return [
            (corners[0], corners[1]),  # UL
            (corners[0], corners[2]),  # UR
            (corners[1], corners[3]),  # LL
            (corners[2], corners[3]),  # LR
        ]

    def sdf_value(self, x, y):
        distances = []
        sides = self.sides()
        for side in sides:
            d = np.linalg.norm(
                np.cross(side[1] - side[0], side[0] - (x, y))
            ) / np.linalg.norm(side[1] - side[0])
            distances.append(d)

        assert len(distances) == 4
        p0 = sides[0][0]
        p3 = sides[2][1]
        if np.argmin(distances) == 0:
            if y < p0[1]:
                distances[0] * -1.0

        elif np.argmin(distances) == 1:
            if x > p0[0]:
                distances[1] * -1.0

        elif np.argmin(distances) == 2:

            if x < p3[0]:
                distances[2] * -1.0
        else:

            if y > p3[1]:
                distances[3] * -1.0
        if x > -0.5 and x < 0.5 and y > -0.5 and y < 0.5:
            return -1.0 * np.min(distances)
        return np.min(distances)

    def on_surface_points(self):
        points = []

        for i in np.linspace(start=0.0, stop=1.0, num=(self.num_points // 4)):
            for side in self.sides():
                p = (1.0 - i) * side[0] + i * side[1]
                points.append(p)

        return points

    def plot(self):
        corner = self.corners()[2]
        _, ax = plt.subplots()

        patches = []

        for d in self.data:
            distance = d[2]

            new_corner = corner - [distance, distance]
            new_width = self.sidelength["a"] + 2 * distance
            new_height = self.sidelength["b"] + 2 * distance

            patches.append((distance, new_corner, new_width, new_height))
            # rec = plt.Rectangle(new_corner,width=new_width,height=new_height,color=self.color_table(distance))

        # ensure that surface is represented within the generated data
        patches.append((0.0, corner, self.sidelength["a"], self.sidelength["b"]))

        for p in sorted(patches, key=lambda x: (x[0]))[::-1]:
            rec = plt.Rectangle(
                xy=p[1],
                width=p[2],
                height=p[3],
                color=self.color_table(p[0]),
                fill=True,
            )
            ax.add_patch(rec)
        # for i,d in enumerate(self.data):
        #     print(i,d)
        #     ax.scatter(d[0],d[1],color="black",s=2)
        ax.scatter(self.data[:, 0], self.data[:, 1], color="black", s=2)
        corners = np.asanyarray(self.corners())
        ax.scatter(corners[:, 0], corners[:, 1], color="red", s=0.9)
        plt.show()

    def generate(self):
        surface = self.on_surface_points()
        long_side = (
            self.sidelength["a"]
            if self.sidelength["a"] > self.sidelength["b"]
            else self.sidelength["b"]
        )
        deltas = [random.gauss(long_side, self.delta) for _ in range(self.num_points)]
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

    square = SDFSquare2D(
        dist_func=f"lambda x,y: np.sqrt(x**2 + y**2)",
        parameters=["x,y"],
        num_points=100,
        delta=0.5,
        sidelength={"a": 1.0, "b": 1.0},
        center=np.asarray([0.0, 0.0]),
        distances=np.asarray([-0.5, 0.0, 0.5]),
    )
    square.plot()
    square.as_json(filename="2DSDF_Rectangle_BETA.json")
