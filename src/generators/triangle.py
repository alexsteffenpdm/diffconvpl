from base import SDFGenerator2D, Setting
from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

class SDFTriangle2D(SDFGenerator2D):
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

        #assert "center" in kwargs
        if kwargs["radius"] is not None:
            assert kwargs["corners"] is None
            self.radius: float = kwargs["radius"]
            self.corners: np.array = self.perimetric_corners()
            self.center:np.array = self.perimetric_center()
        if kwargs["corners"] is not None:
            assert kwargs["radius"] is None
            self.corners: np.array = self.parse_corners(kwargs["corners"])
            self.center: np.array = self.perimetric_center()
        self.data:np.array = self.generate()

    def parse_corners(self,corners):
        corners = np.fromstring(corners,sep=",")
        assert len(corners) == 6
        corners = corners.reshape(3,2)
        self.center: np.array = np.asarray(
            [
                corners[0][0] + corners[1][0] + corners[2][0] / 3,
                corners[0][1] + corners[1][1] + corners[2][1] / 3,
            ]
        )
        return corners

    def perimetric_center(self):
        center:np.array = np.zeros(2)
        for c in self.corners:
            center[0] += c[0]
            center[1] += c[1]
        return np.array([center[0] / 3,center[0] / 3])


    def perimetric_corners(self):
        corners = [random.randint(0,self.num_points +1) for _ in range(3)]
        return np.asanyarray([
            (
                np.cos(2 * np.pi / self.num_points * c) * self.radius,
                np.sin(2 * np.pi / self.num_points * c) * self.radius,
            )
            for c in corners
        ])


    def sides(self):
        return np.asanyarray(
            [
                [self.corners[0],self.corners[1]],
                [self.corners[0],self.corners[2]],
                [self.corners[1],self.corners[2]]
            ]
        )

    def side_normals(self):
        normals = []

        for side in self.sides():
            x1,y1 = side[0][0], side[0][1]
            x2,y2 = side[1][0], side[1][1]
            x3,y3 = self.center[0], self.center[1]

            k = ((y2-y1) * (x3-x1) - (x2-x1) * (y3-y1)) / ((y2-y1)**2 + (x2-x1)**2)
            x4 = (x3 - k * (y2-y1)) 
            y4 = (y3 + k * (x2-x1))
            #print(x4-self.center[0],y4-self.center[1],np.linalg.norm([(x4-self.center[0]),(y4-self.center[1])]))
            p4 = [(x4-self.center[0]),(y4-self.center[1])] / np.linalg.norm([(x4-self.center[0]),(y4-self.center[1])])
            normals.append(p4)
        return np.asanyarray(normals)

    def corner_normals(self):
        normals = []
        for i,corner in enumerate(self.corners):
            s1 = corner - self.corners[(i+1) % 3]
            s2 = corner - self.corners[(i+2) % 3]
            normals.append((s1+s2)/2)
        return normals

    def line_intersection(self,point: np.array,corners:List[int]):
        x1,y1 = point[0], point[1]
        x2,y2 = self.center[0], self.center[1]
        x3,y3 = self.corners[corners[0]][0], self.corners[corners[0]][1]
        x4,y4 = self.corners[corners[1]][0], self.corners[corners[1]][1]
        
        # Source: https://mathworld.wolfram.com/Line-LineIntersection.html
        if ((x1-x2) * (y3-y4) - (y1 -y2) * (x3-x4)) != 0:
            px = (
                    (x1*y2 -y1*x2) * (x3 -x4) - (x1-x2) * (x3*y4 -y3*x4)
                ) / (
                    (x1-x2) * (y3-y4) - (y1 -y2) * (x3-x4)
                )
            
            py = (
                    (x1*y2 -y1*x2) * (y3 -y4) - (y1-y2) * (x3*y4 -y3*x4)
                ) / (
                    (x1-x2) * (y3-y4) - (y1 -y2) * (x3-x4)
                )

            return True,np.array([px,py])
        else:
            return False,np.array([None,None])

    def sdf_value(self, x, y):
        d = np.Inf
        for indices in ([0,1],[0,2],[1,2]):
            intersected, point = self.line_intersection(np.array([x,y]),indices)
            if intersected:
                dt = np.linalg.norm((np.array([x,y]) - point))
                if d > abs(dt):                
                    d = dt

        return d

  
    def generate(self):
        self.num_points -= 3
        points = []
        lambdas = np.linspace(start=0, stop=1, num= 2 + (self.num_points // 3))[1:-1]
        for i in lambdas:
            for side,normal in zip(self.sides(),self.side_normals()):
                p = (1.0 - i) * side[0] + i * side[1]
                if self.setting == Setting.DISTANCE:
                    d = self.sdf_value(p[0],p[1])
                    points.append([p[0],p[1],d])
                elif self.setting == Setting.SURFACENORMALS:                    
                    points.append(np.asarray([p[0],p[1],normal[0],normal[1]]))

        for corner,normal in zip(self.corners,self.corner_normals()):
            normal = normal / np.linalg.norm(normal)
            if self.setting == Setting.DISTANCE:
                d = self.sdf_value(corner[0],corner[1])
                points.append([corner[0],corner[1],d])
            elif self.setting == Setting.SURFACENORMALS:
                points.append(np.asarray([corner[0],corner[1],normal[0],normal[1]]))    
                
        return np.asanyarray(points)

    def plot_normals(self):
        _, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], color="black", s=2)
        ax.scatter(self.corners[:, 0], self.corners[:, 1], color="red", s=0.9)

        for d in self.data:           
            ax.arrow(d[0],d[1],d[2],d[3],length_includes_head=True,head_width=0.025)

        ax.axis("equal")
        plt.show()
        

    def plot_distances(self):
        return super().plot_distances()

    def plot(self):
        return super().plot()
    
    def color_table(self, distance: float):
        return super().color_table(distance)
    
    def as_json(self, filename):
        return super().as_json(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF Data Generator - Triangle")
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
        help="Defines the radius of the circle, on which the corner points of the triangle reside. (Exclusive vs 'corners' option)"
    )

    parser.add_argument(
        "--corners",
        metavar=np.array,
        default=None,
        action="store",
        nargs="?",
        help="Defines the coordinates of the corner points. Format: x1,y2,x2,y2,x3,y3 (Exclusive vs 'radius' option) "
    )

    parser.add_argument(
        "--datapoints",
        metavar=int,
        default=100,
        action="store",
        nargs="?",
        help="Set the amount of datapoints generated."
    )

    args = parser.parse_args()
    _setting:Setting = Setting.UNSET
    
    if args.setting not in ["distance","normals"]:
        raise ValueError(f"Unkown Setting {args.setting}")
    elif args.setting == "distance":
        _setting = Setting.DISTANCE
    elif args.setting == "normals":
        _setting = Setting.SURFACENORMALS
    assert int(args.datapoints) % 3 == 0 and int(args.datapoints) >=3,"Cannot build a triangle with less than 3 points."

    triangle = SDFTriangle2D(
        dist_func=f"lambda x,y: x+y",
        parameters=["x","y"],
        num_points=int(args.datapoints),
        delta=0.0,
        radius=float(args.radius) if args.radius is not None else None,
        corners=args.corners if args.corners is not None else None,
        #center=np.asarray([0.0, 0.0]),
        distances=np.asarray([-0.5, 0.0, 0.5]),
        setting=_setting
    )

    triangle.plot()
    triangle.as_json(filename="2DSDF_Triangle_BETA.json")