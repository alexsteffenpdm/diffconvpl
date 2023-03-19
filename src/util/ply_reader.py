import numpy as np

import trimesh.exchange
import trimesh.intersections

import matplotlib.pyplot as plt


class PlyReader():
    def __init__(self,filepath:str,filetype:str,pp:np.array,pn:np.array):
        self.plane_point = pp
        self.plane_normal = pn
        self.filepath: str = filepath
        self.filetype: str = filetype
        self.vertices: np.array = None
        self.lines: np.array = None
        self.parse()

    def parse(self):
        mesh = trimesh.exchange.load.load(
            file_obj=self.filepath,
            file_type=self.filetype,
            resolver=None,
            force=None
        )
        pointcloud = trimesh.intersections.mesh_plane(
            mesh=mesh,
            plane_normal=self.plane_normal,
            plane_origin=self.plane_point
        )
        self.lines = pointcloud      
        tmp = []
        for p in pointcloud:
            tmp.append(p[0])
            tmp.append(p[1])
        self.vertices = np.asanyarray(tmp)

    def plot(self,identifier:str):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111,projection="3d")
        if identifier == "vertices":
            for v in self.vertices:
                ax.scatter(v[0],v[1],v[2])
        elif identifier == "lines":
            for l in self.lines:
                ax.scatter(l[0][0],l[0][1],l[0][2])
                ax.scatter(l[1][0],l[1][1],l[1][2])
        plt.show()
