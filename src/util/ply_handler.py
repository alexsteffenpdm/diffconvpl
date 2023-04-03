import numpy as np
import os
import trimesh.exchange
import trimesh.intersections
from tqdm import tqdm

import matplotlib.pyplot as plt
if os.getenv("TESTING") == 'True':
    from mesh_precalculator import create_edges,create_faces
else:
    from .mesh_precalculator import create_edges,create_faces

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

class PlyWriter():
    def __init__(self,filepath:str,vertices:np.array,faces:list[str]):
        self.filepath: str = filepath
        self.vertices:np.array = vertices
        self.faces:list[str] = faces
        self.header:str = f"""ply
format ascii 1.0
comment made by Alexander Steffen
comment this file is used for visualization of an
comment SDF map created with Max-Affine functions
element vertex {len(vertices) if self.vertices is not None else 0}
property float32 x
property float32 y
property float32 z
element face {len(faces) if self.faces is not None else 0}
property list uint8 int32 vertex_indices
end_header
"""
    def write(self,):
        
        ply_path = os.path.join(os.getcwd(),"data","generated","blender_files")
        if not os.path.exists(ply_path):
            os.makedirs(ply_path)
        with open(os.path.join(ply_path,self.filepath),"w") as ply_file:
            ply_file.write(self.header)

            print(f"Writing verticies to {self.filepath}")
            [
                ply_file.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                for v in tqdm(self.vertices)
            ]
            
            print(f"Writing faces to {self.filepath}")
            [ ply_file.write(f"4 {f}\n") for f in tqdm(self.faces) ]
                


if __name__ == "__main__":
    _min = -1.0
    _max = 1.0
    spacing = 0.5
    func = lambda x,y: np.sqrt(x**2 + y**2) - 0.4
    points:np.array = int((abs(_min)+abs(_max)) / spacing) + 1
    domain: np.array = np.linspace(_min,_max,points)
    x,y = np.meshgrid(domain,domain)
    
    x_f = x.flatten()
    y_f = y.flatten()
    print(x.shape)
    z = [func(xi,yi) for xi,yi in zip(x_f,y_f)]
    verts = np.asanyarray([[xi,yi,zi] for xi,yi,zi in zip(x_f,y_f,z)])
    print(verts.shape)

    edges = create_edges(x)
    faces = create_faces(edges,x.shape)
    writer = PlyWriter("test_ply.ply",verts,faces)
    writer.write()

    os.environ["TESTING"] = 'False'



    
    
