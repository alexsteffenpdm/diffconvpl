import numpy as np
import os
import bpy
import json
import shutil

arguments = {
    "read":["z_height","filepath_obj","filepath_mtl","noise"],
    "write":["data","gridsize","name"],
}

ORIGINAL_BLEND_PATH = os.path.join(os.getcwd(),"src","assets","scene.blend")
NEW_BLEND_PATH = os.path.join(os.getcwd(),"data","generated","blender_files","scenes")

class BlenderSceneHandler:
    def __init__(self, mode:str, **kwargs):
        match mode:
            case "layercut":
                assert set(kwargs) == set(arguments["read"])
                self.z_height:float = kwargs["z_height"]
                self.filepath_obj:str = kwargs["filepath_obj"]
                self.filepath_mtl:str = kwargs["filepath_mtl"]
                self.noise:float = kwargs["noise"]
                
            case "create":
                assert set(kwargs) == set(arguments["write"])
                self.name:str = kwargs["name"]
                self.data:np.ndarray = kwargs["data"]
                self.gridsize:list[int] = kwargs["gridsize"]              
                
            case _:
                print("No Mode specified. Exiting now.")
                exit(0)

    def make_faces(self,) -> list[int]:
        n = self.gridsize[0]
        m = self.gridsize[1]
        mat = np.arange(0, (n * m), 1).reshape(n, m)
        circular_indices = [0, 1, 3, 2]
        faces = []
        for _n in range(n - 1):
            for _m in range(m - 1):
                face = [v for _,v in sorted(zip(circular_indices,mat[_n : _n + 2 : 1, _m : _m + 2 : 1].flatten().tolist()))]
                faces.append(face)
        return faces

    def layercut(self,) -> None:
        return
       
    def create(self,) -> None:

        shutil.copy(ORIGINAL_BLEND_PATH,os.path.join(NEW_BLEND_PATH,f"{self.name}.blend"))
        bpy.ops.wm.open_mainfile(filepath=os.path.join(NEW_BLEND_PATH,f"{self.name}.blend"))

        quad_faces = self.make_faces()
        tri_faces = []
        for face in quad_faces:
            tri_faces.extend([[face[0], face[1], face[2]], [face[0], face[2], face[3]]])
        tri_faces = np.array(tri_faces)
        mesh = bpy.data.meshes.new(name="Mesh")
        mesh.from_pydata(self.data, [], tri_faces)
        mesh.update()
        obj = bpy.data.objects.new("Approximation", mesh)
        bpy.context.scene.collection.objects.link(obj)

        original_materials = bpy.data.materials
        new_materials = bpy.data.materials
        for mat in original_materials:
            if mat.name not in new_materials:
                new_mat = bpy.data.materials.new(name=mat.name)
                new_mat.use_nodes = mat.use_nodes
                new_mat.node_tree = mat.node_tree
                new_materials.append(new_mat)
       

        material_name = "SDF_Material"
        material = bpy.data.materials.get(material_name)
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(NEW_BLEND_PATH,f"{self.name}.blend"))

if __name__ == "__main__":
    indict = None

    with open(os.path.join(os.getcwd(),"tmp.json"),"r") as infile:
        indict = json.load(infile)

    os.remove(os.path.join(os.getcwd(),"tmp.json"))

    if indict["mode"] == "create":
        handler = BlenderSceneHandler(
            mode=indict["mode"],
            data=indict["data"],
            name=indict["name"],
            gridsize=indict["gridsize"],
            )
        handler.create()
    elif indict["mode"] == "layercut":
        raise NotImplementedError("This function is not implemented yet, but will be implemented in Version 0.2.1.")