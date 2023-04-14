import json
import os
import shutil
from typing import Any, Optional

import bpy
import numpy as np
from tqdm import tqdm

ARGUMENTS = {
    "layercut": ["z_height", "filepath_scene", "noise"],
    "write": ["data", "gridsize", "name"],
}

ORIGINAL_BLEND_PATH = os.path.join(os.getcwd(), "src", "assets", "scene.blend")
NEW_BLEND_PATH = os.path.join(
    os.getcwd(), "data", "generated", "blender_files", "scenes"
)


class BlenderSceneHandler:
    def __init__(self, mode: str, **kwargs):
        match mode:
            case "layercut":
                assert set(kwargs) == set(ARGUMENTS["layercut"])
                self.z_height: float = kwargs["z_height"]
                self.filepath_scene: str = kwargs["filepath_scene"]
                self.noise: float = kwargs["noise"]

            case "create":
                assert set(kwargs) == set(ARGUMENTS["write"])
                self.name: str = kwargs["name"]
                self.data: np.ndarray = kwargs["data"]
                self.gridsize: list[int] = kwargs["gridsize"]

            case _:
                print("No Mode specified. Exiting now.")
                exit(0)

    def make_faces(
        self,
    ) -> list[list[int]]:
        n = self.gridsize[0]
        m = self.gridsize[1]
        mat = np.arange(0, (n * m), 1).reshape(n, m)
        circular_indices = [0, 1, 3, 2]
        faces = []
        for _n in range(n - 1):
            for _m in range(m - 1):
                face = [
                    v
                    for _, v in sorted(
                        zip(
                            circular_indices,
                            mat[_n : _n + 2 : 1, _m : _m + 2 : 1].flatten().tolist(),
                        )
                    )
                ]
                faces.append(face)
        return faces

    def remap_edges(self, vertices, edges):
        indices = [i for i, v in enumerate(vertices) if v is not None]
        return [[indices.index(e[0]), indices.index(e[1])] for e in edges]

    def layercut(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        bpy.ops.wm.open_mainfile(filepath=self.filepath_scene)
        bpy.context.window.scene = bpy.data.scenes[0]

        mesh_object = bpy.data.objects["Target"]

        bpy.ops.mesh.primitive_cube_add(location=(0, 0, self.z_height - 1.0))

        cube_object = bpy.context.active_object
        s_x = mesh_object.dimensions.x / cube_object.dimensions.x + 0.1
        s_y = mesh_object.dimensions.y / cube_object.dimensions.y + 0.1
        cube_object.scale = (s_x, s_y, 1.0)
        bpy.ops.object.transform_apply(scale=True)

        bool_modifier = mesh_object.modifiers.new(name="Intersection", type="BOOLEAN")
        bool_modifier.operation = "INTERSECT"
        bool_modifier.object = cube_object

        bpy.ops.object.select_all(action="DESELECT")
        mesh_object.select_set(True)
        bpy.context.view_layer.objects.active = mesh_object
        bpy.ops.object.modifier_apply(modifier=bool_modifier.name)

        bpy.data.objects.remove(cube_object, do_unlink=True)

        cut_cube_vertices: list[Optional[list[Any]]] = []
        cut_cube_edges = []
        for v in mesh_object.data.vertices:
            v_global = mesh_object.matrix_world @ v.co
            if round(v_global.z, 4) == self.z_height:
                cut_cube_vertices.append([v_global.x, v_global.y, v_global.z])
            else:
                cut_cube_vertices.append(None)

        for e in mesh_object.data.edges:
            v1 = e.vertices[0]
            v2 = e.vertices[1]
            if cut_cube_vertices[v1] is not None and cut_cube_vertices[v2] is not None:
                cut_cube_edges.append([v1, v2])
        cut_cube_edges = self.remap_edges(cut_cube_vertices, cut_cube_edges)
        cut_cube_vertices = [v for v in cut_cube_vertices if v is not None]

        return np.asanyarray(cut_cube_vertices)[:, :2], np.asanyarray(cut_cube_edges)

    def create(
        self,
    ) -> None:
        shutil.copy(
            ORIGINAL_BLEND_PATH, os.path.join(NEW_BLEND_PATH, f"{self.name}.blend")
        )
        bpy.ops.wm.open_mainfile(
            filepath=os.path.join(NEW_BLEND_PATH, f"{self.name}.blend")
        )

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

        file_path = f"C:\\Users\\skyfe\\Development\\diffconvpl\\data\\generated\\blender_files\\datapoints_{self.name}.txt"

        datapoints_collection = bpy.data.collections.get("datapoints")
        if datapoints_collection is None:
            datapoints_collection = bpy.data.collections.new("datapoints")
            bpy.context.scene.collection.children.link(datapoints_collection)

        sphere_diameter = 0.01
        sphere_mat = bpy.data.materials.new(name="Datapoints_Material")
        sphere_mat.diffuse_color = (0.0, 0.0, 0.0, 1.0)

        with open(file_path) as f:
            centroids = [tuple(map(float, line.strip().split())) for line in f]

        orig_sphere = bpy.ops.mesh.primitive_uv_sphere_add(
            radius=sphere_diameter / 2, enter_editmode=False, align="WORLD"
        )

        for centroid in tqdm(centroids):
            sphere_mesh = orig_sphere.copy()
            sphere_mesh = bpy.context.active_object.data

            sphere_obj = bpy.data.objects.new(
                f"Sphere ({centroid[0]}, {centroid[1]}, {centroid[2]})", sphere_mesh
            )
            datapoints_collection.objects.link(sphere_obj)

            sphere_obj.location = centroid
            sphere_obj.active_material = sphere_mat

        bpy.ops.wm.save_as_mainfile(
            filepath=os.path.join(NEW_BLEND_PATH, f"{self.name}.blend")
        )
        print(
            f"Saved approximation in: {os.path.join(NEW_BLEND_PATH,f'{self.name}.blend')}"
        )
        os.remove(file_path)
        return


if __name__ == "__main__":
    indict = None

    with open(os.path.join(os.getcwd(), "tmp.json")) as infile:
        indict = json.load(infile)

    os.remove(os.path.join(os.getcwd(), "tmp.json"))

    if indict["mode"] == "create":
        handler = BlenderSceneHandler(
            mode=indict["mode"],
            data=indict["data"],
            name=indict["name"],
            gridsize=indict["gridsize"],
        )
        handler.create()
    elif indict["mode"] == "layercut":
        handler = BlenderSceneHandler(
            mode=indict["mode"],
            filepath_scene=indict["filepath_scene"],
            noise=indict["noise"],
            z_height=indict["z_height"],
        )
        handler.layercut()
