import bpy

file_path = "C:\\Users\\skyfe\\Development\\diffconvpl\\data\\generated\\blender_files\\datapoints_28-03-2023_18-11-46.txt"

datapoints_collection = bpy.data.collections.new("datapoints")
bpy.context.scene.collection.children.link(datapoints_collection)


# Define custom properties for the collection to control sphere size and color
datapoints_collection["sphere_diameter"] = 0.01
datapoints_collection["sphere_color"] = (0.0, 0.0, 0.0,1.0)

# Read the centroid positions from the file
with open(file_path, "r") as f:
    centroids = [tuple(map(float, line.strip().split())) for line in f]

# Create a sphere for each centroid position and add it to the "datapoints" collection
for centroid in centroids:
    # Create a sphere mesh
    bpy.ops.mesh.primitive_uv_sphere_add(radius=datapoints_collection["sphere_diameter"] / 2, enter_editmode=False, align='WORLD', location=centroid)
    
    # Create a sphere object
    sphere = bpy.context.active_object
    
    # Set the sphere's name
    sphere.name = f"Sphere ({centroid[0]}, {centroid[1]}, {centroid[2]})"
    
    # Add the sphere to the "datapoints" collection
    datapoints_collection.objects.link(sphere)
    
    # Set the sphere's color based on the custom property
    sphere.color = datapoints_collection["sphere_color"]
