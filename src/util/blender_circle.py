import bpy
import math

# Define the number of datapoints and the radius of the circle
num_datapoints = 100
circle_radius = 0.2

# Define the sphere radius
sphere_radius = 0.05

# Create a new collection named "datapoints"
datapoints_collection = bpy.data.collections.new("original_data")
bpy.context.scene.collection.children.link(datapoints_collection)

# Calculate the angle between each datapoint
angle_step = 2 * math.pi / num_datapoints

# Create a sphere for each datapoint and add it to the "datapoints" collection
for i in range(num_datapoints):
    # Calculate the angle for this datapoint
    angle = i * angle_step
    
    # Calculate the position of this datapoint on the circle
    x = circle_radius * math.cos(angle)
    y = circle_radius * math.sin(angle)
    z = 0
    
    # Create a sphere mesh
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, enter_editmode=False, align='WORLD', location=(x, y, z))
    
    # Create a sphere object
    sphere = bpy.context.active_object
    
    # Set the sphere's name
    sphere.name = f"Sphere {i}"
    
    # Add the sphere to the "datapoints" collection
    datapoints_collection.objects.link(sphere)
