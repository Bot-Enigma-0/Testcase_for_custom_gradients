import vtk
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def extract_surface_nodes(vtu_file, x_min, y_min, x_max, y_max, output_csv, output_vtu):
    # Load the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file)
    reader.Update()
    
    unstructured_grid = reader.GetOutput()
    
    # Store the original point IDs
    original_points = unstructured_grid.GetPoints()
    original_point_ids = np.arange(unstructured_grid.GetNumberOfPoints())

    # Convert vtkUnstructuredGrid to vtkPolyData (surface geometry)
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(reader.GetOutputPort())
    geometry_filter.Update()
    
    boundary_poly_data = geometry_filter.GetOutput()
    
    if boundary_poly_data.GetNumberOfPoints() == 0:
        raise ValueError("No points found in the geometry. Check the VTU file.")

    # Extract boundary edges using vtkFeatureEdges
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(boundary_poly_data)
    feature_edges.BoundaryEdgesOn()  # Enable boundary edge extraction
    feature_edges.FeatureEdgesOff()  # Disable feature edges
    feature_edges.NonManifoldEdgesOff()  # Disable non-manifold edges
    feature_edges.ManifoldEdgesOff()  # Disable manifold edges
    feature_edges.Update()
    
    boundary_edges = feature_edges.GetOutput()
    
    # Define the bounding box check function
    def is_within_range(point):
        x, y, _ = point
        return x_min <= x <= x_max and y_min <= y <= y_max

    node_ids = []
    coordinates = []

    # Create a KDTree to map the filtered points back to the original mesh points
    kdtree = cKDTree(np.array([original_points.GetPoint(i) for i in original_point_ids]))

    # Loop over points in the boundary edges and filter by bounding box
    for i in range(boundary_edges.GetNumberOfPoints()):
        point = boundary_edges.GetPoint(i)
        if is_within_range(point):
            # Find the nearest original point
            distance, idx = kdtree.query(point)
            original_id = original_point_ids[idx]
            node_ids.append(original_id)
            coordinates.append((point[0], point[1], point[2]))  # Keep z-coordinate intact

    if not node_ids:
        raise ValueError("No nodes found within the specified bounding box.")

    # Save the filtered nodes to a CSV file
    df = pd.DataFrame({
        'NodeID': node_ids,
        'x': [coord[0] for coord in coordinates],
        'y': [coord[1] for coord in coordinates],
        'z': [coord[2] for coord in coordinates],
    })
    df.to_csv(output_csv, index=False)
    
    # Write the filtered nodes back to a new VTU file
    filtered_points = vtk.vtkPoints()
    for coord in coordinates:
        filtered_points.InsertNextPoint(coord)

    filtered_poly_data = vtk.vtkPolyData()
    filtered_poly_data.SetPoints(filtered_points)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtu)
    writer.SetInputData(filtered_poly_data)
    writer.Write()

# Example usage
vtu_file = "flow_0.vtu"
x_min, y_min, x_max, y_max = 2.0, 0.0, 10.0, 1.0
output_csv = "output_nodes.csv"
output_vtu = "filtered_surface.vtu"

extract_surface_nodes(vtu_file, x_min, y_min, x_max, y_max, output_csv, output_vtu)
