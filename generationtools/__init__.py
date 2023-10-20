import mesh2sdf
import trimesh
from trimesh import Trimesh
import open3d as o3d
import numpy as np

def mesh_to_sdf_tensor(mesh: Trimesh, resolution:int = 64):
    """Creates a normalized signed distance function from a provided mesh, using a voxel grid

    Args:
        mesh (Trimesh): The mesh to convert, can be (non) watertight
        resolution (int, optional): the voxel resolution. Defaults to 64.

    Returns:
        sdf, mesh: the (res, res, res) np.array sdf and the fixed mesh
    """

    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    # fix mesh
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, resolution, fix=(not mesh.is_watertight), level=2 / resolution, return_mesh=True)
    
    mesh.vertices = mesh.vertices / scale + center
    return sdf, mesh

def get_tri_pixel_value(tri, image):
    avPos = np.round(np.average(tri, axis=0))
    val = image[avPos[1].astype(int)][avPos[0].astype(int)]
    return val

def get_point_colors_open3d(mesh, points):
    # Create a scene and add the triangle mesh
    lMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(lMesh)  # we do not need the geometry ID for mesh

    mesh_texture = np.asarray(mesh.textures[0])
    mesh_uvs = np.asarray(mesh.triangle_uvs) * np.array([mesh_texture.shape[1],mesh_texture.shape[0]])
    mesh_tris = np.asarray(mesh.triangles)
    mesh_tris_values = []
    for i in range(len(mesh_tris)):
        points_uv = np.array([mesh_uvs[3*i],mesh_uvs[3*i+1],mesh_uvs[3*i+2]])
        mesh_tris_values.append(get_tri_pixel_value(points_uv, mesh_texture ))
    
    newColors = []
    for queryPoint in points:
        query_point = o3d.core.Tensor([queryPoint], dtype=o3d.core.Dtype.Float32)
        # We compute the closest point on the surface for the point at position [0,0,0].
        ans = scene.compute_closest_points(query_point)
        # We get the triangle index of the closest point
        idx = ans['primitive_ids'][0].item()
        #print("Sampled point: " + str(queryPoint) + ", with triangle index: " + str(idx) + " and color: " + str(mesh_tris_values[idx]))
        newColors.append(mesh_tris_values[idx])
    return newColors


def get_point_colors_trimesh(mesh, points):
    
    # get the indexes of the closest triangle for each point triangles [n,1]
    _,_,triangleIds = trimesh.proximity.closest_point(mesh, points)
    # get the 3 vertex indices of each triangle [n,3]
    vertices = mesh.faces[triangleIds]
    # get the uv coordinate of each vertex [n,3,2]
    uvCoordinates = mesh.visual.uv[vertices]
    # get the average coordinate of each uv triangle [n,2]
    uvCenters = np.average(uvCoordinates, axis = 1)
    # get uv color of each uv center [n,4]
    pointColors = mesh.visual.material.to_color(uvCenters)
    return pointColors