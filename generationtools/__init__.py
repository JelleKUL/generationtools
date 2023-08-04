import mesh2sdf
import trimesh
from trimesh import Trimesh


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