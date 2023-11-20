import mesh2sdf
import trimesh
from trimesh import Trimesh
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch

# Open3d

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

# Trimesh

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

def mesh_to_voxelgrid_trimesh(mesh: Trimesh, resolution: int = 64, hollow =True):
    
    # Normalize the mesh
    scale = 1 / np.max(mesh.extents)
    center = mesh.centroid
    transformMtx =  np.vstack((np.hstack((np.identity(3) * scale, center.reshape((3,1)) )), [0,0,0,1]))
    mesh.apply_transform(transformMtx)

    # Voxelize the mesh
    voxelSize = 1/(resolution-1) # the mesh was scaled to one
    voxelScale = voxelSize / scale
    voxelMesh = mesh.voxelized(voxelSize)
    if(hollow): voxelMesh = voxelMesh.hollow() # hollow to add colors to the grid
    
    # Get voxel colors
    voxelPoints = voxelMesh.points
    voxelColors = get_point_colors_trimesh(mesh, voxelPoints)
    ids = voxelMesh.points_to_indices(voxelPoints)
    colorGrid=np.zeros([voxelMesh.shape[0],voxelMesh.shape[1],voxelMesh.shape[2],4])
    for i in range(len(voxelPoints)):
        colorGrid[ids[i][0],ids[i][1], ids[i][2],:] = voxelColors[i]
    

    return voxelMesh, colorGrid, voxelScale

def show_mask_annotations(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    nr = len(sorted_anns)
    i = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.ones(3) / nr * i, [1]])
        img[m] = color_mask
        i+=1
    ax.imshow(img)
    print("Detected " + str(i) + " patches")

def isolate_mask(image, mask):
    img = image.copy()
    img[~mask,:] = [0,0,0]
    return img

def load_jpeg_from_file(image, image_size, cuda=True):
    img_transforms = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    img = img_transforms(image)
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input