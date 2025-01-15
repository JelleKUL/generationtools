import mesh2sdf
import trimesh
from trimesh import Trimesh
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch
import numbers
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import interp1d, interpn, RegularGridInterpolator
from skimage.measure import marching_cubes
import os
import scipy

# File control

def make_dir_if_not_exist(path):
    if(not os.path.exists(path)):
        print("Folder does not exist, creating the folder: " + path)
        os.mkdir(path)

# Open3d

# Compute barycentric coordinates (u, v, w) for
# point p with respect to triangle (a, b, c)
def carthesian_to_barycentric( p,  a,  b,  c):

    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u,v,w

def barycentric_to_carthesian(a, b, c, u, v, w):
    p = u*a + v*b + c*w
    return p


def get_tri_pixel_value(tri: np.array, image: np.array)-> np.array:
    """Returns the pixel color value of the center of a uv triangle

    Args:
        tri (np.array()): 2x3 array of (x,y) uv coordinates
        image (np.array()): nxn array of pixel values

    Returns:
        np.array(): the pixel value
    """
    avPos = np.round(np.average(tri, axis=0))
    val = image[avPos[1].astype(int)][avPos[0].astype(int)]
    return val

def get_point_pixel_value(tri: np.array, tri3D:np.array, point: np.array, image: np.array)-> np.array:
    """Returns the pixel color of the 3D point

    Args:
        tri (np.array()):  2x3 array of (x,y) uv coordinates of the triangle
        tri3D (np.array()): 3x3 array of (x,y,z) 3D coordinates of the triangle
        point (np.array()): 3x1 array of (x,y,z) 3D coordinates
        image (np.array()): nxn array of pixel values

    Returns:
        np.array(): 2x1 array the pixel value
    """

    u,v,w = carthesian_to_barycentric(point, tri3D[0],  tri3D[1], tri3D[2])
    p2d = barycentric_to_carthesian( tri[0], tri[1], tri[2], u,v,w)

    val = image[p2d[1].astype(int)][p2d[0].astype(int)]
    return val

def get_point_pixel_colors_open3d(mesh: o3d.geometry.TriangleMesh, points:np.array, getDistance = False) -> np.array:
    """Returns the color of all the points from a mesh

    Args:
        mesh (o3d.geometry.TriangleMesh): The source colored mesh
        points (np.array): The uncolored points

    Returns:
        np.array: The point colors
    """
    # Create a scene and add the triangle mesh
    lMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(lMesh)  # we do not need the geometry ID for mesh

    mesh_texture = np.asarray(mesh.textures[0])
    mesh_textures = np.asarray(mesh.textures)
    mesh_uvs = np.asarray(mesh.triangle_uvs) * np.array([mesh_texture.shape[1],mesh_texture.shape[0]]) # multiply by the size of the texture map
    mesh_tris = np.asarray(mesh.triangles)
    mesh_verts = np.asarray(mesh.vertices)

    newColors = []
    if(getDistance):
        distances = []
    query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    # We compute the closest points on the surface.
    ans = scene.compute_closest_points(query_points)
    print("Closest points computed")
    for i in range(len(points)):
        # We get the triangle index of the closest point
        idx = ans['primitive_ids'][i].item()
        # We get the 3D location of the closest surface point
        surfacePoint = ans['points'][i].numpy()
        if(getDistance):
            distances.append(np.linalg.norm(points[i] - surfacePoint,axis=-1))
        #print(surfacePoint)
        # We get the 3D positions of the 3 points forming the triangle
        currentTri = mesh_tris[idx]
        a3d,b3d,c3d = mesh_verts[currentTri[0]], mesh_verts[currentTri[1]],mesh_verts[currentTri[2]]
        auv, buv, cuv = mesh_uvs[3*idx], mesh_uvs[3*idx+1], mesh_uvs[3*idx+2]
        u,v,w = carthesian_to_barycentric(surfacePoint, a3d,b3d,c3d)
        projectedPoint = barycentric_to_carthesian(auv, buv, cuv, u,v,w)
        #print("Sampled point: " + str(queryPoint) + ",with closest point: "+ str(surfacePoint)+" with triangle index: " + str(idx))
        #print(a3d,b3d,c3d)
        #print(surfacePoint)
        #print(auv, buv, cuv)
        #print(projectedPoint)
        newColors.append(mesh_texture[projectedPoint[1].astype(int)][projectedPoint[0].astype(int)])
    if(getDistance):
        return newColors, distances
    return newColors


def get_point_triangle_colors_open3d(mesh: o3d.geometry.TriangleMesh, points:np.array) -> np.array:
    """Returns the color of all the points from a mesh

    Args:
        mesh (o3d.geometry.TriangleMesh): The source colored mesh
        points (np.array): The uncolored points

    Returns:
        np.array: The point colors
    """
    # Create a scene and add the triangle mesh
    lMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(lMesh)  # we do not need the geometry ID for mesh

    mesh_texture = np.asarray(mesh.textures[0])
    mesh_uvs = np.asarray(mesh.triangle_uvs) * np.array([mesh_texture.shape[1],mesh_texture.shape[0]]) # multiply by the size of the texture map
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

def mesh_to_sdf_tensor(mesh: Trimesh, resolution:int = 64, recenter: bool = True, scaledownFactor = 1):
    """Creates a normalized signed distance function from a provided mesh, using a voxel grid

    Args:
        mesh (Trimesh): The mesh to convert, can be (non) watertight
        resolution (int, optional): the voxel resolution. Defaults to 64.

    Returns:
        sdf, mesh: the (res, res, res) np.array sdf and the fixed mesh
    """

    # normalize mesh
    vertices = mesh.vertices
    if(recenter):
        center = mesh.centroid
    else : center = 0
    scale = 2 /  np.max(mesh.extents) * scaledownFactor
    vertices = (vertices - center) * scale

    # fix mesh
    sdf, sdf_mesh = mesh2sdf.compute(
        vertices, mesh.faces, resolution, fix=(not mesh.is_watertight), level=2 / resolution, return_mesh=True)
    
    sdf_mesh.vertices = mesh.vertices / scale + center
    mesh.vertices = vertices /2
    return sdf, mesh

def get_point_colors_trimesh(mesh, points):
    # get the indexes [n] and coordinate [n,3] of the closest triangle and point for each sample point
    closestPoints,_,triangleIds =  trimesh.proximity.closest_point(mesh, points)
    # get the 3 vertex indices of each triangle [n,3]
    faces = mesh.faces[triangleIds]
    # get the uv coordinate of each vertex [n,3,2]
    uvCoordinates = mesh.visual.uv[faces]
    # get the barycentric coordinate of the closest point
    bary_coords = trimesh.triangles.points_to_barycentric(triangles=mesh.vertices[faces], points=closestPoints)
    # Interpolate UV coordinates using barycentric weights
    uv_points = np.einsum("ij,ijk->ik", bary_coords, uvCoordinates)
    # get uv color of each uv center [n,4]
    pointColors = mesh.visual.material.to_color(uv_points)
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

def sdf_to_mesh(sdf, spacing, center = True):
    vertices, faces, normals, _ = marching_cubes(sdf, level=0.0, spacing=(spacing,spacing, spacing))
    # Create a Trimesh object
    if(center):
        vertices = vertices - np.array([0.5,0.5,0.5])
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return new_mesh


def create_voxel_grid(size, center = True):
    # Set the starting point of the grid
    if(center):
        start = -0.5
    else:
        start = 0

    voxel_coordinates = np.linspace(start + 1/size/2, start + 1 - (1/size/2), size)  # n evenly spaced points in [0, 1]
    x, y, z = np.meshgrid(voxel_coordinates, voxel_coordinates, voxel_coordinates, indexing='ij')
    # Stack the coordinates into an (n x n x n x 3) array
    voxel_grid = np.stack((x, y, z), axis=-1)
    return voxel_grid

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

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def normalize_mesh(mesh):
    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds

    # Calculate the center of the bounding box
    center = (max_bound + min_bound) / 2.0

    # Calculate the scale factor to normalize the mesh to fit within [-1, 1]
    max_extent = max(max_bound - center)
    scale_factor = 1.0 / max_extent

    # Translate and scale the vertices
    mesh.vertices -= center
    mesh.vertices *= scale_factor

    return mesh


def scale_mesh_to_unity_cube(mesh):
    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds

    # Calculate the scaling factors for each axis
    scale_factors = 1.0 / (max_bound - min_bound)

    # Translate the mesh to the origin
    mesh.vertices -= min_bound

    # Scale the mesh
    mesh.vertices *= scale_factors

    return mesh

def create_grid_points_from_xyz_bounds(min_x, max_x, min_y, max_y ,min_z, max_z, res):
    x = np.linspace(min_x, max_x, res)
    y = np.linspace(min_y, max_y, res)
    z = np.linspace(min_z, max_z, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=False)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def shoot_holes(vertices, n_holes, dropout, mask_faces=None, faces=None,
                rng=None):
    """Generate a partial shape by cutting holes of random location and size.

    Each hole is created by selecting a random point as the center and removing
    the k nearest-neighboring points around it.

    Args:
        vertices: The array of vertices of the mesh.
        n_holes (int or (int, int)): Number of holes to create, or bounds from
            which to randomly draw the number of holes.
        dropout (float or (float, float)): Proportion of points (with respect
            to the total number of points) in each hole, or bounds from which
            to randomly draw the proportions (a different proportion is drawn
            for each hole).
        mask_faces: A boolean mask on the faces. 1 to keep, 0 to ignore. If
                    set, the centers of the holes are sampled only on the
                    non-masked regions.
        faces: The array of faces of the mesh. Required only when `mask_faces`
               is set.
        rng: (optional) An initialised np.random.Generator object. If None, a
             default Generator is created.

    Returns:
        array: Indices of the points defining the holes.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(n_holes, numbers.Integral):
        n_holes_min, n_holes_max = n_holes
        n_holes = rng.integers(n_holes_min, n_holes_max)

    if mask_faces is not None:
        valid_vertex_indices = np.unique(faces[mask_faces > 0])
        valid_vertices = vertices[valid_vertex_indices]
    else:
        valid_vertices = vertices

    # Select random hole centers.
    center_indices = rng.choice(len(valid_vertices), size=n_holes)
    centers = valid_vertices[center_indices]

    n_vertices = len(valid_vertices)
    if isinstance(dropout, numbers.Number):
        hole_size = n_vertices * dropout
        hole_sizes = [hole_size] * n_holes
    else:
        hole_size_bounds = n_vertices * np.asarray(dropout)
        hole_sizes = rng.integers(*hole_size_bounds, size=n_holes)

    # Identify the points indices making up the holes.
    kdtree = KDTree(vertices, leafsize=200)
    to_crop = []
    for center, size in zip(centers, hole_sizes):
        _, indices = kdtree.query(center, k=size)
        to_crop.append(indices)
    to_crop = np.unique(np.concatenate(to_crop))
    return to_crop


def interpolate_value_range(values,axis: int, idx_range, newRange):
    """
    Interpolates a range of slices along a specified axis in a 3D array.

    Args:
        valies (np.ndarray): Input 3D array.
        idx_range (tuple): Start and end indices of the range to interpolate (inclusive start, exclusive end).
        newRange (int): New size for the interpolated range along the specified axis.
        axis (int): Axis along which to perform the interpolation (0, 1, or 2).

    Returns:
        np.ndarray: 3D array with the interpolated range inserted.
    """
    # Move the chosen axis to the front for easier interpolation
    values = np.moveaxis(values, axis, 0)
    
    # define the slices
    startIdx = idx_range[0]
    endIdx = idx_range[1]
    oldRange = endIdx-startIdx
    slice = values[startIdx:endIdx,:,:]

    # Original and new coordinates along the axis
    original_m = np.linspace(0, 1, oldRange)  # Normalized original m-axis (0 to 1)
    new_m_coords = np.linspace(0, 1, newRange)  # Normalized new m-axis (0 to 1)

    # Interpolate along the x-axis for each n x n slice
    interpolated_array = np.empty((newRange, *slice.shape[1:]))
    for i in range( slice.shape[1]):
        for j in range(slice.shape[2]):
            if(slice[:, i, j].ndim > 1):
                # The to-be interpolated value is an array of its own
                interpolated_array[:, i, j] = interpolate_coordinates(slice[:, i, j], newRange)
            else:    
                # Interpolate along the m-axis for each (i, j) point
                interp_func = interp1d(original_m, slice[:, i, j], kind='linear', bounds_error=False, fill_value="extrapolate")
                interpolated_array[:, i, j] = interp_func(new_m_coords)
    
    # Restore the original axis order and insert the interpolated range
    interpolated_array = np.moveaxis(interpolated_array, 0, axis)
    values = np.moveaxis(values, 0, axis)  # Move back the axis to its original position
    
    # Split the original array into two parts: before and after the insertion point
    before_insertion = np.take(values, range(startIdx), axis=axis)
    after_insertion = np.take(values, range(endIdx, values.shape[axis]), axis=axis)

    # Concatenate the arrays with the insertion
    result_array = np.concatenate((before_insertion, interpolated_array, after_insertion), axis=axis)
    return result_array

# Function to interpolate 3D coordinates
def interpolate_coordinates(coords, new_size):
    """
    Interpolates a list of 3D coordinates to increase its size.
    
    Args:
        coords (np.ndarray): Array of shape (n, m), where n is the number of mD points.
        new_size (int): Desired number of interpolated points.

    Returns:
        np.ndarray: Interpolated array of shape (new_size, m).
    """
    coords = np.array(coords)  # Ensure input is a NumPy array
    n = len(coords)
    original_indices = np.linspace(0, n - 1, n)
    interpolated_indices = np.linspace(0, n - 1, new_size)
    
    # Interpolate x, y, z coordinates separately
    interpolated_coords = np.zeros((new_size, coords.shape[1]))
    for i in range(coords.shape[1]):  # Loop over x, y, z
        interpolated_coords[:, i] = np.interp(interpolated_indices, original_indices, coords[:, i])
    
    return interpolated_coords

def interpolate_vertex_colors(vertices, colors):
    """
    Interpolates the color values for the mesh vertices based on a normalized 3D color grid.

    The input color grid is assumed to be a normalized cube with size 1 and a center at 0 
    (i.e., the grid spans from -0.5 to 0.5 along each axis).

    Args:
        vertices (np.ndarray): Vertices of the mesh (Nx3 array).
        colors (np.ndarray): 3D grid of RGB colors (shape: DxWxHx3) defined in a normalized cube.

    Returns:
        np.ndarray: Interpolated colors for each vertex (Nx3 array).
    """
    grid_shape = colors.shape[:3]
    color_channels = colors.shape[3]

    # Define normalized grid points in the range [-0.5, 0.5]
    grid_coords = [
        np.linspace(-0.5, 0.5, grid_shape[i]) for i in range(3)
    ]

    # Create interpolators for each color channel (R, G, B)
    interpolators = [
        RegularGridInterpolator(
            grid_coords,
            colors[..., i],
            bounds_error=False,
            fill_value=0
        )
        for i in range(color_channels)
    ]

    # Interpolate each channel at the vertex positions
    interpolated_colors = np.column_stack([interp(vertices) for interp in interpolators])

    return interpolated_colors

def create_colored_mesh_from_sdf_and_colors(sdf, colors):
    """
    Creates a mesh from an SDF grid and assigns vertex colors based on the color grid.

    Args:
        sdf (np.ndarray): 3D numpy array of the signed distance field.
        colors (np.ndarray): 3D numpy array of RGB colors with the same shape as sdf.

    Returns:
        trimesh.Trimesh: A Trimesh object with vertex colors.
    """
    # Ensure the dimensions of the SDF and color grid match
    assert sdf.shape == colors.shape[:3], "SDF and color grid dimensions must match!"

    # Use marching cubes to extract the mesh from the SDF
    vertices, faces, _, _ = marching_cubes(sdf, level=0)

    # Normalize the vertex positions to the grid index space
    grid_shape = np.array(sdf.shape)
    vertices /= (grid_shape - 1)

    # Map vertices to the color grid space
    vertices *= (np.array(colors.shape[:3]) - 1)
    
    # Interpolate colors from the grid to the vertices
    vertex_colors = np.array([
        colors[
            int(round(v[0])),
            int(round(v[1])),
            int(round(v[2]))
        ] for v in vertices
    ], dtype=np.uint8)

    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors = vertex_colors

    return mesh

def repeat_value_range(values, axis: int, repeatRange, nrOfRepeats):
    """
    Repeats a specified range of values along a given axis in a multi-dimensional array,
    ensuring that the entire slice is repeated as a whole.

    Parameters:
    ----------
    values : numpy.ndarray
        The input multi-dimensional array from which the slice will be repeated.
    
    axis : int
        The axis along which to repeat the values.
    
    repeatRange : tuple of int
        A tuple (start, end) specifying the range of indices along the chosen axis that 
        will be repeated. The slice from `values[repeatRange[0]:repeatRange[1]]` will be 
        repeated as a whole.

    nrOfRepeats : int
        The number of times to repeat the selected range along the specified axis.

    Returns:
    -------
    numpy.ndarray
        A new array with the repeated values inserted along the specified axis.

    Example:
    --------
    values = np.random.rand(5, 4, 3)
    repeatRange = (1, 3)
    nrOfRepeats = 2
    axis = 0

    result = repeat_value_range(values, axis, repeatRange, nrOfRepeats)
    """
    # Extract the slice along the specified axis
    slices = [slice(None)] * values.ndim  # Create a list of slices for all axes
    slices[axis] = slice(repeatRange[0], repeatRange[1])  # Define the slice for the axis
    slice_to_repeat = values[tuple(slices)]  # Extract the slice

    # Repeat the slice as a whole
    repeated_slice = np.concatenate([slice_to_repeat] * nrOfRepeats, axis=axis)

    # Split the original array into two parts: before and after the repeat range
    before_insertion = np.take(values, indices=range(repeatRange[0]), axis=axis)
    after_insertion = np.take(values, indices=range(repeatRange[1], values.shape[axis]), axis=axis)

    # Concatenate the arrays along the specified axis
    result_array = np.concatenate((before_insertion, repeated_slice, after_insertion), axis=axis)

    return result_array

# Create planes
def create_transparent_plane(position, axis=0, size=1, color=[1, 0, 0, 0.5]):
    """
    Creates a transparent plane at a specific position and axis.

    Args:
        position (float): Position along the specified axis.
        axis (str): Axis along which the plane lies (0,1,2).
        size (float): Side length of the square plane.
        color (list): RGBA color of the plane (last value is transparency).

    Returns:
        trimesh.Trimesh: Plane as a trimesh object.
    """
    # Define the base vertices for a plane in the z-axis
    vertices = np.array([
        [0.0, 0.0, 0],
        [ 1, 0.0, 0.0],
        [ 1,  1, 0.0],
        [0.0,  1, 0.0],
    ])
    vertices  = (vertices - np.array([0.5,0.5,0.5])) * size

    # Move the plane to the specified axis
    if axis == 0:
        vertices = vertices[:, [2, 1, 0]]  # Swap z and x
        vertices[:, 0] += position  # Move along x-axis
    elif axis == 1:
        vertices = vertices[:, [0, 2, 1]]  # Swap z and y
        vertices[:, 1] += position  # Move along y-axis
    elif axis == 2:
        vertices[:, 2] += position  # Move along z-axis
    else:
        raise ValueError("Invalid axis. Choose from 0, 1 or 2")

    # Define faces (two triangles to form a square)
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    # Create the plane with vertex colors
    plane = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=color)
    return plane