# 3D generative model generation

## Concepts

### Fixed sized representation of 3d models

<<<<<<< HEAD
Due to the structure of deep learning models, the input size of the data is rigid. This is not a problem for Images, since they can be easily resized.

#### Voxel Grids

The space is divided in a regular voxel grid, where each point is represented by an occupied voxel.

=======
#### Voxel Grids

>>>>>>> a9bddbee114391038075ad176d5efd74c1ddb707
#### INR

Neural implicit representations
create a neural network to represent every position in the data (pixel coordinate of sound timestamp) to the correct value

<<<<<<< HEAD
### [Nerfs](./Neural-Radiance-Fields.md)

3d representations based on rgb images from an arbitrary viewpoint. See NERFS for more info.
=======
### Nerfs

3d representations based on rgb images from an arbitrary viewpoint
>>>>>>> a9bddbee114391038075ad176d5efd74c1ddb707

### Texture Fields

https://github.com/autonomousvision/texture_fields

Maps a 3D location to a color value

While these can easily produce consistent 3D meshes, they are not very detailed and look more like pointcloud colors stretched over the mesh

### Latent space encoding

reducing the object representation size by reducing the amount of dimentsions needed to define it.

this reduced representation also makes it easier to interpolate between different objects

Generally created with an auto encoder.

<<<<<<< HEAD

## Existing Models

### Stable Dreamfusion

https://github.com/ashawkey/stable-dreamfusion

### Magic 3D

https://research.nvidia.com/labs/dir/magic3d/

### Shape-E

https://github.com/openai/shap-e
=======
## Existing models

### (stable) Dream Fusion

### Magic 3D

### Shap-E
>>>>>>> a9bddbee114391038075ad176d5efd74c1ddb707
