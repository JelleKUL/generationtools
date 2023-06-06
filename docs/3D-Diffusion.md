# Different types of object generation

## Fixed sized representation of 3d models

### Voxel Grids

### INR

Neural implicit representations
create a neural network to represent every position in the data (pixel coordinate of sound timestamp) to the correct value

#### Nerfs

3d representations based on rgb images from an arbitrary vieuwpoint

#### Texture Fields

https://github.com/autonomousvision/texture_fields

Maps a 3D location to a color value

While these can easily produce consistent 3D meshes, they are not very detailed and look more like pointcloud colors stretched over the mesh

### Latent space encoding

reducing the object representation size by reducing the amount of dimentsions needed to define it.

this reduced representation also makes it easier to interpolate between different objects

Generally created with an auto encoder.