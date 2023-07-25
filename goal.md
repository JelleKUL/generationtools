# Goal

The main goal of this research is to accurately complete occluded parts of existing 3D models in 3D scans and separate them from the environment to make them interactive.


## Paper 1: Object and texture completion of partial 3D scans

### What?
Completing 3D models of partial 3D scanned objects, both using existing geometry completion networks and finding new texture completion thingy. The main contribution will be using the existing partial texture data to complete the rest of the mesh.

### Where?
1) CVPR 24 Seattle (03/11/23)
2) SIGGRAPH 24 Colorado (26/01/24)

### How?
Use existing 3D geometry generators, since they all rely on either SDF's or points we will evaluate them separately and need to convert the meshes.

#### Geometry Models to test
- [AutoSDF](https://github.com/yccyenchicheng/AutoSDF) 2022 (VQ-VAE & transformer)
- [Point-Voxel Diffusion](https://github.com/alexzhou907/PVD) 2022 (3D-diffuser)
- [PoinTR](https://github.com/yuxumin/PoinTr) 2021 (3D-transformer)
- [SDFusion](https://github.com/yccyenchicheng/SDFusion) 2023 (latent-diffuser)
- [LION](https://github.com/IGLICT/TM-NET) 2022 (latent-diffuser)

#### SDF / pointcloud to mesh
- [Marching cubes]()
- [Shape_as_points](https://github.com/autonomousvision/shape_as_points)
- [NERF]()
- [Vis2Mesh](https://github.com/gdaosu/vis2mesh)

#### Texture generators to test
- [Texture fields](https://github.com/autonomousvision/texture_fields) 2019 
    > Can predict a full texture based on a single 2d view
- [Implicit Feature Networks for Texture Completion from Partial 3D Data](https://github.com/jchibane/if-net_texture)
    > IF-NET completing partial scans of humans, both geometry and texture (2020)
- [Texturify: Generating Textures on 3D Shape Surfaces](https://nihalsid.github.io/texturify/) 2022
    > Texturify learns to generate geometry-aware textures for untextured collections of 3D objects. Our method trains from only a collection of images and a collection of untextured shapes, which are both often available, without requiring any explicit 3D color supervision or shape-image correspondence. Textures are created directly on the surface of a given 3D shape, enabling generation of high-quality, compelling textured 3D shapes.
- [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper) 2023
    > TEXTure takes an input mesh and a conditioning text prompt and paints the mesh with high-quality textures, using an iterative diffusion-based process. In the paper we show that TEXTure can be used to not only generate new textures but also edit and refine existing textures using either a text prompt or user-provided scribbles.

#### Combined Models
- [GET-3D](https://github.com/nv-tlabs/GET3D) 2022
- [TM-NET](https://github.com/IGLICT/TM-NET) 2021


## Paper 2:  Generative Scene Completion

### What?
Converting a scanned environment into a collection of environment and object models that are filled in from all sides and occlusions

[3D Semantic Scene Completion: A Survey](https://doi.org/10.1007/s11263-021-01504-5)

### Where?
1) ICCV 24 Milan (07/03/24)
2) SIGGRAPH ASIA 24 Tokyo (01/05/24)

### How?
Combine the object completion with the segmentation model and room cleanup model to create a full room with moveable furniture

#### Improving the reconstruction
- [CIRCLE: Convolutional Implicit Reconstruction and Completion for Large-scale Indoor Scene](https://github.com/otakuxiang/circle)
    > CIRCLE is a framework for large-scale scene completion and geometric refinement based on local implicit signed distance function.

#### Existing Models
[SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans](https://github.com/angeladai/spsg)
> completes rgb-d scans by generating new camera views in obscured areas.
> Uses per voxel color information and a TSDF