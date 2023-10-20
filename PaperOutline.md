# Object and texture completion of partial 3D scanned objects

## Introduction

Completing 3D models of partial 3D scanned objects, both using existing geometry completion networks and finding new texture completion thingy. The main contribution will be using the existing partial texture data to complete the rest of the mesh.

## Background and Related Work

- A lot of research in geometry completion
- A lot of research in object texturing
- A lot of research on image inpainting
- Not a lot on texture completion

### Geometry completion

#### Point based
- [Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion](https://github.com/jchibane/if-net)
    > IF-NET Implicit feature based point prediction

#### Implicit
- [AutoSDF](https://github.com/yccyenchicheng/AutoSDF) 
    > 2022 (VQ-VAE & transformer)
- [ShapeFormer: Transformer-based Shape Completion via Sparse Representation](https://github.com/qheldiv/shapeformer)
    > transformar based 

### Object texturing

#### Implicit
- [Texture Fields: Learning Texture Representations in Function Space](https://doi.org/10.48550/arXiv.1905.07259)
    > Neural texture representation 

### Image inpainting

- [Stable Diffusion](https://github.com/Stability-AI/generative-models)

### Texture completion

#### Point based
- [Implicit Feature Networks for Texture Completion from Partial 3D Data](https://github.com/jchibane/if-net_texture)
    > IF-NET completing partial scans of humans, both geometry and texture (2020)

- [TSCom-Net: Coarse-to-Fine 3D Textured Shape Completion Network](https://doi.org/10.48550/arXiv.2208.08768)
    > Texture inpainting based on predicted colors


## Methodology

### Implicit surface generation
Convert the incomplete meshes to voxelgrids as input for the networks

### Material segmentation
Use material detection to segment the uv mep of the object into distinct materials

### Geometry prediction

#### Occlusion detection

#### Surface meshing

### Texture inpainting

#### UV unwrapping

#### inpainting

## Experiments

### Datasets

- [ShapeNet](https://shapenet.org/)
- [Point-Voxel Diffusion](https://github.com/alexzhou907/PVD) 2022 (3D-diffuser)

### Geometry reconstruction

### Texture reconstruction

## Discussion

## Conclusion