# Generation Tools
The main goal of this research is to accurately complete occluded parts of existing 3D models in 3D scans and separate them from the environment to make them interactive.

## Part 1: Object and texture completion of partial 3D scans

### What?
Completing 3D models of partial 3D scanned objects, both using existing geometry completion networks and finding new texture completion thingy. The main contribution will be using the existing partial texture data to complete the rest of the mesh.

### How?
Use existing 3D geometry generators, since they all rely on either SDF's or points we will evaluate them separately and need to convert the meshes.

- [AutoSDF](https://github.com/yccyenchicheng/AutoSDF) 2022 (VQ-VAE & transformer)
- [Marching cubes]()
- [Implicit Feature Networks for Texture Completion from Partial 3D Data](https://github.com/jchibane/if-net_texture)
    > IF-NET completing partial scans of humans, both geometry and texture (2020)

  
#### Datasets
- [3DObjTex.v1](https://cvi2.uni.lu/sharp2022/challenge1/)
- [Matterport3d](https://niessner.github.io/Matterport/)
- [ShapeNet](https://shapenet.org/)
- [ScanNet](http://www.scan-net.org/)
- [PartNet](https://partnet.cs.stanford.edu/)
- [Real World Textured Things (RWTT)](https://texturedmesh.isti.cnr.it/)


#### Combined Models
- [GET-3D](https://github.com/nv-tlabs/GET3D) 2022
- [TM-NET](https://github.com/IGLICT/TM-NET) 2021




## Part 2:  Generative Scene Completion

### What?
Converting a scanned environment into a collection of environment and object models that are filled in from all sides and occlusions


### How?
Combine the object completion with the segmentation model and room cleanup model to create a full room with moveable furniture

[3D Semantic Scene Completion: A Survey](https://doi.org/10.1007/s11263-021-01504-5)

#### Datasets
- [Replica-Dataset](https://github.com/facebookresearch/Replica-Dataset) 500
- [Matterport3d](https://niessner.github.io/Matterport/)
- [ScanNet](http://www.scan-net.org/)
- [3D-front](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
- [Hypersim](https://github.com/apple/ml-hypersim)
- [scanNet++][https://cy94.github.io/scannetpp]


#### Improving the reconstruction
- [CIRCLE: Convolutional Implicit Reconstruction and Completion for Large-scale Indoor Scene](https://github.com/otakuxiang/circle)
    > CIRCLE is a framework for large-scale scene completion and geometric refinement based on local implicit signed distance function.

#### Existing Models

- [SG-NN: Sparse Generative Neural Networks for Self-Supervised Scene Completion of RGB-D Scans](https://github.com/angeladai/sgnn) 2020

- [SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans](https://github.com/angeladai/spsg) 2021
    > completes rgb-d scans by generating new camera views in obscured areas.
    > Uses per voxel color information and a TSDF

- [RfD-Net: Point Scene Understanding by Semantic Instance Reconstruction](https://github.com/GAP-LAB-CUHK-SZ/RfDNet) 2021

- [Point Scene Understanding via Disentangled Instance Mesh Reconstruction](https://github.com/ashawkey/dimr) 2022