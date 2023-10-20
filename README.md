# Generation Tools
Tools to generate 2D and 3D objects

## Documentation

This work contains background information about the field of generative AI and diffusion based generation.

1. [What are Generative Machine Learning Models?](./docs/Generative-Models.md)
2. [What is Image Diffusion?](./docs/Image-Diffusion.md)
3. [What Are NEural Radiance Fields (NERF)?](./docs/Neural-Radiance-Fields.md)
4. [What is Diffusion Based 3D Model Generation?](./docs/3D-Diffusion.md)


## Workflow

- we have a incomplete 3D textured mesh
- convert the mesh to an sdf
- use nn to complete the geometry
- if-nets are trained to complete specific shapes -> humans / or to fill holes of generic objects
- use the texture fields to also complete the texture based on a rough example


## Signed distance functions

- Traditional methods of 3D shape representation include: meshes, pointclouds, voxelgrids.
- meshes are not structured enough and pointclouds and voxelgrids do not define a surface.
- A alternative is Signed distance functions [SDF], these already exist for geometric primitives, but recent advances in machine learning have allowed more complex meshes to also be possible [DeepSDF].
- SDF's define a continuous signed distance field which is defined as the distance to closest point on the surface from any point in space. the magnitude of a point in the field represents the distance to the surface boundary and the sign indicates whether the region is inside (-) or outside (+) of the shape [SDF].
- SDF's are implicit representations, so they require alternative rendering methods to be visualised. The most popular today are:
  - Sphere tracing [SphereTracing]
  - Marching cubes [MarchingCubes]
  - Differentiable rendering [DifferentialRendering]
- A popular variation of the SDF in machine learning is the TSDF (truncated signed distance field) where the boundaries and size are clearly defined. This makes sure the shape if the data remains consistent over the different objects [TSDF].
- The texture of an object can be represented explicitly using different methods:
  - UV texture map
      > A separate file where each face is mapped on the uv plane and the texture is stored as a 2D image.
  - Vertex colors
      > These lack detail since they can only represent a single color per vertex
- These methods either lack a clear spacial relation or detail, so studies have tried to also define the texture of an object using implicit functions.
- Texture-fields are a deep-learning based continuous 3D function that can still represent high frequency detail while being compatible with deep learning networks [TextureFields].
- Joint Geometry and texture representation
  - Voxels represent a single datapoint in a regular grid. This datapoint can be a single value like a 1 or 0, or can be of a higher dimention like a 3 dimentional vector to represent the color. [Geometry and Attribute Compression for Voxel Scenes]
   - Neural radiance fields [NERF] in contrast to SDF's can handle more complex visual representations of a scene. As they are view dependant density functions. Neural radiance fields (NeRFs) are a technique that generates 3D representations of an object or scene from 2D images by using machine learning. The technique involves encoding the entire object or scene into a neural network, which predicts the radiance at any point in the space to generate novel 3D views from different angles.
   - Texture UV Radiance Fields (TUVF) generates textures in a learnable UV sphere space rather than directly on the 3D shape. This allows the texture to be disentangled from the underlying shape and transferable to other shapes that share the same UV space [TUVF: Learning Generalizable Texture UV Radiance Fields]

[SDF]:(https://books.google.be/books?hl=en&lr=&id=T3SSqIVnS4YC&oi=fnd&pg=PR12&dq=.+Introduction+to+Implicit+Surfaces.+Morgan+Kaufmann&ots=BxE-OmZIYM&sig=EFnBEfAr3kYPVX-kfMaY7EFEkkc#v=onepage&q=.%20Introduction%20to%20Implicit%20Surfaces.%20Morgan%20Kaufmann&f=false)

[DeepSDF]:(https://doi.org/10.48550/arXiv.1901.05103)

[SphereTracing]:(https://doi.org/10.1007/s003710050084)

[MarchingCubes]:(https://doi.org/10.1109/TVCG.2003.1207437)

[DifferentialRendering]:(https://doi.org/10.1145/3528223.3530139)

[TSDF]:(http://dx.doi.org/10.1109/ISMAR.2011.6092378)

[TextureFields]:(https://doi.org/10.48550/arXiv.1905.07259)

[TSCom]:(https://doi.org/10.48550/arXiv.2208.08768)

[Geometry and Attribute Compression for Voxel Scenes]:(https://doi.org/10.1111/cgf.12841)

[NERF]:(https://doi.org/10.48550/arXiv.2003.08934)

[TUVF: Learning Generalizable Texture UV Radiance Fields]:(https://arxiv.org/abs/2305.03040)