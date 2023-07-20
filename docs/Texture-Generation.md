# Texture generation

Textures can be generated using different formats:
- Uv-Mapping
- Voxel-colours
- MLP (multi layer perception) Neural implicit representations
- rendered vieuws at certain viewpoints
- direct surface mapping

[Texture Fields: Learning Texture Representations in Function Space](https://openaccess.thecvf.com/content_ICCV_2019/html/Oechsle_Texture_Fields_Learning_Texture_Representations_in_Function_Space_ICCV_2019_paper.html)

[Implicit Feature Networks for Texture Completion from Partial 3D Data](https://arxiv.org/abs/2009.09458)
> IF-NET completing partial scans of humans, both geometry and texture (2020)

[SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans](https://arxiv.org/abs/2006.14660)
> completing a whole scene at once, by training a with synthetic less complete data (2021)
> 
> Thus, our generative 3D model predicts a 3D scene
reconstruction represented as a truncated signed distance
function with per-voxel colors (TSDF), where we leverage
a differentiable renderer to compare the predicted geometry
and color to the original RGB-D frames.

[Texturify: Generating Textures on 3D Shape Surfaces](https://arxiv.org/abs/2204.02411)
> Texturify learns to generate geometry-aware textures for untextured collections
of 3D objects. Our method produces textures that when rendered to various 2D image
views, match the distribution of real image observations. Texturify enables training
from only a collection of images and a collection of untextured shapes, which are both
often available, without requiring any explicit 3D color supervision or shape-image
correspondence. Textures are created directly on the surface of a given 3D shape,
enabling generation of high-quality, compelling textured 3D shapes.

[TUVF: Learning Generalizable Texture UV Radiance Fields](https://arxiv.org/abs/2305.03040)