# Texture generation

A lot of the completion networks use textureless datasets, this creates a disconnect between geometry and texture

Textures can be generated using different formats:
- Uv-Mapping
- Voxel-colours
- MLP (multi layer perception) Neural implicit representations
- rendered vieuws at certain viewpoints
- direct surface mapping

[Texture Fields: Learning Texture Representations in Function Space](https://openaccess.thecvf.com/content_ICCV_2019/html/Oechsle_Texture_Fields_Learning_Texture_Representations_in_Function_Space_ICCV_2019_paper.html)
>In recent years, substantial progress has been achieved in learning-based reconstruction of 3D objects. At the same time, generative models were proposed that can generate highly realistic images. However, despite this success in these closely related tasks, texture reconstruction of3D objects has received little attention from the research community and state-of-the-art methods are either limited to comparably low resolution or constrained experimental setups. A major reason for these limitations is that common representations of texture are inefficient or hard to interface for modern deep learning techniques. In this paper, we pro- pose Texture Fields, a novel texture representation which is based on regressing a continuous 3D function parameterized with a neural network. Our approach circumvents limiting factors like shape discretization and parameterization, as the proposed texture representation is independent of the shape representation of the 3D object. We show that Texture Fields are able to represent high frequency texture and naturally blend with modern deep learning techniques. Experimentally, we ﬁnd that Texture Fields compare favorably to state-of-the-art methods for conditional texture reconstruction of3D objects and enable learning of probabilistic generative models for texturing unseen 3D models. We believe that Texture Fields will become an important building block for the next generation of generative 3D models.

This paper proposes a new way to represent textures on 3D models, by learning a continuous function to map a 3D point in space to a color

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