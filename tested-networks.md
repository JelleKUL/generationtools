# Tested Networks

## 3D Geometry Generation

### AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation
[[Paper](https://arxiv.org/abs/2203.09516)]
[[Github](https://github.com/yccyenchicheng/AutoSDF)]
> model the distribution over 3D shapes as a nonsequential autoregressive distribution over a discretized,
low-dimensional, symbolic grid-like latent representation of
3D shapes.

The method uses a volumetric Truncated-Signed Distance Field (T-SDF)
for representing a 3D shape and learns a Transformer based neural autoregressive model. 

The current implementation relies on removing a part of the complete sdf in order to indicate which part needs to be replaced. 

#### Questions?
- Can we turn an incomplete mesh into a good enough SDF so it can be completed
- Can we use a more complex boundary condition to indicate which parts need to be generated
- Can we incorporate Texture into the learning process?