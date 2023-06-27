# Neural Radiance Fields

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis]

![image](./img/nerfImage.png)
*The nerf workflow*

Neural radiance fields (NeRFs) are a technique that generates 3D representations of an object or scene from 2D images by using machine learning. The technique involves encoding the entire object or scene into a neural network, which predicts the radiance at any point in the space to generate novel 3D views from different angles.

Great for reflective materials
meshing is pretty much the same as normal photogrammetry
using marching cubes
no special rendering
it looks like it can render reflective surfaces, but this is not true when we look at the generated meshes

## Concepts

### View synthesis

you can give it a series of photographs of a scene and it will be able to create, i.e. synthesize, a “photograph” from another viewpoint.  The new “photo” will look like it is of the same location and same things in the others photos.

### Volumetric rendering

The scene is represented as a continuous volumetric scene function (radiance field).
A view is rendered by ray-marching using that function.

### Ray-marching

Incrementally casting a ray in a direction and checking the distance to the object. The ray will move forward with a magnitude of the calculated distance, this will ensure it does not overshoot the surface. This process is repeated until the distance is very small, or a max amount of iterations is reached.

## Workings

- Generate a sampled set of 3D points—by marching camera rays through the scene.
- Produce an output set of densities and colors—by inputting your sampled points with their corresponding 2D viewing directions into the neural network.
- Accumulate your densities and colors into a 2D image—by using classical volume rendering techniques.

## Implementations

[Awesome Neural Radiance Fields]

https://captures.lumalabs.ai/me

https://github.com/NVlabs/instant-ngp

## Sources

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis]:
(https://doi.org/10.48550/arXiv.2003.08934)

[Awesome Neural Radiance Fields]: 
(https://github.com/awesome-NeRF/awesome-NeRF)