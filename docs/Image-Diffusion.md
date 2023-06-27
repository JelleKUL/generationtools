# Image Diffusion

Diffusion is one of [a number of generative neural networks](./Generative-models.md) to create new content out of existing training data.

Diffusion models are inspired by non-equilibrium thermodynamics ([Deep Unsupervised Learning using Nonequilibrium Thermodynamics]). They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. ([What are Diffusion Models?]) 

So basically adding a bunch of random noise steps to an image and learning to reverse that process.


## Concepts

### Gaussian Noise

A normalized noise field defined by  by 2 parameters: a mean $μ$ and a variance $σ² ≥ 0$ with probability density function: 

$$ pdf(x) = {1 \over σ\sqrt{2π}}e^{(x — μ)^2 \over 2σ^2} $$

### Markov Chain

A series of operations where the next step is only defined by the last one. This means every step can be recreated by the preceding step.

## Step by step process

- Forward Diffusion Step
  - Adding Gaussian noise to an existing image
- Reverse Diffusion Step
  - Start with pure gaussian noise
  - let the network predict the final image
  - add -1 steps of gaussian noise back to the predicted image
  - repeat until the final image is reached

### Forward Diffusion Step

The training data is created by iteratively adding gaussian noise $q$ to the original image, in theory, should this step be repeated an infinite amount of times, the resulting images will only consist of pure gaussian noise. In practice, around 1000 steps are needed to get sufficiently random noise as a starting point.


### Adding noise to the image

Each new (slightly noisier) image at time step $t$ is drawn from a conditional Gaussian distribution with:

$$ 
μ_t = \sqrt{1-β_t}*x_{t-1}
\\
σ_t^2 = β_t
$$

$$ x_t = \sqrt{1-β_t}*x_{t-1} + \sqrt{β_t} * ϵ $$

where $ϵ$ is a integer between 0 and the number of steps ([The Annotated Diffusion Model]).
we can easily sample a random step because gaussian noise is additive. 


### Generating Random Images

- we train a network to predict the noise added to an image
- Then perform an inverse step to remove the noise from that image
- We iteratively add a bit of random gaussian noise to an image, because it is gaussian it can be easily combined, so we can easily get the n'th image directly instead of generating all the ones before as well.
- The generation is performed by starting with a random noise image, letting the network predict the final image and then adding a little bit less noise back to that image (one step less). This process is repeated until we arrive at the final step.

### Conditioned Image generation

- We embed the prompt, transformer style, by tokenizing it and adding it to the loss function

#### Clip embedding

[CLIP](https://github.com/openai/CLIP)

- generating prompt image pair embeddings (transforming from text to numbers a computer can understand)


### Classifier free Guidance

- also add the not conditioned image to the output and compare the differences between the 2 noisy images. 
- Amplify the difference to better hone in on the prompt

### Upscalers

- to save on processing time, images are generated at a very small resolutions and another network is trained to upscale the image

### Auto-encoders

- You can convert the noise image to latent space which is a lower dimensional representation 
- This is similar to the VAE's but the diffusion is taking place at the latent space step.


## Existing Models

### Dall-E 2

[Hierarchical Text-Conditional Image Generation with CLIP Latents]

Open ai based transformer and GAN, source code not available, but accessible through an API

### Imagen

[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding]

google designed diffuser, not publicly available

### Stable Diffusion

[High-Resolution Image Synthesis with Latent Diffusion Models]

The diffusion model is trained in latent space (much more efficient) instead of on the full pixel image.

### Midjourney 

Uses a StyleGAN2 to generate images
Private company, accessible through a discord server


## Sources

[What are Diffusion Models?]: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
[The Annotated Diffusion Model]: https://huggingface.co/blog/annotated-diffusion

[High-Resolution Image Synthesis with Latent Diffusion Models]: https://doi.org/10.48550/arXiv.2112.10752
[Deep Unsupervised Learning using Nonequilibrium Thermodynamics]: https://doi.org/10.48550/arXiv.1503.03585

https://colab.research.google.com/drive/1roZqqhsdpCXZr8kgV_Bx_ABVBPgea3lX?usp=sharing


[Hierarchical Text-Conditional Image Generation with CLIP Latents]: (https://doi.org/10.48550/arXiv.2204.06125)
[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding]: (https://doi.org/10.48550/arXiv.2205.11487)