# Image Diffusion

Diffusion is one of [a number of generative neural networks](./Generative-models.md) to create new content out of existing training data.

Diffusion models are inspired by non-equilibrium thermodynamics ([Deep Unsupervised Learning using Nonequilibrium Thermodynamics]). They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. ([What are Diffusion Models?]) 

So basically adding a bunch of random noise steps to an image and learning to reverse that process.


## Concepts

### Gaussian Noise

A normalised noise field defined by  by 2 parameters: a mean $μ$ and a variance $σ² ≥ 0$ with probability density function: 

$$ pdf(x) = {1 \over σ\sqrt{2π}}e^{(x — μ)^2 \over 2σ^2} $$

### Markov Chain

The next sample is only reliant on the previous one. 


## Step by step

- Forward diffusion step
  - gaussian noise function $q$ in a markov chain of steps
- reverse diffusion step

### Adding noise to the image

Basically, each new (slightly noisier) image at time step $t$ is drawn from a conditional Gaussian distribution with:

$$ 
μ_t = \sqrt{1-β_t}*x_{t-1}
\\
σ_t^2 = β_t
$$

$$ x_t = \sqrt{1-β_t}*x_{t-1} + \sqrt{β_t} * ϵ $$

where $ϵ$ is a integer between 0 and the number of steps ([The Annotated Diffusion Model]).
we can easily sample a random step because gaussian noice is additive. 


### Generating Random Images

- we train a network to predict the noise added to an image
- Then perform an inverse step to remove the noise from that image
- We itteratively add a bit of random gaussian noise to an image, because it is gaussian it can be easily combined, so we can easily get the n'th image directly instead of generating all the ones before as well.
- The generation is performed by starting with a random noise image, letting the network predict the final image and then adding a little bit less noise back to that image (one step less). This process is repeated intil we arrive at the final step.

### Conditioned Image generation

- We embed the prompt, transformer style, by tokenizing it and adding it to the loss function

### Clip embedding

[CLIP](https://github.com/openai/CLIP)

- generating prompt image pair embeddings (transforming from text to numbers a computer can understand)


### Classifier free Guidance

- also add the not conditioned image to the output and compare the differences between the 2 noisy images. 
- Amplify the difference to better hone in on the prompt

### Upscalers

- to save on processing time, images are generated at a very small resolutiona and another network is trained to upscale the image

### Auto-encoders

You can convert the noise image to latent space which is a lower dimentional representation 


## Existing Models

### Dall-E 2

Open ai based transformer, source code not available, but accasible through an API

### Imagen
[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://doi.org/10.48550/arXiv.2205.11487)

google designed diffuser, not publically available

### Stable Diffusion

[High-Resolution Image Synthesis with Latent Diffusion Models]

The diffusion model is trained in latent space (much more efficient) instead of on the full pixel image.

### Midjourney 

Private company, accessible 


## Sources

[What are Diffusion Models?]: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
[The Annotated Diffusion Model]: https://huggingface.co/blog/annotated-diffusion

[High-Resolution Image Synthesis with Latent Diffusion Models]: https://doi.org/10.48550/arXiv.2112.10752
[Deep Unsupervised Learning using Nonequilibrium Thermodynamics]: https://doi.org/10.48550/arXiv.1503.03585

https://colab.research.google.com/drive/1roZqqhsdpCXZr8kgV_Bx_ABVBPgea3lX?usp=sharing


