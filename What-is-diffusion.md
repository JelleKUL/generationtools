# Diffusion

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise.

So basically adding a bunch of random noise steps to an image and learning to reverse that 

To speed up the process, the noise is applied to a compressed low dimensional latent representation

## Concepts

### Markov Chain

The next sample is only reliant on the previous one

## Step by step

### Generating Random Images

- we train a network to predict the noise added to an image
- Then perform an inverse step to remove the noise from that image
- We itteratively add a bit of random gaussian noise to an image, because it is gaussian it can be easily combined, so we can easily get the n'th image directly instead of generating all the ones before as well.
- The generation is performed by starting with a random noise image, letting the network predict the final image and then adding a little bit less noise back to that image (one step less). This process is repeated intil we arrive at the final step.

### Conditioned Image generation

- We embed the promt, transformer style, by tokenizing it and adding it to the loss function

### Clip embedding

- generating prompt image pair embeddings (transforming from text to mumbers a computer can understand)


### Classifier free Guidance

- also add the not conditioned image to the output and compare the differences between the 2 noisy images. 
- Amplify the difference to better hone in on the prompt

### Upscalers vs Latent

- to save on processing time, images are generated at a very small resolutiona and another network is trained to upscale the image
-auto encoder:  or you can convert the noise image to latent space which is much more detailed with less data

## Sources

https://doi.org/10.48550/arXiv.1503.03585

https://colab.research.google.com/drive/1roZqqhsdpCXZr8kgV_Bx_ABVBPgea3lX?usp=sharing



