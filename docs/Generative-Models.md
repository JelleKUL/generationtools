# Generative Models

The process of creating new/ novel data from existing data by leveraging machine learning is called Generative Machine learning. Over the years, a number of different aproaches have been developped.

Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D assets and efficient architectures for denoising 3D data, neither of which currently exist.

This page explains the different types on machine learning networks that are used to generate Images.

![image](./img/three-generative-models.png)
*An overview of the three existing neural networks*

## GAN

[Generative Adversarial Networks]

2 networks are trained simultaniously where the first model $G$ is trained to genererate a result and the second model $D$ is trained to discriminate between the real and fake data. While these generate great results, they can be very hard to train.

- Uses 2 networks: Generator and Discriminator
- Provides fast and detailed results
- trained with: Adversarial loss / minimax the classification loss, when we train the generator, we want to maximise the classification error, and the other way around
- is hard to train and can easily collapse on a small subset of the data.


## VAE

Variational Autoencoder: [Reducing the Dimensionality of Data with Neural Networks]

Autoencoder is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation. It consists of an Encoder network, which converts the data to a lower dimentional *latent space* and a Decoder network, which converts the *latent space* back into the original dimensions. the latent space is structure as a probality function for a given value x to return a Z

- VAEs store latent attributes as probability distributions
- 2 networks: encoder, decoder
- trained with: minimising L2 loss
  - The loss of the autoencoder is to minimize both the reconstruction loss (how similar the autoencoder’s output to its input) and its latent loss (how close its hidden nodes were to a normal distribution).

## Flow-based 

Flow-based Generative Models [Variational Inference with Normalizing Flows]

both GAN and VAE lack the exact evaluation and inference of the probability distribution. This resulted in low-quality results in VAEs and challenges in the GAN training process such as model collapse and vanishing gradients, etc.

a stack of invertable transormations is applied. a flow model f is constructed as an invertible transformation that maps the high-dimensional random variable x to a standard Gaussian latent variable z=f(x), as in nonlinear independent component analysis

- no noise needed for generation
- we train a single model to encode the image into latent space, and use the inverse of that to decode again
- the latent space has the same dimentionality as the original image, but is normally distributed
- it is a flow because it is a series of inversable transformations
- train for probability of real data

## Transformers

[Attention Is All You Need]

mimic the human brain with neural pathways

- positional encoding: every word has a identifier that the model can train on
- self attention: learn a meaning of a word based on the context in which it is used

transforms one input to another, useful for tranforming text to numerical measurable information

## Diffusion

[Deep Unsupervised Learning using Nonequilibrium Thermodynamics]

Diffusion based network

- 1 networks: reverse diffuser
- trained with: minimising L2 loss

See [full diffusion article](./Image-Diffusion.md.md)


## Sources

[Generative Adversarial Networks]: 
https://doi.org/10.1145/3422622

[Reducing the Dimensionality of Data with Neural Networks]: 
https://doi.org/10.1126/science.1127647

[Variational Inference with Normalizing Flows]: 
(https://arxiv.org/abs/1505.05770)

[Attention Is All You Need]: 
(https://doi.org/10.48550/arXiv.1706.03762)

[Deep Unsupervised Learning using Nonequilibrium Thermodynamics]:
(https://doi.org/10.48550/arXiv.1503.03585)