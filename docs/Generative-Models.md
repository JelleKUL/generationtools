# Generative Models

This page explains the different types on machine learning networks that are used to generate Images 

## GAN

[Generative Adversarial Networks]

2 networks are trained simultaniously where the first model $G$ is trained to genererate a result and the second model $D$ is trained to discriminate between the real and fake data. While these generate great results, they can be very hard to train.

## VAE

Variational Autoencoder: [Reducing the Dimensionality of Data with Neural Networks]

Autoencoder is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation. It consists of an Encoder network, which converts the data to a lower dimentional *latent space* and a Decoder network, which converts the *latent space* back into the original dimensions.

## Flow-based 

Flow-based Generative Models

## Transformers

mimic the human brain with neural pathways

## [Diffusion](./Image-Diffusion.md.md)

Diffusion based network



## Sources

[Generative Adversarial Networks]: https://doi.org/10.48550/arXiv.1406.2661

[Reducing the Dimensionality of Data with Neural Networks]: https://doi.org/10.1126/science.1127647