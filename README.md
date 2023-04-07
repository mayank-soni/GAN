# GANs 101 and its Applications

This repository is for AI Singapore's AIAP Batch 12 Group Presentation Topic: "GANs 101 and its Applications".

In this repository, we generated fake celebrity images based on the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) using 3 different Generative Adversarial Networks (GANs), namely,

- DCGAN (Deep Convolution GAN)
- WGAN (Wasserstein GAN)
- LSGAN (Least Squares GAN)

We also further explored the possibility of transfer learning on GANs in an attempt to reduce time and resources to train a model from scratch.

The results of our findings is briefly discussed below. Please refer to our article:todo for the full details.

## Installation

todo: see conda file

## Usage

todo: add notes on how to run the python notebook, specifically how we can toggle the mode param

todo: add binder link

## Results

todo: add image

### DCGAN

### WGAN

### LSGAN

### Transfer Learning

Comparing results from scratch with pretrained model with same hyperparameters such as learning rate and batch sizes.

Here is the result of LSGAN from scratch at different epochs:
![Results From Scratch](assets/tl-results-from-scratch.png)

Here is the result of LSGAN + Pretrained model using [vgg16_bn](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16_bn.html) at different epochs.

![Transfer Learning Results with](assets/tl-results-pretrained.png)

The pretrained model performs worse than the scratch model upon visual inspection, possibly due to differences between the source data and target dataset. Hyperparameter tuning may improve the pretrained model, and using a lower layer of vgg16_bn could be explored.

## Authors

This program is developed by apprentices from Batch 12 of AI Singapore's Apprenticeship Program, with contributions from the following people (in alphabetical order):

- [Mayank Soni](https://github.com/mayank-soni)
- [Quek Zhi Qiang](https://github.com/qzq92)
- [Soh Zhan Hong (Paul)](https://github.com/PaulSZH95)
- [Teng Kok Wai (Walter)](https://github.com/davzoku)

## References

- [[1406.2661] Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [[1701.00160] NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
- [[1701.07875] Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [[1611.04076] Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- [[1805.01677] Transferring GANs: generating images from limited data](https://arxiv.org/abs/1805.01677)
- [[1411.1792] How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- [Understanding Generative Adversarial Networks (GANs) | by Joseph Rocca | Towards Data Science](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)
- [How to Identify and Diagnose GAN Failure Modes - MachineLearningMastery.com](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)

## Further Exploration

Check out the work from other groups too:

- [liong-t/aiap12_group1_sharing: AIAP Sharing Project Topic](https://github.com/liong-t/aiap12_group1_sharing)
- [AlvinNg89/AIAP-RAG-Chatbot: Chatbot for AI Singapore AI Apprenticeship Program](https://github.com/AlvinNg89/AIAP-RAG-Chatbot)
- [AIAP-Group-Sharing-5/Image-segmentation-using-U-Net](https://github.com/AIAP-Group-Sharing-5/Image-segmentation-using-U-Net)
