# ASIF
### Coupled Data Turns Unimodal Models to Multimodal without Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noranta4/ASIF/blob/main/ASIF_colab_demo.ipynb)

This repository contains a demo of ASIF by me (first author of the paper).

It is a self-contained notebook useful to run ASIF models based on different backbones and datasets, sufficient to reproduce the main results reported in the paper within minutes. The free GPU runtime of colab is sufficient to run all the code, dataset embeddings are precomputed and downloaded from my google drive.

### Notebook content
- Setup of the ASIF model
- DEMO1: Zero-shot classification experiment (Fig. 5 in the paper)
- DEMO2: Calculate the similarity between uploaded image and text inputs
- DEMO3: Interpretability demo: deep dive into a classification
- DEMO4: Universal classifier using images from your webcam 

## Paper

Paper: [*ASIF: Coupled Data Turns Unimodal Models to Multimodal Without Training*](https://arxiv.org/abs/2210.01738)

By: [Antonio Norelli](https://noranta4.com/),
[Marco Fumero](https://gladia.di.uniroma1.it/authors/fumero/),
[Valentino Maiorca](https://gladia.di.uniroma1.it/authors/maiorca/)\,
[Luca Moschella](https://luca.moschella.dev/)\,
[Emanuele Rodolà](https://gladia.di.uniroma1.it/authors/rodola/),
[Francesco Locatello](https://www.francescolocatello.com/)

<img src="https://github.com/noranta4/ASIF/blob/main/asif_teaser.JPG" alt="Image" width="800">

**TLDR**: The meaning was already there: connecting text and images without training a neural network to do so.

**Abstract**: CLIP proved that aligning visual and language spaces is key to solving many vision tasks without explicit training, but required to train image and text encoders from scratch on a huge dataset. LiT improved this by only training the text encoder and using a pre-trained vision network. In this paper, we show that a common space can be created without any training at all, using single-domain encoders (trained with or without supervision) and a much smaller amount of image-text pairs. Furthermore, our model has unique properties. Most notably, deploying a new version with updated training samples can be done in a matter of seconds. Additionally, the representations in the common space are easily interpretable as every dimension corresponds to the similarity of the input to a unique entry in the multimodal dataset. Experiments on standard zero-shot visual benchmarks demonstrate the typical transfer ability of image-text models. Overall, our method represents a simple yet surprisingly strong baseline for foundation multi-modal models, raising important questions on their data efficiency and on the role of retrieval in machine learning.


## Instructions

Click on the big blue button [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noranta4/ASIF/blob/main/ASIF_colab_demo.ipynb) and run all the cells of the notebook. That's it.

You can adjust backbones and datasets through the convenient drop-down menus. 

## Cite
If you liked our work and want to cite it in yours:
```
@article{norelli2022asif,
  title     = {ASIF: coupled data turns unimodal models to multimodal without training},
  author    = {Antonio Norelli and Marco Fumero and Valentino Maiorca and Luca Moschella and E. Rodolà and Francesco Locatello},
  journal   = {ARXIV.ORG},
  year      = {2022},
  doi       = {10.48550/arXiv.2210.01738},
  bibSource = {Semantic Scholar https://www.semanticscholar.org/paper/3703547b3efd1040a6fcf0b05a3624e900364ae8}
}
```
