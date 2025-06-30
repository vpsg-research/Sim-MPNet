<div align="center">
<h1> [ICCV 2025] Sim-MPNet: Learning Category-Aware Localization with Similarity Memory Priors for Medical Image Segmentation </h1>
</div>

## 🎈 News

- [2025.3.5] Training and inference code released
- [2025.6.25] Our work has been accepted by ICCV2025！

- ## ⭐ Abstract
- In recent years, it has been found that “grandmother cells” in the primary visual cortex (V1) of macaques can directly recognize visual input with complex shapes. This inspires us to examine the value of these cells in promoting the research of medical image segmentation. In this paper, we design a Similarity Memory Prior Network (Sim-MPNet) for medical image segmentation. Specifically, we propose a Dynamic Memory Weights–Loss Attention (DMW-LA), which matches and remembers the category features of specific lesions or organs in medical images through the similarity memory prior in the prototype memory bank, thus helping the network to learn subtle texture changes between categories. DMW-LA also dynamically updates the similarity memory prior in reverse through Weight-Loss Dynamic (W-LD) update strategy, effectively assisting the network directly extract category features. In addition, we propose the Double-Similarity Global Internal Enhancement Module (DS-GIM) to deeply explore the internal differences in the feature distribution of input data through cosine similarity and euclidean distance. Extensive experiments on four public datasets show that Sim-MPNet has better segmentation performance than other state-of-the-art methods.

## 🚀 Introduction

<div align="center">
    <img width="400" alt="image" src="figures/Motivation.png?raw=true">
</div>

Motivation. (a) Schematic diagram of the primate visual system. (b) We use similarity memory priors stored in the prototype memory bank to imitate “grandmother cells” in V1, and utilize these features to directly match various organs in medical images.

<div align="center">
    <img width="600" alt="image" src="figures/Introduction.png?raw=true">
</div>

Compared with the existing works. DMW-LA uses similarity memory priors to directly identify and extract category features of organs.

## 📻 Overview

<div align="center">
<img width="700" alt="image" src="figures/Sim-MPNet.png?raw=true">
</div>

Overview of Sim-MPNet. The network consists of two encoders, one decoder and one skip connection, and it efficiently models intra-class and global dependencies of input features through the dual encoder structure.

## 🎮 Getting Started

### 1. Install Environment

```
torch==2.6.0
numpy==1.24.4
timm==1.0.15
scipy==1.15.3
einops==0.8.1
faiss-gpu==1.7.2
scikit-learn==1.7.0
tensorboardX==2.6.4
medpy==0.5.2
seaborn==0.13.2
segmentation_mask_overlay==0.4.4
thop==0.1.1.post2209072238
h5py==3.14.0
torchsummaryX==1.3.0
imgaug==0.4.0
```

### 2. Prepare Datasets

- Download datasets: Synapse from this [link](https://www.synapse.org/Synapse:syn3193805/wiki/217789), SegPC-2021 form this [link](https://www.kaggle.com/datasets/sbilab/segpc2021dataset), ACDC form this [link](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), ISIC-2018 from this [link](https://challenge.isic-archive.com/data/#2018).

- Folder organization: put datasets into ./data folder.

### 3. Pretrained Model

-You should download the pretrained MaxViT models from [link](https://drive.google.com/drive/folders/1k-s75ZosvpRGZEWl9UEpc_mniK3nL2xq), and then put it in the './pretrained_pth/maxvit/' folder for initialization.

### 4. Train the Sim-MPNet

```
python -W ignore train_synapse.py
```

### 5. Test the Sim-MPNet

```
python -W ignore test_synapse.py
```
## ✨ Quantitative comparison

<div align="center">
<img width="700" alt="image" src="figures/compare1.png?raw=true">
</div>

<div align="center">
<img width="700" alt="image" src="figures/compare2.png?raw=true">
</div>
<div align="center">
    We compare Sim-MPNet with thirteen state-of-the-art methods, evaluate the segmentation performance of ACDC, SegPC-2021 and ISIC-2018 datasets, and assess the generalization of Sim-MPNet on Synapse dataset.
</div>

## 🖼️ Visualization

<div align="center">
<img width="700" alt="image" src="figures/Visualization.png?raw=true">
</div>

<div align="center">
    Multi-organ and cell segmentation. Comparison of visualization results between Sim-MPNet and other SOTA methods on Synapse and SegPC-2021. The different segmentation targets are represented by different colors.
</div>

##  Citations

```
@inproceedings{simmpnet,
  title={Sim-MPNet: Learning Category-Aware Localization with Similarity Memory Priors for Medical Image Segmentation},
  author={Hao Tang, Zhiqing Guo, Liejun Wang and Chao Liu},
  booktitle={International Conference on Computer Vision (ICCV)},
  month={June},
  year={2025}
}
```
