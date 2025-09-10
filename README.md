# SV-Softmax: Large-Margin Softmax Loss using Synthetic Virtual Class

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **SV-Softmax: Large-Margin Softmax Loss using Synthetic Virtual Class**

In our paper, we introduce a margin adaptive synthetic virtual Softmax loss with virtual prototype insertion strategy, which emphasizes the importance of misclassified hard samples. The proposed method dynamically selects between synthetic and virtual class prototypes based on the classification correctness, leading to improved discriminative feature learning.

## Method Overview

The proposed unified virtual class loss framework is:

$$
\begin{align}
L &= \frac{1}{N}\sum_{i=1}^{N} L_{i}
 =-\frac{1}{N}\sum_{i=1}^{N} \log\frac{e^{w_{y_{i}}^{T}z_{i}}}{\sum_{j=1}^{C}e^{w_{j}^{T}z_{i}}+e^{w_{v}^{T}z_{i}}}, \\
w_{v} &=
\begin{cases}
    w_{synth} = \frac{\|w_{y_i}\|h}{\|h\|}, &\text{if } \  w_{y_i} z_i \geq \max_{j \neq y_i} w_j z_i \\
    w_{virt} = \frac{\|w_{y_i}\|z_i}{\|z_i\|}, &\text{if } \ w_{y_i}z_i < \max_{j \neq y_i} w_j z_i \\
\end{cases} ,\\
h &= m\frac{z_i}{\|z_i\|} - (1-m)\frac{w_{y_i}}{\|w_{y_i}\|}, \quad m\in[0,1].
\end{align}
$$

**Key Features:**

- **Adaptive Selection**: Dynamically chooses between synthetic and virtual prototypes based on classification correctness
- **Margin Control**: Adjustable margin parameter `m` for controlling the synthetic prototype direction
- **Hard Sample Focus**: Emphasizes learning from misclassified samples through virtual class insertion

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision 0.13+

### Setup

1. Clone the repository:

```bash
git clone https://github.com/10zhou/SV-Softmax.git
cd SV-Softmax
```

2. Create and activate a conda environment:

```bash
conda create -n lm python=3.8.18
conda activate lm
pip install -r requirements.txt
```

## Usage

### Quick Start

Train SV-Softmax on CIFAR-10 with ResNet-18:

```bash
python train_demo.py --arch resnet18 --method svsoftmax --epochs 200 --batch-size 128 --margin 0.6
```

### Training Options

The implementation supports multiple loss functions for comparison:

```bash
# SV-Softmax (Ours)
python train_demo.py --method svsoftmax --margin 0.6

# Baseline methods
python train_demo.py --method ce              # Cross Entropy
python train_demo.py --method arcface         # ArcFace
python train_demo.py --method normface        # NormFace  
python train_demo.py --method virtual_softmax # Virtual Softmax
```

### Available Arguments

- `--arch`: Model architecture (`resnet18`, `resnet34`)
- `--method`: Loss function (`svsoftmax`, `ce`, `arcface`, `normface`, `virtual_softmax`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--margin`: Margin parameter for SV-Softmax (default: 0.6)
- `--data-root`: Path to dataset directory
- `--fp16`: Enable mixed precision training

## Supported Datasets

- **CIFAR-10**: Automatically downloaded and preprocessed
- **Custom datasets**: Modify the data loading functions in `train_demo.py`

## Implementation Details

### Loss Functions

The repository includes implementations of various loss functions:

- **Cross Entropy** (`CrossEntropyLoss`): Standard softmax
- **NormFace** (`NormFaceLoss`): Normalized softmax
- **CosFace** (`CosFaceLoss`): Cosine margin loss
- **ArcFace** (`ArcFaceLoss`): Angular margin loss
- **Virtual Softmax** (`VirtualSoftmax`): Baseline virtual class method
- **SV-Softmax** (`SVSoftmaxLoss`): Our proposed method
- There are other implementations of other methods, but additional adaptations are required.


## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{SVSoftmax,
title = {Large-margin Softmax loss using synthetic virtual class},
author = {Jiuzhou Chen and Xiangyang Huang and Shudong Zhang},
journal = {Neural Networks},
volume = {193},
pages = {108068},
year = {2026},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2025.108068},
url = {https://www.sciencedirect.com/science/article/pii/S0893608025009487}
}
```


## Acknowledgments

- Thanks to [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) for the facial recognition training framework implementation
- Thanks to [Iductive Bias](https://github.com/tkasarla/max-separation-as-inductive-bias) and [Largeest Margin](https://openreview.net/forum?id=hqkhcFHOeKD) for the visual classification training framework implementation
- Thanks to [VirtualSoftmax](https://github.com/sungwool/virtual_softmax) for the virtual class method implementation

## Contact

For questions and feedback, please contact:

- Email: <10zhounb@gmail.com>
- Issues: [GitHub Issues](https://github.com/your-username/SV-Softmax/issues)

---

**Note**: This is a research implementation. For production use, please ensure thorough testing and validation.


