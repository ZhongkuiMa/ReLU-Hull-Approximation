# ReLU Hull Approximation

## Introduction

This project provides algorithms to calculate ReLU Hull, which is a convex hull or approximation of a polytope after ReLU transformation. These algorithms are used to neural network verification and can be embedded in current state-of-art tool PRIMA in [ERAN](https://github.com/eth-sri/eran) project.

## Selective Constraints Identification (SCI)

Here we give a fast, pricise and scalable algorithm, **selective constraints identification** (SCI) and its enhanced version, SCIPLUS. This algorithm get a ReLU hull approximation by selectively identify faces (facets) of the exact convex hull. Our method has more advantages compared to the exact or approximate methods in ERAN (e.g. SBLM+PDDM).

Two types of experiments are given. One is for comparing the approximation precision (volumes of resulting convex hull/approximation), efficiency (run time) and scalability (acceptable dimension); another is for verifying local robustness of a neural network with $l_{\infty}$ norm bound.

## Requirements

Same as [ERAN](https://github.com/eth-sri/eran) project. 

All neural network files are from ERAN project and only `.onnx` files are supported. Only MNIST and CIFAR10 are supported datasets.

## Installation

1. Install [ERAN](https://github.com/eth-sri/eran). We put ERAN and ELINA in a same directory, which is different from the default configuration of ERAN. You need to manually adjust a few of our code, or choose to adjust the configuration location of ELINA.
2. Copy all our files to the ERAN folder.

## Usage

### Compare Algorithms

Change the directory to `experiment_volume` and run the following command:

```bash
python3 estimate_volume.py
```

You may need to change the specific codes to run the desired experiment.

### Verify Neural Network

Change the directory to `experiment_gpu` (using GPUPoly or GPURefinePoly) or`experiment_cpu` (using DeepPoly or RefinePoly) and input a command like

```bash
python3 experiment.py --net_file ../nets/onnx/mnist/mnist_relu_3_50.onnx --dataset mnist --domain refinegpupoly --ns 20 --k 3 --s 1 --convex_method sci
```

We provide a parser different from that of ERAN, and it has the following main arguments:

- `--net_file`: The network file name/path (.onnx)
- `--dataset`: The dataset (mnist or cifar10)
- `--domain`: The domain name (deeppoly, refinepoly, gpupoly or gpurefinepoly)
- `--epsilon`: The Epsilon for $l_\infty$ perturbation
- `--samples_num`: The number of samples to test
- `--samples_start`: The first index of samples to test
- `--ns`: The number of variables to group by k-relu
- `--k`: The group size of a k-relu
- `--s`: The overlap size between two k-relu groups
- `--convex_method`: The method to calculate k-relu

## Contributors

Zhongkui Ma (contact) - zhongkui.ma@uq.net.au

Jiaying Li

Guangdong Bai