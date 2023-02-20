# ReLU Hull Approximation

## Introduction

This project provides algorithms to calculate ReLU Hull, which is a convex hull or approximation of a polytope after ReLU transformation. These algorithms are used to neural network verification and can be embedded in current state-of-art tool PRIMA<sup>[1]</sup> in [ERAN](https://github.com/eth-sri/eran) project.

### Our Algorithm: Selective Constraints Identification (SCI)

Here we give a fast, pricise and scalable algorithm, **selective constraints identification** (SCI) and its enhanced version, SCIPLUS. This algorithm get a ReLU hull approximation by selectively identify faces (facets) of the exact convex hull. Our method has more advantages compared to the exact or approximate methods in PRIMA (e.g. SBLM+PDDM).

Two types of experiments are given. One is for comparing the approximation precision (volumes of resulting convex hull/approximation), efficiency (run time) and scalability (acceptable dimension); another is for verifying local robustness of a neural network with $l_{\infty}$ norm bound.

## Requirements

Same as [ERAN](https://github.com/eth-sri/eran) project. 

All neural network files are from ERAN project and only `.onnx` files are supported. Only MNIST and CIFAR10 are supported datasets.

## Code Description

Folder `experiment_cpu` contains code operating on a CPU for neural network verification.

Folder `experiment_cpu` contains code operating on a GPU for neural network verification.

Folder `experiment_volume` contains code for comparing different ReLU Hull approximation algorithms.

Folder `relu_hull` contains code for our algorithm.

Folder `tf_verify_sci` is similar to `tf_verify` in ERAN with necessary adjustment for our algorithm.

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

Change the directory to `experiment_gpu` (using GPUPoly or GPURefinePoly<sup>[4]</sup>) or`experiment_cpu` (using DeepPoly or RefinePoly<sup>[2]</sup>) and input a command like

```bash
python3 experiment.py --net_file ../nets/onnx/mnist/mnist_relu_3_50.onnx --dataset mnist --domain refinegpupoly --epsilon 0.1 --ns 20 --k 3 --s 1 --convex_method sci
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
- `--convex_method`: The method to calculate k-relu, there are four options, cdd (exact method)<sup>[2]</sup>, fast (approximate method, SBLM+PDDM, in PRIMA<sup>[1]</sup>), sci (our basic method) and sciplus (our enhanced method)

## Related Research

[1] Müller, Mark Niklas, et al. "PRIMA: general and precise neural network certification via scalable convex hull approximations." *arXiv preprint arXiv:2103.03638* (2021).

[2] Singh, Gagandeep, et al. "An abstract domain for certifying neural networks." *Proceedings of the ACM on Programming Languages* 3.POPL (2019): 1-30.

[3] Singh, Gagandeep, et al. "Beyond the single neuron convex barrier for neural network certification." *Advances in Neural Information Processing Systems* 32 (2019).

[4] Müller, Christoph, et al. "Neural network robustness verification on gpus." *CoRR, abs/2007.10868* (2020).

## Contributors

Zhongkui Ma (The University of Queensland) - zhongkui.ma@uq.net.au

Jiaying Li (Microsoft)

Guangdong Bai (The University of Queensland)