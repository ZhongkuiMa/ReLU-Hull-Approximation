#!/bin/bash

python3 experiment.py --net_file ../nets/onnx/mnist/mnist_relu_3_50.onnx --dataset mnist --domain refinepoly --epsilon 0.1 --ns 20 --k 3 --s 1 --convex_method sci --samples_start 1 --samples_num 5