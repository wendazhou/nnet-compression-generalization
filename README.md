# Compressibility and Generalization in Large-Scale Deep Learning

This repository contains everything necessary to reproduce the experiments
in our paper: [Compressibility and Generalization in Large-Scale Deep Learning](https://arxiv.org/abs/1804.05862)
(Wenda Zhou, Victor Veitch, Morgane Austern, Ryan P. Adams and Peter Orbanz).


## LeNet-5 pruning

The directory [compression_lenet](https://github.com/wendazhou/nnet-compression-generalization/tree/master/scripts/compression_lenet/)
contais the scripts which implement the training, pruning, quantization and evaluation of the LeNet-5 network.


## MobileNet Pruning

The directory [compression-mobilenet](https://github.com/wendazhou/nnet-compression-generalization/tree/master/scripts/compression-mobilenet)
contains the scripts which implement the pruning, quantization, and evaluation of noise stability of our network.
The typical compression pipeline would be to take an already trained network (such as those available [here](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html))
and prune then quantize the network. Please see the [readme](https://github.com/wendazhou/nnet-compression-generalization/blob/master/scripts/compression-mobilenet/README.md)
for more details on compressing MobileNet.

## CIFAR-10 randomization tests

The directory [randomization-cifar](https://github.com/wendazhou/nnet-compression-generalization/tree/master/scripts/randomization-cifar)
contains the scripts which implement the training and pruning of a ResNet-56 on the CIFAR-10 dataset
with a portion of the labels randomized.
