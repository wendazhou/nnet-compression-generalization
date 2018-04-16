# Compressibility and Generalization in Large-Scale Deep Learning

This repository contains everything necessary to reproduce the experiments
in our paper: Compressibility and Generalization in Large-Scale Deep Learning
(Wenda Zhou, Victor Veitch, Morgane Austern, Ryan P. Adams and Peter Orbanz).


## MobileNet Pruning

The directory [compression-mobilenet](https://github.com/wendazhou/nnet-compression-generalization/tree/master/scripts/compression-mobilenet)
contains the scripts which implement the pruning, quantization, and evaluation of noise stability of our network.
The typical compression pipeline would be to take an already trained network (such as those available [here](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html))
and prune then quantize the network. Please see the [readme](https://github.com/wendazhou/nnet-compression-generalization/blob/master/scripts/compression-mobilenet/README.md)
for more details on compressing MobileNet.

## Label randomization and pruning

The directory [randomization-cifar](https://github.com/wendazhou/nnet-compression-generalization/tree/master/scripts/randomization-cifar)
contains the scripts which implement the training and pruning of a ResNet on a randomized version of the CIFAR-10 dataset.
