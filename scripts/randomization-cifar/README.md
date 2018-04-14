# Randomization and compression

This directory contains scripts to explore the connection between compressibility and generalization by attempting to prune
networks trained on progressively randomized datasets. The `train_resnet_random.py` file provides code to train a ResNet model
on the CIFAR-10 dataset with an arbitrary level of label randomization. The output may then be passed to the
`prune_resnet_dns.py` script to prune the model to a given target sparsity.

In general, we observe that as the label randomization increases, the training accuracy of the compressed models degrade
more quickly as a function of the compression ratio imposed to the network.
