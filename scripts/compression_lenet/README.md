# Compressing LeNet

This directory contains the necessary scripts to train and compress LeNet-5 for the MNIST dataset.
We have also provided a pre-trained and compressed checkpoint in `artifacts/lenet`, and a script
to compute the corresponding bound.

## Witnessing the generalizatino bound

We have provided a pre-trained and compressed checkpoint corresponding to the
To witness the bound with the parameters described in the paper, simply run
```{bash}
python -m scripts.compression_lenet.summary --checkpoint-path artifacts/lenet --scale-posterior 0.05
```
from the main directory.

## Training and compression

We have also provided scripts to train and compress a network from scratch. These scripts are the `baseline`,
`prune_dns` and `quantize` scripts, and respectively train a network, prune it using dynamic network surgery,
and quantize the network. Note that these are not optimally tuned as the optimal tuning is somewhat tricky.