# Compressing MobileNet

This directory contains the necessary files to compress MobileNet from a pre-trained state. We suggest starting
from the pre-trained models provided by Google [here](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html).
Unfortunately these checkpoints are not directly compatible with the scripts provided here, but we have provided an utility
to convert those checkpoints. You may simply run (from the main directory) and provide your input checkpoint:
```{bash}
python -m nnet.models.mobilenet.rename_pretrained -i mobilenet_v1_0.50_224.ckpt -o converted/mobilenet_050.ckpt
```

The first step is the pruning step, and can be run by invoking:
```{bash}
python -u prune_mobilenet_dns.py train --dataset $DATASET_PATH --train-dir $TRAIN_DIR \
  --depth-multiplier 0.5 --warm-start $CONVERTED_CHECKPOINT
```
which will start the pruning process from the given pre-trained checkpoint. You can tune the target sparsity by editing
the `_make_train_op` function, and choosing the target sparsity for depthwise and pointwise variables separately, along
with the sparsity scaling as a function of layer size. This step can be fairly compute intensive, especially for the larger
versions of MobileNet which train slower.

The second step is to quantize the network, and can be run by invoking:
```{bash}
python -u quantize_mobilenet.py train --dataset $DATASET_PATH --train-dir $QUANTIZE_DIR --depth-multiplier 0.5 \
  --warm-start $PRUNED_CHECKPOINT --num-bits-pointwise 6 --num-bits-depthwise 6 --num-bits-dense 5 --use-codebook
```
By default this function uses a simple linear discretization, but better performance can be obtained by using a trained codebook.
We also observe that there the weights in the final fully connected layers are more tolerant to discretization. Finally, note
that you may need to edit the model definition to use a smaller momentum for batch normalization unless you wish to train
quantization for extended period of times as the moving statistics of the batch normalization may not adjust fast enough.
