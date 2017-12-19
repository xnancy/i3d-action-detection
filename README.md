# I3D models on Something-Something 

## Overview

Base repository from trained models reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman. The paper was posted on arXiv in May 2017, and will be published as a
CVPR 2017 conference paper w/ additional pre-processing for 20BN Something-Something dataset. 

### Provided checkpoints

The default model has been pre-trained on ImageNet and then Kinetics; other
flags allow for loading a model pre-trained only on Kinetics and for selecting
only the RGB or Flow stream. The script `multi_evaluate.sh` shows how to run all
these combinations, generating the sample output in the `out/` directory.

The directory `data/checkpoints` contains the four checkpoints that were
trained. The ones just trained on Kinetics are initialized using the default
Sonnet / TensorFlow initializers, while the ones pre-trained on ImageNet are
initialized by bootstrapping the filters from a 2D Inception-v1 model into 3D,
as described in the paper. Importantly, the RGB and Flow streams are trained
separately, each with a softmax classification loss. During test time, we
combine the two streams by adding the logits with equal weighting, as shown in
the `evalute_sample.py` code.

We train using synchronous SGD using `tf.train.SyncReplicasOptimizer`. For each
of the RGB and Flow streams, we aggregate across 64 replicas with 4 backup
replicas. During training, we use 0.5 dropout and apply BatchNorm, with a
minibatch size of 6. The optimizer used is SGD with a momentum value of 0.9, and
we use 1e-7 weight decay. The RGB and Flow models are trained for 115k and 155k
steps respectively, with the following learning rate schedules.

RGB:

*   0 - 97k: 1e-1
*   97k - 108k: 1e-2
*   108k - 115k: 1e-3

Flow:

*   0 - 97k: 1e-1
*   97k - 104.5k: 1e-2
*   104.5k - 115k: 1e-3
*   115k - 140k: 1e-1
*   140k - 150k: 1e-2
*   150k - 155k: 1e-3

This is because the Flow models were determined to require more training after
an initial run of 115k steps.

The models are trained using the training split of Kinetics. On the Kinetics
test set, we obtain the following top-1 / top-5 accuracy:

Model          | ImageNet + Kinetics | Kinetics
-------------- | :-----------------: | -----------
RGB-I3D        | 71.1 / 89.3         | 68.4 / 88.0
Flow-I3D       | 63.4 / 84.9         | 61.5 / 83.4
Two-Stream I3D | 74.2 / 91.3         | 71.6 / 90.0


