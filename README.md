## Pytorch Adversarial Training on CIFAR-10

* This repository provides simple PyTorch implementations for adversarial training methods on CIFAR-10.
* This repository shows fine reproduced accuracies which are similar to the accuracies in the original papers.
* If you have questions about this repository, please send an e-mail to me (dongbinna@postech.ac.kr) or make an issue.

### Experiment Settings

* The basic experiment setting used in this repository are followed the setting used in [Madry Laboratory](https://github.com/MadryLab/cifar10_challenge).
* Dataset: CIFAR-10 (10 classes)
* Attack method: PGD attack (Epsilon size is 0.0314 in L-infinity bound)
* Training batch size: 128
* Weight decay: 0.0002
* Momentum: 0.9
* Learning rate adjustment
  1) 0.1 for epoch [0, 100)
  2) 0.01 for epoch [100, 150)
  3) 0.001 for epoch [150, 200)

### Basic Training

* The basic training method adpots ResNet-18 architecture.
* This architecture is smaller than Madry Laboratory, but performances are similar.

### PGD Adversarial Training

* This defense method was proposed by Aleksander Madry in ICLR 2018.

### Interpolated Adversarial Training (IAT)

* This defense method was proposed by Alex Lamb in AISec 2019.
