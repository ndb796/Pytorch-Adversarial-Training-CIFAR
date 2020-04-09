### Adversarial Training

* This repository provides simple PyTorch implementations for adversarial training methods on CIFAR-10.
* This repository shows fine reproduced accuracies which are similar to the accuracies in the original papers.
* The basic experiment setting used in this repository are the same methods used in [Madry Laboratory].
 * Dataset: CIFAR-10 (10 classes)
 * Attack method: PGD attack (Epsilon size is 0.0314 in L-infinity bound)
* If you have a question about these source codes, please send an e-mail to me (dongbinna@postech.ac.kr).

### Basic Training

* The basic training method adpots ResNet-18 architecture.

### PGD Adversarial Training

* This defense method was proposed by Aleksander Madry in ICLR 2018

### Interpolated Adversarial Training (IAT)

* This defense method was proposed by Alex Lamb in AISec 2019
