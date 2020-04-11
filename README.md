## Pytorch Adversarial Training on CIFAR-10

* This repository provides simple PyTorch implementations for adversarial training methods on CIFAR-10.
* This repository shows fine accuracies which are similar to the accuracies in the original papers.
* If you have questions about this repository, please send an e-mail to me (dongbinna@postech.ac.kr) or make an issue.

### Experiment Settings

* The basic experiment setting used in this repository follows the setting used in [Madry Laboratory](https://github.com/MadryLab/cifar10_challenge).
* Dataset: CIFAR-10 (10 classes)
* Attack method: PGD attack (Epsilon size is 0.0314 in L-infinity bound)
* Training batch size: 128
* Weight decay: 0.0002
* Momentum: 0.9
* Learning rate adjustment
  1) 0.1 for epoch [0, 100)
  2) 0.01 for epoch [100, 150)
  3) 0.001 for epoch [150, 200)
* The ResNet-18 architecture used in this repository is smaller than Madry Laboratory, but performance is similar.

### Training Methods

#### 1. Basic Training

* The basic training method adopts ResNet-18 architecture proposed by Kaiming He in [CVPR 2016](https://arxiv.org/pdf/1512.03385.pdf).
<pre>
python3 basic_training.py
</pre>

#### 2. PGD Adversarial Training

* This defense method was proposed by Aleksander Madry in [ICLR 2018](https://arxiv.org/pdf/1706.06083.pdf).
<pre>
python3 pgd_adversarial_training.py
</pre>

#### 3. Interpolated Adversarial Training (IAT)

* This defense method was proposed by Alex Lamb in [AISec 2019](https://arxiv.org/pdf/1906.06784.pdf).
<pre>
python3 interpolated_adversarial_training.py
</pre>

#### 4. Basic Training with Robust Dataset

* Normal dataset can be splited into robust dataset and non-robust dataset.
* Construction method for robust dataset proposed by Andrew Ilyas in [NIPS 2019](https://arxiv.org/pdf/1905.02175.pdf).
<pre>
python3 basic_training_with_robust_dataset.py
</pre>

#### 5. Basic Training with Non-robust Dataset

* Normal dataset can be splited into robust dataset and non-robust dataset.
* Construction method for non-robust dataset proposed by Andrew Ilyas in [NIPS 2019](https://arxiv.org/pdf/1905.02175.pdf).
<pre>
python3 basic_training_with_non_robust_dataset.py
</pre>

### How to Test

* The attack method is PGD attack (Epsilon size is 0.0314 in L-infinity bound).
* All pre-trained models are provided in this repository :)
<pre>
python3 test.py
</pre>
