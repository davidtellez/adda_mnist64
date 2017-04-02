### ADDA MNIST 64x64

Given the MNIST 64x64 handwritten recognition dataset, we define the following two sets as "source" and "target":

<p align="center">
<img src="/resources/source_domain.png" alt="Source MNIST domain" height="400">
<img src="/resources/target_domain.png" alt="Target MNIST domain" height="400">
</p>

The goal is to maximize the accuracy of a classifier on the "target" set. However, **labels are only available for the "source" set**. We formulate the task as a **domain adaptation problem in adversarial terms**. Two networks, the classifier and the domain discriminator, compete to optimize opposite objectives. 

In particular, the discriminator predicts whether a sample image belongs to the "source" or the "target" domains, **accessing features from the input images through the classifier network only**. At the same time, the classifier has two simultaneous objectives: a) recognizing digits from the "source" domain in a supervised fashion, and b) **fooling the discriminator** by maximizing its classification error.

The idea behind adversarial domain adaptation is that **the classifier will eventually learn to hide features that are useful to discriminate between domains**. By doing so, it becomes robust against domain differences and improves its classification accuracy in the "target" set.

### Results



### Usage

- Simply execute ```python train.py```

### Requisites

- [Anaconda Python 2.5](https://www.continuum.io/downloads)
- [Lasagne 0.2.dev1](http://lasagne.readthedocs.io/)
- [Theano 0.9](http://deeplearning.net/software/theano/)
- GPU for fast training

### References

- [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
- [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464)
