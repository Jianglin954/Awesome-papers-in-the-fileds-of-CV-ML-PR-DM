# Awesome papers in the filed of CV, ML, DM.

![PRs Welcome](https://img.shields.io/badge/PhD-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This is a collection of papers that inspire me a lot. Feel free to fork. 


## Contents

- [Graph Neural Network](#GNN)
  - [Survey](#GNNSurvey) 
  - [Latent Graph Learning](#LGL) 
  - [WL Test](#wltest)
- [Transformers](#Transformers)
  - [Survey](#TransformersSurvey)
  - [Vision Transformers](#ViT)
  - [Graph Transformers](#GraphTransformers)
- [High Dynamic Range Imaging](#HDR)
- [Point Cloud](#pointcloud)
- [Causal Inference](#cause)
- [Clustering](#clustering)
  - [Shallow Clustering](#shallowClustering)
  - [Deep Clustering](#deepclustering)
- [Self-Supervised Learning](#ssl)
- [Learning with Label Noise](#labelnoise)
- [Network Compression](#networkcompression)
  - [Survey](#NetCompreSrvey)
  - [Pruning](#pruning)
- [Vision Language Pretraining](#VLP)
- [Hashing](#hashing)
  - [Survey](#hashingSurvey)
  - [Shallow Hashing](#shallowHash)
  - [Deep Hashing](#deepHash)
- [Domain Adaptation](#DA)
- [Matrix Completion](#matrixcompletion)
- [CUR Decomposition](#cur)




<a name="GNN" />

## Graph Neural Network


<a name="GNNSurvey" />

### Survey

1. **A Comprehensive Survey on Graph Neural Networks.** *Zonghan Wu et al, IEEE TNNLS 2021.* [paper](https://ieeexplore.ieee.org/document/9046288)


<a name="LGL" />

### Latent Graph Learning

1. **Differentiable Graph Module (DGM) for Graph Convolutional Networks.** *Anees Kazi et al, IEEE TPAMI 2023.* [paper](https://ieeexplore.ieee.org/document/9763421)


<a name="wltest" />

### WL Test

1. **Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning.** *Pan Li et al, NeurIPS 2020.* [paper](https://ieeexplore.ieee.org/document/9046288)


<a name="Transformers" />

## Transformers


<a name="TransformersSurvey" />

### Survey


1. **A Survey on Vision Transformer.** *Kai Han et al, IEEE TPAMI 2022.*  [[PDF]](https://ieeexplore.ieee.org/document/9716741)

<a name="ViT" />

### Vision Transformers


1. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** *Ze Liu et al, ICCV 2021.*  [[PDF]](https://ieeexplore.ieee.org/document/9710580)

<a name="GraphTransformers" />

### Graph Transformers

1. **Graph Transformer Networks.** *Seongjun Yun et al, NeurIPS 2019.*  [[PDF]](https://arxiv.org/abs/1911.06455)



<a name="HDR" />

## High Dynamic Range Imaging

1. **Ghost-free High Dynamic Range Imaging with Context-Aware Transformer.** *Zhen Liu et al, ECCV 2022.*  [[PDF]](https://arxiv.org/abs/2208.05114)   
Notes: This is the first work that introduces Transformer for HDR imaging. 



<a name="pointcloud" />

## Point Cloud

1. **Dynamic Graph CNN for Learning on Point Clouds.** *Yue Wang et al, ACM TOG 2019.*  [[PDF]](https://dl.acm.org/doi/10.1145/3326362)


<a name="cause" />

## Causal Inference

1. **.** * et al, ICCV 2019.*  [[PDF]]()


<a name="clustering" />

## Clustering


<a name="shallowClustering" />

### Shallow Clustering

1. **Sparse Subspace Clustering: Algorithm, Theory, and Applications.** *Ehsan Elhamifar et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/document/6482137)   
Notes: This papers proposes **Sparse Subspace Clustering (SSC)** which introduces sparse representation into the subspace clustering problem, and define the **Self-Expressiveness** property: each data point in a union of subspaces can be efficiently reconstructed by a combination of other points in the dataset.



<a name="deepclustering" />

### Deep Clustering

1. **A Critique of Self-Expressive Deep Subspace Clustering.** *Benjamin David Haeffele et al, ICLR 2021.*  [[PDF]](https://openreview.net/forum?id=FOyuZ26emy)    
Notes: This papers show that many previous deep subspace networks are ill-posed, and their performance improvement is largely attributable to an ad-hoc post-processing step.

1. **Deep Subspace Clustering Networks.** *Pan Ji et al, NIPS 2017.*  [[PDF]](https://arxiv.org/abs/1709.02508)   
Notes: This is the first deep subspace clustering network, however, it has been proved to be ill-posed by [paper](https://openreview.net/forum?id=FOyuZ26emy).


<a name="ssl" />

## Self-Supervised Learning

1. **.** * et al, ICCV 2019.*  [[PDF]]()



<a name="cause" />

## Causal Inference

1. **.** * et al, ICCV 2019.*  [[PDF]]()





<a name="labelnoise" />

## Learning with Label Noise

1. **.** * et al, ICCV 2019.*  [[PDF]]()







<a name="networkcompression" />

## Network Compression


<a name="NetCompreSrvey" />

### Survey


1. **Recent Advances on Neural Network Pruning at Initialization.** *Huan Wang et al, IJCAI 2022.*  [[PDF]](https://www.ijcai.org/proceedings/2022/786)  
Notes: This is the first survey on pruning at initialization.



<a name="pruning" />

### Pruning

1. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** *Jonathan Frankle et al,  ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1803.03635.pdf)   
Notes: This paper proposes the **Lottery Ticket Hypothesis**: A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the
original network after training for at most the same number of iterations.

1. **Dual Lottery Ticket Hypothesis.** *Yue Bai et al,  ICLR 2022.*  [[PDF]](https://openreview.net/pdf?id=fOsN52jn25l)   
Notes: This paper proposes the **Dual Lottery Ticket Hypothesis**:  A randomly selected subnetwork from a randomly
initialized dense network can be transformed into a trainable condition, where the transformed subnetwork can be trained in isolation and achieve better at least comparable performance to LTH and other strong baselines.



<a name="VLP" />

## Vision Language Pretraining 


1. **Learning Transferable Visual Models From Natural Language Supervision.** *Alec Radford et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf)




<a name="hashing" />
## Hashing


<a name="hashingSurvey" />
### Survey

1. **A Survey on Learning to Hash.** *Jingdong Wang et al, IEEE TPAMI 2018.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) 

1. **A Survey on Deep Hashing Methods.** *Xiao Luo et al, ACM TKDD 2022.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) 


<a name="shallowHash" />
### Shallow Hashing


1. **Locality-Sensitive Hashing Scheme based on p-Stable Distributions.** *Mayur Datar et al, SCG 2004.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/997817.997857)   
Notes: This paper proposes a novel **Locality-Sensitive Hashing (LSH)** for the Approximate Nearest Neighbor Problem under L<sub>p</sub> norm, based on p-stable distributions.  The key idea is to hash the points using several hash functions so as to ensure that, for each function, the probability of collision is much higher for objects which are close to each other than for those which are far apart. Then, one can determine near neighbors by hashing the query point and retrieving elements stored in buckets containing that point.



1. **Spectral Hashing.** *Yair Weiss et al, NIPS 2008.*  [[PDF]](https://proceedings.neurips.cc/paper/2008/file/d58072be2820e8682c0a27c0518e805e-Paper.pdf)   
Notes: This paper proposes **Spectral Hashing (SH)** where the bits are calculated by thresholding a subset of eigenvectors of the Laplacian of the similarity graph. The basic idea is to embed the data in a Hamming space such that the neighbors in the original data space remain neighbors in the Hamming space. 


1. **Hashing with Graphs.** *Wei Liu et al, ICML 2011.*  [[PDF]](https://icml.cc/2011/papers/6_icmlpaper.pdf)   
Notes: This paper proposes **Anchor Graph Hashing (AGH)** which builds an approximate neighborhood graph using Anchor Graphs, resulting in O(n) time for graph construction. The graph is sufficiently sparse with performance approaching to the true KNN graph as the number of anchors increases.



1. **Iterative Quantization: A Procrustean Approach to Learning Binary Codes for Large-Scale Image Retrieval.** *Yunchao Gong et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/document/6296665)   
Notes: This paper proposes **Iterative Quantization (ITQ)** that finds a rotation of zero-centered data so as to minimize the quantization error of mapping this data to the vertices of a zero-centered binary hypercube. The optimization problem is intrinsically an [Orthogonal Procrustes problem](https://web.stanford.edu/class/cs273/refs/procrustes.pdf).



1. **The Power of Asymmetry in Binary Hashing.** *Behnam Neyshabur et al, NIPS 2013.*  [[PDF]](https://proceedings.neurips.cc/paper/2013/file/84438b7aae55a0638073ef798e50b4ef-Paper.pdf)   
Notes: This paper proves that shorter and more accurate hash codes can be obtained by using two distinct code maps. The asymmetry here is defined in function space, i.e., adopting different hashing functions for similarity calculation.



1. **Sparse Projections for High-dimensional Binary Codes.** *Yan Xia et al, CVPR 2015.*  [[PDF]](https://ieeexplore.ieee.org/document/7298954)   
Notes: This paper proposes **Sparse Projections (SP)** which imposes orthogonal and L<sub>0</sub> constraints on the hashing projection matrix. For the resulting non-convex optimization problem, they adopt the variable-splitting and penalty techniques.


1. **Supervised Discrete Hashing.** *Fumin Shen et al, CVPR 2015.*  [[PDF]](https://ieeexplore.ieee.org/document/7298598)   
Notes: This paper proposes **Supervised Discrete Hashing (SDH)** that solves the discrete optimization without any relaxations using a discrete cyclic coordinate descent (DCC) algorithm. The assumption is that good hash codes are optimal for linear classification. 


1. **Fast Supervised Discrete Hashing.** *Jie Gui et al, TPAMI 2018.*  [[PDF]](https://ieeexplore.ieee.org/document/7873258)   
Notes: This paper proposes **Fast Supervised Discrete Hashing (FSDH)** that regress the class labels of training data to the corresponding hash codes to accelerate the SDH. It avoids iterative hash code-solving step of the DCC algorithm. 



<a name="deepHash" />
### Deep Hashing

1. **Feature Learning based Deep Supervised Hashing with Pairwise Labels.** *Wu-Jun Li et al, IJCAI 2016.*  [[PDF]](https://www.ijcai.org/Proceedings/16/Papers/245.pdf)   
Notes: This paper proposes the first deep hashing method called **Deep Pairwise-Supervised Hashing (DPSH)** for applications with pairwise labels, which can perform simultaneous feature learning and hash-code learning. This paper can be regarded as a deep learning extension of [Latent Factor Hashing (LFH)](https://dl.acm.org/doi/pdf/10.1145/2600428.2609600).



1. **Asymmetric Deep Supervised Hashing.** *Qing-Yuan Jiang et al, AAAI 2018.*  [[PDF]](https://arxiv.org/abs/1707.08325)   
Notes: This paper proposes the first asymmetric deep hashing method called **Asymmetric Deep Supervised Hashing(ADSH)**, which can treats query points and database points in an asymmetric way. Specifically, ADSH learns a deep hash function only for query points while the hash codes for
database points are directly learned.


1. **Deep Supervised Hashing with Anchor Graph.** *Yudong Chen et al, ICCV 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/9010953)   
Notes: This paper proposes **Deep Anchor Graph Hashing (DAGH)**, which adopts an anchor graph to learn the hash codes of the whole training samples directly during training. Since in different epochs the anchors used are different, the entire training samples will be trained if given enough epochs. This paper can also be regarded as an asymmetric deep hashing method. 





<a name="DA" />
## Domain Adaptation

1. **Universal Domain Adaptation.** *Kaichao You et al, CVPR 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8954135)   
Notes: This paper introduces **Universal Domain Adaptation (UDA)** that requires no prior knowledge on the label sets of source and target domains.









<a name="matrixcompletion" />

## Matrix Completion

1. **Speedup Matrix Completion with Side Information: Application to Multi-Label Learning.** *Miao Xu et al, NIPS 2013.*  [[PDF]](https://proceedings.neurips.cc/paper/2013/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)   
Notes: This paper explicitly explores the side information of data for matrix completion, with which the number of observed entries needed for a perfect recovery of matrix M can be dramatically reduced from $O(n ln^2 n)$ to $O(ln n)$.


1. **Graph Convolutional Matrix Completion.** *Rianne van den Berg et al, KDD 2018.*  [[PDF]](https://arxiv.org/pdf/1706.02263.pdf)   
Notes: This paper considers matrix completion for recommender systems from the point of view of
link prediction on graphs.


<a name="cur" />

## CUR Decomposition

1. **Joint Active Learning with Feature Selection via CUR Matrix Decomposition.** *Changsheng Li et al, IEEE TPAMI 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8367893)   
Notes: This work performs sample selection and feature selection simulteneously based on CUR decomposition.
