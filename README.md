# Awesome papers in the fields of computer vision, machine learning, pattern recognition, and data mining.


![Paper Reading](https://img.shields.io/badge/PhD-Paper_Reading-green)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Paper Reading](https://img.shields.io/badge/Fields-CV_ML_PR_DM-blue)

This is a collection of awesome papers I have read (carefully or roughly) in the fields of computer vision, machine learning, pattern recognition, and data mining (where the notes only represent my personal views). The collection will be continuously updated, so stay tuned. Any suggestions and comments are welcome (jianglinlu@outlook.com). 


## Contents
- [Manifold Learning](#mainfold)
- [Sparse Representation](#sparseRepre)
- [Low-Rank Representation](#LRR)
- [Clustering](#clustering)
  - [Shallow Clustering](#shallowClustering)
  - [Deep Clustering](#deepclustering)
- [Learning to Hash](#hashing)
  - [Shallow Hashing](#shallowHash)
  - [Deep Hashing](#deepHash)
- [Domain Adaptation](#DA)
- [Convolutional Neural Network](#CNN)
- [Transformers](#Transformers)
  - [Vision Transformers](#ViT)
  - [Graph Transformers](#GraphTransformers)
- [Graph Neural Network](#GNN)
  - [Latent Graph Learning](#LGL) 
  - [Self-Supervised GNN](#SSLGNN) 
  - [Weisfeiler-Lehman Test](#wltest)
  - [Deeper GNN](#deeperGNN)
- [Network Compression](#networkcompression)
  - [Pruning](#pruning)
  - [Knowledge Distillation](#knowDistil)
  - [Network Quantization](#quantization)
  - [Low-Rank Factorization](#LRF)
- [Learning with Label Noise](#labelnoise)
- [Low-Level Vision](#llv)
  - [High Dynamic Range Imaging](#HDR)
  - [Image Super-Resolution](#ImageSR)
  - [Image Low-Light Enhancement](#ImageLLE)
- [Vision Language Pretraining](#VLP)
- [Point Cloud](#pointcloud)
- [Causal Inference](#cause)
- [Others](#others)
  - [Procrustes Problem](#procrustes)
  - [CUR Decomposition](#cur)
  - [Matrix Completion](#matrixcompletion)
  - [PAC Learning](#PACLearning)
  - [Quantum Computing](#quantumcomputing)
- [Learning Sources](#learningsources)
- [Acknowledgement](#acknowledgement)





<a name="mainfold" />

## Manifold Learning [[Back to Top]](#)

1. **A Global Geometric Framework for Nonlinear Dimensionality Reduction.** *Joshua B. Tenenbaum et al, Science 2000.*  [[PDF]](https://www.science.org/doi/10.1126/science.290.5500.2319) [[Author]](http://web.mit.edu/cocosci/josh.html)   
Notes: This is a classical paper that proposes **Isometric Feature Mapping (ISOMAP)** for nonlinear dimensionality reduction, which contains three step including neighborhood graph construction, shortest paths computing, and low-dimensional embedding.

1. **Nonlinear Dimensionality Reduction by Locally Linear Embedding.** *Sam T. Roweis et al, Science 2000.*  [[PDF]](https://www.science.org/doi/10.1126/science.290.5500.2323)  [[Author]](https://cs.nyu.edu/~roweis/)   
Notes: This is a classical paper that proposes **Locally Linear Embedding (LLE)** for nonlinear dimensionality reduction, which, being different from ISOMAP, assumes that each data point and its neighbors lie on or close to a locally linear patch of the mainfold. The local geometry of these patches is characterized by linear coefficients that reconstruct each data point from its neighbors. 

1. **Laplacian Eigenmaps for Dimensionality Reduction and Data Representation.** *Mikhail Belkin et al, Neural Computation 2003.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/6789755) [[Author]](http://misha.belkin-wang.org/)    
Notes: This is a classical paper that proposes **Laplacian Eigenmaps (LE)** for nonlinear dimensionality reduction and data representation, which uses the notion of Laplacian of the graph to compute a low-dimensional representation of the data set that optimally preserves local neighborhood information in a certain sense.

1. **Locality Preserving Projections.** *Xiaofei He et al, NIPS 2003.*  [[PDF]](https://proceedings.neurips.cc/paper/2003/file/d69116f8b0140cdeb1f99a4d5096ffe4-Paper.pdf) [[Author]](http://www.cad.zju.edu.cn/home/xiaofeihe/)   
This paper proposes **Locality Preserving Projections (LPP)**, which computes a linear projection matrix that maps the data point to a subspace. The linear transformation optimally preserves local neighborhood information in a certain sense. This work can be regarded as a linear extension of Laplacian Eigenmaps (LE).

1. **Neighborhood Preserving Embedding.** *Xiaofei He et al, ICCV 2005.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1544858) [[Author]](http://www.cad.zju.edu.cn/home/xiaofeihe/)    
This paper proposes **Neighborhood Preserving Embedding (NPE)**, which aims at preserving the local neighborhood structure on the data manifold. Here, the locality or local structure means that each data point can be represented as a linear combination of its neighbors. This work can be regarded as a linear extension of Locally Linear Embedding (LLE).

1. **Graph Embedding and Extensions: A General Framework for Dimensionality Reduction.** *Shuicheng Yan et al, IEEE TPAMI 2007.*  [[PDF]](https://ieeexplore.ieee.org/document/4016549) [[Author]](https://yanshuicheng.ai/)     
Notes: This paper proposes a general framework called **Graph Embedding** for linear dimensionality reduction, in which an intrinsic graph characterizes the intraclass compactness while a penalty graph characterizes the interclass separability. 

1. **Spectral Regression for Efficient Regularized Subspace Learning.** *Deng Cai et al, ICCV 2007.*  [[PDF]](https://ieeexplore.ieee.org/document/4016549) [[Author]](http://www.cad.zju.edu.cn/home/dengcai/)   
Notes: This paper proposes **Spectral Regression (SR)** for subspace learning, which casts the problem of learning the projective functions into a regression framework and avoids the eigen-decomposition of dense matrices. It is worth noting that different kinds of regularizers can be naturally incorporated into SR such as L<sub>1</sub> regularization.








<a name="sparseRepre" />

## Sparse Representation [[Back to Top]](#)

1. **Regression Shrinkage and Selection Via the Lasso.** *Rob Tibshirani, Journal of the Royal Statistical Society 1996.*  [[PDF]](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.2517-6161.1996.tb02080.x) [[Author]](http://tibshirani.su.domains/)   
Notes: This is a classical paper that proposes **Least absolute shrinkage and selection operator (LASSO)** for linear regression, which minimizes the residual sum of squares subject to the sum of the absolute value of the coefficients being less than a constant. AKA L<sub>1</sub> penalty. 

1. **Regularization and Variable Selection via the Elastic Net.** *Hui Zou et al, Journal of the royal statistical society 2005.*  [[PDF]](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1467-9868.2005.00503.x) [[Author]](http://users.stat.umn.edu/~zouxx019/)   
Notes: This paper proposes **Elastic Net** for regularization and variable selection, which encourages a grouping effect where strongly correlated predictors tend to be in or out of the model together. The Elastic Net combines L<sub>2</sub> and L<sub>1</sub> regularizations together, and can be viewed as a generalization of LASSO.

1. **Sparse Principal Component Analysis.** *Hui Zou et al, Journal of Computational and Graphical Statistics 2006.*  [[PDF]](https://hastie.su.domains/Papers/spc_jcgs.pdf) [[Author]](http://users.stat.umn.edu/~zouxx019/)   
Notes: This paper proposes **Sparse Principal Component Analysis (SPCA)** that introduces the LASSO or Elastic Net into Principal Component Analysis (PCA) to produce modified principal components with sparse loadings. It formulates PCA as a regression-type optimization problem and then obtains sparse loadings by imposing the LASSO or Elastic Net constraint on the regression coefficients. The Theorem 4 of **Reduced Rank Procrustes Rotation** is useful. 

1. **Robust Face Recognition via Sparse Representation.** *John Wright et al, IEEE TPAMI 2009.*  [[PDF]](https://ieeexplore.ieee.org/document/4483511) [[Author]](http://www.columbia.edu/~jw2966/)   
Notes:

1. **Robust principal component analysis?.** *Emmanuel J. Cand`es et al, Journal of the ACM 2011.*  [[PDF]](https://dl.acm.org/doi/abs/10.1145/1970392.1970395) [[Author]](https://candes.su.domains/)   
Notes:


### Resources

1. **SLEP: Sparse Learning with Efficient Projections.** *Jun Liu et al, Arizona State University 2009.*  [[PDF]](https://media.gradebuddy.com/documents/411659/9e8ca2d6-1223-47a2-81b1-02c81e2f40ce.pdf) [[Resource]](http://www.yelabs.net/software/SLEP/) [[Author]](https://sites.google.com/site/junliupage/)  
Notes: This paper develops a **Sparse Learning with Efficient Projections (SLEP)** package written in Matlab for sparse representation learning.















<a name="LRR" />

## Low-Rank Representation [[Back to Top]](#)

1. **Robust Subspace Segmentation by Low-Rank Representation.** *Guangcan Liu et al, ICML 2010.*  [[PDF]](https://icml.cc/Conferences/2010/papers/521.pdf) [[Author]](https://sites.google.com/site/guangcanliu/)   
Notes:

1. **Robust Recovery of Subspace Structures by Low-Rank Representation.** *Guangcan Liu et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6180173) [[Author]](https://sites.google.com/site/guangcanliu/)   
Notes:

1. **The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices.** *Zhouchen Lin et al, arXiv 2013.*  [[PDF]](https://arxiv.org/abs/1009.5055) [[Author]](https://zhouchenlin.github.io/)   
Notes:




















<a name="clustering" />

## Clustering [[Back to Top]](#)


<a name="shallowClustering" />

### Shallow Clustering

1. **Large Scale Spectral Clustering with Landmark-Based Representation.** *Xinlei Chen et al, AAAI 2011.*  [[PDF]](http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/2011_AAAI-LSC.pdf) [[Author]](https://xinleic.xyz/)   
Notes: This paper adopts anchor graph for spectral clustering.

1. **Sparse Subspace Clustering: Algorithm, Theory, and Applications.** *Ehsan Elhamifar et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/document/6482137) [[Author]](https://khoury.northeastern.edu/home/eelhami/)   
Notes: This papers proposes **Sparse Subspace Clustering (SSC)** which introduces sparse representation into the subspace clustering problem, and define the **Self-Expressiveness** property: each data point in a union of subspaces can be efficiently reconstructed by a combination of other points in the dataset.

1. **Clustering and Projected Clustering with Adaptive Neighbors.** *Feiping Nie et al, KDD 2014.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/2623330.2623726) [[Author]](https://sites.google.com/site/feipingnie/)  
Notes: This paper proposes **Clustering with Adaptive Neighbors (CAN)** to learn the data similarity matrix and clustering structure simultaneously. It is worth noting that they present an effective method to determine the regularization parameter considering the locality of the data.







<a name="deepclustering" />

### Deep Clustering

1. **A Critique of Self-Expressive Deep Subspace Clustering.** *Benjamin David Haeffele et al, ICLR 2021.*  [[PDF]](https://openreview.net/forum?id=FOyuZ26emy) [[Author]](https://www.cis.jhu.edu/~haeffele/)   
Notes: This papers show that many previous deep subspace networks are ill-posed, and their performance improvement is largely attributable to an ad-hoc post-processing step.

1. **Deep Subspace Clustering Networks.** *Pan Ji et al, NIPS 2017.*  [[PDF]](https://arxiv.org/abs/1709.02508) [[Author]](https://sites.google.com/view/panji530)   
Notes: This is the first deep subspace clustering network, however, it has been proved to be ill-posed by the [paper](https://openreview.net/forum?id=FOyuZ26emy).








<a name="hashing" />

## Learning to Hash [[Back to Top]](#)


<a name="shallowHash" />

### Shallow Hashing

1. **Locality-Sensitive Hashing Scheme based on p-Stable Distributions.** *Mayur Datar et al, SCG 2004.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/997817.997857) [[Author]](https://www.linkedin.com/in/mayur-datar-b0a65018/)   
Notes: This paper proposes a novel **Locality-Sensitive Hashing (LSH)** for the Approximate Nearest Neighbor Problem under L<sub>p</sub> norm, based on p-stable distributions.  The key idea is to hash the points using several hash functions so as to ensure that, for each function, the probability of collision is much higher for objects which are close to each other than for those which are far apart. Then, one can determine near neighbors by hashing the query point and retrieving elements stored in buckets containing that point.

1. **Spectral Hashing.** *Yair Weiss et al, NIPS 2008.*  [[PDF]](https://proceedings.neurips.cc/paper/2008/file/d58072be2820e8682c0a27c0518e805e-Paper.pdf) [[Author]](https://www.cs.huji.ac.il/~yweiss/)   
Notes: This paper proposes **Spectral Hashing (SH)** where the bits are calculated by thresholding a subset of eigenvectors of the Laplacian of the similarity graph. The basic idea is to embed the data in a Hamming space such that the neighbors in the original data space remain neighbors in the Hamming space. 

1. **Hashing with Graphs.** *Wei Liu et al, ICML 2011.*  [[PDF]](https://icml.cc/2011/papers/6_icmlpaper.pdf) [[Author]](https://sites.google.com/view/cuweiliu)   
Notes: This paper proposes **Anchor Graph Hashing (AGH)** which builds an approximate neighborhood graph using Anchor Graphs, resulting in O(n) time for graph construction. The graph is sufficiently sparse with performance approaching to the true KNN graph as the number of anchors increases.

1. **Iterative Quantization: A Procrustean Approach to Learning Binary Codes for Large-Scale Image Retrieval.** *Yunchao Gong et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/document/6296665) [[Author]](https://www.linkedin.com/in/yunchao-gong-150a32a/)   
Notes: This paper proposes **Iterative Quantization (ITQ)** that finds a rotation of zero-centered data so as to minimize the quantization error of mapping this data to the vertices of a zero-centered binary hypercube. The optimization problem is intrinsically an [Orthogonal Procrustes problem](https://web.stanford.edu/class/cs273/refs/procrustes.pdf).

1. **The Power of Asymmetry in Binary Hashing.** *Behnam Neyshabur et al, NIPS 2013.*  [[PDF]](https://proceedings.neurips.cc/paper/2013/file/84438b7aae55a0638073ef798e50b4ef-Paper.pdf) [[Author]](https://www.neyshabur.net/)   
Notes: This paper proves that shorter and more accurate hash codes can be obtained by using two distinct code maps. The asymmetry here is defined in function space, i.e., adopting different hashing functions for similarity calculation.

1. **Sparse Projections for High-dimensional Binary Codes.** *Yan Xia et al, CVPR 2015.*  [[PDF]](https://ieeexplore.ieee.org/document/7298954) [[Author]](https://www.linkedin.com/in/xiayan-ustc/)   
Notes: This paper proposes **Sparse Projections (SP)** which imposes orthogonal and L<sub>0</sub> constraints on the hashing projection matrix. For the resulting non-convex optimization problem, they adopt the variable-splitting and penalty techniques.

1. **Supervised Discrete Hashing.** *Fumin Shen et al, CVPR 2015.*  [[PDF]](https://ieeexplore.ieee.org/document/7298598) [[Author]](https://scholar.google.com.au/citations?user=oqYL6fQAAAAJ&hl=en)  
Notes: This paper proposes **Supervised Discrete Hashing (SDH)** that solves the discrete optimization without any relaxations using a discrete cyclic coordinate descent (DCC) algorithm. The assumption is that good hash codes are optimal for linear classification. 

1. **Fast Supervised Discrete Hashing.** *Jie Gui et al, TPAMI 2018.*  [[PDF]](https://ieeexplore.ieee.org/document/7873258) [[Author]](https://guijiejie.github.io/)   
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

1. **Deep Cross-Modal Hashing.** *Qing-Yuan Jiang et al, CVPR 2017.*  [[PDF]](https://ieeexplore.ieee.org/document/9010953)   
Notes: This paper proposes the first deep cross-modal hashing called **Deep Cross-Modal Hashing (DCMH)** which can be regarded as a cross-modal extension of [Deep Pairwise-Supervised Hashing (DPSH)](https://www.ijcai.org/Proceedings/16/Papers/245.pdf).


### Survey

1. **A Survey on Learning to Hash.** *Jingdong Wang et al, IEEE TPAMI 2018.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) 

1. **A Survey on Deep Hashing Methods.** *Xiao Luo et al, ACM TKDD 2022.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) 




<a name="DA" />

## Domain Adaptation [[Back to Top]](#)
1. **My Personal Learning Notes on Domain Adaptation.** *Jianglin Lu.*  [[PDF]](https://jianglin954.github.io/Awesome-papers-in-the-fileds-of-CV-ML-DM/files/Introduction_of_Two_Domain_Adaptation_Methods.pdf)  

1. **Domain Adaptation under Target and Conditional Shift.** *Kun Zhang et al, ICML 2013.*  [[PDF]](http://proceedings.mlr.press/v28/zhang13d.pdf)   
Notes: This paper exploits importance reweighting or sample transformation to find the learning machine that works well on test data, and propose to estimate the weights or transformations by reweighting or transforming training data to reproduce the covariate distribution on the test domain.

1. **Domain Adaptation with Conditional Transferable Components.** *Mingming Gong et al, ICML 2016.*  [[PDF]](http://proceedings.mlr.press/v48/gong16.pdf)   
Notes:

1. **Universal Domain Adaptation.** *Kaichao You et al, CVPR 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8954135)   
Notes: This paper introduces **Universal Domain Adaptation (UDA)** that requires no prior knowledge on the label sets of source and target domains.













































<a name="CNN" />

## Convolutional Neural Network [[Back to Top]](#)







































<a name="Transformers" />

## Transformers [[Back to Top]](#)





<a name="ViT" />

### Vision Transformers

1. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** *Ze Liu et al, ICCV 2021.*  [[PDF]](https://ieeexplore.ieee.org/document/9710580)




<a name="GraphTransformers" />

### Graph Transformers

1. **Graph Transformer Networks.** *Seongjun Yun et al, NeurIPS 2019.*  [[PDF]](https://arxiv.org/abs/1911.06455)




### Survey


1. **A Survey on Vision Transformer.** *Kai Han et al, IEEE TPAMI 2022.*  [[PDF]](https://ieeexplore.ieee.org/document/9716741)



































<a name="GNN" />

## Graph Neural Network [[Back to Top]](#)


1. **My Personal Learning Notes on Graph Neural Network.** *Jianglin Lu.* [[PDF]](https://jianglin954.github.io/Awesome-papers-in-the-fileds-of-CV-ML-DM/files/Introduction_of_GNN.pdf)



<a name="LGL" />

### Latent Graph Learning

1. **Differentiable Graph Module (DGM) for Graph Convolutional Networks.** *Anees Kazi et al, IEEE TPAMI 2023.* [[PDF]](https://ieeexplore.ieee.org/document/9763421)






<a name="SSLGNN" />

### Self-Supervised GNN


1. **Self-Supervised Representation Learning via Latent Graph Prediction.** *Yaochen Xie et al, ICML 2022.* [[PDF]](https://arxiv.org/pdf/2202.08333.pdf)   
Notes: This paper proposes **LaGraph**, a predictive SSL framework for representation learning of graph data, based on self-supervised latent graph prediction. It makes two assumptions: a. the observed feature vector of each node in an obsereved graph is independently generated from a certain distribution conditioned on the corresponding latent graph; b. the conditional distribution of the observed graph is centered at the latent graph. 




<a name="wltest" />

### Weisfeiler-Lehman Test

1. **Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning.** *Pan Li et al, NeurIPS 2020.* [[PDF]](https://ieeexplore.ieee.org/document/9046288)







<a name="deeperGNN" />

### Deeper GNN

1. **Towards Deeper Graph Neural Networks.** *Meng Liu et al, KDD 2020.* [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403076) [[Author]](https://mengliu1998.github.io/)   
Notes: 







### Survey

1. **A Comprehensive Survey on Graph Neural Networks.** *Zonghan Wu et al, IEEE TNNLS 2021.* [[PDF]](https://ieeexplore.ieee.org/document/9046288)

1. **Self-Supervised Learning of Graph Neural Networks: A Unified Review.** *Yaochen Xie et al, IEEE PAMI 2023.* [[PDF]](https://arxiv.org/pdf/2102.10757.pdf) [[Author]](https://ycremar.github.io/)  



































<a name="networkcompression" />

## Network Compression [[Back to Top]](#)




<a name="pruning" />

### Pruning

1. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** *Jonathan Frankle et al,  ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1803.03635.pdf)   
Notes: This paper proposes the **Lottery Ticket Hypothesis**: A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the
original network after training for at most the same number of iterations.

1. **Dual Lottery Ticket Hypothesis.** *Yue Bai et al,  ICLR 2022.*  [[PDF]](https://openreview.net/pdf?id=fOsN52jn25l)   
Notes: This paper proposes the **Dual Lottery Ticket Hypothesis**:  A randomly selected subnetwork from a randomly
initialized dense network can be transformed into a trainable condition, where the transformed subnetwork can be trained in isolation and achieve better at least comparable performance to LTH and other strong baselines.


<a name="knowDistil" />

### Knowledge Distillation







<a name="quantization" />

### Network Quantization






<a name="LRF" />

### Low-Rank Factorization




### Survey


1. **Recent Advances on Neural Network Pruning at Initialization.** *Huan Wang et al, IJCAI 2022.*  [[PDF]](https://www.ijcai.org/proceedings/2022/786)  
Notes: This is the first survey on pruning at initialization.












<a name="labelnoise" />

## Learning with Label Noise [[Back to Top]](#)

1. **.** * et al, ICCV 2019.*  [[PDF]]()






























<a name="ssl" />

## Self-Supervised Learning [[Back to Top]](#)

1. **.** * et al, ICCV 2019.*  [[PDF]]()



































<a name="llv" />

## Low-Level Vision [[Back to Top]](#)


<a name="HDR" />

### High Dynamic Range Imaging

1. **Ghost-free High Dynamic Range Imaging with Context-Aware Transformer.** *Zhen Liu et al, ECCV 2022.*  [[PDF]](https://arxiv.org/abs/2208.05114)   
Notes: This is the first work that introduces Transformer for HDR imaging. 





#### Survey 

1. **Deep Learning for HDR Imaging: State-of-the-Art and Future Trends.** *Lin Wang et al, IEEE TPAMI 2021.*  [[PDF]](https://arxiv.org/pdf/2110.10394.pdf) [[Author]](https://addisonwang2013.github.io/vlislab/linwang.html)   
 





<a name="ImageSR" />

### Image Super-Resolution




<a name="ImageLLE" />

### Image Low-Light Enhancement


















<a name="VLP" />

## Vision Language Pretraining [[Back to Top]](#)


1. **Learning Transferable Visual Models From Natural Language Supervision.** *Alec Radford et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf)




















<a name="pointcloud" />

## Point Cloud [[Back to Top]](#)

1. **Dynamic Graph CNN for Learning on Point Clouds.** *Yue Wang et al, ACM TOG 2019.*  [[PDF]](https://dl.acm.org/doi/10.1145/3326362)












<a name="cause" />

## Causal Inference [[Back to Top]](#)

1. **.** * et al, ICCV 2019.*  [[PDF]]()




















<a name="others" />

## Others [[Back to Top]](#)



<a name="procrustes" />

### Procrustes Problem

1. **A Generalized Solution of the Orthogonal Procrustes Problem.** *Peter H. Schönemann, Psychometrika 1966.*  [[PDF]](https://web.stanford.edu/class/cs273/refs/procrustes.pdf)   
Notes: This is a classical paper that proposes a generalized solution to the **Orthogonal Procrustes Problem**, which is applicable to the case where the matrices involved are of less than full column rank. 

1. **<font color=blue>Generalized Embedding Regression: A Framework for Supervised Feature Extraction.</font>** *Jianglin Lu et al, IEEE TNNLS 2022.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9248650)   
Notes: My first-author paper attempts to unify previous hand-crafted feature extraction methods in a **Generalized Embedding Regression (GER)** framework. Based on GER, a new supervised feature extraction method is further proposed, which adopts the penalty graph Laplacian as the constraint matrix of a generalized orthogonal constraint. We theoretically demonstrate that the resulted optimization subproblem is intrinsically an unbalanced Procrustes problem, and elaborately design an iterative algorithm to solve it with convergence guarantee. Although the topic is somewhat out-of-date, the optimization makes me excited.




<a name="labelPropagation" />

### Label Propagation

1. **Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions** *Xiaojin Zhu, ICML 2003.*  [[PDF]](https://mlg.eng.cam.ac.uk/zoubin/papers/zgl.pdf)   
Notes: 

1. **Label Propagation Through Linear Neighborhoods** *Fei Wang, ICML 2006.*  [[PDF]](https://dl.acm.org/doi/10.1145/1143844.1143968)   
Notes: 









<a name="cur" />

### CUR Decomposition

1. **Joint Active Learning with Feature Selection via CUR Matrix Decomposition.** *Changsheng Li et al, IEEE TPAMI 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8367893)   
Notes: This work performs sample selection and feature selection simulteneously based on CUR decomposition.

1. **Robust CUR Decomposition: Theory and Imaging Applications.** *HanQin Cai et al, SIAM 2021.*  [[PDF]](https://arxiv.org/pdf/2101.05231.pdf)   
Notes: This paper considers the use of Robust PCA in a CUR decomposition framework.




<a name="matrixcompletion" />

### Matrix Completion

1. **Speedup Matrix Completion with Side Information: Application to Multi-Label Learning.** *Miao Xu et al, NIPS 2013.*  [[PDF]](https://proceedings.neurips.cc/paper/2013/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)   
Notes: This paper explicitly explores the side information of data for matrix completion, with which the number of observed entries needed for a perfect recovery of matrix M can be dramatically reduced from $O(n \ln^2 n)$ to $O(\ln n)$.

1. **Graph Convolutional Matrix Completion.** *Rianne van den Berg et al, KDD 2018.*  [[PDF]](https://arxiv.org/pdf/1706.02263.pdf)   
Notes: This paper considers matrix completion for recommender systems from the point of view of
link prediction on graphs.






<a name="PACLearning" />

### Probably Approximately Correct (PAC) Learning





<a name="quantumcomputing" />

### Quantum Computing

1. **My Personal Learning Notes on Quantum Computing.** *Jianglin Lu.* [[PDF]](https://jianglin954.github.io/files/Quantum%20Computing.pdf)







<a name="learningsources" />

## Learning Sources <a href="#top">[Back to Top]</a>

1. **UvA Deep Learning Tutorials.** [[Website]](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html)   

1. **PyTorch Image Models (timm) Documentation** [[Website]](https://github.com/rwightman/pytorch-image-models)

1. **PyTorch Geometric (PyG) Documentation** [[Website]](https://pytorch-geometric.readthedocs.io/en/latest/)

1. **Deep Graph Library (DGL) Tutorials and Documentation** [[Website]](https://docs.dgl.ai/en/latest/)

1. **PyTorch Lightning Documentation** [[Website]](https://pytorch-lightning.readthedocs.io/en/stable/)

1. **Qiskit Machine Learning Documentation** [[Website]](https://qiskit.org/documentation/machine-learning/index.html)

1. **Interpretable Machine Learning** [[Website]](https://christophm.github.io/interpretable-ml-book/)








<a name="acknowledgement" />

## Acknowledgement <a href="#top">[Back to Top]</a>

<!-- I would like to express my most sincere appreciation to those who provided me with unwavering encouragement and insightful suggestions throughout the course of my research career.   

[Yun Raymond Fu](http://www1.ece.neu.edu/~yunfu/)              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[Jie Zhou]()                                                   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Alexander Lapin](https://kpfu.ru/Alexandr.Lapin?p_lang=2)     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[Pan Li](https://sites.google.com/view/panli-purdue/home)	   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[Xi-Zhao Wang](http://www.hebmlc.org/en/)                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[Tongliang Liu](https://tongliang-liu.github.io/)              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
[Yudong Chen]()                                                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
-->