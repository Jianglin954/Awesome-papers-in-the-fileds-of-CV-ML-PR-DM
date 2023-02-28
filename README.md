# Awesome papers in the fields of computer vision, machine learning, pattern recognition, and data mining.


![Paper Reading](https://img.shields.io/badge/PhD-Paper_Reading-green)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Paper Reading](https://img.shields.io/badge/Fields-CV_ML_PR_DM-blue)

This is a collection of awesome papers I have read (carefully or roughly) in the fields of computer vision, machine learning, pattern recognition, and data mining (where the notes only represent my personal views). The collection will be continuously updated, so stay tuned. Any suggestions and comments are welcome (jianglinlu@outlook.com). 


## Contents
- [Manifold Learning](#Manifold)
  - [Nonlinear Dimensionality Reduction](#NonlinearDR)
  - [Subspace Learning](#subspaceLearning) 
  - [Graph Construction](#graphConstruction)
- [Sparse Representation](#sparseRepre)
- [Low-Rank Representation](#LRR)
- [Clustering](#clustering)
  - [Shallow Clustering](#shallowClustering)
  - [Deep Clustering](#deepclustering)
- [Learning to Hash](#hashing)
  - [Shallow Hashing](#shallowHash)
  - [Deep Hashing](#deepHash)
- [Domain Adaptation](#DA)
  - [Shallow Domain Adaptation](#shallowDA)
  - [Deep Domain Adaptation](#deepDA)
- [Convolutional Neural Network](#CNN)
- [Transformers](#Transformers)
  - [Vision Transformers](#ViT)
  - [Graph Transformers](#GraphTransformers)
- [Graph Neural Network](#GNN)
  - [Spectral-based GNN](#SpectralGNN)
  - [Spatial-based GNN](#SpatialGNN)
  - [Graph Pooling](#GraphPooling)
  - [Graph Structure Learning](#LGL) 
  - [Self-Supervised GNN](#SSLGNN)
  - [GNN Pre-training](#GNNPreTrain)
  - [GNN Adversarial Attacks](#GNNAA) 
  - [GNN Pruning](#GNNpruning)
  - [Graph Domain Adaptation](#GraphDomainAdaptation)
  - [Graph Data Augmentation](#GraphDataAugmentation)
  - [Graph Generation](#GraphGeneration)
  - [Weisfeiler-Lehman Test](#wltest)
  - [Graph Information Bottleneck](#graphInfoBottle)
  - [Deeper GNN](#deeperGNN)
  - [Few-Shot Learning on Graph](#fewshotGNN)
- [Diffusion Models](#DiffusionModels)
- [Network Compression](#networkcompression)
  - [Pruning](#pruning)
  - [Knowledge Distillation](#knowDistil)
  - [Network Quantization](#quantization)
  - [Low-Rank Factorization](#LRF)
- [Learning with Label Noise](#labelnoise)
  - [Statistically Inconsistent Classifiers](#SICnoise)
  - [Statistically Consistent Classifiers](#SCCnoise)
- [Contrastive Learning](#CLR)
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
  - [Optimization Methods](#Optimization)
  - [PAC Learning](#PACLearning)
  - [Information Theory](#informationTheory)
  - [Quantum Computing](#quantumcomputing)
- [Learning Sources](#learningsources)





<a name="Manifold" />

## Manifold Learning [[Back to Top]](#)



<a name="NonlinearDR" />

### Nonlinear Dimensionality Reduction

1. **A Global Geometric Framework for Nonlinear Dimensionality Reduction.** *Joshua B. Tenenbaum et al, Science 2000.*  [[PDF]](https://www.science.org/doi/10.1126/science.290.5500.2319) [[Author]](http://web.mit.edu/cocosci/josh.html)   
Notes: This is a classical paper that proposes **Isometric Feature Mapping (ISOMAP)** for nonlinear dimensionality reduction, which contains three step including neighborhood graph construction, shortest paths computing, and low-dimensional embedding.

1. **Nonlinear Dimensionality Reduction by Locally Linear Embedding.** *Sam T. Roweis et al, Science 2000.*  [[PDF]](https://www.science.org/doi/10.1126/science.290.5500.2323)  [[Author]](https://cs.nyu.edu/~roweis/)   
Notes: This is a classical paper that proposes **Locally Linear Embedding (LLE)** for nonlinear dimensionality reduction, which, being different from ISOMAP, assumes that each data point and its neighbors lie on or close to a locally linear patch of the manifold. The local geometry of these patches is characterized by linear coefficients that reconstruct each data point from its neighbors. 

1. **Laplacian Eigenmaps for Dimensionality Reduction and Data Representation.** *Mikhail Belkin et al, Neural Computation 2003.*  [[PDF]](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf) [[Author]](http://misha.belkin-wang.org/)    
Notes: This is a classical paper that proposes **Laplacian Eigenmaps (LE)** for nonlinear dimensionality reduction and data representation, which uses the notion of Laplacian of the graph to compute a low-dimensional representation of the data set that optimally preserves local neighborhood information in a certain sense.

1. **Locality Preserving Projections.** *Xiaofei He et al, NIPS 2003.*  [[PDF]](https://proceedings.neurips.cc/paper/2003/file/d69116f8b0140cdeb1f99a4d5096ffe4-Paper.pdf) [[Author]](http://www.cad.zju.edu.cn/home/xiaofeihe/)   
This paper proposes **Locality Preserving Projections (LPP)**, which computes a linear projection matrix that maps the data point to a subspace. The linear transformation optimally preserves local neighborhood information in a certain sense. This work can be regarded as a linear extension of Laplacian Eigenmaps (LE).

1. **Neighborhood Preserving Embedding.** *Xiaofei He et al, ICCV 2005.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1544858) [[Author]](http://www.cad.zju.edu.cn/home/xiaofeihe/)    
This paper proposes **Neighborhood Preserving Embedding (NPE)**, which aims at preserving the local neighborhood structure on the data manifold. Here, the locality or local structure means that each data point can be represented as a linear combination of its neighbors. This work can be regarded as a linear extension of Locally Linear Embedding (LLE).

1. **Graph Embedding and Extensions: A General Framework for Dimensionality Reduction.** *Shuicheng Yan et al, IEEE TPAMI 2007.*  [[PDF]](https://ieeexplore.ieee.org/document/4016549) [[Author]](https://yanshuicheng.ai/)     
Notes: This paper proposes a general framework called **Graph Embedding** for linear dimensionality reduction, in which an intrinsic graph characterizes the intraclass compactness while a penalty graph characterizes the interclass separability. 


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="subspaceLearning" />

### Subspace Learning


1. **Spectral Regression for Efficient Regularized Subspace Learning.** *Deng Cai et al, ICCV 2007.*  [[PDF]](https://ieeexplore.ieee.org/document/4016549) [[Author]](http://www.cad.zju.edu.cn/home/dengcai/)   
Notes: This paper proposes **Spectral Regression (SR)** for subspace learning, which casts the problem of learning the projective functions into a regression framework and avoids the eigen-decomposition of dense matrices. It is worth noting that different kinds of regularizers can be naturally incorporated into SR such as L<sub>1</sub> regularization.


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="graphConstruction" />

### Graph Construction

1. **Graph Construction and $b$-Matching for Semi-supervised Learning.** *Tony Jebara et al, ICML 2009.*  [[PDF]](https://icml.cc/Conferences/2009/papers/188.pdf) [[Author]](http://www.cs.columbia.edu/~jebara/)

1. **Influence of Graph Construction on Semi-supervised Learning.** *Celso Sousa et al, ECML PKDD 2013.*  [[PDF]](https://link.springer.com/chapter/10.1007/978-3-642-40994-3_11) [[Author]](https://scholar.google.com/citations?user=i8jXj9kAAAAJ&hl=zh-CN&oi=sra)

1. **How to Learn a Graph from Smooth Signals.** *Vassilis Kalofolias et al, AISTATS 2016.*  [[PDF]](http://proceedings.mlr.press/v51/kalofolias16.pdf) [[Author]](https://scholar.google.ch/citations?user=Bz1RQ8MAAAAJ&hl=en)


1. **A Quest for Structure: Jointly Learning the Graph Structure and Semi-Supervised Classification.** *Xuan Wu et al, CIKM 2018.*  [[PDF]](https://arxiv.org/pdf/1909.12385.pdf) [[Author]]()    
Notes: This paper proposes **Parallel Graph Learning (PG-Learn) for the graph construction step of semi-supervised learning. The two main ingredients include a) a gradient-based optimization of the edge weights (different kernel bandwidths in each dimension) and b) a parallel hyperparameter search algorithm. It adopts LGC algorithm and the corresponding solution can be found without explicitly taking any matrix inverse and instead using the power method. 




1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()














<a name="sparseRepre" />

## Sparse Representation [[Back to Top]](#)

1. **Regression Shrinkage and Selection Via the Lasso.** *Rob Tibshirani, Journal of the Royal Statistical Society 1996.*  [[PDF]](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.2517-6161.1996.tb02080.x) [[Author]](http://tibshirani.su.domains/)   
Notes: This is a classical paper that proposes **Least absolute shrinkage and selection operator (LASSO)** for linear regression, which minimizes the residual sum of squares subject to the sum of the absolute value of the coefficients being less than a constant. AKA L<sub>1</sub> penalty. 

1. **Regularization and Variable Selection via the Elastic Net.** *Hui Zou et al, Journal of the royal statistical society 2005.*  [[PDF]](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1467-9868.2005.00503.x) [[Author]](http://users.stat.umn.edu/~zouxx019/)   
Notes: This paper proposes **Elastic Net** for regularization and variable selection, which encourages a grouping effect where strongly correlated predictors tend to be in or out of the model together. The Elastic Net combines L<sub>2</sub> regularization and L<sub>1</sub> regularization together, and can be viewed as a generalization of LASSO.

1. **Sparse Principal Component Analysis.** *Hui Zou et al, Journal of Computational and Graphical Statistics 2006.*  [[PDF]](https://hastie.su.domains/Papers/spc_jcgs.pdf) [[Author]](http://users.stat.umn.edu/~zouxx019/)   
Notes: This paper proposes **Sparse Principal Component Analysis (SPCA)** that introduces the LASSO or Elastic Net into Principal Component Analysis (PCA) to produce modified principal components with sparse loadings. It formulates PCA as a regression-type optimization problem and then obtains sparse loadings by imposing the LASSO or Elastic Net constraint on the regression coefficients. The Theorem 4 of **Reduced Rank Procrustes Rotation** is useful. 

1. **Robust Face Recognition via Sparse Representation.** *John Wright et al, IEEE TPAMI 2009.*  [[PDF]](https://ieeexplore.ieee.org/document/4483511) [[Author]](http://www.columbia.edu/~jw2966/)   
Notes:

1. **Robust principal component analysis?.** *Emmanuel J. Cand`es et al, Journal of the ACM 2011.*  [[PDF]](https://dl.acm.org/doi/abs/10.1145/1970392.1970395) [[Author]](https://candes.su.domains/)   
Notes:


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




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



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()













































<a name="clustering" />

## Clustering [[Back to Top]](#)


<a name="shallowClustering" />

### Shallow Clustering

1. **Large Scale Spectral Clustering with Landmark-Based Representation.** *Xinlei Chen et al, AAAI 2011.*  [[PDF]](http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/2011_AAAI-LSC.pdf) [[Author]](https://xinleic.xyz/)   
Notes: This paper adopts an **Anchor Graph** for spectral clustering.

1. **Sparse Subspace Clustering: Algorithm, Theory, and Applications.** *Ehsan Elhamifar et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/document/6482137) [[Author]](https://khoury.northeastern.edu/home/eelhami/)   
Notes: This papers proposes **Sparse Subspace Clustering (SSC)** which introduces sparse representation into the subspace clustering problem, and define the **Self-Expressiveness** property: each data point in a union of subspaces can be efficiently reconstructed by a combination of other points in the dataset.

1. **Clustering and Projected Clustering with Adaptive Neighbors.** *Feiping Nie et al, KDD 2014.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/2623330.2623726) [[Author]](https://sites.google.com/site/feipingnie/)  
Notes: This paper proposes **Clustering with Adaptive Neighbors (CAN)** to learn the data similarity matrix and clustering structure simultaneously. It is worth noting that they present an effective method to determine the regularization parameter considering the locality of the data.


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="deepclustering" />

### Deep Clustering

1. **Deep Subspace Clustering Networks.** *Pan Ji et al, NIPS 2017.*  [[PDF]](https://arxiv.org/abs/1709.02508) [[Author]](https://sites.google.com/view/panji530)   
Notes: This is the first deep subspace clustering network, however, it has been proved to be ill-posed by the [paper](https://openreview.net/forum?id=FOyuZ26emy).


1. **A Critique of Self-Expressive Deep Subspace Clustering.** *Benjamin David Haeffele et al, ICLR 2021.*  [[PDF]](https://openreview.net/forum?id=FOyuZ26emy) [[Author]](https://www.cis.jhu.edu/~haeffele/)   
Notes: This papers show that many previous deep subspace networks are ill-posed, and their performance improvement is largely attributable to an ad-hoc post-processing step.

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()












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

1. **Feature Learning based Deep Supervised Hashing with Pairwise Labels.** *Wu-Jun Li et al, IJCAI 2016.*  [[PDF]](https://www.ijcai.org/Proceedings/16/Papers/245.pdf) [[Author]](https://cs.nju.edu.cn/lwj/index.htm)  
Notes: This paper proposes the first deep hashing method called **Deep Pairwise-Supervised Hashing (DPSH)** for applications with pairwise labels, which can perform simultaneous feature learning and hash-code learning. This paper can be regarded as a deep learning extension of [Latent Factor Hashing (LFH)](https://dl.acm.org/doi/pdf/10.1145/2600428.2609600).

1. **Asymmetric Deep Supervised Hashing.** *Qing-Yuan Jiang et al, AAAI 2018.*  [[PDF]](https://arxiv.org/abs/1707.08325) [[Author]](https://jiangqy.github.io/)   
Notes: This paper proposes the first asymmetric deep hashing method called **Asymmetric Deep Supervised Hashing(ADSH)**, which can treats query points and database points in an asymmetric way. Specifically, ADSH learns a deep hash function only for query points while the hash codes for
database points are directly learned.

1. **Deep Supervised Hashing with Anchor Graph.** *Yudong Chen et al, ICCV 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/9010953) [[Author]]()    
Notes: This paper proposes **Deep Anchor Graph Hashing (DAGH)**, which adopts an **Anchor Graph** to learn the hash codes of the whole training samples directly during training. Since in different epochs the anchors used are different, the entire training samples will be trained if given enough epochs. This paper can also be regarded as an asymmetric deep hashing method. 

1. **Deep Cross-Modal Hashing.** *Qing-Yuan Jiang et al, CVPR 2017.*  [[PDF]](https://ieeexplore.ieee.org/document/9010953) [[Author]](https://jiangqy.github.io/)   
Notes: This paper proposes the first deep cross-modal hashing called **Deep Cross-Modal Hashing (DCMH)** which can be regarded as a cross-modal extension of [Deep Pairwise-Supervised Hashing (DPSH)](https://www.ijcai.org/Proceedings/16/Papers/245.pdf).


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




### Survey

1. **A Survey on Learning to Hash.** *Jingdong Wang et al, IEEE TPAMI 2018.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) [[Author]](https://jingdongwang2017.github.io/) 

[//]: 1. **A Survey on Deep Hashing Methods.** *Xiao Luo et al, ACM TKDD 2022.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) [[Author]]() 












































<a name="DA" />

## Domain Adaptation [[Back to Top]](#)

1. **My Personal Learning Notes on Domain Adaptation.** *Jianglin Lu.*  [[PDF]](https://jianglin954.github.io/Awesome-papers-in-the-fileds-of-CV-ML-PR-DM/files/Introduction_of_Two_Domain_Adaptation_Methods.pdf)  



<a name="shallowDA" />

### Shallow Domain Adaptation

1. **Domain Adaptation under Target and Conditional Shift.** *Kun Zhang et al, ICML 2013.*  [[PDF]](http://proceedings.mlr.press/v28/zhang13d.pdf) [[Author]](https://www.andrew.cmu.edu/user/kunz1/index.html)    
Notes: This paper exploits importance reweighting or sample transformation to find the learning machine that works well on test data, and propose to estimate the weights or transformations by reweighting or transforming training data to reproduce the covariate distribution on the test domain.

1. **Domain Adaptation with Conditional Transferable Components.** *Mingming Gong et al, ICML 2016.*  [[PDF]](http://proceedings.mlr.press/v48/gong16.pdf) [[Author]](https://mingming-gong.github.io/)    
Notes:


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="deepDA" />

### Deep Domain Adaptation

1. **Universal Domain Adaptation.** *Kaichao You et al, CVPR 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8954135) [[Author]](https://youkaichao.github.io/about)   
Notes: This paper introduces **Universal Domain Adaptation (UDA)** that requires no prior knowledge on the label sets of source and target domains.


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





### Survey

1. **A Survey on Transfer Learning.** *Sinno Jialin Pan et al, IEEE TKDE 2010.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5288526) [[Author]](http://www.cse.cuhk.edu.hk/~sinnopan/) 



1. **A Comprehensive Survey on Transfer Learning.** *Fuzhen Zhuang et al, Proceedings of the IEEE 2021.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9134370) [[Author]](https://fuzhenzhuang.github.io/index.html) 











































<a name="CNN" />

## Convolutional Neural Network [[Back to Top]](#)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


















































<a name="Transformers" />

## Transformers [[Back to Top]](#)





<a name="ViT" />

### Vision Transformers

1. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** *Ze Liu et al, ICCV 2021.*  [[PDF]](https://ieeexplore.ieee.org/document/9710580) [[Author]](https://zeliu98.github.io/)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="GraphTransformers" />

### Graph Transformers

1. **Graph Transformer Networks.** *Seongjun Yun et al, NeurIPS 2019.*  [[PDF]](https://arxiv.org/abs/1911.06455) [[Author]](https://scholar.google.com/citations?user=8-MZ2RwAAAAJ&hl=en)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




### Survey

1. **A Survey on Vision Transformer.** *Kai Han et al, IEEE TPAMI 2022.*  [[PDF]](https://ieeexplore.ieee.org/document/9716741) [[Author]](https://scholar.google.com/citations?user=vThoBVcAAAAJ&hl=en&oi=sra)




































<a name="GNN" />

## Graph Neural Network [[Back to Top]](#)


1. **My Personal Learning Notes on Graph Neural Network.** *Jianglin Lu.* [[PDF]](https://jianglin954.github.io/Awesome-papers-in-the-fileds-of-CV-ML-PR-DM/files/Introduction_of_GNN.pdf) 



<a name="SpectralGNN" />

### Spectral-based GNN



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="SpatialGNN" />

### Spatial-based GNN

1. **Inductive Representation Learning on Large Graphs.** *William L. Hamilton et al, NeurIPS 2017.*  [[PDF]](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) [[Author]](https://www.cs.mcgill.ca/~wlh/)   
Notes: This paper proposes **SAmple and aggreGatE (GraphSAGE)** for inductive node embedding.

1. **Graph Attention Networks.** *Petar Veličković et al, ICLR 2018.*  [[PDF]](https://arxiv.org/abs/1710.10903) [[Author]](https://petar-v.com/)   
Notes: 


1. **Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Learning.** *Qimai Li et al, AAAI 2018.*  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/11604) [[Author]](https://liqimai.github.io/)


1. **Adaptive Sampling Towards Fast Graph Representation Learning.** *Wenbing Huang et al, NeurIPS 2018.* [[PDF]](https://proceedings.neurips.cc/paper/2018/file/01eee509ee2f68dc6014898c309e86bf-Paper.pdf) [[Author]](https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=31&ruccode=ADIIMVRnBzFXMFdnVTAIOw%3D%3D&ln=en) 


1. **Grale: Designing Networks for Graph Learning.** *Jonathan Halcrow et al, KDD 2020.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403302) [[Author]](https://scholar.google.com/citations?user=2zZucy4AAAAJ&hl=en&oi=ao)   


1. **Graph Neural Networks with Adaptive Residual.** *Xiaorui Liu et al, NeurIPS 202.*  [[PDF]](https://proceedings.neurips.cc/paper/2021/file/50abc3e730e36b387ca8e02c26dc0a22-Paper.pdf) [[Author]]()





1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="GraphPooling" />

### Graph Pooling

1. **Hierarchical Graph Representation Learning with Differentiable Pooling.** *Rex Ying et al, NeurIPS 2018.*  [[PDF]](https://proceedings.neurips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf) [[Author]](https://cs.stanford.edu/people/rexy/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="LGL" />

### Graph Structure Learning (AKA, Latent Graph Learning)

1. **Deep Convolutional Networks on Graph-Structured Data.** *Mikael Henaff et al, arXiv 2015.*  [[PDF]](https://arxiv.org/pdf/1506.05163.pdf) [[Author]](http://www.mikaelhenaff.net/)   
Notes: 


1. **Adaptive Graph Convolutional Neural Networks.** *Ruoyu Li et al, AAAI 2018.* [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/11691) [[Author]](https://scholar.google.co.uk/citations?user=S23gEPsAAAAJ&hl=en)  
Notes: The bottlenecks of current graph CNNS: a. restrict graph degree; b. required identical graph structure shared among inputs; C. fixed graph constructed without training; d. incapability of learning from topological structure. This paper proposes **Adaptive Graph Convolution Network (AGCN)** that feeds on original data of diverse graph structures. AGCN seems to be designed primarily for graph classification. Besides, AGCN needs an initial graph and suffers from the limitation of transductive models as described in [DGM](https://ieeexplore.ieee.org/document/9763421).

1. **Topology Optimization based Graph Convolutional Network.** *Liang Yang et al, IJCAI 2019.*  [[PDF]](https://www.ijcai.org/proceedings/2019/0563.pdf) [[Author]](http://yangliang.github.io/)   
Notes: This paper proposes **Topology Optimization based GCN (TO-GCN)** to jointly learn the network topology and the parameters of fully connected network. The refinement of the network topology is modeled as a **[Label Propagation](#labelPropagation)** process where the network topology is modeled as the multiplication of the predicted label matrix with its transpose matrix. The TO-GCN also penalizes the high similarities between the nodes from different classes. 

    
1. **Semi-Supervised Learning With Graph Learning-Convolutional Networks.** *Bo Jiang et al, CVPR 2019.*  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) [[Author]](https://scholar.google.com/citations?user=n-aTwuMAAAAJ&hl=zh-CN)   
Notes: This paper proposes **Graph Learning-Convolutional Network (GLCN)** for semi-supervised task, which integrates both graph learning and graph convolution in a unified network architecture such that both given and estimated labels are incorporated to provide weakly supervised information for graph structure refinement. The graph learning function is similar to [GAT](https://arxiv.org/abs/1710.10903) and the graph learning loss is similar to [CAN](https://dl.acm.org/doi/pdf/10.1145/2623330.2623726). The graph learned in the sense of probability is dense and lack sparse structure.

1. **Large Scale Graph Learning from Smooth Signals.** *Vassilis Kalofolias et al, ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1710.05654.pdf) [[Author]](https://scholar.google.ch/citations?user=Bz1RQ8MAAAAJ&hl=en)  
Notes: This papers uses approximate nearest neighbor techniques for large scale graph learning from smooth signals. Also refer to [paper](http://proceedings.mlr.press/v51/kalofolias16.pdf).  

1. **Learning Discrete Structures for Graph Neural Networks.** *Luca Franceschi et al, ICML 2019.*  [[PDF]](http://proceedings.mlr.press/v97/franceschi19a/franceschi19a.pdf) [[Author]](https://scholar.google.com/citations?user=NCls8VMAAAAJ&hl=en&oi=ao)  
Notes: This paper proposes **Learning Discrete Structures (LDS)** to learn the graph structure and the parameters of GCNs by approximately solving a bilevel program that learns a discrete probability distribution of the edges of the graph. Given two objective functions $F$ and $L$, the outer and inner objectives, and two sets of variables, $\theta \in \mathcal{R}^{m}$ and $\omega \in \mathcal{R}^{d}$, the outer and inner variables, a **Bilevel Program** is given by: $\min_{\theta, \omega_{\theta}}F(\omega_{\theta}, \theta)$ such that $\omega_{\theta} \in \arg \min_{\omega} L(\omega, \theta)$. LDS only works in the transductive setting and the graph topology learned cannot be controlled due to the sampling strategy. 

1. **Graph Structure Learning for Robust Graph Neural Networks** *Wei Jin et al, KDD 2020.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) [[Author]](http://cse.msu.edu/~jinwei2/)   
Notes: This paper proposes **Property GNN (Pro-GNN)** that explores graph properties of sparsity, low rank and feature smoothness to defend adversarial attacks. Pro-GNN simultaneously learns the clean graph structure from perturbed graph and GNN parameters to defend against adversarial attacks. This paper assumes that the graph structure has already been perturbed before training GNNs while the node features are not changed.

1. **Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings.** *Yu Chen et al, NeurIPS 2020.* [[PDF]](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf) [[Author]](http://academic.hugochan.net/)  
Notes: This paper proposes **Iterative Deep Graph Learning (IDGL)** that learns graph structure and graph embedding simultaneously. The graph learning problem is transferred as a similarity metric learning problem and an adaptive graph regularization is leveraged (assume that the optimized graph structure is potentially a shift from the initial graph structure). IDGL adopts multi-head self-attention with $\epsilon$-neighborhood sparsification for graph construction. An **Anchor Graph** based version is also proposed and the corresponding node-anchor message passing strategy is provided. IDGL works on (semi-)supervised tasks and needs an initial $kNN$ graph construction.

1. **Latent-Graph Learning for Disease Prediction.** *Luca Cosmo et al, MICCAI 2020.*  [[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_62) [[Author]](https://www.dsi.unive.it/~cosmo/)   
Notes: This paper proposes an end-to-end trainable graph learning architecture that automatically learns to predict an underlying patients-graph. The edge weight is learned through a sigmoid-like function with two trainable parameters. This method can work in inductive setting since it does not directly optimize a graph for a given population but rather learn a function that predicts the graph from input features. The graph learned is directly used only in a classification loss without any regularization. Besides, the global threshold and the Euclidean space embedding may not be necessarily optimal. 

1. **Graph-Revised Convolutional Network.** *Donghan Yu et al, ECML PKDD 2020.*  [[PDF]](https://arxiv.org/pdf/1911.07123.pdf) [[Author]](https://plusross.github.io/)   
Notes: This paper proposes **Graph-Revised Convolutional Network (GRCN)**, where a GCN-based graph revision module is introduced for predicting missing edges and revising edge weights w.r.t. downstream tasks via joint optimization. The similarity graph is calculated based on node embedding using certain kernel function (specifically, using dot product in their implementation for simplicity). The **[Representer Theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)** is provided to show that, under certain conditions, the optimal regression function can be expressed as a linear combination of kernel functions defined on training samples. Compared with the graph revision in [GAT](https://arxiv.org/abs/1710.10903) and [GLCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) which use entrywise product, GRCN adopts the entrywise addition operator in order for new edges to be considered. A graph sparsification process is also proposed and the gradients will only backpropagate through the top-$K$ values. In GRCN, an initial graph is required.   


1. **SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks.** *Bahare Fatemi et al, NeurIPS 2021.* [[PDF]](https://proceedings.neurips.cc/paper/2021/file/bf499a12e998d178afd964adf64a60cb-Paper.pdf) [[Author]](https://baharefatemi.github.io/homepage/)  
Notes: This paper proposes **Simultaneous Learning of Adjacency and GNN Parameters with Self-supervision (SLAPS)** for semi-supervised classification, which provides more supervision for inferring a graph structure through self-supervision. The authors also identify a **Supervision Starvation** problem in latent graph learning: the edges between pairs of nodes that are far from labeled nodes receive insufficient supervision. To solve this, a multi-task learning framework is designed by supplementing the classification task with a self-supervised task (which is based on the hypothesis that a graph structure is suitable for predicting the node feature is also suitable for predicting the node labels). Also refer to [paper](http://proceedings.mlr.press/v119/you20a/you20a.pdf).

1. **Graph Structure Learning with Variational Information Bottleneck.** *Qingyun Sun et al, AAAI 2022.*  [[PDF]](https://arxiv.org/pdf/2112.08903.pdf) [[Author]](https://sunqysunqy.github.io/)   
Notes: This paper proposes **Variational Information Bottleneck guided Graph Structure Learning (VIB-GSL)** that advances the **Information Bottleneck** principle for graph structure learning. 


1. **Robust Graph Structure Learning via Multiple Statistical Tests.** *Yaohua Wang et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2210.03956.pdf) [[Author]](https://scholar.google.com/citations?user=TRAwmsgAAAAJ&hl=zh-CN)


1. **pyGSL: A Graph Structure Learning Toolkit.** *Max Wasserman et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2211.03583.pdf) [[Author]](https://github.com/maxwass)


1. **Learning Continuous Graph Structure with Bilevel Programming for Graph Neural Networks.** *Minyang Hu et al, IJCAI 2022.*  [[PDF]](https://www.ijcai.org/proceedings/2022/0424.pdf) [[Author]]()
2. 

1. **Learning Graph Structure from Convolutional Mixtures.** *Max Wasserman et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2205.09575.pdf) [[Author]](https://github.com/maxwass)


1. **Self-organization Preserved Graph Structure Learning with Principle of Relevant Information.** *Qingyun Sun et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2301.00015.pdf) [[Author]](https://sunqysunqy.github.io/)

1. **Regularized Graph Structure Learning with Semantic Knowledge for Multi-variates Time-Series Forecasting.** *Hongyuan Yu et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2210.06126.pdf) [[Author]]()

1. **DBGSL: Dynamic Brain Graph Structure Learning.** *Alexander Campbell et al, .*  [[PDF]](https://arxiv.org/pdf/2209.13513.pdf) [[Author]]()

1. **Position-aware Structure Learning for Graph Topology-imbalance by Relieving Under-reaching and Over-squashing.** *Qingyun Sun et al, CIKM 2022.*  [[PDF]](https://arxiv.org/pdf/2208.08302.pdf) [[Author]](https://sunqysunqy.github.io/)


1. **Semi-Supervised Clustering via Dynamic Graph Structure Learning.** *Huaming Ling et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2209.02513.pdf) [[Author]]()


1. **Boosting Graph Structure Learning with Dummy Nodes.** *Xin Liu et al, ICML 2022.*  [[PDF]](https://arxiv.org/pdf/2206.08561.pdf) [[Author]](https://cse.hkust.edu.hk/~xliucr/)

1. **GPN: A Joint Structural Learning Framework for Graph Neural Networks.** *Qianggang Ding et al, AAAI 2022.*  [[PDF]](https://arxiv.org/pdf/2205.05964.pdf) [[Author]](http://mrdqg.com/)


1. **Multi-view graph structure learning using subspace merging on Grassmann manifold.** *Razieh Ghiasi et al, Multimedia Tools and Applications 2022.*  [[PDF]](https://arxiv.org/pdf/2204.05258.pdf) [[Author]]()


1. **Differentiable Graph Module (DGM) for Graph Convolutional Networks.** *Anees Kazi et al, IEEE TPAMI 2023.* [[PDF]](https://ieeexplore.ieee.org/document/9763421) [[Author]](https://campar.in.tum.de/Main/AneesKazi.html)   
Notes: The current GNNs are often restricted to the transductive setting and rely on the assumption that underlying graph is known and fixed. This paper proposes **Differentiable Graph Module (DGM)** that infers the graph directly from the data. Specifically, DGM is a learnable function that predicts edge probabilities in the graph which are optimal for the downstream task. **Latent Graph**: the graph itself is not be explicitly given.      


1. **ASGNN: Graph Neural Networks with Adaptive Structure.** *Zepeng Zhang et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2210.01002.pdf) [[Author]](https://home.zepengzhang.com/)






1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()









<a name="SSLGNN" />

### Self-Supervised GNN

1. **Graph Contrastive Learning with Augmentations.** *Yuning You et al, NeurIPS 2020.*  [[PDF]]() [[Author]]()

1. **When Does Self-Supervision Help Graph Convolutional Networks?** *Yuning You et al, ICML 2020.*  [[PDF]]() [[Author]]()

1. **Self-Supervised Representation Learning via Latent Graph Prediction.** *Yaochen Xie et al, ICML 2022.* [[PDF]](https://arxiv.org/pdf/2202.08333.pdf) [[Author]](https://ycremar.github.io/)   
Notes: This paper proposes **LaGraph**, a predictive SSL framework for representation learning of graph data, based on self-supervised latent graph prediction. It makes two assumptions: a. the observed feature vector of each node in an observed graph is independently generated from a certain distribution conditioned on the corresponding latent graph; b. the conditional distribution of the observed graph is centered at the latent graph. 

1. **Automated Self-Supervised Learning for Graphs.** *Wei Jin et al, ICLR 2022.*  [[PDF]](https://arxiv.org/pdf/2106.05470.pdf) [[Author]](http://cse.msu.edu/~jinwei2/)





<a name="GNNPreTrain" />

### GNN Pre-Training

1. **Strategies for Pre-training Graph Neural Networks.** *Weihua Hu et al, ICLR 2020.*  [[PDF]](https://arxiv.org/pdf/1905.12265.pdf) [[Author]](https://weihua916.github.io/)

1. **GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training.** *Jiezhong Qiu et al, KDD 2020.*  [[PDF]](https://arxiv.org/pdf/2006.09963.pdf) [[Author]](http://jiezhongqiu.com/)






<a name="GNNAA" />

### GNN Adversarial Attacks


1. **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective.** *Kaidi Xu et al, IJCAI 2019.*  [[PDF]](https://arxiv.org/pdf/1906.04214.pdf) [[Author]](https://kaidixu.com/)

1. **Empowering Graph Representation Learning with Test-Time Graph Transformation.** *Wei Jin et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=Lnxl5pr018) [[Author]](http://cse.msu.edu/~jinwei2/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="GNNpruning" />

### GNN Pruning

1. **A Unified Lottery Ticket Hypothesis for Graph Neural Networks.** *Tianlong Chen et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/chen21p/chen21p.pdf) [[Author]](https://tianlong-chen.github.io/about/)  
Notes:

 
1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="GraphDomainAdaptation" />

### Graph Domain Adaptation


1. **Graph Domain Adaptation via Theory-Grounded Spectral Regularization.** Yuning You* et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=OysfLgrk8mk) [[Author]](https://yyou1996.github.io/)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="GraphDataAugmentation" />

### Graph Data Augmentation

1. **Data Augmentation for Graph Neural Networks.** *Tong Zhao et al, AAAI 2021.*  [[PDF]](https://arxiv.org/pdf/2006.06830.pdf) [[Author]](https://tzhao.io/)    
Notes: 

1. **Local Augmentation for Graph Neural Networks.** *Songtao Liu et al, ICML 2022.*  [[PDF]](https://arxiv.org/pdf/2109.03856.pdf) [[Author]](https://songtaoliu0823.github.io/)   
Notes: 


1. **Graph Data Augmentation for Graph Machine Learning: A Survey.** *Tong Zhao et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2202.08871.pdf) [[Author]](https://tzhao.io/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()

<a name="GraphGeneration" />

### Graph Generation

[Graph Generation](#GraphGeneration)

1. **Fast Graph Generation via Spectral Diffusion.** *Tianze Luo et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2211.08892.pdf) [[Author]](https://www.linkedin.com/in/tianze-luo-40720a82/)




1. **A Survey on Deep Graph Generation: Methods and Applications.** *Yanqiao Zhu et al, LoG 2022.*  [[PDF]](https://arxiv.org/pdf/2203.06714.pdf) [[Author]](https://sxkdz.github.io/)


1. **Generative Diffusion Models on Graphs: Methods and Applications.** *Wenqi Fan et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2302.02591.pdf) [[Author]](https://wenqifan03.github.io/)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="wltest" />

### Weisfeiler-Lehman Test

1. **How Powerful are Graph Neural Networks?** *Keyulu Xu et al, ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1810.00826.pdf) [[Author]](https://people.csail.mit.edu/keyulux/)

1. **Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning.** *Pan Li et al, NeurIPS 2020.* [[PDF]](https://ieeexplore.ieee.org/document/9046288) [[Author]](https://sites.google.com/view/panli-purdue/home)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="graphInfoBottle" />

### Graph Information Bottleneck 


1. **Graph Information Bottleneck.** *Tailin Wu et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/ebc2aa04e75e3caabda543a1317160c0-Paper.pdf) [[Author]](https://tailin.org/)
2. 
1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()







<a name="deeperGNN" />

### Deeper GNN

1. **Towards Deeper Graph Neural Networks.** *Meng Liu et al, KDD 2020.* [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403076) [[Author]](https://mengliu1998.github.io/)   
Notes: 



<a name="fewshotGNN" />

### Few-Shot Learning on Graph

1. **Few-Shot Learning on Graphs.** *Chuxu Zhang et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2203.09308.pdf) [[Author]](https://chuxuzhang.github.io/index.html)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="heterogeneousGNN" />

### Heterogeneous Graph Neural Network



1. **Heterogeneous Graph Neural Network.** *Chuxu Zhang et al, KDD 2019.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3292500.3330961) [[Author]](https://chuxuzhang.github.io/index.html)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()







### Survey

1. **Geometric Deep Learning: Going beyond Euclidean Data.** *Michael Bronstein et al, IEEE SPM 2017.* [[PDF]](https://ieeexplore.ieee.org/abstract/document/7974879) [[Author]](https://www.cs.ox.ac.uk/people/michael.bronstein/) 

1. **A Comprehensive Survey on Graph Neural Networks.** *Zonghan Wu et al, IEEE TNNLS 2021.* [[PDF]](https://ieeexplore.ieee.org/document/9046288) [[Author]](https://scholar.google.com/citations?user=SzH0tgMAAAAJ&hl=en&oi=sra)

1. **Self-Supervised Learning of Graph Neural Networks: A Unified Review.** *Yaochen Xie et al, IEEE TPAMI 2023.* [[PDF]](https://arxiv.org/pdf/2102.10757.pdf) [[Author]](https://ycremar.github.io/)  

1. **A Survey on Graph Structure Learning: Progress and Opportunities.** *Yanqiao Zhu et al, arXiv 2022.* [[PDF]](https://arxiv.org/pdf/2103.03036.pdf) [[Author]](https://sxkdz.github.io/) 












<a name="DiffusionModels" />

## Diffusion Models [[Back to Top]](#)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



### Survey 

1. **Diffusion Models in Vision: A Survey.** *Florinel-Alin Croitoru et al, TPAMI 2022.*  [[PDF]](https://arxiv.org/pdf/2209.04747.pdf) [[Author]](https://scholar.google.com/citations?user=RyD1dScAAAAJ&hl=zh-CN&oi=sra)























<a name="networkcompression" />

## Network Compression [[Back to Top]](#)




<a name="pruning" />

### Pruning

1. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** *Jonathan Frankle et al,  ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1803.03635.pdf) [[Author]](http://www.jfrankle.com/)   
Notes: This paper proposes the **Lottery Ticket Hypothesis**: A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the
original network after training for at most the same number of iterations.

1. **Dual Lottery Ticket Hypothesis.** *Yue Bai et al,  ICLR 2022.*  [[PDF]](https://openreview.net/pdf?id=fOsN52jn25l) [[Author]](https://yueb17.github.io/)   
Notes: This paper proposes the **Dual Lottery Ticket Hypothesis**:  A randomly selected subnetwork from a randomly
initialized dense network can be transformed into a trainable condition, where the transformed subnetwork can be trained in isolation and achieve better at least comparable performance to LTH and other strong baselines.


<a name="knowDistil" />

### Knowledge Distillation

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="quantization" />

### Network Quantization

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="LRF" />

### Low-Rank Factorization

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


### Survey


1. **Recent Advances on Neural Network Pruning at Initialization.** *Huan Wang et al, IJCAI 2022.*  [[PDF]](https://www.ijcai.org/proceedings/2022/786) [[Author]](http://huanwang.tech/)  
Notes: This is the first survey on pruning at initialization.






















































<a name="labelnoise" />

## Learning with Label Noise [[Back to Top]](#)



<a name="SCCnoise" />

### Statistically Consistent Classifiers

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="SICnoise" />

### Statistically Inconsistent Classifiers



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




















































<a name="CLR" />

## Contrastive Learning [[Back to Top]](#)

1. **A Simple Framework for Contrastive Learning of Visual Representations.** *Ting Chen et al, ICML 2020.*  [[PDF]](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) [[Author]](https://scholar.google.com/citations?user=KoXUMbsAAAAJ&hl=en)   
Notes: 


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


































































<a name="llv" />

## Low-Level Vision [[Back to Top]](#)


<a name="HDR" />

### High Dynamic Range Imaging

1. **Ghost-free High Dynamic Range Imaging with Context-Aware Transformer.** *Zhen Liu et al, ECCV 2022.*  [[PDF]](https://arxiv.org/abs/2208.05114) [[Author]]()   
Notes: This is the first work that introduces Transformer for HDR imaging. 


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



#### Survey 

1. **Deep Learning for HDR Imaging: State-of-the-Art and Future Trends.** *Lin Wang et al, IEEE TPAMI 2021.*  [[PDF]](https://arxiv.org/pdf/2110.10394.pdf) [[Author]](https://addisonwang2013.github.io/vlislab/linwang.html)   
 




<a name="ImageSR" />

### Image Super-Resolution

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


<a name="ImageLLE" />

### Image Low-Light Enhancement

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



















































<a name="VLP" />

## Vision Language Pretraining [[Back to Top]](#)


1. **Learning Transferable Visual Models From Natural Language Supervision.** *Alec Radford et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) [[Author]](https://scholar.google.com/citations?user=dOad5HoAAAAJ&hl=en)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()














































<a name="pointcloud" />

## Point Cloud [[Back to Top]](#)

1. **Dynamic Graph CNN for Learning on Point Clouds.** *Yue Wang et al, ACM TOG 2019.*  [[PDF]](https://dl.acm.org/doi/10.1145/3326362) [[Author]](https://yuewang.xyz/)



1. **Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling.** *Jiancheng Yang et al, CVPR 2019.*  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Modeling_Point_Clouds_With_Self-Attention_and_Gumbel_Subset_Sampling_CVPR_2019_paper.pdf) [[Author]](https://jiancheng-yang.com/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


























<a name="cause" />

## Causal Inference [[Back to Top]](#)

1. **Advances in Variational Inference.** *Cheng Zhang et al, TPAMI 2019.*  [[PDF]](https://arxiv.org/pdf/1711.05597.pdf) [[Author]](https://cheng-zhang.org/)

1. **A Causal View on Robustness of Neural Networks.** *Cheng Zhang et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/02ed812220b0705fabb868ddbf17ea20-Paper.pdf) [[Author]](https://cheng-zhang.org/)

1. **Relating Graph Neural Networks to Structural Causal Models.** Matej Zečević* et al, arXiv 2021.*  [[PDF]](https://qiniu.pattern.swarma.org/pdf/arxiv/2109.04173.pdf) [[Author]]()


1. **Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure.** *Shaohua Fan et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2209.14107.pdf) [[Author]](https://scholar.google.com.hk/citations?user=3LxcBjkAAAAJ&hl=zh-CN)

1. **The Causal Structure of Domain Invariant Supervised Representation Learning.** *Zihao Wang et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2208.06987v4.pdf) [[Author]](https://www.linkedin.com/in/zihao-wang-2b1645123/)


1. **Generative Causal Explanations for Graph Neural Networks.** *Wanyu Lin et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/lin21d/lin21d.pdf) [[Author]](https://wanyu-lin.github.io/)





2. 
1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


































<a name="others" />

## Others [[Back to Top]](#)



<a name="procrustes" />

### Procrustes Problem

1. **A Generalized Solution of the Orthogonal Procrustes Problem.** *Peter H. Schönemann, Psychometrika 1966.*  [[PDF]](https://web.stanford.edu/class/cs273/refs/procrustes.pdf) [[Author]](https://en.wikipedia.org/wiki/Peter_Sch%C3%B6nemann)   
Notes: This is a classical paper that proposes a generalized solution to the **Orthogonal Procrustes Problem**, which is applicable to the case where the matrices involved are of less than full column rank. 

1. **<font color=blue>Generalized Embedding Regression: A Framework for Supervised Feature Extraction.</font>** *Jianglin Lu et al, IEEE TNNLS 2022.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9248650)   
Notes: My first-author paper attempts to unify previous hand-crafted feature extraction methods in a **Generalized Embedding Regression (GER)** framework. Based on GER, a new supervised feature extraction method is further proposed, which adopts the penalty graph Laplacian as the constraint matrix of a generalized orthogonal constraint. We theoretically demonstrate that the resulted optimization subproblem is intrinsically an unbalanced Procrustes problem, and elaborately design an iterative algorithm to solve it with convergence guarantee. Although the topic is somewhat out-of-date, the optimization makes me excited.




<a name="labelPropagation" />

### Label Propagation

1. **Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions** *Xiaojin Zhu, ICML 2003.*  [[PDF]](https://mlg.eng.cam.ac.uk/zoubin/papers/zgl.pdf) [[Author]](https://pages.cs.wisc.edu/~jerryzhu/)   
Notes: 

1. **Label Propagation Through Linear Neighborhoods** *Fei Wang, ICML 2006.*  [[PDF]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a7cd9ab6aaabf54350554f3f64c0bf19f85e65e5) [[Author]](https://wcm-wanglab.github.io/index.html)   
Notes: 









<a name="cur" />

### CUR Decomposition

1. **Joint Active Learning with Feature Selection via CUR Matrix Decomposition.** *Changsheng Li et al, IEEE TPAMI 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8367893) [[Author]](https://cs.bit.edu.cn/szdw/jsml/gjjgccrc/lcs_e253eb02bdf246c4a88e1d2499212546/index.htm)   
Notes: This work performs sample selection and feature selection simultaneously based on CUR decomposition.

1. **Robust CUR Decomposition: Theory and Imaging Applications.** *HanQin Cai et al, SIAM 2021.*  [[PDF]](https://arxiv.org/pdf/2101.05231.pdf) [[Author]](https://hqcai.org/)   
Notes: This paper considers the use of Robust PCA in a CUR decomposition framework.




<a name="matrixcompletion" />

### Matrix Completion

1. **Speedup Matrix Completion with Side Information: Application to Multi-Label Learning.** *Miao Xu et al, NIPS 2013.*  [[PDF]](https://proceedings.neurips.cc/paper/2013/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf) [[Author]](https://researchers.uq.edu.au/researcher/26509)   
Notes: This paper explicitly explores the side information of data for matrix completion, with which the number of observed entries needed for a perfect recovery of matrix M can be dramatically reduced from $O(n \ln^2 n)$ to $O(\ln n)$.


1. **Matrix Completion on Graphs.** *Vassilis Kalofolias et al, arXiv 2014.*  [[PDF]](https://arxiv.org/pdf/1408.1717.pdf) [[Author]](https://scholar.google.ch/citations?user=Bz1RQ8MAAAAJ&hl=en)  

1. **Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks.** *Federico Monti et al, NeurIPS 2017.*  [[PDF]](https://proceedings.neurips.cc/paper/2017/file/2eace51d8f796d04991c831a07059758-Paper.pdf) [[Author]](https://scholar.google.com/citations?hl=en&user=NUdNFucAAAAJ)

1. **Graph Convolutional Matrix Completion.** *Rianne van den Berg et al, KDD 2018.*  [[PDF]](https://arxiv.org/pdf/1706.02263.pdf) [[Author]](https://www.microsoft.com/en-us/research/people/rvandenberg/)   
Notes: This paper considers matrix completion for recommender systems from the point of view of
link prediction on graphs.


1. **Inductive Matrix Completion Based on Graph Neural Networks.** *Muhan Zhang et al, ICLR 2020.*  [[PDF]](https://arxiv.org/pdf/1904.12058.pdf) [[Author]](https://muhanzhang.github.io/)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()












<a name="Optimization" />

### Optimization Methods

1. **Optimization Methods for Large-Scale Machine Learning.** *L ́eon Bottou et al, SIAM 2018.*  [[PDF]](https://epubs.siam.org/doi/epdf/10.1137/16M1080173) [[Author]](https://leon.bottou.org/start)   
Notes: This paper provides a review and commentary on the past, present, and future of numerical optimization algorithms in the context of machine learning applications. 

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="PACLearning" />

### Probably Approximately Correct (PAC) Learning

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="informationTheory" />

### Information Theory

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()











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






