# isciml
Imaging using SCIentific Machine Learning (ISCIML)

This is a user manual for generating training data for three-dimensional imaging via inversion of magnetic data. The method is general, and can be applied to many practical scenarios including, but not limited to, imaging onshore and offshore buried pipelines, unexploded ordnance (UXO) detection, mineral exploration, etc.

## High level overview of impact on use cases
Artificial Intelligence (AI) and Machine Learning (ML) applications have become active focus areas in many industry applications and geophysical imaging and inversion is no exception. Key bottlenecks manifest in the following manner:

- Time associated with the simulation of geophysical response from complex realistic targets is nontrivial and frequently off-limits for practical scale deployment.
- Large three-dimensional models require large allocation of run-time memory which oftentimes won’t fit on a single or even simple 2-4 CPU clusters. In addition, the training for such models requires a substantial number of these to be accessible to the algorithm at the same time. This requires efficient utilization of high-performance computing clusters.
- The cost associated with parallel, high-performance computation, whether cloud-based or on-premises, is often prohibitively high.

Key changes to the form of the simulated input data used for training, and the corresponding design of the architecture of the hidden layers enable ~ O(n) reduction in the computational complexity of the training architecture. In addition, multi-GPU Distributed Deep Learning (DDL) algorithms (Balu et al. 2021) optimized specifically for machine learning problems are deployed for rapid training of the preferred deep learning architecture which is then used for predicting the expected subsurface material property via inference on actual field data. 
