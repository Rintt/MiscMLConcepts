README
Machine Learning Algorithms Implementation

This repository contains a collection of machine learning algorithms that cover both supervised and unsupervised learning techniques. It includes implementations for classification, clustering, dimensionality reduction, regression, and utility functions.

    Principal Component Analysis (PCA)
        Project data onto lower dimensions, helpful for noise reduction and explorative analysis.

    Locally Linear Embedding (LLE)
        Dimensionality reduction method that preserves the local geometry using nearest-neighbor graphs.Supports both k-nearest neighbors anddistance  based neighborhood

    Gaussian Mixture Model (GMM)
        Implements the Expectation-Maximization (EM) algorithm for fitting Gaussian Mixture Models to the data

    K-Means Clustering
        A standard K-Means algorithm for clustering data into k groups. Additionally, the Hierarchical Agglomerative Clustering (kmeans_agglo) is included

    Kernel Ridge Regression (KRR)
        A regression method that uses kernel functions for non-linear regression tasks

    Cross-Validation for Model Selection
        This function helps in selecting the best model parameters  based on performance metrics.

    Gamma Index Calculation
        The gammaidx function computes the gamma index for each data point, based on distances to its nearest neighbors. Evaluates clustering quality.

    Area Under the Curve (AUC) Calculation
        The auc function computes the Area Under the Curve (AUC) for binary classification tasks

    Neural Network Classifier
        A fully connected feedforward neural network implemented using PyTorch. The neural network supports ReLU activations, softmax outputs, dropout regularization, and gradient-based optimization. It is suitable for classification tasks.
    Support Vector Machines (SVM)
        Quadratic Programming-based SVM (svm_qp)
