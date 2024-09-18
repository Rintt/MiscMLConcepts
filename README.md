README
Machine Learning Algorithms Implementation

This repository contains a collection of machine learning algorithms that cover both supervised and unsupervised learning techniques. It includes implementations for classification, clustering, dimensionality reduction, regression, and utility functions for model evaluation and cross-validation.
Implemented Algorithms and Functions

    Principal Component Analysis (PCA)
        This dimensionality reduction technique computes principal components of a dataset. It allows for data projection onto lower dimensions, which is helpful in noise reduction and exploratory data analysis.

    Locally Linear Embedding (LLE)
        A dimensionality reduction method that preserves the local geometry of the data using nearest-neighbor graphs. The algorithm supports both k-nearest neighbors (knn) and epsilon-ball neighborhood rules for constructing the embedding.

    Gaussian Mixture Model (GMM)
        Implements the Expectation-Maximization (EM) algorithm for fitting Gaussian Mixture Models to the data. It can be initialized using K-Means for faster convergence. Visualization tools are included to display Gaussian components and covariance ellipses.

    K-Means Clustering
        A standard K-Means algorithm for clustering data into k groups. Additionally, the Hierarchical Agglomerative Clustering (kmeans_agglo) method merges clusters iteratively based on the K-Means criterion.

    Kernel Ridge Regression (KRR)
        A regression method that uses kernel functions (linear, polynomial, Gaussian) for non-linear regression tasks. The implementation includes cross-validation for hyperparameter selection, enabling the best performance on different datasets.

    Cross-Validation for Model Selection
        The repository includes a cv function for model selection using k-fold cross-validation. This function helps in selecting the best model parameters, such as kernel types and regularization strengths, based on performance metrics.

    Gamma Index Calculation
        The gammaidx function computes the gamma index for each data point, based on distances to its nearest neighbors. This function is useful for evaluating clustering quality.

    Area Under the Curve (AUC) Calculation
        The auc function computes the Area Under the Curve (AUC) for binary classification tasks. It also includes an option to visualize the Receiver Operating Characteristic (ROC) curve.

    Neural Network Classifier
        A fully connected feedforward neural network implemented using PyTorch. The neural network supports ReLU activations, softmax outputs, dropout regularization, and gradient-based optimization. It is suitable for classification tasks.
    Support Vector Machines (SVM)
        Quadratic Programming-based SVM (svm_qp): Solves the SVM optimization problem using quadratic programming. Supports linear, polynomial, and Gaussian (RBF) kernels.
        Scikit-learn SVM (svm_sklearn): A wrapper around Scikit-learn’s SVM, making it easier to use and compare with custom implementations.
