import scipy.linalg as la
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import numpy as np
import torch
from torch import  nn, optim
from torch.nn import Parameter, ParameterList, Module
import sklearn.svm
from torch.utils.data import DataLoader, TensorDataset

class svm_qp:
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None

    def compute_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = np.zeros((n, m))
        if self.kernel == 'linear':
            K = np.dot(X, Y.T)
        elif self.kernel == 'polynomial':
            K = (1 + np.dot(X, Y.T)) ** self.kernelparameter
        elif self.kernel == 'gaussian':
            sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
            K = np.exp(-0.5 * (1 / self.kernelparameter**2) * sq_dists)
        return K

    def fit(self, X, y):
        n = X.shape[0]
        y = y * 2 - 1  # Convert y from {0, 1} to {-1, 1}
        K = self.compute_kernel(X)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones((n, 1)))
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        A = matrix(y, (1, n), 'd')
        b = matrix(0.0)
        
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).flatten()
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alpha_sv = alphas[sv]
        self.X_sv = X[sv]
        self.Y_sv = y[sv]
        self.b = np.mean([self.Y_sv[i] - np.sum(self.alpha_sv * self.Y_sv * K[ind[i], ind]) for i in range(len(self.X_sv))])

    def predict(self, X):
        if self.X_sv is None:
            raise ValueError("Model has not been fitted yet")
        K = self.compute_kernel(X, self.X_sv)
        decision = np.dot(K, self.alpha_sv * self.Y_sv) + self.b
        return np.sign(decision)


class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
            self.clf = sklearn.svm.SVC(C=C, kernel=kernel, gamma=1./(1./2. * kernelparameter**2))
        else:
            # Ensure 'degree' is set as an integer only for polynomial kernel
            if kernel == 'polynomial':
                self.clf = sklearn.svm.SVC(C=C, kernel=kernel, degree=int(kernelparameter), gamma='scale', coef0=1)
            else:
                self.clf = sklearn.svm.SVC(C=C, kernel=kernel, gamma='scale', coef0=1)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


def plot_boundary_2d(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)  # Make sure the output of predict is correctly sized
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.show()

def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X**2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2, 0)[:, np.newaxis]
        Y2 = sum(Y**2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if Y.isinstance(bool) and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter**2))
    else:
        raise Exception('unspecified kernel')
    return K


class neural_network(nn.Module):
    def __init__(self, layers, lr=0.1, p=0.1, lam=0.1):
        super(neural_network, self).__init__()
        self.layers = nn.ModuleList()
        self.p = p
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            weight = nn.Parameter(torch.randn(layers[i], layers[i + 1]) * np.sqrt(0.1))
            bias = nn.Parameter(torch.randn(layers[i + 1]) * np.sqrt(0.1))
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers[-1].weight = weight
            self.layers[-1].bias = bias

        self.dropout = nn.Dropout(p)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=lam)

    def relu(self, X, W, b):
        Z = torch.mm(X, W) + b
        A = torch.relu(Z)
        # Apply dropout
        if self.training:  # Only apply dropout during training
            A = self.dropout(A)
        return A

    def softmax(self, X, W, b):
        Z = torch.mm(X, W) + b
        return torch.log_softmax(Z, dim=1)

    def forward(self, X):
        for i, layer in enumerate(self.layers[:-1]):
            X = self.relu(X, layer.weight, layer.bias)
        return self.softmax(X, self.layers[-1].weight, self.layers[-1].bias)

    def loss(self, output, y):
        return nn.functional.nll_loss(output, y)

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        for epoch in range(nsteps):
            for data, target in loader:
                self.train()  # Set the model to training mode
                self.optimizer.zero_grad()
                output = self.forward(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

            if plot and epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item()}')

        if plot:
            plt.figure()
            plt.plot(loss.item(), label='Training Loss')
            plt.legend()
            plt.show()

    def predict(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self.forward(X).max(1)[1]

