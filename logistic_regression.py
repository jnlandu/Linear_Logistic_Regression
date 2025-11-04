import numpy as np
import matplotlib.pyplot as plt

import gradient_descent


class LogisticRegression:
    """
    This class provides functionality for training a logistic regression model
    on labeled binary classification data using gradient descent optimization.
    It supports both batch gradient descent and stochastic gradient descent.
    The user can specify  the number of epochs, the learning rate for the
    algorithm and beta, if they want to apply momentum.

    Attributes
    ----------
    x: np.ndarray
        outputs the features matrix to be used in training the linear regression model.
    y: np.ndarray
        outputs the target vector to be used in training the linear regression model.
    epochs: int
        outputs the number of epochs set to run in the implementation of the gradient descent algorithm.
    lr: float
        outputs the learning rate used (to be used) in the algorithm.
    batch: int
        number of datasets to be used for each iteration in training the model. An output of None indicates batch
        gradient descent was used; 1 indicates stochastic gradient descent algorithm was implemented.
    beta: float
        outputs the value for beta (momentum) used.
    weights: np.ndarray
        outputs the weights determined from the implementation of the gradient descent algorithm.
    losses: np.ndarray
        outputs the average loss for each epoch from the implementation of the gradient descent algorithm.

    Methods
    --------
    add_ones:
        adds a column of ones to the feature matrix to account for the intercept.
    sigmoid:
        computes the sigmoid function for the given features matrix.
    predict:
        classifies predictions from the sigmoid function into appropriate class using a threshold of 0.5.
    cross_entropy:
        computes the loss for a given weights. Used in gradient descent algorithm to compute weights.
    grad_function:
        gradient of cross entropy. Used in gradient descent algorithm to compute weights.
    fit:
        implements gradient descent algorithm using inputs to determine best weights. Also changes loss attribute
        of the object.
    accuracy:
        computes the level of accuracy for the predictor on given features matrix and target vector.
        Multiplying output by 100 gives accuracy in percentages.
    """

    def __init__(self, x, y, epochs=100, lr=0.1, beta=0, batch=None):
        """
        Initialize regression model for a given input matrix X and target vector y.
        :param x: np.ndarray
            input matrix
        :param y: np.ndarray
            target vector
        :param epochs: int
            number of epochs
        :param lr: float
            learning rate
        :param beta: float
            momentum
        :param batch: int

        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.batch = self.x.shape[0] if batch is None else batch
        self.losses = []
        self.weights = None

    def add_ones(self, x):
        """
        adds a column of ones to the feature matrix to account for the intercept.
        :param x: np.ndarray
            features matrix
        :return: np.ndarray
        """
        ones_vector = np.ones((x.shape[0], 1))
        return np.hstack((ones_vector, x))

    def sigmoid(self, x):
        """
        computes the sigmoid function for the given features matrix. returns an N x 1 matrix
        of probabilities.
        :param x: np.ndarray
            features matrix
        :return: np.ndarray
        """
        if x.shape[1] != self.weights.shape[0]:
            x = self.add_ones(x)
        z = x @ self.weights
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        """
        classifies predictions from the sigmoid function into appropriate class using a threshold of 0.5.
        :param x: np.ndarray
            features matrix
        :return: np.ndarray
        """
        probas = self.sigmoid(x)
        output = (probas >= 0.5).astype(int)
        return output

    def cross_entropy(self, x, y, w):
        """
        computes the loss for a given weights. Used in gradient descent algorithm to compute weights.
        :param x: np.ndarray
            features matrix
        :param y: np.ndarray
            target vector
        :param w: np.ndarray
            weights
        :return: float
        """
        y = y.reshape(-1, 1)
        z = x @ w
        y_pred = 1 / (1 + np.exp(-z))
        loss = -1 * np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def grad_function(self, x, y, w):
        """
        gradient of cross entropy. Used in gradient descent algorithm to compute weights.
        :param x: np.ndarray
            features matrix
        :param y: np.ndarray
            target vector
        :param w: np.ndarray
            weights
        :return: float
        """
        y = y.reshape(-1, 1)
        z = x @ w
        y_pred = 1 / (1 + np.exp(-z))
        return -1 / x.shape[0] * x.T @ (y - y_pred)

    def fit(self):
        """
        implements gradient descent algorithm using inputs to determine best weights. Also changes loss attribute
        of the object.
        :return: None
        """
        grad_descent = gradient_descent.GradientDescent(self.add_ones(self.x), self.y, self.epochs, self.lr,
                                                        self.batch, self.beta, loss_function=self.cross_entropy,
                                                        grad_function=self.grad_function)
        grad_descent.fit()
        self.weights = grad_descent.weights
        self.losses = grad_descent.losses

    def accuracy(self, x=None, y_true=None):
        """
        computes the level of accuracy for the predictor on given features matrix and target vector.
        Multiplying output by 100 gives accuracy in percentages.
        :param x: np.ndarray
            features matrix
        :param y_true: np.ndarray
            target vector
        :return: float
        """
        x = self.x if x is None else x
        y_true = self.y if y_true is None else y_true
        y_pred = self.predict(x)
        y_true = y_true.reshape((-1, 1))
        accuracy_value = np.sum(y_true == y_pred) / y_true.shape[0]
        return accuracy_value

    def plot_decision_boundary(self, X=None, y=None, w=None, b=None, title='Decision Boundary'):
        """
        plots the feature matrix and target vector with decision boundary.
        :param X: np.ndarray
            features matrix
        :param y: np.ndarray
            target vector
        :param w: np.ndarray
            weights
        :param b: np.ndarray
            intercept
        :param title: str
            adds title to graph
        :return:
        """
        # z = w1x1 + w2x2 + w0
        # one can think of the decision boundary as the line x2=mx1+c
        # Solving we find m and c
        x = self.x if X is None else X
        y = self.y if y is None else y
        x1 = [X[:, 0].min(), X[:, 0].max()]
        m = -w[1] / w[2]
        c = -b / w[2]
        x2 = m * x1 + c

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.xlim([-2, 3])
        plt.ylim([0, 2.2])
        plt.xlabel("feature 1")
        plt.ylabel("feature 2")
        plt.plot(x1, x2, 'y-')
        plt.title(title)
        plt.show()
