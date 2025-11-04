import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


class GradientDescent:
    """
    This class implements the gradient descent algorithm. Depending on the values passed, batch, mini-batch or
    stochastic gradient descent is implemented. If beta is 0, the algorithms are implemented without momentum, otherwise
    it is implemented with momentum.

    Attributes
    ----------
    epochs: int
        outputs the number of epochs set to run in the implementation of the gradient descent algorithm.
    lr: float
        outputs the learning rate used (to be used) in the algorithm.
    batch: int
        number of datasets to be used for each iteration in training the model. An output of None indicates batch
        gradient descent was used; 1 indicates stochastic gradient descent algorithm was implemented.
    beta: float
        outputs the value for beta (momentum) used.
    loss_function: function
        outputs function used as loss function in the implementation of the algorithm.
    grad_function: function
        outputs the gradient of the loss function.
    n_features: int
        the number of features or columns of the input or features matrix.
    n_samples: int
        the number of rows or samples in the input or features matrix.
    weights: np.ndarray
        outputs the weights of the model after training. Accessible after training
    loss: float
        outputs the loss after training. Accessible after training.
    grad: float
        outputs the value of gradient of the loss function after training. Accessible after training.

    Methods
    -------
    initialize_weights:
        initializes the weights before the training algorithm starts.
    initialize_momentum:
        initializes the momentum before the training algorithm starts.
    compute_loss:
        computes loss for an iteration or epoch.
    compute_grad:
        computes value of the gradient of the loss for an iteration or epoch.
    """

    def __init__(self, x, y, epochs=1000, lr=0.1, batch=None, beta=0.0, loss_function=None, grad_function=None):
        """
        initializes an object of class GradientDescent. This will be used in implementing gradient descent algorithm.
        The following params can be specified:
        :param x: np.ndarray, pd.Series, pd.DataFrame.
                The features matrix to be used in the training of the weights of the model.
        :param y: np.ndarray, pd.Series, pd.DataFrame.
                The target matrix to be used in the training of the weights of the model.
        :param epochs: int | Optional | default = 1000
                The number of epochs.
        :param lr: 'float' | Optional | default = 0.1
                The learning rate.
        :param batch: int | Optional | default = None
                If batch is None, batch Gradient descent is implemented. If it is 1, stochastic gradient descent
                is implemented, otherwise minibatch is implemented with each batch containing the specified data
                points. Default is None, i.e, batch gradient descent.
        :param beta: float | Optional | default = 0.0
                specifies the momentum to be used in the training. Default is 0.0. This implements the algorithm
                without momentum.
        :param loss_function: function
            The loss function to be used in the training. Must be specified.
        :param grad_function: function
            The gradient of the loss function to be used in training. Must be specified.
        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.beta = beta
        self.loss_function = loss_function
        self.grad_function = grad_function
        self.n_features = self.x.shape[1]
        self.n_samples = self.x.shape[0]
        self.tolerance = 0.001
        self.weights = None
        self.grad = None
        self.loss = None
        self.momentum = None
        self.losses = None

    def initialize_weights(self):
        """
        initializes the weights before the training algorithm starts. A 2-d array with one column and rows equal to
        the 'number of features + 1' is created. This sets the weight attribute to the weights created.
        :return: None.
        """
        self.weights = np.zeros((self.x.shape[1], 1))

    def initialize_momentum(self):
        """
        initializes momentum. This sets an attribute of the object, self.momentum.
        :return: None
        """
        self.momentum = 0

    def make_prediction(self):
        """
        makes prediction using the input array and the set of weights.
        :return: np.ndarray
        """
        return self.x @ self.weights

    def compute_loss(self, x, y):
        """
        computes loss given a features matrix, target matrix and loss function
        :param x: features matrix
        :param y: target matrix
        :return: float.
        """
        self.loss = self.loss_function(x, y, self.weights)
        return self.loss

    def compute_grad(self, x, y):
        """
        computes the gradient for an iteration using the given features matrix, target matrix and gradient function.
        :param x: features matrix
        :param y: target matrix
        :return: float
        """
        self.grad = self.grad_function(x, y, self.weights)

    def get_momentum(self):
        """
        updates momentum for an iteration or epoch.
        :return: None
        """
        self.momentum = self.momentum * self.beta + (1 - self.beta) * self.grad

    def update_weights(self, x_i, y_i):
        """
        updates weight for an iteration or epoch.
        :return: None
        """
        self.compute_grad(x_i, y_i)
        self.get_momentum()
        self.weights = self.weights - self.lr * self.momentum

    def fit(self):
        """Computes gradient descent.
        If batch = 1, stochastic gradient descent is carried out.
        If beta is zero, Gradient descent is done without momentum.
        :return None"""

        self.losses = []
        self.initialize_weights()
        avg_loss = float("inf")
        indexes = list(range(self.x.shape[0]))
        epoch = 0
        while epoch < self.epochs and avg_loss > self.tolerance:
            self.initialize_momentum()
            epoch_loss = 0
            shuffle(indexes)
            for i in range(0, self.n_samples, self.batch):
                x_i = self.x[indexes[i: i + self.batch]]
                y_i = self.y[indexes[i: i + self.batch]]
                epoch_loss += self.compute_loss(x_i, y_i)
                self.update_weights(x_i, y_i)

            avg_loss = epoch_loss / self.n_samples
            self.losses.append(avg_loss)
            epoch += 1

    def plot_losses(self):
        """
        plots average losses computed for each epoch in the gradient descent algorithm.
        :return:
        """
        plt.plot(self.losses)
        plt.show()
