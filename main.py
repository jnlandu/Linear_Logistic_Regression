from linear_regression import LinearRegression
from logistic_regression import LogisticRegression

from sklearn.datasets import make_classification, make_moons
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


def train_test_split(X, y):
    np.random.seed(0)
    train_size = 0.8
    n = int(len(X) * train_size)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_idx = indices[: n]
    test_idx = indices[n:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_test, y_test


# Generate data for linear regression implementation
def generate_data(n=1000, seed=0):
    np.random.seed(seed)
    x = np.linspace(-5.0, 5.0, n).reshape(-1, 1)
    y = (29 * x + 30 * np.random.rand(n, 1)).squeeze().reshape((-1, 1))
    x = np.hstack((np.ones_like(x), x))
    return x, y


# Linear Regression Test 1
x, y = generate_data()
lin_x_train, lin_y_train, lin_x_test, lin_y_test = train_test_split(x, y)
print(f"x_train:{lin_x_train.shape}, y_train: {lin_y_train.shape}, x_test: {lin_x_test.shape}, "
      f"y_test: {lin_y_test.shape}")

lin_reg = LinearRegression(lin_x_train, lin_y_train)
lin_reg.fit()
print('Test 1 model weights', lin_reg.weights)
print('Test 1 training r-square', lin_reg.r_square())
print('Test 1 test r-square', lin_reg.r_square(lin_x_test, lin_y_test))
plt.plot(lin_reg.losses)
plt.title('Linear Regression Test 1 losses.')
plt.show()
lin_reg.plot_function(title='Test 1 Observed values vs Predicted values')

# Linear Regression Test 2
x, y = generate_data(n=500, seed=2)
lin_x_train, lin_y_train, lin_x_test, lin_y_test = train_test_split(x, y)
print(f"x_train:{lin_x_train.shape}, y_train: {lin_y_train.shape}, x_test: {lin_x_test.shape}, "
      f"y_test: {lin_y_test.shape}")

lin_reg = LinearRegression(lin_x_train, lin_y_train)
lin_reg.fit()
print('Test 2 model weights', lin_reg.weights)
print('Test 2 training r-square', lin_reg.r_square())
print('Test 2 test r-square', lin_reg.r_square(lin_x_test, lin_y_test))
plt.plot(lin_reg.losses)
plt.title('Linear Regression Test 2 losses.')
plt.show()
lin_reg.plot_function(title="Test 2 Observed values vs Predicted values")


# Test 1
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1,
                           n_clusters_per_class=1)
log_x_train, log_y_train, log_x_test, log_y_test = train_test_split(X, y)
log_reg = LogisticRegression(log_x_train, log_y_train, epochs=1000, lr=0.1)
log_reg.fit()
print('Test 1 weights', log_reg.weights)
print('Test 1 Training accuracy', log_reg.accuracy())
print('Test 1 Test accuracy', log_reg.accuracy(log_x_test, log_y_test))
plt.plot(log_reg.losses)
plt.title('Logistic Regression Test 1 losses.')
plt.show()
log_reg.plot_decision_boundary(log_x_train, log_y_train, log_reg.weights, log_reg.weights[0],
                               title='Test 1 Training data')
log_reg.plot_decision_boundary(log_x_test, log_y_test, log_reg.weights, log_reg.weights[0],
                               title='Test 1 Test data')

# Test 2
X, y = make_moons(n_samples=100, noise=0.24)
log_x_train, log_y_train, log_x_test, log_y_test = train_test_split(X, y)
log_reg = LogisticRegression(log_x_train, log_y_train, epochs=1000, lr=0.1)
log_reg.fit()
print('Test 2 weights', log_reg.weights)
print('Test 2 Training accuracy', log_reg.accuracy())
print('Test 2 Test accuracy', log_reg.accuracy(log_x_test, log_y_test))
plt.plot(log_reg.losses)
plt.title('Logistic Regression Test 2 losses.')
plt.show()
log_reg.plot_decision_boundary(log_x_train, log_y_train, log_reg.weights, log_reg.weights[0],
                               title='Test 2 Training data')
log_reg.plot_decision_boundary(log_x_test, log_y_test, log_reg.weights, log_reg.weights[0], title='Test 2 Test data')
