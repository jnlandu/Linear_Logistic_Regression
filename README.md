# Implemenation of Linear and Logistic Regression with Gradient Descent in Python from Scratch

## Introduction

This project provides Python implementations of linear regression, logistic regression, and 
gradient descent algorithms from scratch. These classes can be used for supervised learning tasks such as
regression and classification.

## Overview

The project includes the following components:
- `LinearRegression`: A class for performing linear regression using gradient descent.
- `LogisticRegression`: A class for performing logistic regression using gradient descent.
- `GradientDescent`: A class for implementing gradient descent optimization algorithms, including batch,
mini-batch, and stochastic gradient descent with or without momentum.

Each component is implemented in a separate Python file for modularity and ease of maintenance.

Installation and Requirements
To use this project, clone the repository and install the required dependencies using pip:

```Copy code
git clone https://github.com/Ignatiusboadi/Linear_Logistic_Regression.git
cd Linear_Logistic_Regression
pip install -r requirements.txt
```

This project requires the following dependencies:
- NumPy
- Matplotlib
- Scikit-learn (for generating synthetic datasets)


## Example Usage
To instantiate the LinearRegression class with a specific gradient descent optimization method,
use the following parameters:

- features: The feature matrix for training the model.
- target: The target vector for training the model.
- epochs (optional): default = 100 | Number of epochs to be run 
- lr (optional): default = 0.1 | learning rate
- batch_size (optional): default = None | Specifies the batch size for mini-batch gradient descent. 
By default, batch gradient descent is implemented. Passing 1 leads to implementation of Stochastic 
gradient descent, whilst passing a number other than 1 leads to implementation of mini-batch gradient 
descent with figure passed as batch size . 
- momentum (optional): default = 0.0 | Specifies the momentum parameter for gradient descent with momentum.

```Copy code
model = LinearRegression(x_train, y_train, epochs=100, lr=0.1, batch=1, beta=0.9)
model.fit()
print('model weights', model.weights)
print('Training r-square', model.r_square())
print('Test r-square', model.r_square(x_test, y_test))
plt.plot(model.losses)
plt.title('Linear Regression Test 1 losses.')
plt.show()
model.plot_function(title='Linear regression Test 1 Actual and Predicted')
```

This can also be done for implementation of logistic regression on a dataset. See the main.py file for more
examples.

## Contributing
Contributions to this project are welcome! If you'd like to contribute, please fork the repository,
make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.