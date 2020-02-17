import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Load training and testing data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize the dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_sex_x_flatten / 255.


# Computes sigmoid
def sigmoid(z):
    """
    In -- z: A scalar or numpy array of any size.
    
    Out -- s:  Sigmoid of the input array.
    """
    s = 1. / (1 + np.exp(-z))
    return s


# Creates a vector of zeros and sets b = 0
def initialize_with_zeros(dim):
    '''
    In -- dim:  Size of the w vector we want.
    
    Out -- w:   Vector of zeros with shape (dim, 1)
    Out -- b:   Scalar of 0.
    '''
    w = np.zeros(shape(dim, 1), dtype=np.float32)
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


# Implements the cost function and its gradient for propagation
def propogate(w, b, X, Y):
    '''
    In -- w: Weights array that has dimensions (12288, 1)
    In -- b: Bias scalar
    In -- X: Raw data array with dimensions (12288, n), where n is the number of examples
    In -- Y: True label vector (0 = not a cat, 1 = cat)
    
    Out -- cost: Negative log-likelihood cost for logistic regression
    Out -- dw: Gradient of the loss with respect to w
    Out -- db: Gradient of the loss with respect to b
    '''
    m = X.shape[1]
    
    # FORWARD PROPAGATION
    # Compute activation
    A = sigmoid(np.dot(w.T, X) + b)
    
    # Compute cost
    cost = (-1. / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)), axis=1)

    # BACKWARD PROPAGATION
    dw = (1. / m) * np.dot(X, ((A - Y).T))
    db = (1. / m) * np.sum(A - Y, axis=1)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw, "db": db}
    return grads, cost


# Optimizes w & b by running a gradient descent algorithm
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    '''
    In -- w: Weights array that has dimensions (12288, 1)
    In -- b: Bias scalar
    In -- X: Raw data array with dimensions (12288, n), where n is the number of examples
    In -- Y: True label vector (0 = not a cat, 1 = cat)
    In -- num_iterations: Number of iterations of the optimization loop
    In -- learning_rate: Learning rate of the gradient descent update rule
    In -- print_cost: If "True", print the cost every 100 steps
    
    Out -- params: Dictionary containing weights (w) and biases (b)
    Out -- grads: Dictionary containing the gradients of the weights and bias with respect to the cost function
    Out -- costs: List of all the costs computed during the optimization
    '''
    costs = []

    for i in range(num_iterations):
        
        # Cost and gradient calculation
        grads, cost = propagate(w=w, b=b, X=X, Y=Y)
        
        # Retrieve derivatives
        dw = grads['dw']
        db = grads['db']

        # Update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training samples
        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


# Using learned logistic regression parameters, predict whether the label is 0 or 1
def predict(w, b, X):
    '''
    In -- w: Weights (array)
    In -- b: Bias (scalar)
    In -- X: Data (array)

    Out -- Y_prediction: A numpy array containing all predictions for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    # Convert probabilities into actual predictions
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


# Putting it all together
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    '''
    In -- X_train: Training set array with dimensions (12288, m_train)
    In -- Y_train: Training labels for X_train (0 = not cat, 1 = cat); dimensions: (1, m_train)
    In -- X_test: Testing set array with dimensions (12288, m_test)
    In -- Y_test: Testing labels for X_test (0 = not cat, 1 = cat); dimensions: (1, m_test)
    In -- num_iterations: Hyperparameter representing the number of iterations to optimize the parameters
    In -- learning_rate: Hyperparameter representing the learning rate used in the update rule of optimize()
    In -- print_cost: If "True", print the cost every 100 steps

    Out -- d: Dictionary containing information about the model.
    '''
    # Initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d


# Run it!
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
print(d)

# Graph it!
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
