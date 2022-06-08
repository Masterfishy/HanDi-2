import numpy as np
from tqdm import tqdm

"""
This file contains an implementation of a multi-class multi-layer perceptron
"""

def sigmoid(x):
    """
    Sigmoid activation function that returns the activation value and a matrix
    of the derivatives of the element-wise sigmoid operation.

    :param x: ndarray of inputs to be sent to the sigmoid function
    :type x: ndarray
    :return: tuple of (1) activation outputs and (2) the gradients of the
                sigmoid function, both ndarrays of the same shape as x
    :rtype: tuple
    """
    y = 1.0 / (1.0 + np.exp(-x))
    grad = np.multiply(y, (1 - y))

    return y, grad


def nll(score, labels):
    """
    Loss function that computes the negative log-likelihood of the labels by
    interpreting score as probabilities.

    :param score: ndarray of positive class probabilities
    :type score: ndarray
    :param labels: length-n array of sample labels
    :type labels: array
    :return: ndarray of the gradients of the NLL with respect to the scores 
                with the same shape as the score input
    :rtype: ndarray
    """
    x, y = score.shape

    # Gradients
    grad = np.zeros((x,y))
    for i in range(x):
        grad[i] = np.divide(score[i] - (labels == i), score[i] * (1 - score[i]))

    return grad


def mlp_predict(data, model):
    """
    Predict binary-class labels for a batch of test points.

    :param data: ndarray where each column is an observation
    :type data: ndarray
    :param model: learned model from mlp_train
    :type model: dict
    :return: array of predicted class labels
    :rtype: array
    """
    n = data.shape[1]
    weights = model['weights']
    activation_function = model['activation_function']

    num_layers = len(weights)

    activation_derivatives = []
    activations = [np.vstack((data, np.ones((1, n))))]

    # Pass data through activation layers
    for layer in range(num_layers):
        new_activations, activation_derivative = activation_function(weights[layer].dot(activations[layer]))
        activations.append(new_activations)
        activation_derivatives.append(activation_derivative)

    # Final predictions in the final layer of the activations
    scores = activations[-1]
    labels = np.argmax(scores, axis=0)

    return labels, scores, activations, activation_derivatives


def mlp_objective(model, data, labels, loss_function):
    """
    Compute the gradients of a data pass given the model weights.

    :param model: dict containing the current model weights
    :type model: dict
    :param data: ndarray where each column is an observation
    :type data: ndarray:
    :param labels: length-n array of ground-truth labels
    :type labels: array
    :param loss_function: a function to evaluate the loss and gradient
    :type loss_function: function
    :return: ndarray of gradients for each weight matrix
    :rtype: ndarray
    """
    n = labels.size

    # Forward propagation
    weights = model['weights']
    num_layers = len(weights)
    _, scores, activations, activation_derivatives = mlp_predict(data, model)

    # Back propagation
    
    layer_delta = [None] * num_layers
    layer_gradients = [None] * num_layers

    gradient = loss_function(scores, labels)

    layer_delta[-1] = gradient

    # back propagate error to previous layers
    for i in reversed(range(num_layers - 1)):
        layer_delta[i] = weights[i + 1].transpose().dot(layer_delta[i + 1] * activation_derivatives[i + 1])

    # compute gradient for each layer
    for i in range(num_layers):
        layer_gradients[i] = (layer_delta[i] * activation_derivatives[i]).dot(activations[i].transpose())

    return layer_gradients


def mlp_train(data, labels, params, model=None):
    """
    Train a multi-layer perceptron with gradient descent and back-propagation.

    :param data: ndarray where each column is an observation
    :type data: ndarray
    :param labels: length-n array containing ground-truth labels
    :type labels: array
    :param params: dictionary containing hyper parameters
    :type params: dict
    :return: learned model containing 'weights' list
    :rtype: dict
    """
    input_dim = data.shape[0]

    # If no model is given, initialize new model based on params
    if not model:
        model = dict()

        num_hidden_units = params['num_hidden_units']
        model['num_hidden_units'] = num_hidden_units
        model['weights'] = list()

        # create input layer
        model['weights'].append(.1 * np.random.randn(num_hidden_units[0], input_dim + 1))

        # create hidden layers
        for layer in range(1, len(num_hidden_units)):
            model['weights'].append(.1 * np.random.randn(num_hidden_units[layer], num_hidden_units[layer - 1]))

        # create output layer
        model['weights'].append(.1 * np.random.randn(len(set(labels)), num_hidden_units[-1])) # maybe consider changing how we know how many class types we have

        model['activation_function'] = params['activation_function']
        model['lambda'] = params['lambda']

    loss_function = params['loss_function']
    num_layers = len(model['weights'])
    lam = params['lambda']
    learning_rate = params['learning_rate']

    # Training
    for i in range(params['max_iter']):
        grad = mlp_objective(model, data, labels, loss_function)

        total_change = 0

        for j in range(num_layers):
            change = -learning_rate * (grad[j] + lam * model['weights'][j])
            total_change += np.sum(np.abs(change))

            # clip change to [-0.1, 0.1] to avoid numerical overflow
            change = np.clip(change, -0.1, 0.1)

            model['weights'][j] += change

        if total_change < 1e-8:
            print("Exiting because total change was %e, a sign that we have reached a local minimum." % total_change)
            break

    return model
