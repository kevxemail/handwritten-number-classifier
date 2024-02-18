import pickle
import sys

import numpy as np

"""
Sigmoid activation function
"""
def sigmoid(num):
    return 1 / (1 + np.exp(-num))
activation = np.vectorize(sigmoid)

"""
Goes through the neural network and makes a prediction
"""
def predict(weights, biases, x):
    a = list() # let a be a Python list of numpy matrices that will hold the output at each layer
    a.append(x)
    n = len(weights) - 1 # Number of perceptron layers in the neural network
    for i in range(1, n+1): # Go from layers 1 to n
        a.append(activation(weights[i] @ a[i-1] + biases[i])) # Find dot product, resulting in matrix with 1 column. Then add biases to each, then run through activation function
        # *****Each row in the weight matrix represents a perceptron in that layer. so matrix multi. on that row gives you the output of that individual perceptron******
    return a[n]
"""
Returns the number that the 10x1 matrix outputted by the neural network represents
"""
def process_res(res):
    # Find the index with the maximum value which is what our prediction will be
    max_row_index = np.argmax(res, axis=0)
    return max_row_index


def test_accuracy(test_data, weights, biases):
    incorrect = 0
    for i in range(len(test_data)):
        result = test_data[i][1]
        # print(result)
        res = predict(weights, biases, test_data[i][0])
        if result != process_res(res):
            incorrect += 1
    return incorrect


with open('test_data.pickle', 'rb') as f: # Load the pickled ,training and test data sets
    test_data = pickle.load(f)

weights = "weights/weights_epoch" + sys.argv[1] + ".pickle"
biases = "biases/bias_epoch" + sys.argv[1] + ".pickle"

# Load the weights and biases from the pickle file

with open(weights, 'rb') as f: 
    weights = pickle.load(f)

with open(biases, 'rb') as f:
    biases = pickle.load(f)

print(test_accuracy(test_data, weights, biases))