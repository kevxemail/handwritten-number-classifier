import os
import pickle
import numpy as np

"""
When inputted an array with numbers representing the size of each layer, generate that network with random numbers as the weights and biases initially
"""
def generate_network(network):
    res = len(network)
    weights = [None] # Placeholder
    biases = [None]
    for i in range(1, len(network)):
        rows, cols = network[i], network[i-1] # Since we dot product weights * input, the # rows is the number of layers in the current index, and the number of columns is the number of rows from previous layer's output
        w = 3.0 * np.random.rand(rows, cols) - 1.5 # Generates numpy matrix with values from [-1.5, 1.5). Normally does [0, 1), but multiply by 3 and subtract by 1.5 gives max 1.5 and minimum -1.5
        weights.append(w)

        b = 3.0 * np.random.rand(rows, 1) - 1.5  # Biases will always be a column vector with the current layer's size
        biases.append(b)

    return weights, biases

"""
Sigmoid activation function used in perceptron outputs
"""
def sigmoid(num):
    return 1 / (1 + np.exp(-num))
activation = np.vectorize(sigmoid)

"""
Derivative of sigmoid function (which is just A(X) * (1-A(X)))
This is used for backpropagation when we are calculating partial derivatives for gradient descent
The idea is we are trying to MINIMIZE the relationship between weights/biases and calculated error
"""
def d_sigmoid(num): # Derivative of sigmoid activation function used to calculate deltas from the video
    return sigmoid(num) * (1 - sigmoid(num))
d_activation = np.vectorize(d_sigmoid)

"""
Backpropagates along the neural network, adjusting the weights and biases
If we wanted to do gradient ascent, I'd have to multiply by negative 1 for my deltas
This is because when I derived it by hand it naturally was positive for gradient descent
"""
def backpropagation(weights, biases, training_data, test_data, learning_rate, epochs):
    # Create directories if they don't exist
    os.makedirs("weights", exist_ok=True)
    os.makedirs("biases", exist_ok=True)

    for epoch in range(epochs):
        for x, y in training_data:
            a = list([x]) # Add the input as the first "output" we've had
            dotList = list([None]) # Dummy spot, use this to keep track of the outputs matrices of each layer BEFORE activation function
            layer_count = len(weights)-1 # How many layers, -1 becase of the placeholder None at the start

            # Pass the input through the neural network
            for layer in range(1, layer_count+1): # Avoid the dummy node for both

                dotL = np.dot(weights[layer], a[layer-1]) + biases[layer] # [previous layer's output matrix] @ [weights of layer] + [biases] 
                aL = activation(dotL) # vector after passing it through the activation function
                a.append(aL) # Add the information to their respective lists
                dotList.append(dotL)

            # Calculate the delta last
            deltaN = d_activation(dotList[-1]) * (y-a[-1]) # Derivative_Sigmoid(last dot matrix) pairwise multiply by the difference between expected output and our output

            delta_list = [None] * (layer_count+1) # Initialize this completely to begin with to make our lives easier, since the first node will be a dummy node and adding stuff and getting the right index is a pain
            delta_list[-1] = deltaN
            # Go through the layers and calculate their respective delta values
            # Layers N-1 to 0
            for layer in range(layer_count-1, 0, -1):
                delta = d_activation(dotList[layer]) * (weights[layer+1].transpose() @ delta_list[layer+1]) # Use the previous delta value to calculate the current one
                delta_list[layer] = delta # Store it

            for layer in range(1, layer_count+1): # Apply the delta values to the weights and biases before starting the next training data
                biases[layer] = biases[layer] + (learning_rate * delta_list[layer]) # b = b + lambda * delta(layer)
                weights[layer] = weights[layer] + learning_rate * np.dot(delta_list[layer], a[layer-1].transpose()) # w = w + lamba * (delta(layer) x a[layer-1].transpose())

        incorrect = test_accuracy(test_data, weights, biases)
        learning_rate = incorrect/len(test_data) # Adjust the learning rate based on how many we are still getting wrong

        if learning_rate < 0.0001: # Don't want it too low
            learning_rate = 0.0005

        if epoch % 5 == 0: # Every 5 epochs, pickle the updated weights and biases and save it
            weight_name = os.path.join("weights", "weights_epoch" + str(epoch) + ".pickle")
            biases_name = os.path.join("biases", "bias_epoch" + str(epoch) + ".pickle")
            with open(weight_name, 'wb') as f:
                pickle.dump(weights, f)
            with open(biases_name, 'wb') as f:
                pickle.dump(biases, f)
        print("Epoch", epoch+1, "complete", "Incorrect:", incorrect)

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

"""
Tests the accuracy of the neural network based on the test_data and by using our current weights and biases to make a prediction
"""
def test_accuracy(test_data, weights, biases):
    incorrect = 0
    for i in range(len(test_data)):
        result = test_data[i][1]
        # print(result)
        res = predict(weights, biases, test_data[i][0])
        if result != process_res(res):
            incorrect += 1
    return incorrect



with open('train_data.pickle', 'rb') as f:
    train_data = pickle.load(f)

with open('test_data.pickle', 'rb') as f:
    test_data = pickle.load(f)

initial_weights, initial_biases = generate_network([784,300,100,10])

# print(test_accuracy(test_data, initial_weights, initial_biases))
# for i in range(1, len(initial_weights)):
#     print(initial_weights[i].shape)

# for i in range(1, len(initial_biases)):
#     print(initial_biases[i].shape)
backpropagation(initial_weights, initial_biases, train_data, test_data, 0.01, 100)
