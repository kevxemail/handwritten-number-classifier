import numpy as np
import pickle
"""
Processes the mnist_train.csv dataset and returns an array where each index holds [784x1 numpy array representing the pixel inputs, 10x1 output array representing which number it is. It will all be 0 except index down the column which the actual number should be.
"""
def process_train_data():
    data = open("mnist_train.csv").read()
    data = [line.strip() for line in data.splitlines()] # Split into lines and remove leading/trailing spaces & \n

    train_data = list()
    for line in data:
        curr = line.split(",") # Split into array of characters
        for i in range(len(curr)):
            curr[i] = int(curr[i]) # Cast it to an integer since we read it from a file

        # We divide by 255 below because sigmoid operates on values 0 to 1
        input = np.array(curr[1:]) / 255 # Turn the input into a numpy array
        input = np.reshape(input, (-1, 1)) # Need to turn it from 1x784 to 784x1, -1 used by numpy to automatically infer the number of rows based on size of original, 1 represents I only want 1 column

        # The first position in curr is the label
        output = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        output[curr[0], 0] = 1 # Modify the correct row
        train_data.append([input, output])
    return train_data

"""
Processes the mnist_test.csv dataset and returns an array where each index holds [784x1 numpy array representing the pixel inputs, 10x1 output array representing which number it is. It will all be 0 except index down the column which the actual number should be.
"""
def process_test_data():
    data = open("mnist_test.csv").read()
    data = [line.strip() for line in data.splitlines()] # Split into lines and remove leading/trailing spaces & \n

    test_data = list()
    for line in data:
        curr = line.split(",") # Split into array of characters
        for i in range(len(curr)):
            curr[i] = int(curr[i]) # Cast it to an integer since we read it from a file
        input = np.array(curr[1:]) / 255 # Turn the input into a numpy array
        input = np.reshape(input, (-1, 1)) # Need to turn it from 1x784 to 784x1, -1 used by numpy to automatically infer the number of rows based on size of original, 1 represents I only want 1 column

        # For output just get the actual number
        test_data.append([input, curr[0]])
    return test_data

train_data = process_train_data()
test_data = process_test_data()

with open('train_data.pickle', 'wb') as f:
    pickle.dump(train_data, f)

with open('test_data.pickle', 'wb') as f:
    pickle.dump(test_data, f)
    