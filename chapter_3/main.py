import numpy as np
import nnfs
from nnfs.datasets import spiral_data


# Adding Layers
inputs = [[1, 2, 3, 2.5],
            [2., 5., -1., 2],
            [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# the hidden layer is when the process repeats with the new inputs being the outputs of the previous layer. In this case, there are three neurons in the input layer therefore there are three outputs. These outputs are the inputs to the hidden layer which is why there is a shape of 3x3 for the weights matrix. 

weights2 = [
                [0.1, -0.14, 0.5],
                [-0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]
                ]

biases2 = [-1.0, 2.0, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# the output of the first layer is the input of the seocond layer

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(f'layer2_outputs: {layer2_outputs}')


# training data

# sets the random seed to 0 (by the default), creates a float32 dtype default, and overrides the original dot product from NumPy (therefore wont work with code above)
nnfs.init()
class Layer_Dense:

    # layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases, we will be creating randomly although this might be different if you use a pre trained model

        #  Initializing weights to be (inputs, neurons) to prevent transposing every forward pass
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # the biases are initialized to zero
        self.biases = np.zeros((1, n_neurons))

        # no outputs yet
        self.output = None

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def __repr__(self):
        # used to print the object in a more readable way
        return (f"Layer_Dense(weights={self.weights}, biases={self.biases}, output={self.output})")


# create a dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values (or neurons)
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

print(dense1)