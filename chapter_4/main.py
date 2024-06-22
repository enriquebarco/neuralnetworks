# What we have previously is a linear model, where there is no 'activation' or 'deactivation' of the neurons, because the output is a simple linear transformation of the input.

# In a neural network, we need to introduce non-linearity in the network, because without it, the neural network would be no more powerful than a linear regression model.

# Real world problems often times cannot be described by a linear model, so we need to introduce non-linearity in the network to enable it to model complex, real-world relationships and patterns.

# How do we turn our linear model into a non-linear model? With an activation function.

# The purpose of an activation function serves is to mimic a neuron “firing” or “not firing” based on input information.

# The activation function we will use is the rectified linear activation function (ReLU)

# ReLU is simply: if x is greater than 0 the neuron is firing (active), if x is less than 0 the neuron is not firing (deactivated).

# ReLU activation function in Python:
def relu(x):
    return max(0, x)

# When graphically representing the neural network, weights will impact the slope, while biases will impact the y-intercept.

import numpy as np
from nnfs.datasets import spiral_data
from chapter_3.main import Layer_Dense


class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        # calculate output values from input
        self.output = np.maximum(0, inputs)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function, takes output of previous layer
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print(activation1.output[:5])