import numpy as np
# Adding Layers
inputs = [
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
        ]
weights = [
             [0.2, 0.8, -0.5, 1.0],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]
             ]
biases = [2.0, 3.0, 0.5]

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

print(f'layer2_outputs: {layer2_outputs}')