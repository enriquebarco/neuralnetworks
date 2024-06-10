
# Coding Our First Neurons

# single neuron with 4 inputs
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# This neuron sums each input multiplied by that input’s weight, then adds the bias

output = (inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias)

print(f'single neuron: {output}')

# layers are just a group of neurons, connected by the same inputs but each having their own weights and biases

inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5
outputs = [
          # Neuron 1:
          inputs[0]*weights1[0] +
          inputs[1]*weights1[1] +
          inputs[2]*weights1[2] +
          inputs[3]*weights1[3] + bias1,
          # Neuron 2:
          inputs[0]*weights2[0] +
          inputs[1]*weights2[1] +
          inputs[2]*weights2[2] +
          inputs[3]*weights2[3] + bias2,
          # Neuron 3:
          inputs[0]*weights3[0] +
          inputs[1]*weights3[1] +
          inputs[2]*weights3[2] +
          inputs[3]*weights3[3] + bias3]
print(f'layer: {outputs}')

# A tensor object is an object that can be represented as an array

# A dot product of two vectors is a sum of products of consecutive vector elements. Both vectors must be of the same size (have an equal number of elements).

# vectors a and b
a = [1, 2, 3]
b = [2, 3, 4]

# To obtain the dot product:
dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(f'dot product: {dot_product}')

import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = np.dot(weights, inputs) + bias
print(f'numpy dot product: {output}')

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]


layer_outputs = np.dot(weights, inputs) + biases
print(f'numpy layer: {layer_outputs}')

# matrix product performs dot products on rows of the first matrix and the columns of second matrix

# transposition is when you make a matrix rows become the columns or vice versa (with numpy T)

# example of a numpy row vector
a = [1,2,3]
print(f'row vector: {np.array([a])}')

a = [1, 2, 3]
b = [2, 3, 4]

# example of transposing to a column vector
row_vector_a = np.array([a])
column_vector_b = np.array([b]).T

print(f'normal dot product: {np.dot(a, b)}')
print(f'matrix product: {np.dot(row_vector_a, column_vector_b)}')

# a batch is a set of inputs that are processed simultaneously. we want to process multiple inputs at once to speed up the process and prevent overfitting (no generalization, only memorization)
inputs = [[1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(f'batch: {layer_outputs}')