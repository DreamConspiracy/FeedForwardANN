import math
import random

# Learning constant
alpha = 0.1

# The activation function of all neurons in the network.
def activationFunction(x):
	return 1 / (1 + math.exp(-x))

# The derivative of the above activation function. It is assumed that activationFunction(x) = y.
def activationDerivative(x, y):
	return y*(1 - y)

# An object for a single instance of a feed forward ANN. The constructor take a list of integers,
# s.t. size[i] is the number of neurons in layer i WITHOUT the bias neuron. edges is a three
# dimensional list s.t. edges[n][i][j] is either True of False depending on if there exists an edge
# between neutron i in layer n and neuron j in layer n + 1. weights is formatted the same as edges,
# however each element denotes the weight of the edge. If edges[n][i][j] is False, weights[n][i][j]
# MUST be 0. If weights is not provided it is randomly generated.
class FFANN:
	def __init__(self, size, edges, weights = None):
		self.size = size # Does NOT include bias neurons
		self.edges = edges

		if weights is not None:
			self.weights = weights
		else:
			self.weights = [[[0 if not edges[i][j][k] else random.random()*2 - 1
						for k in range(0, size[i+1])]
						for j in range(0, size[i])]
						for i in range(0, len(size) - 1)]

		for n in range(0, len(self.edges)):
			self.edges[n].append([True for i in range(0, self.size[n+1])])
			self.weights[n].append([1 for i in range(0, self.size[n+1])])

	# This function evaluates the output of the network for a given input. A list is returned with
	# the final output values.
	def evaluate(self, input):
		x = [[0 for j in range(0, i)] for i in self.size]
		y = [[0 for j in range(0, i)] for i in self.size]

		y[0] = input

		for i in range(1, len(self.size)):
			for j in range(0, self.size[i]):
				xValue = 0
				for k in range(0, self.size[i-1]):
					xValue += self.weights[i-1][k][j]*y[i-1][k]
				xValue += self.weights[i-1][-1][j] # Bias neuron

				x[i][j] = xValue
				y[i][j] = activationFunction(x[i][j])

		return y[-1]

	# This function trains the network on a test input and target output.
	# Assumes quadratic error function.
	def backPropagate(self, input, target):
		x = [[0 for j in range(0, i)] for i in self.size]
		y = [[0 for j in range(0, i)] for i in self.size]

		y[0] = input 

		for i in range(1, len(self.size)):
			for j in range(0, self.size[i]):
				xValue = 0
				for k in range(0, self.size[i-1]):
					xValue += self.weights[i-1][k][j]*y[i-1][k]
				xValue += self.weights[i-1][-1][j] # Bias neuron

				x[i][j] = xValue
				y[i][j] = activationFunction(x[i][j])

		delta = [[0 for j in range(0, i)] for i in self.size] # Partials of c with respect to pre-
															  # activation values.

		for i in range(0, self.size[-1]):
			delta[-1][i] = (y[-1][i] - target[i])*activationDerivative(x[-1][i], y[-1][i])

		for n in range(len(self.size) - 2, 0, -1):
			for i in range(0, self.size[n]):
				d = 0
				for j in range(0, self.size[n+1]):
					d += delta[n+1][j]*self.weights[n][i][j]*activationDerivative(x[n][i], y[n][i])

				delta[n][i] = d

		for n in range(0, len(self.size) - 1):
			for i in range(0, self.size[n]):
				for j in range(0, self.size[n+1]):
					if self.edges[n][i][j]:
						self.weights[n][i][j] -= alpha*delta[n+1][j]*y[n][i]

		for n in range(0, len(self.size) - 1):
			for i in range(0, self.size[n+1]):
				self.weights[n][-1][i] -= alpha*delta[n+1][i] # Change bias neurons

# The code below is a very basic demonstration of this implementation. The neural network learns to
# recognize the difference between numbers greater than and numbers less than 0.5. It takes a single
# input in the range [0, 1) and it's first output indicates whether the number is at least 0.5
# and the second output indicates whether it's less than 0.5. There are two error functions, one
# calculating error as a vector difference and the other as an error percent.
size = [1, 2, 2]
edges = [[[True, True]], [[True, True], [True, True]]]

ffann = FFANN(size, edges)

for i in range(0, 10):
	# Supervised learning cases
	for j in range(0, 1000):
		x = random.random()

		ffann.backPropagate([x], [1, 0] if x >= 0.5 else [0, 1])

	# Test cases
	e = 0
	for j in range(0, 100000):
		x = random.random()

		out = ffann.evaluate([x])
		target = [1, 0] if x >= 0.5 else [0, 1]

#		e += math.hypot(out[0] - target[0], out[1] - target[1]) # Vector difference error
		e += 1 if (out[1] > out[0]) ^ (target[1] > target[0]) else 0 # Error percentage error

	print(str(i) + ": " + str(e/100000.0))