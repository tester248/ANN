import numpy as np
import matplotlib.pyplot as plt

def linear(x):
	return x
def binary_step(x):
	return np.where(x >= 0, 1, 0)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def tanh(x):
	return np.tanh(x)
def relu(x):
	return np.maximum(0, x)

def main():
	x = np.linspace(-10, 10, 400)

	# Binary Step
	plt.figure()
	plt.plot(x, binary_step(x))
	plt.title("Binary Step Activation Function")
	plt.xlabel("Input")
	plt.ylabel("Output")
	plt.grid(True)
	plt.show()

	# Sigmoid
	plt.figure()
	plt.plot(x, sigmoid(x))
	plt.title("Sigmoid Activation Function")
	plt.xlabel("Input")
	plt.ylabel("Output")
	plt.grid(True)
	plt.show()

	# Tanh
	plt.figure()
	plt.plot(x, tanh(x))
	plt.title("Tanh Activation Function")
	plt.xlabel("Input")
	plt.ylabel("Output")
	plt.grid(True)
	plt.show()

	# ReLU
	plt.figure()
	plt.plot(x, relu(x))
	plt.title("ReLU Activation Function")
	plt.xlabel("Input")
	plt.ylabel("Output")
	plt.grid(True)
	plt.show()

	# Linear
	plt.figure()
	plt.plot(x, linear(x))
	plt.title("Linear Activation Function")
	plt.xlabel("Input")
	plt.ylabel("Output")
	plt.grid(True)
	plt.show()

if __name__ == "__main__":
	main()

