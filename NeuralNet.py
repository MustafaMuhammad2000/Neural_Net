import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


class neural_network:
    # Layer nodes describes the shape of the neural network (how many nodes in each layer)
    # lr is the learning rate, used to scale the amount the weights/biases shift each time the machine is trained
    def __init__(self, layerNodes, lr):
        self.layerNums = len(layerNodes)
        self.learningRate = lr

        # Weights for each layer
        # The number of rows is the number of hidden units of that layer (neurons)
        # The number of columns is the number of features/rows of the previous layer (neurons of last layer)
        self.weightLayers = []
        # Biases for each layer
        # The number of rows is the number of hidden units of that layer (neurons)
        # Only one column
        self.biasLayers = []
        # Populating layers (not input layer) with weights/biases
        for i in range(1, self.layerNums):
            weight = np.random.rand(layerNodes[i], layerNodes[i - 1])*2-1
            bias = np.random.rand(layerNodes[i], 1)*2-1
            self.weightLayers.append(weight)
            self.biasLayers.append(bias)

    def train(self, input_data, target_data):
        # Transposing input and target to convert to 1 column array (aka a layer)
        input_layer = np.reshape(input_data, (len(input_data), 1))
        target_layer = np.reshape(target_data, (len(target_data), 1))

        # Feed Forward
        # Creating activation data for layers
        # activation_data is data yet to be put through the activation function
        # Whilst sigmoid_data is post activation function data
        activation_data = []
        sigmoid_data = []

        # looping forward through the layers
        for i in range(self.layerNums - 1):
            # Input layer (create first activation layer)
            if i == 0:
                temp_a = self.weightLayers[i].dot(input_layer) + self.biasLayers[i]
                temp_s = sigmoid(temp_a)
            # Hidden+output layers
            else:
                temp_a = self.weightLayers[i].dot(sigmoid_data[i - 1]) + self.biasLayers[i]
                temp_s = sigmoid(temp_a)

            activation_data.append(temp_a)
            sigmoid_data.append(temp_s)
        # Back propagation
        # Creating an array of errors for back propagation (array will be reversed due to moving backwards)
        errors = []
        # delta = array of (error)*(derivative of activation function)*(learning rate)
        # Use delta for weight and bias adjustments, since delta involves the error gradient
        deltas = []

        for i in reversed(range(self.layerNums - 1)):
            # As error array consists only of error signals, length is self.layerNums-2
            # as well as -1 due to indexing in arrays
            error_signal_layer = self.layerNums-2-1-i

            # (Hidden layers) error = weight from previous neuron * error signal of current neuron
            if i != self.layerNums-2:
                temp_error = self.weightLayers[i+1].T.dot(errors[error_signal_layer])
                temp_delta = temp_error*sigmoid_deriv(activation_data[i])*self.learningRate

            # (Output layer) error = expected-output
            else:
                temp_error = target_layer-sigmoid_data[i]
                temp_delta = temp_error*sigmoid_deriv(activation_data[i])*self.learningRate

            errors.append(temp_error)
            deltas.append(temp_delta)
            # Updating weight and biases
            bias_adjustment = deltas[len(deltas)-1]
            # Weight adjustment for hidden layers
            if i != 0:
                # (i-1) index accessed as don't want to use output layers activation data
                # since trying to adjust weights/biases to effect output layers activation
                weight_adjustment = deltas[len(deltas)-1].dot(sigmoid_data[i-1].T)
            # Weight adjustment for input layer
            else:
                weight_adjustment = deltas[len(deltas)-1].dot(input_layer.T)
            self.biasLayers[i] += bias_adjustment
            self.weightLayers[i] += weight_adjustment

    # A feed forward once function, that returns the output layer (the result)
    # Already explained feed forward within train function, thus no comments for test
    def test(self, input_data):
        input_layer = np.reshape(input_data, (len(input_data), 1))

        activation_data = []
        sigmoid_data = []

        for i in range(self.layerNums - 1):
            if i == 0:
                temp_a = self.weightLayers[i].dot(input_layer) + self.biasLayers[i]
                temp_s = sigmoid(temp_a)
            else:
                temp_a = self.weightLayers[i].dot(sigmoid_data[i - 1]) + self.biasLayers[i]
                temp_s = sigmoid(temp_a)

            activation_data.append(temp_a)
            sigmoid_data.append(temp_s)
        # Returns the output layer (the result)
        return sigmoid_data[len(sigmoid_data)-1]







