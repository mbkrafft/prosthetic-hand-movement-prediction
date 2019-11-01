from random import random
import numpy as np
import math
import matplotlib.pyplot as plt


class mlp:
    def __init__(self, inputs, targets, nhidden):
        # activation constant
        self.beta = 1

        # learning rate
        self.eta = 0.1

        self.momentum = 0.0

        self.network = self.initializeNetwork(
            inputs[0].size, nhidden, targets[0].size)

    def initializeNetwork(self, numberOfInputs, numberOfHidden,
                          numberOfOutputs):
        network = []

        # create layer of hidden neurons with random weights
        # (+ 1 because bias node)
        hiddenLayer = [{'weights': [random() for _ in
                                    range(numberOfInputs + 1)]}
                       for _ in range(numberOfHidden)]

        network.append(hiddenLayer)

        # create layer of output neurons with random weights
        # (+ 1 because bias node)
        outputLayer = [{'weights': [random() for _ in
                                    range(numberOfHidden + 1)]}
                       for _ in range(numberOfOutputs)]
        network.append(outputLayer)

        return network

    # determines if we should keep training or stop
    def earlystopping(self, inputs, targets, valid, validtargets):
        # start training
        self.train(inputs, targets)

        # error after first run. +1 to make first comparison ok
        error = self.sumOfSquaresError(valid, validtargets) + 1

        plotValidError = []
        plotTrainError = []

        stop = False

        # will show errors of training and validaton set throughout training
        print("Valid: \t Training:")

        while stop == False:
            newError = self.sumOfSquaresError(valid, validtargets)
            plotValidError.append(newError)

            errorTrainingSet = self.sumOfSquaresError(inputs, targets)
            plotTrainError.append(errorTrainingSet)

            # print current errors
            print(f'{round(newError, 2)} \t {round(errorTrainingSet, 2)}')

            # stop when validation set error stops improving
            # - to avoid overfitting
            if error - newError < 0.01:
                idx = [x * 10 for x in range(len(plotTrainError))]
                plt.plot(idx, plotValidError, label='Validation set error')
                plt.plot(idx, plotTrainError, label='Training set error')

                plt.ylabel('sum-of-squares error')
                plt.xlabel('number of epochs')

                plt.legend()
                plt.show()

                stop = True
            else:
                error = newError
                self.train(inputs, targets)

    def sumOfSquaresError(self, inputs, targets):
        error = 0

        for idx, row in enumerate(inputs):
            output = self.forward(row)
            target = targets[idx]

            error += sum(map(self.differenceSquared, output, target))

        return error * 0.5

    def differenceSquared(self, a, b):
        return (a - b)**2

    # runs the network through training for a set amount of epochs
    def train(self, inputs, targets, iterations=10):
        for _ in range(iterations):
            for idx, row in enumerate(inputs):
                # apply input vectors and calculate all activation functions, a and u
                output = self.forward(row)

                # evaluate deltas for all output units and backpropagate error
                # through the network
                self.backpropagate(output, targets[idx])

                # # update weights
                self.updateWeights(inputs[idx])

    # calulates activations and out put of all neurons in the network
    def forward(self, inputs):
        for _, layer in enumerate(self.network):
            # add bias node with value -1 as the last input
            inputs = np.append(inputs, -1)
            newInputs = []

            for neuron in layer:
                # get sum of all (input*weight)
                activation = self.activate(inputs, neuron['weights'])

                # output of the neuron is the result of the activation
                neuron['output'] = self.activationFunction(activation)

                """ NOT SURE IF I SHOULD USE THIS OR NOT - change _ to idx if used
                # linear if output layer, sigmoid if hidden layer
                if (idx == len(self.network)):
                    neuron['output'] = self.outputActivationFunction(
                        activation)
                else:
                   neuron['output'] = self.activationFunction(activation)
                """

                # add output into what will become the new inputs to
                # the next layer
                newInputs.append(neuron['output'])

            # let all the outputs of the last layer be
            # the inputs to the new layer
            inputs = newInputs

        # return the outputs of the last layer
        return inputs

    # calculates total input*weight signal of a neuron
    def activate(self, inputs, weights):
        activation = 0

        for i in range(len(weights)):
            activation += (weights[i] * inputs[i])

        return activation

    # determines how much a neuron should fire based on activation
    def activationFunction(self, activation):
        return 1 / (1 + math.exp(- self.beta * activation))

    # slope of the activation function * activation constant
    def activationFunctionDerivative(self, output):
        return self.beta * output * (1 - output)

    def outputActivationFunction(self, activation):
        return activation

    # runs output errors back through the network, updating deltas
    def backpropagate(self, outputs, targets):
        # going backwards through the layers
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []

            # hidden layer(s)
            if i != (len(self.network) - 1):
                for j in range(len(layer)):
                    error = 0
                    # backpropagated error
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])

                    errors.append(error)
            else:
                # getting all errors for output layer
                for idx, neuron in enumerate(layer):
                    errors.append(targets[idx] - neuron['output'])

            # calculating deltas from the errors
            for idx, neuron in enumerate(layer):
                neuron['delta'] = errors[idx] * \
                    self.activationFunctionDerivative(neuron['output'])

    # updates weights based on deltas
    def updateWeights(self, inputs):
        for idx, layer in enumerate(self.network):
            if idx != 0:
                # adding previous layers output as inputs
                inputs = [neuron['output'] for neuron in self.network[idx - 1]]

            for neuron in layer:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.eta * \
                        neuron['delta'] * inputs[j]

                # updating bias node weight - last node of the layer
                neuron['weights'][-1] = self.eta * neuron['delta']

    # takes a validation set and gives the actual outputs of the network
    # and target outputs and gives the matrix showing when the network
    # provides the correct result
    def confusion(self, inputs, targets):
        matrix = self.createMatrix(targets)

        for idx, target in enumerate(targets):
            output = self.forward(inputs[idx])

            # hard-max for to determine classification
            actual = np.argmax(target) + 1
            predicted = output.index(max(output)) + 1

            matrix[actual][predicted] += 1

        totalAcc = 0
        for i, line in enumerate(matrix):
            # skip first line
            if i == 0:
                continue

            # amount of correct predictions
            right = line[i]
            wrong = sum(line[1::]) - right

            line[9] = round((right / (right + wrong)) * 100, 2)

            totalAcc += line[9]

        for line in matrix:
            print(*line, sep='\t')

        print(f'Total accuracy of network: {round(totalAcc / len(targets[0]),2)}')

    def createMatrix(self, targets):
        matrix = [[i] for i in range(len(targets[0]) + 1)]

        for idx, line in enumerate(matrix):
            for i in range(len(targets[0]) + 1):
                if (idx != 0):
                    line.append(0)
                else:
                    line.append(i + 1)

        matrix[0][0] = '-'
        matrix[0][9] = '%'

        return matrix
