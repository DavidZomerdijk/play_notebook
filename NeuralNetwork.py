# -*- coding: utf-8 -*-
"""
Neural Network

Classes:

- Neuron: where all the calculations are done
- Layer: facilitates communication between the neuron and the network
- NNetwork: performs higher level commands integral to the neural network

Author: Riaan Zoetmulder
assisted by: Christina Zavou
"""

import numpy as np
import math as cm
import matplotlib.pyplot as plt
import cPickle as pickle
import copy
from random import shuffle
import random


class Neuron():
    """
    class that creates the individual neurons
    """
    def __init__(self, weights, numInLayer, learningRate, momentum):
        self.learnRate =- learningRate
        self.numWeights = weights + 1
        self.numberInLayer = numInLayer
        self.weights = np.random.uniform(low =-1, high =1, size = self.numWeights)
        self.weights[0] = 1.0
        self.input = np.random.uniform(low =0, high =1, size = self.numWeights)
        self.delta = 0
        self.output = 0
        self.currentGradient = np.zeros(self.numWeights)
        self.previousGradient = np.zeros(self.numWeights)
        self.momentum = 0
        self.previouserror=0
        
        
    # method to set the input of the inputVector
    def setInput(self, inputVector):
        self.input = inputVector
        
    # Method to calculate the output
    def sigmoid(self):
        power = np.dot(self.weights, self.input)
        self.output = (1.0/(1.0 + cm.exp(-power)))
        return self.output
    
    # Method to return the weights of each neuron
    def getWeights(self):
        return self.weights
    
    # Method for OUTPUT LAYER delta
    def getDelta(self):
        return self.delta
    
    # set the gradient for the OUTPUT Layer
    def setGradientOutput(self, target):
        self.PreviousGradient = self.currentGradient
        self.previousGradient[0] = 0.0
        self.delta = (self.output - target)*self.output*(1-self.output)
        self.currentGradient = self.delta *self.input
        
    # set the Gradient for the HIDDEN Layer
    def setGradientHidden(self, deltaPrev, weightsPrev):
        self.previousGradient = self.currentGradient
        self.previousGradient[0] = 0.0
        sumHidden = np.dot(deltaPrev, weightsPrev)
        self.delta = self.output*(1-self.output)* sumHidden
        self.currentGradient = self.delta* self.input
    
    # Method to update the weights
    def updateWeights(self):
        self.weights = self.weights + self.learnRate * self.currentGradient + self.momentum*self.previousGradient \
      
        
        
    
    
class Layer():
    
    """
    class that controls the neurons and communicates with NN
    """
    
    def __init__(self, NeuronNumber, previousLayer, learningRate, momentum):
        self.NNumber = NeuronNumber
        self.Layer = []
        
        # Create all the neurons in the layer
        for x in range(0, self.NNumber):
            self.Layer.append(Neuron(previousLayer, x, learningRate, momentum))
    
    # Method for gathering the outputs from all the Neurons in layer
    def getOutput(self, inputVector):
        
        outputList = []
        for neuron in self.Layer:
            neuron.setInput(inputVector)
            outputList.append(neuron.sigmoid())
            
        return np.asarray(outputList, float)
    
    # Method for doing backpropagation for the OUTPUT Layer
    def backPropOutput(self, target):
        
        for neuronNumber in range(0,len(self.Layer)):
            
            self.Layer[neuronNumber].setGradientOutput(target.item(neuronNumber))
            self.Layer[neuronNumber].updateWeights()
    
    # Function for doing backpropagation in the hidden layers
    def backPropHidden(self, deltaOutput, weightsOutput):
        weightsPairs = []
        for rows in range(0, int(np.shape(weightsOutput)[0])):
            weightsPair = []
            for columns in range(0, int(np.shape(weightsOutput)[1])):
                weightsPair.append(weightsOutput[rows][columns])
            weightsPairs.append(weightsPair)
        
        # sum over weights vertically
        weightsArray = np.asarray(weightsPairs).sum(axis=0)
        for neuronNumber in range(0, len(self.Layer)):
            self.Layer[neuronNumber].setGradientHidden(np.sum(deltaOutput), weightsArray.item(neuronNumber))
            self.Layer[neuronNumber].updateWeights()
            
            
    # Method for collecting all the deltas from the outputNeurons
    def getDeltas(self):
        
        deltas = []
        for neuron in self.Layer:
            deltas.append(neuron.getDelta())
            
        return np.asarray(deltas)
    
    # Method for collecting the weights from the outputNeurons
    def getWeights(self):
        weightsOutput = []
        for neuron in self.Layer:
            weightsOutput.append(neuron.getWeights())
        
        return np.asarray(weightsOutput)
        
    
            
    
class NNetwork():
    
    """
    Class that controls all the neural network functions
    and communicates with the layers
    """
    
    def __init__(self, inputneurons, hidden, output, learningRate = 0.1, momentum = 0):
        self.input = inputneurons
        self.hidden = hidden
        self.output = output
        
        # Create layer objects
        self.Layers = []
        self.Layers.append(Layer(hidden, inputneurons, learningRate, momentum))
        self.Layers.append(Layer(output, hidden, learningRate, momentum))

    # Function to do forward propagation
    def forwardPropagation(self, NetworkInput):
        firstInput = np.concatenate([[1.0],NetworkInput])
        
        tempInput = self.Layers[0].getOutput(firstInput)
        
        secondInput = np.concatenate([[1.0], tempInput])
        
        output = self.Layers[1].getOutput(secondInput)
        
        return output
        
    def backPropagation(self, data, target, iterations):
        
        for x in range(0, iterations):
            totalError = 0
            
            
            # Iterate over all data points
            for vectorNumber in range(0, int(np.shape(data)[0])):
                
                #Do forward propagation
                outputNetwork = self.forwardPropagation(data[vectorNumber])
                
                # Calculate the error
                totalError += self.errorFunction(outputNetwork, target[vectorNumber])
                
                # Backward pass to outputLayer
                self.Layers[1].backPropOutput(target[vectorNumber])
                    
                # collect weights and delta from previous layer
                deltaOutput = self.Layers[1].getDeltas()
                weightsOutput = self.Layers[1].getWeights()
                
                # Backward pass to hiddenLayer
                self.Layers[0].backPropHidden(deltaOutput, weightsOutput)
                
            
            print "TotalError = ", totalError, "Iteration: ", x
        
    # Function for calculating the error
    def errorFunction(self, outputNetwork, targetValue):
        return 0.5*cm.pow(np.sum(np.subtract(outputNetwork, targetValue)), 2)


if __name__ == "__main__":
    myNN = NNetwork(1, 25, 1, 0.1, 0.9)
    
    # generate some toy data
    x = np.linspace(0, 10, 200).tolist()
    shuffle(x)
    inputData = []
    targetData = []
    for value in range(0,len(x)):
        inputData.append([x[value]])
        targetData.append([0.1 +((0.5*(cm.sin(x[value])+ 1))/1.25)])
    
    # do Backpropagation
    myNN.backPropagation(np.asarray(inputData), np.asarray(targetData), 1000)
    
    # predict some values
    testData = []
    for x in range(0, len(inputData)):
        testData.append(myNN.forwardPropagation(inputData[x]))
    
    # show the output
    plt.plot(inputData,targetData,'.')
    plt.plot(inputData,testData,'.')

    
    
        
        
    
    

    
    