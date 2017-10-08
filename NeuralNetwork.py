# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:30:55 2017

@author: adityamagarde
"""

import numpy as np
import pandas as pd


#Importing dataset
dataset = pd.read_csv('one.csv')
datasetDimensions = dataset.shape

#Data Preprocessing
X = dataset.iloc[:, 0:3].values
X[:, 0:2] = X[:, 0:2]/100
yReal = dataset.iloc[:, 3]
yReal = np.reshape(yReal, (1499,1))
    #One_Hot _Encoding Y
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features = 'all')
y = oneHotEncoder.fit_transform(yReal).toarray()
y = y[:, 1:]


#SIGMOID FUNCTION
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/((1 + np.exp(-z))**2)

class NeuralNetwork(object):
    numberOfLayers = 3
    inputLayerSize = 3
    hiddenLayerSize = 4
    outputLayerSize = 5

#Initialising weights and biases
    def __init__(self):
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.W1b = np.random.uniform(size=(1, self.hiddenLayerSize))
        self.W2b = np.random.uniform(size=(1, self.outputLayerSize))
        self.debug = 1
        print('INITIALISED : \nW1 = ', self.W1, '\nW2 = ', self.W2)

#Calculate the values of Z2, A2, Z3, output
    def initialiseAZ(self, X):
        self.Z2 = np.dot(X, self.W1)
        self.Z2 = self.Z2 + self.W1b
        self.A2 = sigmoid(self.Z2)
        self.Z3 = np.dot(self.A2, self.W2)
        self.Z3 = self.Z3 + self.W2b
        self.output = sigmoid(self.Z3)
        #for debugging
        if(self.debug == 1):
            print('\nZ2 = ', self.Z2, '\nA2 = ', self.A2, '\nZ3 = ', self.Z3, '\nouput = ', self.output)
            self.debug = 0

#Finding the cost
    def calculateCost(self):
        cost = np.sum(self.output - y, axis=1)
        cost = np.reshape(cost, (1499, 1))
        return cost

#Cost Function
    def costFunction(self):
        costMatrix = self.output - y
        print(costMatrix, '\n')
        cost = (1/2)*np.sum(((self.output - y)**2), axis=1)
        cost = np.reshape(cost, (1499, 1))
        return cost, costMatrix

#Backpropogation
    def backprop(self):
            self.initialiseAZ(X)
            delta3 = np.multiply(sigmoidPrime(self.Z3), (self.output-y))
            dJdW2 = np.dot(self.A2.T, delta3)
            delta2 = np.dot(delta3, self.W2.T)*sigmoidPrime(self.Z2)
            dJdW1 = np.dot(X.T, delta2)
            scalar = 0.0025
            self.W1 = self.W1 - scalar*dJdW1
            self.W2 = self.W2 - scalar*dJdW2
            self.W1b = self.W1b - scalar*sum(delta2)
            self.W2b = self.W2b - scalar*sum(delta3)
            return self.W1, self.W2, self.W1b, self.W2b

#MAIN CODE
myNeuralNetwork = NeuralNetwork()
intialWeight1, initialWeight2, intialWeight1Bias,intialWweight2Bias = myNeuralNetwork.backprop()
initialCost, initialCostMatrix = myNeuralNetwork.costFunction()

for i in range(1,20000):
    weight1, weight2, weight1Bias, weight2Bias = myNeuralNetwork.backprop()
    cost, costMatrix = myNeuralNetwork.costFunction()
    print('\tCost = ', cost)

print('\n\nInitial Cost = ', initialCost)


#Till here we have the values of optimised weights
#Here the input should include the bias unit 1 i.e the input should be like 1 0 0
print('Enter the values of input: ')
x = input()
x = x.split(' ')
x = np.array(x, dtype='float64')
x = np.reshape(x, (1,3))

testZ2 = np.dot(x, weight1)
testZ2 = testZ2 + weight1Bias
testA2 = sigmoid(testZ2)
testZ3 = np.dot(testA2, weight2)
testZ3 = testZ3 + weight2Bias
testOutput = sigmoid(testZ3)

#testOutput[:] = testOutput[:]>=max(testOutpu

print('\nOUTPUT = ', testOutput)
