# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:30:55 2017

@author: adityamagarde
"""

import numpy as np
import pandas as pd
#IMPORTING DATASET
dataset = pd.read_csv(r'F:\Files\Projects\Untitled Folder\FingerCount\Datasets\one.csv')
datasetDimensions = dataset.shape

#DATA PREPROCESSING
X = dataset.iloc[:, 0:3].values
X[:, 0:2] = X[:, 0:2]/100
#rowsOnes = np.ones((datasetDimensions[0], 1), dtype='int64')
#X = np.concatenate((rowsOnes, X), axis=1)

yReal = dataset.iloc[:, 3]
yReal = np.reshape(yReal, (1499,1))

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features = 'all')
y = oneHotEncoder.fit_transform(yReal).toarray()
y = y[:, 1:]

intitialdJdW1 = 1
intitialdJdW2 = 1

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
#CREATE AND INITIALISE THE WEIGHT MATRICES
    def __init__(self):
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) 
        print(self.W1)
        print(self.W2)
#INITIALISE THE VALUES OF Z2, A2, Z3, OUTPUT
    def initialiseAZ(self, X):
        self.Z2 = np.dot(X, self.W1)
        self.A2 = sigmoid(self.Z2)    
#ADDING THE BIAS UNIT IN A2 OR THE HIDDEN-LAYER
        #shapeOfX = X.shape
        #rowsOnes = np.ones((shapeOfX[0], 1), dtype='int64')
        #self.A2 = np.concatenate((rowsOnes, self.A2), axis=1)    
        self.Z3 = np.dot(self.A2, self.W2)
        self.output = sigmoid(self.Z3)    
#FINDING THE COST
    def costFunction(self):
            return (1/2)*sum((self.output - y)**2)    
#DEFINE THE BACKPROP
    def backprop(self):
            self.initialiseAZ(X)
            delta3 = np.multiply(sigmoidPrime(self.Z3), (self.output - y))
            dJdW2 = np.dot(self.A2.T, delta3)
            delta2 = np.dot(delta3, self.W2.T)*sigmoidPrime(self.Z2)
            dJdW1 = np.dot(X.T, delta2)
            print('dJdW1 = ', dJdW1, 'dJdW2 =', dJdW2)
            scalar = 1
            self.W1 = self.W1 - scalar*dJdW1
            self.W2 = self.W2 - scalar*dJdW2            
            return dJdW1, dJdW2, self.W1, self.W2
        
#MAIN CODE
myNeuralNetwork = NeuralNetwork()
initialdJdW1, initialdJdW2, intialWeight1, initialWeight2 = myNeuralNetwork.backprop()
intialCost = myNeuralNetwork.costFunction()
        
for i in range(1,250):
    weight1, weight2, main_dJdW1, main_dJdW2 = myNeuralNetwork.backprop()
    cost = myNeuralNetwork.costFunction()
    print('W1 = ', weight1, 'W2 = ', weight2, 'Cost = ', cost)

print('initialDJDW1 = ', initialdJdW1, 'intialDJDW2 = ', initialdJdW2)

#WE NOW TAKE THE INPUT AND PERFORM THE PREDICTIONS


