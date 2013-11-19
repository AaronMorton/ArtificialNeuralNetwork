'''
Created on Nov 12, 2013

An artifical neural network class.

@author: Aaron Morton
'''

import numpy
import math

class ANN():
    def __init__(self,hiddenLayers):
        '''
            Create internal representation of the network, using the hiddenLayers array.
            hiddenLayers(i) represents the number of nodes in layer i. 
            layer 0 is the input layer, and the last layer is the output layer
            Weights are randomly generated b/w 0 and 1
        '''
        
        #create layerArray matrix, initialized with biases of 0 for each node
        #the biases for the input layer should always be 0 anyway (I think)
        hiddenLayersWeights = []
        for i in range(len(hiddenLayers)):
            layer = []
            for j in range(hiddenLayers[i]):
                layer.append(0)
            hiddenLayersWeights.append(layer)
        self.layerArray = hiddenLayersWeights
        
        #create list of weight matrices, with random weights for each connection
        #one weight matrix for each connection b/w layers (e.g) one less than number of layers
        matrixList = []
        for i in range(len(hiddenLayers)-1):
            weightMatrix = numpy.matrix(numpy.random.rand(hiddenLayers[i+1],hiddenLayers[i]))
            matrixList.append(weightMatrix)
        self.weightMatrixList = matrixList
        
    def getWeightMatrixList(self):
        return self.weightMatrixList
    
    def getHiddenLayers(self):
        return self.layerArray
    
    def feed(self,inputList):
        '''
            Takes a list of input values, which must be the same length as self.layerArray(0)
            Produces a list of output values, which will be the same length as self.layerArray(-1)
            Right now, uses simple sigmoidal function at each node.
        '''
        
        'inputFeed variable to be reused in the big loop, initialized to the inputFeed to the network'
        inputFeed = inputList
        
        'main external loop, pass along activation at each layer, and pass it along'
        for i in range(len(self.weightMatrixList)):
            weightMatrix = self.weightMatrixList[i]
            accumInput = []
            'main internal loop, calculate weighted activation sum at each node in current layer'
            for j in range(len(self.layerArray[i+1])):
                #set the accumulator to initially be the bias of this node
                acc = (self.layerArray[i+1])[j]
                for k in range(len(inputFeed)):
                    acc += weightMatrix[j,k]*inputFeed[k]
                accumInput.append(sigmoid(acc))
            inputFeed = accumInput
            
        return inputFeed
            
            
  
def sigmoid(x):
    '''
    A simple sigmoidal function for applying thresholding to the input at each node. Is simple to differentiate, and is
    based of off Roberts slides here: http://www.cems.uvm.edu/~rsnapp//teaching/cs295ml/notes/backpropagation.pdf
    '''
    return (2/math.pi) * math.atan(x)
    
def main():
    testAnn = ANN([5,30,12,5])
#     weights = testAnn.getWeightMatrixList()
#     for mat in weights:
#         print(mat)
#         print(numpy.shape(mat))
#     print(testAnn.getHiddenLayers())
    print(testAnn.feed([700,40,28,4000,17]))
    print(testAnn.feed([2,1,1,2,1]))    
    print(testAnn.feed([2,1,1,2,1]))    
    
main()