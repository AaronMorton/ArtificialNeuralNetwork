'''
Created on Nov 12, 2013

An artifical neural network class.

@author: Aaron Morton
'''

import numpy

class ANN():
    def __init__(self,hiddenLayers):
        '''
            Create internal representation of the network, using the hiddenLayers array.
            hiddenLayers(i) represents the number of nodes in layer i. 
            layer 0 is the input layer, and the last layer is the output layer
            Weights are randomly generated b/w 0 and 1
        '''
        self.layerArray = hiddenLayers
        matrixList = []
        for i in range(len(hiddenLayers)-1):
            weightMatrix = numpy.matrix(numpy.random.rand(hiddenLayers[i+1],hiddenLayers[i]))
            matrixList.append(weightMatrix)
        self.weightMatrixList = matrixList
        
    def getWeightMatrixList(self):
        return self.weightMatrixList
    
    def feed(self,inputList):
        '''
            Takes a list of input values, which must be the same length as self.layerArray(0)
            Produces a list of output values, which will be the same length as self.layerArray(-1)
        '''
        
        'input variable to be reused in the big loop, initialized to the input to the network'
        input = inputList
        
        'main external loop, pass along activation at each layer, and pass it along'
        for i in range(len(self.weightMatrixList)):
            weightMatrix = self.weightMatrixList(i)
            newInput = []
            'main internal loop, calculate weighted activation sum at each node in current layer'
            for j in range(self.layerArray[i+1]):
                acc = 0
                for k in range(len(input)):
                    acc += weightMatrix(j,k)*input(k)
                    
                newInput[j]=acc
                
            'FILL IN REST HERE, CHECK CORRECTNESS OF ABOVE'
            input = newInput
            
            
    
def main():
    testAnn = ANN([5,10,33,1])
    weights = testAnn.getWeightMatrixList()
    for mat in weights:
        print(mat)
        print(numpy.shape(mat))
    
main()