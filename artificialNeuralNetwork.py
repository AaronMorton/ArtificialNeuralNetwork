'''
Created on Nov 12, 2013

An artifical neural network class.

@author: Aaron Morton
'''

import numpy
import math
from arffWrapper import arffWrapper

class ANN():
    def __init__(self,hiddenLayers):
        '''
            Create internal representation of the network, using the hiddenLayers array.
            hiddenLayers(i) represents the number of nodes in layer i. 
            layer 0 is the input layer, and the last layer is the output layer
            Weights are randomly generated b/w 0 and 1
        '''
        
        #create layerArray matrix, initialized with presynaptic activation of 0 for each node
        hiddenLayersWeights = []
        for i in range(len(hiddenLayers)):
            layer = []
            for j in range(hiddenLayers[i]):
                layer.append(0)
            hiddenLayersWeights.append(layer)
        self.layerArray = hiddenLayersWeights
        
        #create list of weight matrices, with random weights for each connection
        #one weight matrix for each connection b/w layers (e.g) one less than number of layers
        #last column in weight matrix is the bias of that node
        matrixList = []
        for i in range(len(hiddenLayers)-1):
            weightMatrix = numpy.matrix(numpy.random.rand(hiddenLayers[i+1],hiddenLayers[i]+1))
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
        
        'put the input as the pre-synaptic activation of the first layer (the input)'
        self.layerArray[0]=inputList;
        
        'main external loop, pass along activation at each layer, and pass it along'
        for i in range(len(self.weightMatrixList)):
            weightMatrix = self.weightMatrixList[i]
            accumInput = []
            'main internal loop, calculate weighted activation sum at each node in current layer'
            for j in range(len(self.layerArray[i+1])):
                #set the accumulator to initially be the bias of this node
                acc = weightMatrix[j,len(inputFeed)]
                for k in range(len(inputFeed)):
                    acc += weightMatrix[j,k]*inputFeed[k]
                #save the pre-synaptic activation into the layerArray
                (self.layerArray[i+1])[j] = acc;
                accumInput.append(sigmoid(acc))
            inputFeed = accumInput
        
        return inputFeed
            
    def objectiveFunction(self,instanceList,classList):
        '''
        For batch update, this produces a total error given a list of instances and classes, using the current state
        of the network.
        '''
        
        'Make the error accumulator, which will always be a number for now, as if there is multiple output values, each is assumed to have equal importance'
        if type(classList[0]) == list:
            errorAccum = 0.0
            listClass = True
        else:
            errorAccum = 0.0
            listClass = False
            
        'Sum up all of the squared errors'
        for i in range(len(instanceList)):
            instance = instanceList[i]
            #If the class is a list of values, sum up the errors across all of the outputs
            if listClass:
                producedValue = self.feed(instance)
                for j in range(len(producedValue)):
                    producedValuePart = producedValue[j]
                    trueValuePart = (classList[i])[j]
                    error = math.pow((producedValuePart - trueValuePart),2)
                    errorAccum += error  
            #If the class is one value, sum up the errors as normal
            else:
                producedValue = self.feed(instance)[0]
                trueValue = classList[i]
                error = math.pow((producedValue - trueValue),2)
                errorAccum += error            
            
        return errorAccum

  
def sigmoid(x):
    '''
    A simple sigmoidal function for applying thresholding to the input at each node. Is simple to differentiate, and is
    based of off Roberts slides here: http://www.cems.uvm.edu/~rsnapp//teaching/cs295ml/notes/backpropagation.pdf
    '''
    return (2/math.pi) * math.atan(x)

def sigmoidDeriv(x):
    '''
    A simple function to return the result of applying the derivative of the recently used sigmoid function to the input
    '''
    return (2/math.pi) * (1 / (1 + math.pow(x,2)))
    
def main():
    data = arffWrapper("export.arff");
    testAnn = ANN([data.getAttrCount(),2,1])
#     weights = testAnn.getWeightMatrixList()
#     for mat in weights:
#         print(mat)
#         print(numpy.shape(mat))
#     print(testAnn.getHiddenLayers())
    print(testAnn.objectiveFunction(data.getInstances(), data.getClasses()))    
main()