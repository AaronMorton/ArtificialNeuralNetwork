'''
Created on Jan 6, 2014

A wrapper for extracting data from ARFF files

@author: Aaron Morton
'''

import random

class arffWrapper():

    def __init__(self,fileName1):
        ''' Open the file '''
        self.fileName = fileName1
        self.file = open(self.fileName,'r')
        
        '''get labels for the data, get pointer to the data'''
        line = self.file.readline()
        self.lblList = []
        self.typeList = []
        while line.lower() != "@data\n":
            if "attribute" in line.lower():
                splitString = str.split(line)
                self.lblList.append(splitString[1])
                self.typeList.append(splitString[2])
            line = self.file.readline()
        
        '''get data HERE DATA TYPE IS HARDCODED IN, IF THIS IS TO BE MADE GENERAL PURPOSE,
            WILL NEED HEAVY MODIFICATION
        '''
        line = self.file.readline()
        self.attrList = []
        self.classList = []
        while line != "":
            self.attrList.append(list(map(float,str.split(line,',')[0:-1])))
            self.classList.append([float((str.split(line,',')[-1]).replace("\n",""))])
            line = self.file.readline()
            
    def getAttrCount(self):
        return len(self.attrList[0])
    
    def getInstCount(self):
        return len(self.attrList)
    
    def getClassSize(self):
        return len(self.getClass(0))
    
    def getInstanceRange(self,index1,index2):
        return self.attrList[index1:index2]

    def getClassRange(self,index1,index2):
        return self.attrList[index1:index2]
    
    def getInstance(self,index):
        return self.attrList[index]
    
    def getClass(self,index):
        return self.classList[index]
    
    def getInstances(self):
        return self.attrList
    
    def getClasses(self):
        return self.classList
    
    def shuffle(self):
        classListShuffle = []
        attrListShuffle = []
        
        indices = list(range(len(self.classList)))
        random.shuffle(indices)
        
        for i in indices:
            classListShuffle.append(self.classList[i])
            attrListShuffle.append(self.attrList[i])
            
        self.classList=classListShuffle
        self.attrList=attrListShuffle
        
            
def main():
    a = arffWrapper("export.arff");
    for i in range(len(a.getClasses())):
        print(str(a.getInstance(i))+" | "+str(a.getClass(i)))
    a.shuffle();
    for i in range(len(a.getClasses())):
        print(str(a.getInstance(i))+" | "+str(a.getClass(i)))
    
main()
        