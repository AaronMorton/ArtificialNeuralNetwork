'''
Created on Jan 6, 2014

A wrapper for extracting data from ARFF files

@author: Aaron Morton
'''

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
    
    def getInstance(self,index):
        return self.attrList[index]
    
    def getClass(self,index):
        return self.classList[index]
    
    def getInstances(self):
        return self.attrList
    
    def getClasses(self):
        return self.classList
            
def main():
    a = arffWrapper("export.arff");
    print(a.classList)
    for b in a.attrList:
        print(b)
    print(a.getAttrCount())
    
        