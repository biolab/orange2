from orngScaleLinProjData import *
from copy import copy
from math import sqrt

class orngScalePolyvizData(orngScaleLinProjData):
    def __init__(self):
        orngScaleLinProjData.__init__(self)
        self.normalizeExamples = 1
        self.anchorData =[]        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        

    # attributeReverse, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, domain = None, scaleFactor = 1.0, jitterSize = 0.0
    def createProjectionAsExampleTable(self, attrList, **settingsDict):
        if self.dataDomain.classVar:
            domain = settingsDict.get("domain") or orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), orange.EnumVariable(self.dataDomain.classVar.name, values = getVariableValuesSorted(self.dataDomain.classVar))])
        else:
            domain = settingsDict.get("domain") or orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar")])
        data = self.createProjectionAsNumericArray(attrList, **settingsDict)
        if data != None:
            return orange.ExampleTable(domain, data)
        else:
            return orange.ExampleTable(domain)
        
    def createProjectionAsNumericArray(self, attrIndices, **settingsDict):
        # load the elements from the settings dict
        attributeReverse = settingsDict.get("reverse", [0]*len(attrIndices))
        validData = settingsDict.get("validData")
        classList = settingsDict.get("classList")
        sum_i     = settingsDict.get("sum_i")
        XAnchors  = settingsDict.get("XAnchors")
        YAnchors  = settingsDict.get("YAnchors")
        scaleFactor = settingsDict.get("scaleFactor", 1.0)
        jitterSize  = settingsDict.get("jitterSize", 0.0)
        removeMissingData = settingsDict.get("removeMissingData", 1)
        
        if validData == None:
            validData = self.getValidList(attrIndices)
        if sum(validData) == 0:
            return None

        if classList == None and self.dataHasClass:
            classList = self.originalData[self.dataClassIndex]  

        if removeMissingData:
            selectedData = numpy.compress(validData, numpy.take(self.noJitteringScaledData, attrIndices, axis = 0), axis = 1)
            if classList != None and len(classList) != numpy.shape(selectedData)[1]:
                classList = numpy.compress(validData, classList)
        else:
            selectedData = numpy.take(self.noJitteringScaledData, attrIndices, axis = 0)
        
        if sum_i == None:
            sum_i = self._getSum_i(selectedData)

        if XAnchors == None or YAnchors == None:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))

        xanchors = numpy.zeros(numpy.shape(selectedData), numpy.float)
        yanchors = numpy.zeros(numpy.shape(selectedData), numpy.float)
        length = len(attrIndices)

        for i in range(length):
            if attributeReverse[i]:
                xanchors[i] = selectedData[i] * XAnchors[i] + (1-selectedData[i]) * XAnchors[(i+1)%length]
                yanchors[i] = selectedData[i] * YAnchors[i] + (1-selectedData[i]) * YAnchors[(i+1)%length]
            else:
                xanchors[i] = (1-selectedData[i]) * XAnchors[i] + selectedData[i] * XAnchors[(i+1)%length]
                yanchors[i] = (1-selectedData[i]) * YAnchors[i] + selectedData[i] * YAnchors[(i+1)%length]

        x_positions = numpy.sum(numpy.multiply(xanchors, selectedData), axis=0)/sum_i
        y_positions = numpy.sum(numpy.multiply(yanchors, selectedData), axis=0)/sum_i
        #x_positions = numpy.sum(numpy.transpose(xanchors* numpy.transpose(selectedData)), axis=0) / sum_i
        #y_positions = numpy.sum(numpy.transpose(yanchors* numpy.transpose(selectedData)), axis=0) / sum_i

        if scaleFactor != 1.0:
            x_positions = x_positions * scaleFactor
            y_positions = y_positions * scaleFactor
        if jitterSize > 0.0:
            x_positions += (numpy.random.random(len(x_positions))-0.5)*jitterSize
            y_positions += (numpy.random.random(len(y_positions))-0.5)*jitterSize

        if classList != None:
            return numpy.transpose(numpy.array((x_positions, y_positions, classList)))
        else:
            return numpy.transpose(numpy.array((x_positions, y_positions)))


    def getProjectedPointPosition(self, attrIndices, values, **settingsDict):
        # load the elements from the settings dict
        attributeReverse = settingsDict.get("reverse", [0]*len(attrIndices))
        useAnchorData = settingsDict.get("useAnchorData")
        XAnchors = settingsDict.get("XAnchors")
        YAnchors = settingsDict.get("YAnchors")
    
        if XAnchors != None and YAnchors != None:
            XAnchors = numpy.array(XAnchors)
            YAnchors = numpy.array(YAnchors)
        elif useAnchorData:
            XAnchors = numpy.array([val[0] for val in self.anchorData])
            YAnchors = numpy.array([val[1] for val in self.anchorData])
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))

        m = min(values); M = max(values)
        if m < 0.0 or M > 1.0:  # we have to do rescaling of values so that all the values will be in the 0-1 interval
            values = [max(0.0, min(val, 1.0)) for val in values]
            #m = min(m, 0.0); M = max(M, 1.0); diff = max(M-m, 1e-10)
            #values = [(val-m) / float(diff) for val in values]
        
        s = sum(numpy.array(values))
        if s == 0: return [0.0, 0.0]

        length = len(values)
        xanchors = numpy.zeros(length, numpy.float)
        yanchors = numpy.zeros(length, numpy.float)
        for i in range(length):
            if attributeReverse[i]:
                xanchors[i] = values[i] * XAnchors[i] + (1-values[i]) * XAnchors[(i+1)%length]
                yanchors[i] = values[i] * YAnchors[i] + (1-values[i]) * YAnchors[(i+1)%length]
            else:
                xanchors[i] = (1-values[i]) * XAnchors[i] + values[i] * XAnchors[(i+1)%length]
                yanchors[i] = (1-values[i]) * YAnchors[i] + values[i] * YAnchors[(i+1)%length]

        x_positions = numpy.sum(numpy.dot(xanchors, values), axis=0) / float(s)
        y_positions = numpy.sum(numpy.dot(yanchors, values), axis=0) / float(s)
        return [x, y]