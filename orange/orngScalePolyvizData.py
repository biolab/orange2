from orngScaleLinProjData import *
from copy import copy
from math import sqrt

class orngScalePolyvizData(orngScaleLinProjData):
    def __init__(self):
        orngScaleLinProjData.__init__(self)
        self.normalizeExamples = 1
        self.anchorData =[]        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.attrLocalValues = {}

    # if we use globalScaling we must also save min and max values for each attribute
    def setData(self, data):
        # first call the original function to scale data
        orngScaleLinProjData.setData(self, data)
        
        if data == None: return

        if self.globalValueScaling:
            for index in range(len(data.domain)):
                if data.domain[index].varType == orange.VarTypes.Discrete:
                    self.attrLocalValues[data.domain[index].name] = [0, len(data.domain[index].values)-1]
                else:
                    self.attrLocalValues[data.domain[index].name] = [self.domainDataStat[index].min, self.domainDataStat[index].max]
        else:
            self.attrLocalValues = self.attrValues


        # attributeReverse, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, domain = None, scaleFactor = 1.0, jitterSize = 0.0
    def createProjectionAsExampleTable(self, attrList, settingsDict = {}):
        domain = settingsDict.get("domain")
        if not domain: domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        data = self.createProjectionAsNumericArray(attrList, settingsDict)
        return orange.ExampleTable(domain, data)

    #def createProjectionAsNumericArray(self, attrIndices, attributeReverse, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, scaleFactor = 1.0, jitterSize = 0.0, removeMissingData = 1):
    def createProjectionAsNumericArray(self, attrIndices, settingsDict = {}):
        # load the elements from the settings dict
        attributeReverse = settingsDict.get("reverse", [0]*len(attrIndices))
        validData = settingsDict.get("validData")
        classList = settingsDict.get("classList")
        sum_i     = settingsDict.get("sum_i")
        XAnchors = settingsDict.get("XAnchors")
        YAnchors = settingsDict.get("YAnchors")
        scaleFactor = settingsDict.get("scaleFactor", 1.0)
        jitterSize = settingsDict.get("jitterSize", 0.0)
        removeMissingData = settingsDict.get("removeMissingData", 1)
        
        if not validData: validData = self.getValidList(attrIndices)

        if not classList:
            classList = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]    

        if removeMissingData:
            selectedData = Numeric.compress(validData, Numeric.take(self.noJitteringScaledData, attrIndices))
            classList = Numeric.compress(validData, classList)
        else:                 selectedData = Numeric.take(self.noJitteringScaledData, attrIndices)
        
        if not sum_i: sum_i = self._getSum_i(selectedData)
        if not (XAnchors and YAnchors):
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))

        xanchors = Numeric.zeros(Numeric.shape(selectedData), Numeric.Float)
        yanchors = Numeric.zeros(Numeric.shape(selectedData), Numeric.Float)
        length = len(attrIndices)

        for i in range(length):
            if attributeReverse[i]:
                xanchors[i] = selectedData[i] * XAnchors[i] + (1-selectedData[i]) * XAnchors[(i+1)%length]
                yanchors[i] = selectedData[i] * YAnchors[i] + (1-selectedData[i]) * YAnchors[(i+1)%length]
            else:
                xanchors[i] = (1-selectedData[i]) * XAnchors[i] + selectedData[i] * XAnchors[(i+1)%length]
                yanchors[i] = (1-selectedData[i]) * YAnchors[i] + selectedData[i] * YAnchors[(i+1)%length]

        x_positions = Numeric.sum(Numeric.multiply(xanchors, selectedData)) / sum_i
        y_positions = Numeric.sum(Numeric.multiply(yanchors, selectedData)) / sum_i

        if scaleFactor != 1.0:
            x_positions = x_positions * scaleFactor
            y_positions = y_positions * scaleFactor
        if jitterSize > 0.0:
            x_positions += (RandomArray.random(len(x_positions))-0.5)*jitterSize
            y_positions += (RandomArray.random(len(y_positions))-0.5)*jitterSize

        return Numeric.transpose(Numeric.array((x_positions, y_positions, classList)))


    def getProjectedPointPosition(self, attrIndices, values, settingsDict = {}):
        # load the elements from the settings dict
        attributeReverse = settingsDict.get("reverse", [0]*len(attrIndices))
        useAnchorData = settingsDict.get("useAnchorData")
        XAnchors = settingsDict.get("XAnchors")
        YAnchors = settingsDict.get("YAnchors")
    
        if XAnchors and YAnchors:
            XAnchors = Numeric.array(XAnchors)
            YAnchors = Numeric.array(YAnchors)
        elif useAnchorData:
            XAnchors = Numeric.array([val[0] for val in self.anchorData])
            YAnchors = Numeric.array([val[1] for val in self.anchorData])
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))

        m = min(values); M = max(values)
        if m < 0.0 or M > 1.0:  # we have to do rescaling of values so that all the values will be in the 0-1 interval
            values = [max(0.0, min(val, 1.0)) for val in values]
            #m = min(m, 0.0); M = max(M, 1.0); diff = max(M-m, 1e-10)
            #values = [(val-m) / float(diff) for val in values]
        
        s = sum(Numeric.array(values))
        if s == 0: return [0.0, 0.0]

        length = len(values)
        xanchors = Numeric.zeros(length, Numeric.Float)
        yanchors = Numeric.zeros(length, Numeric.Float)
        for i in range(length):
            if attributeReverse[i]:
                xanchors[i] = values[i] * XAnchors[i] + (1-values[i]) * XAnchors[(i+1)%length]
                yanchors[i] = values[i] * YAnchors[i] + (1-values[i]) * YAnchors[(i+1)%length]
            else:
                xanchors[i] = (1-values[i]) * XAnchors[i] + values[i] * XAnchors[(i+1)%length]
                yanchors[i] = (1-values[i]) * YAnchors[i] + values[i] * YAnchors[(i+1)%length]

        x_positions = Numeric.sum(Numeric.multiply(xanchors, values)) / float(s)
        y_positions = Numeric.sum(Numeric.multiply(yanchors, values)) / float(s)
        return [x, y]