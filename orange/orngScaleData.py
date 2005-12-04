# OWGraph.py
# extension for the base graph class that is used in all visualization widgets
#from OWGraph import *
import sys, math, os.path, time
import orange

import Numeric, RandomArray, MA
from MA import transpose
from OWTools import *
import OWVisFuncts


# ####################################################################    
# return a list of sorted values for attribute at index index
# EXPLANATION: if variable values have values 1,2,3,4,... then their order in orange depends on when they appear first
# in the data. With this function we get a sorted list of values
def getVariableValuesSorted(data, index):
    if data.domain[index].varType == orange.VarTypes.Continuous:
        print "Invalid index for getVariableValuesSorted"
        return []
    
    values = list(data.domain[index].values)
    intValues = []
    i = 0
    # do all attribute values containt integers?
    try:
        intValues = [int(val) for val in values]
    except:
        return values

    # if all values were intergers, we first sort them ascendently
    intValues.sort()
    return [str(val) for val in intValues]

# ####################################################################
# create a dictionary with variable at index index. Keys are variable values, key values are indices (transform from string to int)
# in case all values are integers, we also sort them
def getVariableValueIndices(data, index):
    if data.domain[index].varType == orange.VarTypes.Continuous:
        print "Invalid index for getVariableValueIndices"
        return {}

    values = getVariableValuesSorted(data, index)
    return dict([(values[i], i) for i in range(len(values))])


    
class orngScaleData:
    def __init__(self):
        self.rawdata = None                     # input data
        self.originalData = None                # input data in a Numeric array
        self.scaledData = None                  # scaled data to the interval 0-1
        self.noJitteringScaledData = None
        self.attributeNames = []                # list of attribute names from self.rawdata
        self.attributeNameIndex = {}            # dict with indices to attributes
        self.domainDataStat = []
        self.attributeFlipInfo = {}             # dictionary with attrName: 0/1 attribute is flipped or not
        self.subsetData = None
        self.globalValueScaling = 0         # do we want to scale data globally
        self.scalingByVariance = 0
        self.jitterSize = 10
        self.jitterContinuous = 0
        
       
    
    # Converts orange.ExampleTable to Numeric.array based on the attribute values.
    # rows correspond to examples, columns correspond to attributes, class values are left out
    # missing values and attributes of types other than orange.FloatVariable are masked
    def orng2Numeric(exampleTable):
        vals = exampleTable.native(0, substituteDK = "?", substituteDC = "?", substituteOther = "?")
        array = Numeric.array(vals, Numeric.PyObject)
        mask = Numeric.where(Numeric.equal(array, "?"), 1, 0)
        Numeric.putmask(array, mask, 1e20)
        return array.astype(Numeric.Float), mask

    # ####################################################################
    # ####################################################################
    # set new data and scale its values to the 0-1 interval
    def setData(self, data, keepMinMaxVals = 0):
        self.attributeFlipInfo = {}
        if not keepMinMaxVals or self.globalValueScaling == 1:
            self.attrValues = {}

        self.rawdata = data
                
        if data == None or len(data) == 0:
            self.originalData = self.scaledData = self.noJitteringScaledData = self.validDataArray = None
            return
        
        self.attributeFlipInfo = dict([(attr.name, 0) for attr in data.domain]) # reset the fliping information

        self.domainDataStat = orange.DomainBasicAttrStat(data)
        self.offsets = []
        self.normalizers = []
        self.attributeNames = [attr.name for attr in data.domain]
        self.attributeNameIndex = dict([(data.domain[i].name, i) for i in range(len(data.domain))])
        
        min = -1; max = -1
        if self.globalValueScaling == 1:
            (min, max) = self.getMinMaxValDomain(data, self.attributeNames)

        arr = transpose(data.toMA("ac")[0])
        averages = MA.average(arr, 1)
        averages = MA.filled(averages, 1)   # replace missing values with 1
        self.validDataArray = Numeric.array(1-arr.mask(), Numeric.Int)  # have to convert to int array, otherwise when we do some operations on this array we get overflow
        self.averages = averages.tolist()
        arr = Numeric.array(MA.filled(arr, averages))
        self.originalData = arr
        self.scaledData = Numeric.zeros([len(data.domain), len(data)], Numeric.Float)
        self.noJitteringScaledData = Numeric.zeros([len(data.domain), len(data)], Numeric.Float)

        # see if the values for discrete attributes have to be resorted 
        for index in range(len(data.domain)):
            attr = data.domain[index]
            
            if data.domain[index].varType == orange.VarTypes.Discrete:
                variableValueIndices = getVariableValueIndices(data, index)
                for i in range(len(data.domain[index].values)):
                    if i != variableValueIndices[data.domain[index].values[i]]:
                        line = arr[index].copy()  # make the array a contiguous, otherwise the putmask function does not work
                        indices = [Numeric.where(line == val, 1, 0) for val in range(len(data.domain[index].values))]
                        for i in range(len(data.domain[index].values)):
                            Numeric.putmask(line, indices[i], variableValueIndices[data.domain[index].values[i]])
                        arr[index] = line   # save the changed array
                        break

                if not self.attrValues.has_key(attr.name):  self.attrValues[attr.name] = [0, len(attr.values)]
                count = self.attrValues[attr.name][1]
                arr[index] = (arr[index]*2.0 + 1.0)/ float(2*count)
                self.offsets.append(-0.5)
                self.normalizers.append(count)
                self.scaledData[index] = arr[index] + (self.jitterSize/(50.0*count))*(RandomArray.random(len(data)) - 0.5)
            else:
                if self.scalingByVariance:
                    self.offsets.append(self.domainDataStat[index].avg)
                    self.normalizers.append(5*self.domainDataStat[index].dev)
                    arr[index] = (arr[index] - offsets[-1]) / normalizers[-1]
                else:
                    if self.attrValues.has_key(attr.name):          # keep the old min, max values
                        min, max = self.attrValues[attr.name]
                    elif self.globalValueScaling == 0:
                        min = self.domainDataStat[index].min
                        max = self.domainDataStat[index].max
                    diff = float(max - min) or 1.0
                    self.attrValues[attr.name] = [min, max]
                    self.offsets.append(min)
                    self.normalizers.append(diff)
                    arr[index] = (arr[index] - float(min)) / diff

                if self.jitterContinuous:
                    line = arr[index] + self.jitterSize/50.0 * (0.5 - RandomArray.random(len(data)))
                    line = Numeric.absolute(line)       # fix values below zero

                    # fix values above 1
                    ind = Numeric.where(line > 1.0, 1, 0)
                    Numeric.putmask(line, ind, 2.0 - Numeric.compress(ind, line))
                    self.scaledData[index] = line
                else:
                    self.scaledData[index] = arr[index]

        self.noJitteringScaledData = arr
        
 
    # ####################################################################
    # compute min and max value for a list of attributes 
    def getMinMaxValDomain(self, data, attrList):
        first = TRUE
        min = -1; max = -1
        for attr in attrList:
            if data.domain[attr].varType == orange.VarTypes.Discrete: continue
            (minVal, maxVal) = self.getMinMaxVal(data, attr)
            if first == TRUE:
                min = minVal; max = maxVal
                first = FALSE
            else:
                if minVal < min: min = minVal
                if maxVal > max: max = maxVal
        return (min, max)
    

    # ####################################################################
    # get min and max value of data attribute at index index
    def getMinMaxVal(self, data, index):
        attr = data.domain[index]

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            print "warning. Computing min, max value for discrete attribute."
            return (0, float(len(attr.values))-1)
        else:
            return (self.domainDataStat[index].min, self.domainDataStat[index].max)
        
    # ####################################################################
    # scale data at index index to the interval 0 to 1
    # min, max - if booth -1 --> scale to interval 0 to 1, else scale inside interval [min, max]
    # jitteringEnabled - jittering enabled or not
    # ####################################################################
    def scaleData(self, data, index, min = -1, max = -1, jitteringEnabled = 1):
        attr = data.domain[index]
        values = []

        arr = Numeric.zeros([len(data)], Numeric.Float)
        
        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # is the attribute discrete
            # we create a hash table of variable values and their indices
            variableValueIndices = getVariableValueIndices(data, index)
            count = float(len(attr.values))
            values = [0, len(attr.values)-1]

            for i in range(len(data)):
                if data[i][index].isSpecial() == 1: continue
                arr[i] = variableValueIndices[data[i][index].value]
                arr = (arr*2 + 1) / float(2*count)
            if jitteringEnabled:
                arr = arr + 0.5 - (self.jitterSize/(50.0*count))*RandomArray.random(len(data))
            
        # is the attribute continuous
        else:
            if min == max == -1:
                min = self.domainDataStat[index].min
                max = self.domainDataStat[index].max
            values = [min, max]
            diff = max - min
            if diff == 0.0: diff = 1    # prevent division by zero
            
            for i in range(len(data)):
                if data[i][index].isSpecial() == 1: continue
                arr[i] = data[i][index].value
            arr = (arr - min) / diff

        return (arr, values)

    # scale example's value at index index to a range between 0 and 1 with respect to self.rawdata
    def scaleExampleValue(self, example, index):
        if example[index].isSpecial():
            print "Warning: scaling example with missing value"
            return 1e20
        if example.domain[index].varType == orange.VarTypes.Discrete:
            d = getVariableValueIndices(example, index)
            return (d[example[index].value]*2 + 1) / float(2*len(d))
        else:
            [min, max] = self.attrValues[example.domain[index].name]
            #if example[index] < min:   return 0
            #elif example[index] > max: return 1
            #else: return (example[index] - min) / float(max - min)
            # warning: returned value can be outside 0-1 interval!!!
            return (example[index] - min) / float(max - min)
        

    def rescaleAttributesGlobaly(self, data, attrList, jittering = 1):
        if len(attrList) == 0: return
        # find min, max values
        (Min, Max) = self.getMinMaxValDomain(data, attrList)

        # scale data values inside min and max
        for attr in attrList:
            if data.domain[attr].varType == orange.VarTypes.Discrete: continue  # don't scale discrete attributes
            index = self.attributeNameIndex[attr]
            scaled, values = self.scaleData(data, index, Min, Max, jitteringEnabled = jittering)
            self.scaledData[index] = scaled
            self.attrValues[attr] = values

    def getAttributeLabel(self, attrName):
        if self.attributeFlipInfo[attrName] and self.rawdata.domain[attrName].varType == orange.VarTypes.Continuous: return "-" + attrName
        return attrName

    def flipAttribute(self, attrName):
        if attrName not in self.attributeNames: return 0
        if self.rawdata.domain[attrName].varType == orange.VarTypes.Discrete: return 0
        if self.globalValueScaling: return 0
            
        index = self.attributeNameIndex[attrName]
        self.attributeFlipInfo[attrName] = not self.attributeFlipInfo[attrName]
        if self.rawdata.domain[attrName].varType == orange.VarTypes.Continuous:
            self.attrValues[attrName] = [-self.attrValues[attrName][1], -self.attrValues[attrName][0]]
    
        self.scaledData[index] = 1 - self.scaledData[index]
        self.noJitteringScaledData[index] = 1 - self.noJitteringScaledData[index]
        return 1


    # get array of 0 and 1 of len = len(self.rawdata). if there is a missing value at any attribute in indices return 0 for that example
    def getValidList(self, indices):
        selectedArray = Numeric.take(self.validDataArray, indices)
        arr = Numeric.add.reduce(selectedArray)
        return Numeric.equal(arr, len(indices))

    # get array with numbers that represent the example indices that have a valid data value
    def getValidIndices(self, indices):
        validList = self.getValidList(indices)
        return Numeric.nonzero(validList)
        
    # returns a number from -max to max
    def rndCorrection(self, max):
        if max == 0: return 0.0
        return (random() - 0.5)*2*max
        
    