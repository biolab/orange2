# OWGraph.py
# extension for the base graph class that is used in all visualization widgets
#from OWGraph import *
import sys, math
import orange
import numpy
from orngDataCaching import *
try:
    import numpy.ma as MA
except:
    import numpy.core.ma as MA


# ####################################################################
# return a list of sorted values for attribute at index index
# EXPLANATION: if variable values have values 1,2,3,4,... then their order in orange depends on when they appear first
# in the data. With this function we get a sorted list of values
def getVariableValuesSorted(variable):
    if variable.varType == orange.VarTypes.Continuous:
        print "getVariableValuesSorted - attribute %s is a continuous variable" % (variable)
        return []

    values = list(variable.values)
    intValues = []

    # do all attribute values containt integers?
    try:
        intValues = [(int(val), val) for val in values]
    except:
        return values

    # if all values were intergers, we first sort them ascendently
    intValues.sort()
    return [val[1] for val in intValues]

# ####################################################################
# create a dictionary with variable at index index. Keys are variable values, key values are indices (transform from string to int)
# in case all values are integers, we also sort them
def getVariableValueIndices(variable, sortValuesForDiscreteAttrs = 1):
    if variable.varType == orange.VarTypes.Continuous:
        print "getVariableValueIndices - attribute %s is a continuous variable" % (str(index))
        return {}

    if sortValuesForDiscreteAttrs:
        values = getVariableValuesSorted(variable)
    else:
        values = list(variable.values)
    return dict([(values[i], i) for i in range(len(values))])


# discretize the domain
# if we have a class remove the examples with missing class value
# discretize the continuous class into discrete class with two values
# discretize continuous attributes using entropy discretization (or equiN if we don't have a class or class is continuous)
def discretizeDomain(data, removeUnusedValues = 1):
    entroDisc = orange.EntropyDiscretization()
    equiDisc  = orange.EquiNDiscretization(numberOfIntervals = 2)
    discAttrs = []

    className = data and len(data) > 0 and data.domain.classVar and data.domain.classVar.name or None
#    if className:
#        data = data.filterref(orange.Filter_hasClassValue())  # remove examples with missing classes

    if not data or len(data) == 0:
        return None

    # if we have a continuous class we have to discretize it before we can discretize the attributes
    if className and data.domain.classVar.varType == orange.VarTypes.Continuous:
        newClass = equiDisc(data.domain.classVar.name, data)
        newClass.name = className
        newDomain = orange.Domain(data.domain.attributes, newClass)
        data = orange.ExampleTable(newDomain, data)

    for attr in data.domain.attributes:
        try:
            name = attr.name
            if attr.varType == orange.VarTypes.Continuous:  # if continuous attribute then use entropy discretization
                if data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete:
                    attr = entroDisc(attr, data)
                else:
                    attr = equiDisc(attr, data)
            if removeUnusedValues:
                attr = orange.RemoveUnusedValues(attr, data)
            attr.name = name
            discAttrs.append(attr)
        except:     # if all values are missing, entropy discretization will throw an exception. in such cases ignore the attribute
            pass

    if className: discAttrs.append(data.domain.classVar)
    return data.select(discAttrs)



class orngScaleData:
    def __init__(self):
        self.rawData = None                     # input data
        self.rawSubsetData = None
        self.attributeNames = []                # list of attribute names from self.rawData
        self.attributeNameIndex = {}            # dict with indices to attributes
        self.attributeFlipInfo = {}             # dictionary with attrName: 0/1 attribute is flipped or not
        
        self.dataHasClass = False
        self.dataHasContinuousClass = False
        self.dataHasDiscreteClass = False
        self.dataClassName = None
        self.dataDomain = None
        self.dataClassIndex = None
        self.haveData = False
        self.haveSubsetData = False

        self.jitterSize = 10
        self.jitterContinuous = 0

        self.attrValues = {}
        self.domainDataStat = []
        self.originalData = self.originalSubsetData = None    # input (nonscaled) data in a numpy array
        self.scaledData = self.scaledSubsetData = None        # scaled data to the interval 0-1
        self.noJitteringScaledData = self.noJitteringScaledSubsetData = None
        self.validDataArray = self.validSubsetDataArray = None

    # take examples from data and subsetData and merge them into one dataset
    def mergeDataSets(self, data, subsetData):
        if data == None and subsetData == None: None
        if subsetData == None:
            fullData = data
        elif data == None:
            fullData = subsetData
        else:
            fullData = orange.ExampleTable(data)
            fullData.extend(subsetData)
        return fullData

    # force the exising data to be rescaled do to changes like, jitterContinuous, jitterSize, ...
    def rescaleData(self):
        self.setData(self.rawData, self.rawSubsetData, skipIfSame = 0)

    # this function has to be called before setData or setSubsetData
    # because it computes the minimum and maximum values in the dataset
    def setData(self, data, subsetData = None, **args):
        if args.get("skipIfSame", 1):
            if ((data == None and self.rawData == None) or (self.rawData != None and data != None and self.rawData.checksum() == data.checksum())) and  \
               ((subsetData == None and self.rawSubsetData == None) or (self.rawSubsetData != None and subsetData != None and self.rawSubsetData.checksum() == subsetData.checksum())):
                    return

        self.domainDataStat = []
        self.attrValues = {}
        self.originalData = self.originalSubsetData = None
        self.scaledData = self.scaledSubsetData = None
        self.noJitteringScaledData = self.noJitteringScaledSubsetData = None
        self.validDataArray = self.validSubsetDataArray = None

        self.rawData = None
        self.rawSubsetData = None
        self.haveData = False
        self.haveSubsetData = False
        self.dataHasClass = False
        self.dataHasContinuousClass = False
        self.dataHasDiscreteClass = False
        self.dataClassName = None
        self.dataDomain = None
        self.dataClassIndex = None
                
        if data == None: return
        fullData = self.mergeDataSets(data, subsetData)
                
        self.rawData = data
        self.rawSubsetData = subsetData

        lenData = data and len(data) or 0
        numpy.random.seed(1)     # we always reset the random generator, so that if we receive the same data again we will add the same noise

        self.attributeNames = [attr.name for attr in fullData.domain]
        self.attributeNameIndex = dict([(fullData.domain[i].name, i) for i in range(len(fullData.domain))])
        self.attributeFlipInfo = {}         # dict([(attr.name, 0) for attr in fullData.domain]) # reset the fliping information
        
        self.dataDomain = fullData.domain
        self.dataHasClass = bool(fullData.domain.classVar)
        self.dataHasContinuousClass = bool(self.dataHasClass and fullData.domain.classVar.varType == orange.VarTypes.Continuous)
        self.dataHasDiscreteClass = bool(self.dataHasClass and fullData.domain.classVar.varType == orange.VarTypes.Discrete)
        self.dataClassName = self.dataHasClass and fullData.domain.classVar.name
        if self.dataHasClass:
            self.dataClassIndex = self.attributeNameIndex[self.dataClassName]
        self.haveData = bool(self.rawData and len(self.rawData) > 0)
        self.haveSubsetData = bool(self.rawSubsetData and len(self.rawSubsetData) > 0)
        
        self.domainDataStat = getCached(fullData, orange.DomainBasicAttrStat, (fullData,))

        sortValuesForDiscreteAttrs = args.get("sortValuesForDiscreteAttrs", 1)

        for index in range(len(fullData.domain)):
            attr = fullData.domain[index]
            if attr.varType == orange.VarTypes.Discrete:
                self.attrValues[attr.name] = [0, len(attr.values)]
            elif attr.varType == orange.VarTypes.Continuous:
                self.attrValues[attr.name] = [self.domainDataStat[index].min, self.domainDataStat[index].max]
        
        # the originalData, noJitteringScaledData and validArray are arrays that we can cache so that other visualization widgets
        # don't need to compute it. The scaledData on the other hand has to be computed for each widget separately because of different
        # jitterContinuous and jitterSize values
        if getCached(data, "visualizationData") and subsetData == None:
            self.originalData, self.noJitteringScaledData, self.validDataArray = getCached(data, "visualizationData")
            self.originalSubsetData = self.noJitteringScaledSubsetData = self.validSubsetDataArray = numpy.array([]).reshape([len(self.originalData), 0])
        else:
            noJitteringData = fullData.toNumpyMA("ac")[0].T
            validDataArray = numpy.array(1-noJitteringData.mask, numpy.short)  # have to convert to int array, otherwise when we do some operations on this array we get overflow
            noJitteringData = numpy.array(MA.filled(noJitteringData, orange.Illegal_Float))
            originalData = noJitteringData.copy()
            
            for index in range(len(data.domain)):
                attr = data.domain[index]
                if attr.varType == orange.VarTypes.Discrete:
                    # see if the values for discrete attributes have to be resorted
                    variableValueIndices = getVariableValueIndices(data.domain[index], sortValuesForDiscreteAttrs)
                    if 0 in [i == variableValueIndices[attr.values[i]] for i in range(len(attr.values))]:
                        line = noJitteringData[index].copy()  # make the array a contiguous, otherwise the putmask function does not work
                        indices = [numpy.where(line == val, 1, 0) for val in range(len(attr.values))]
                        for i in range(len(attr.values)):
                            numpy.putmask(line, indices[i], variableValueIndices[attr.values[i]])
                        noJitteringData[index] = line   # save the changed array
                        originalData[index] = line     # reorder also the values in the original data
                    noJitteringData[index] = (noJitteringData[index]*2.0 + 1.0)/ float(2*len(attr.values))
                    
                elif attr.varType == orange.VarTypes.Continuous:
                    diff = self.domainDataStat[index].max - self.domainDataStat[index].min or 1     # if all values are the same then prevent division by zero
                    noJitteringData[index] = (noJitteringData[index] - self.domainDataStat[index].min) / diff

            self.originalData = originalData[:,:lenData]; self.originalSubsetData = originalData[:,lenData:]
            self.noJitteringScaledData = noJitteringData[:,:lenData]; self.noJitteringScaledSubsetData = noJitteringData[:,lenData:]
            self.validDataArray = validDataArray[:,:lenData]; self.validSubsetDataArray = validDataArray[:,lenData:]
        
        if data: setCached(data, "visualizationData", (self.originalData, self.noJitteringScaledData, self.validDataArray))
        if subsetData: setCached(subsetData, "visualizationData", (self.originalSubsetData, self.noJitteringScaledSubsetData, self.validSubsetDataArray))
            
        # compute the scaledData arrays
        scaledData = numpy.concatenate([self.noJitteringScaledData, self.noJitteringScaledSubsetData], axis = 1)
        for index in range(len(data.domain)):
            attr = data.domain[index]
            if attr.varType == orange.VarTypes.Discrete:
                scaledData[index] += (self.jitterSize/(50.0*max(1,len(attr.values))))*(numpy.random.random(len(fullData)) - 0.5)
                
            elif attr.varType == orange.VarTypes.Continuous and self.jitterContinuous:
                scaledData[index] += self.jitterSize/50.0 * (0.5 - numpy.random.random(len(fullData)))
                scaledData[index] = numpy.absolute(scaledData[index])       # fix values below zero
                ind = numpy.where(scaledData[index] > 1.0, 1, 0)     # fix values above 1
                numpy.putmask(scaledData[index], ind, 2.0 - numpy.compress(ind, scaledData[index]))
        self.scaledData = scaledData[:,:lenData]; self.scaledSubsetData = scaledData[:,lenData:]


  
    # scale example's value at index index to a range between 0 and 1 with respect to self.rawData
    def scaleExampleValue(self, example, index):
        if example[index].isSpecial():
            print "Warning: scaling example with missing value"
            return 0.5     #1e20
        if example.domain[index].varType == orange.VarTypes.Discrete:
            d = getVariableValueIndices(example.domain[index])
            return (d[example[index].value]*2 + 1) / float(2*len(d))
        elif example.domain[index].varType == orange.VarTypes.Continuous:
            diff = self.domainDataStat[index].max - self.domainDataStat[index].min
            if diff == 0: diff = 1          # if all values are the same then prevent division by zero
            return (example[index] - self.domainDataStat[index].min) / diff


    def getAttributeLabel(self, attrName):
        if self.attributeFlipInfo.get(attrName, 0) and self.dataDomain[attrName].varType == orange.VarTypes.Continuous:
            return "-" + attrName
        return attrName

    def flipAttribute(self, attrName):
        if attrName not in self.attributeNames: return 0
        if self.dataDomain[attrName].varType == orange.VarTypes.Discrete: return 0

        index = self.attributeNameIndex[attrName]
        self.attributeFlipInfo[attrName] = 1 - self.attributeFlipInfo.get(attrName, 0)
        if self.dataDomain[attrName].varType == orange.VarTypes.Continuous:
            self.attrValues[attrName] = [-self.attrValues[attrName][1], -self.attrValues[attrName][0]]

        self.scaledData[index] = 1 - self.scaledData[index]
        self.scaledSubsetData[index] = 1 - self.scaledSubsetData[index]
        self.noJitteringScaledData[index] = 1 - self.noJitteringScaledData[index]
        self.noJitteringScaledSubsetData[index] = 1 - self.noJitteringScaledSubsetData[index]
        return 1

    def getMinMaxVal(self, attr):
        if type(attr) == int:
            attr = self.attributeNames[attr]
        diff = self.attrValues[attr][1] - self.attrValues[attr][0]
        return diff or 1.0

    # get array of 0 and 1 of len = len(self.rawData). if there is a missing value at any attribute in indices return 0 for that example
    def getValidList(self, indices, alsoClassIfExists = 1):
        if self.validDataArray == None or len(self.validDataArray) == 0:
            return numpy.array([], numpy.bool)
        
        inds = indices[:]
        if alsoClassIfExists and self.dataHasClass: 
            inds.append(self.dataClassIndex) 
        selectedArray = self.validDataArray.take(inds, axis = 0)
        arr = numpy.add.reduce(selectedArray)
        return numpy.equal(arr, len(inds))

    # get array of 0 and 1 of len = len(self.rawSubsetData). if there is a missing value at any attribute in indices return 0 for that example
    def getValidSubsetList(self, indices, alsoClassIfExists = 1):
        if self.validSubsetDataArray == None or len(self.validSubsetDataArray) == 0:
            return numpy.array([], numpy.bool)
        if alsoClassIfExists and self.dataClassIndex: 
            indices.append(self.dataClassIndex)
        selectedArray = self.validSubsetDataArray.take(indices, axis = 0)
        arr = numpy.add.reduce(selectedArray)
        return numpy.equal(arr, len(indices))

    # get array with numbers that represent the example indices that have a valid data value
    def getValidIndices(self, indices):
        validList = self.getValidList(indices)
        return numpy.nonzero(validList)[0]

    # get array with numbers that represent the example indices that have a valid data value
    def getValidSubsetIndices(self, indices):
        validList = self.getValidSubsetList(indices)
        return numpy.nonzero(validList)[0]

    # returns a number from -max to max
    def rndCorrection(self, max):
        return (random() - 0.5)*2*max

