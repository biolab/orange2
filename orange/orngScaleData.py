# OWGraph.py
# extension for the base graph class that is used in all visualization widgets
#from OWGraph import *
import sys, math
import orange
import numpy
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
    if className:
        data = data.filterref(orange.Filter_hasClassValue())  # remove examples with missing classes

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

        self.globalValueScaling = 0         # do we want to scale data globally
        self.scalingByVariance = 0
        self.jitterSize = 10
        self.jitterContinuous = 0

        self.attrValues = {}
        self.domainDataStat = []
        self.offsets = []
        self.normalizers = []
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
        self.offsets = []
        self.normalizers = []
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
                
        fullData = self.mergeDataSets(data, subsetData)
        if fullData == None: return
        
        self.rawData = data
        self.rawSubsetData = subsetData

        lenData = data and len(data) or 0
        numpy.random.seed(1)     # we always reset the random generator, so that if we receive the same data again we will add the same noise

        self.domainDataStat = orange.DomainBasicAttrStat(fullData)
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

                
        if self.globalValueScaling:
            (Min, Max) = self.getMinMaxValDomain()

        for index in range(len(fullData.domain)):
            attr = fullData.domain[index]
            if attr.varType == orange.VarTypes.Discrete:
                self.attrValues[attr.name] = [0, len(attr.values)]
                self.offsets.append(0.0)
                self.normalizers.append(len(attr.values)-1)
            else:
                if self.scalingByVariance:
                    self.offsets.append(self.domainDataStat[index].avg)
                    self.normalizers.append(max(1e-5, self.domainDataStat[index].dev))
                    self.attrValues[attr.name] = [self.domainDataStat[index].min, self.domainDataStat[index].max]
                else:
                    if not self.globalValueScaling:
                        Min = self.domainDataStat[index].min
                        Max = self.domainDataStat[index].max
                    diff = float(Max - Min) or 1.0
                    self.offsets.append(Min)
                    self.normalizers.append(diff)
                    self.attrValues[attr.name] = [Min, Max]

        sortValuesForDiscreteAttrs = args.get("sortValuesForDiscreteAttrs", 1)
        arr = fullData.toNumpyMA("ac")[0].T
        validDataArray = numpy.array(1-arr.mask, numpy.short)  # have to convert to int array, otherwise when we do some operations on this array we get overflow
        arr = numpy.array(MA.filled(arr, orange.Illegal_Float))
        originalData = arr.copy()
        scaledData = numpy.zeros(originalData.shape, numpy.float)
        
        if data != None:
            for index in range(len(data.domain)):
                attr = data.domain[index]

                if attr.varType == orange.VarTypes.Discrete:
                    # see if the values for discrete attributes have to be resorted
                    variableValueIndices = getVariableValueIndices(data.domain[index], sortValuesForDiscreteAttrs)
                    if 0 in [i == variableValueIndices[attr.values[i]] for i in range(len(attr.values))]:
                        line = arr[index].copy()  # make the array a contiguous, otherwise the putmask function does not work
                        indices = [numpy.where(line == val, 1, 0) for val in range(len(attr.values))]
                        for i in range(len(attr.values)):
                            numpy.putmask(line, indices[i], variableValueIndices[attr.values[i]])
                        arr[index] = line   # save the changed array
                        originalData[index] = line     # reorder also the values in the original data
                        
                    arr[index] = (arr[index]*2.0 + 1.0)/ float(2*len(attr.values))
                    scaledData[index] = arr[index] + (self.jitterSize/(50.0*max(1,len(attr.values))))*(numpy.random.random(len(fullData)) - 0.5)
                else:
                    arr[index] = ((arr[index] - self.offsets[index]) / self.normalizers[index])

                    if self.jitterContinuous:
                        line = arr[index] + self.jitterSize/50.0 * (0.5 - numpy.random.random(len(fullData)))
                        line = numpy.absolute(line)       # fix values below zero

                        # fix values above 1
                        ind = numpy.where(line > 1.0, 1, 0)
                        numpy.putmask(line, ind, 2.0 - numpy.compress(ind, line))
                        scaledData[index] = line
                    else:
                        scaledData[index] = arr[index]

        self.originalData = originalData[:,:lenData]; self.originalSubsetData = originalData[:,lenData:]
        self.scaledData = scaledData[:,:lenData]; self.scaledSubsetData = scaledData[:,lenData:]
        self.noJitteringScaledData = arr[:,:lenData]; self.noJitteringScaledSubsetData = arr[:,lenData:]
        self.validDataArray = validDataArray[:,:lenData]; self.validSubsetDataArray = validDataArray[:,lenData:]

    

    # scale example's value at index index to a range between 0 and 1 with respect to self.rawData
    def scaleExampleValue(self, example, index):
        if example[index].isSpecial():
            print "Warning: scaling example with missing value"
            return 0.5     #1e20
        if example.domain[index].varType == orange.VarTypes.Discrete:
            d = getVariableValueIndices(example.domain[index])
            return (d[example[index].value]*2 + 1) / float(2*len(d))
        else:
            if len(self.offsets) <= index or len(self.normalizers) <= index :
                print "invalid example or attribute index", index, len(self.offsets), len(self.normalizers)
                return 0.0
            return (example[index] - self.offsets[index]) / self.normalizers[index]


    def getAttributeLabel(self, attrName):
        if self.attributeFlipInfo.get(attrName, 0) and self.dataDomain[attrName].varType == orange.VarTypes.Continuous:
            return "-" + attrName
        return attrName

    def flipAttribute(self, attrName):
        if attrName not in self.attributeNames: return 0
        if self.dataDomain[attrName].varType == orange.VarTypes.Discrete: return 0
        if self.globalValueScaling: return 0

        index = self.attributeNameIndex[attrName]
        self.attributeFlipInfo[attrName] = 1 - self.attributeFlipInfo.get(attrName, 0)
        if self.dataDomain[attrName].varType == orange.VarTypes.Continuous:
            self.attrValues[attrName] = [-self.attrValues[attrName][1], -self.attrValues[attrName][0]]

        self.scaledData[index] = 1 - self.scaledData[index]
        self.scaledSubsetData[index] = 1 - self.scaledSubsetData[index]
        self.noJitteringScaledData[index] = 1 - self.noJitteringScaledData[index]
        self.noJitteringScaledSubsetData[index] = 1 - self.noJitteringScaledSubsetData[index]
        return 1

    # compute min and max value for a list of attributes
    def getMinMaxValDomain(self, attrList = None):
        minVal = 1e40
        maxVal = -1e40
        if not self.domainDataStat:
            return 0, 1
        if attrList == None:
            attrList = range(len(self.dataDomain))

        for attr in attrList:
            if self.dataDomain[attr].varType == orange.VarTypes.Discrete: continue
            minVal = min(minVal, self.domainDataStat[attr].min)
            maxVal = max(maxVal, self.domainDataStat[attr].max)
        return (minVal, maxVal)


    # get min and max value of data attribute at index index
    def getMinMaxVal(self, data, attr):
        if data.domain[attr].varType == orange.VarTypes.Discrete:
            return (0, float(len(data.domain[attr].values))-1)
        else:
            return (self.domainDataStat[attr].min, self.domainDataStat[attr].max)


    # get array of 0 and 1 of len = len(self.rawData). if there is a missing value at any attribute in indices return 0 for that example
    def getValidList(self, indices):
        if self.validDataArray == None or len(self.validDataArray) == 0:
            return numpy.array([], numpy.bool)
        try:
            selectedArray = self.validDataArray.take(indices, axis = 0)
        except:
            print "getValidList failure. Invalid indices:", indices
            print self.validDataArray
        arr = numpy.add.reduce(selectedArray)
        return numpy.equal(arr, len(indices))

    # get array of 0 and 1 of len = len(self.rawSubsetData). if there is a missing value at any attribute in indices return 0 for that example
    def getValidSubsetList(self, indices):
        if self.validSubsetDataArray == None or len(self.validSubsetDataArray) == 0:
            return numpy.array([], numpy.bool)
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

