# OWGraph.py
# extension for the base graph class that is used in all visualization widgets
#from OWGraph import *
import sys, math, time
import orange
import numpy
import numpy.core.ma as MA
from OWTools import *
import orngVisFuncts


# ####################################################################
# return a list of sorted values for attribute at index index
# EXPLANATION: if variable values have values 1,2,3,4,... then their order in orange depends on when they appear first
# in the data. With this function we get a sorted list of values
def getVariableValuesSorted(data, index):
    if data.domain[index].varType == orange.VarTypes.Continuous:
        print "getVariableValuesSorted - attribute %s is a continuous variable" % (str(index))
        return []

    values = list(data.domain[index].values)
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
def getVariableValueIndices(data, index, sortValuesForDiscreteAttrs = 1):
    if data.domain[index].varType == orange.VarTypes.Continuous:
        print "getVariableValueIndices - attribute %s is a continuous variable" % (str(index))
        return {}

    if sortValuesForDiscreteAttrs:
        values = getVariableValuesSorted(data, index)
    else:
        values = list(data.domain[index].values)
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
        self.rawdata = None                     # input data
        self.originalData = None                # input data in a numpy array
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
        self.subDataMinMaxDict = {}             # dictionary of tuples. keys are attribute names. values are (min, max) vals for examples in subsetData
        self.validDataArray = None
        self.validSubDataArray = None
        self.attrValues = {}
        self.attrSubValues = {}
        self.normalizers = []


    # Converts orange.ExampleTable to numpy.array based on the attribute values.
    # rows correspond to examples, columns correspond to attributes, class values are left out
    # missing values and attributes of types other than orange.FloatVariable are masked
    def orng2Numeric(exampleTable):
        vals = exampleTable.native(0, substituteDK = "?", substituteDC = "?", substituteOther = "?")
        array = numpy.array(vals, numpy.object)
        mask = numpy.where(numpy.equal(array, "?"), 1, 0)
        numpy.putmask(array, mask, 1e20)
        return array.astype(numpy.float), mask

    # ####################################################################
    # ####################################################################
    # set new data and scale its values to the 0-1 interval or normalize it by subtracting the mean and dividing by the deviation
    def setData(self, data, **args):
        self.attributeFlipInfo = {}
        self.attrValues = {}
        sortValuesForDiscreteAttrs = args.get("sortValuesForDiscreteAttrs", 1)

        self.rawdata = data
        numpy.random.seed(1)     # we always reset the random generator, so that if we receive the same data again we will add the same noise

        if data == None or len(data) == 0:
            self.originalData = self.scaledData = self.noJitteringScaledData = self.validDataArray = None
            self.domainDataStat = [];       self.attributeNames = []
            self.attributeNameIndex = {};   self.attributeFlipInfo = {}
            return

        self.attributeFlipInfo = dict([(attr.name, 0) for attr in data.domain]) # reset the fliping information

        self.domainDataStat = orange.DomainBasicAttrStat(data)
        self.offsets = []
        self.normalizers = []
        self.attributeNames = [attr.name for attr in data.domain]
        self.attributeNameIndex = dict([(data.domain[i].name, i) for i in range(len(data.domain))])

        if self.globalValueScaling:
            (Min, Max) = self.getMinMaxValDomain(data, self.attributeNames)

        arr = MA.transpose(data.toNumpyMA("ac")[0])
        averages = MA.average(arr, 1)
        self.averages = MA.filled(averages, 1)   # replace missing values with 1
        self.validDataArray = numpy.array(1-arr.mask, numpy.int)  # have to convert to int array, otherwise when we do some operations on this array we get overflow
        arr = numpy.array(MA.filled(arr, -99999999))
        self.originalData = arr.copy()
        self.scaledData = numpy.zeros([len(data.domain), len(data)], numpy.float)

        # see if the values for discrete attributes have to be resorted
        for index in range(len(data.domain)):
            attr = data.domain[index]

            if attr.varType == orange.VarTypes.Discrete:
                variableValueIndices = getVariableValueIndices(data, index, sortValuesForDiscreteAttrs)
                for i in range(len(attr.values)):
                    if i != variableValueIndices[attr.values[i]]:
                        line = arr[index].copy()  # make the array a contiguous, otherwise the putmask function does not work
                        indices = [numpy.where(line == val, 1, 0) for val in range(len(attr.values))]
                        for i in range(len(attr.values)):
                            numpy.putmask(line, indices[i], variableValueIndices[attr.values[i]])
                        arr[index] = line   # save the changed array
                        break

                count = len(attr.values)
                self.attrValues[attr.name] = [0, count]
                arr[index] = (arr[index]*2.0 + 1.0)/ float(2*count)
                self.offsets.append(0.0)
                self.normalizers.append(count-1)
                self.scaledData[index] = arr[index] + (self.jitterSize/(50.0*max(1,count)))*(numpy.random.random(len(data)) - 0.5)
            else:
                if self.scalingByVariance:
                    self.offsets.append(self.domainDataStat[index].avg)
                    self.normalizers.append(max(1e-5, self.domainDataStat[index].dev))
                    arr[index] = ((arr[index] - self.offsets[-1]) / self.normalizers[-1])
                    self.attrValues[attr.name] = [MA.minimum(arr[index]), MA.maximum(arr[index])]
                else:
                    if not self.globalValueScaling:
                        Min = self.domainDataStat[index].min
                        Max = self.domainDataStat[index].max
                    diff = float(Max - Min) or 1.0
                    self.attrValues[attr.name] = [Min, Max]
                    self.offsets.append(Min)
                    self.normalizers.append(diff)
                    arr[index] = (arr[index] - float(Min)) / diff

                if self.jitterContinuous:
                    line = arr[index] + self.jitterSize/50.0 * (0.5 - numpy.random.random(len(data)))
                    line = numpy.absolute(line)       # fix values below zero

                    # fix values above 1
                    ind = numpy.where(line > 1.0, 1, 0)
                    numpy.putmask(line, ind, 2.0 - numpy.compress(ind, line))
                    self.scaledData[index] = line
                else:
                    self.scaledData[index] = arr[index]

        self.noJitteringScaledData = arr
#        if self.subsetData:
#            self.setSubsetData(self.subsetData)

    def setSubsetData(self, subData):
        self.subsetData = subData
        self.validSubDataArray = []
        self.attrSubValues = {}
        self.subDataMinMaxDict = {}

#        if not subData or not self.rawdata or subData.domain.checksum() != self.rawdata.domain.checksum():
#            return

        if not subData or not self.rawdata:
            return

        try:
            subData = subData.select(self.rawdata.domain)
        except:
            print "Warning: Subset data domain incompatible with data domain.\nData domain: %s\n Subset data domain: %s\n" % (self.rawdata.domain, subData.domain)
            return

        # create a  valid data array
        arr = MA.transpose(subData.toNumpyMA("ac")[0])
        self.validSubDataArray = numpy.array(1-arr.mask, numpy.int)  # have to convert to int array, otherwise when we do some operations on this array we get overflow

        domainSubDataStat = orange.DomainBasicAttrStat(subData)
        for index in range(len(subData.domain)):
            attr = subData.domain[index]
            if subData.domain[index].varType == orange.VarTypes.Continuous:
                Min = domainSubDataStat[index].min
                Max = domainSubDataStat[index].max
                self.attrSubValues[attr.name] = (Min, Max)
                #if self.scalingByVariance or self.globalValueScaling:
                #    continue
                #normalizer = self.normalizers[index] or 1
                #projMin = (Min - self.offsets[index]) / normalizer
                #projMax = (Max - self.offsets[index]) / normalizer
                #if projMin < 0.0 or projMax > 1.0:
                #    self.subDataMinMaxDict[attr.name] = (min(projMin, 0.0), max(1.0, projMax))
            elif subData.domain[index].varType == orange.VarTypes.Discrete:
                self.attrSubValues[attr.name] = [0, len(attr.values)]

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

        arr = numpy.zeros([len(data)], numpy.float)

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
                arr = arr + 0.5 - (self.jitterSize/(50.0*count))*numpy.random.random(len(data))

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
            return 0.5     #1e20
        if example.domain[index].varType == orange.VarTypes.Discrete:
            d = getVariableValueIndices(example, index)
            return (d[example[index].value]*2 + 1) / float(2*len(d))
        else:
            if len(self.offsets) <= index or len(self.normalizers) <= index :
                print "invalid example or attribute index", index, len(self.offsets), len(self.normalizers)
                return 0.0
            position = (example[index] - self.offsets[index]) / self.normalizers[index]
            if self.subDataMinMaxDict.has_key(self.rawdata.domain[index].name):
                m, M = self.subDataMinMaxDict[self.rawdata.domain[index].name]
                position = (position - m) / float(max(M-m, 1e-10))
            return position


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
        if self.validDataArray == None:
            return numpy.array([], numpy.bool)
        selectedArray = numpy.take(self.validDataArray, indices, axis = 0)
        arr = numpy.add.reduce(selectedArray)
        return numpy.equal(arr, len(indices))

    # get array of 0 and 1 of len = len(self.subsetData). if there is a missing value at any attribute in indices return 0 for that example
    def getValidSubList(self, indices):
        if self.validSubDataArray == None:
            return numpy.array([], numpy.bool)
        selectedArray = numpy.take(self.validSubDataArray, indices, axis = 0)
        arr = numpy.add.reduce(selectedArray)
        return numpy.equal(arr, len(indices))

    # get array with numbers that represent the example indices that have a valid data value
    def getValidIndices(self, indices):
        validList = self.getValidList(indices)
        return numpy.nonzero(validList)[0]

    # get array with numbers that represent the example indices that have a valid data value
    def getValidSubIndices(self, indices):
        validList = self.getValidSubList(indices)
        return numpy.nonzero(validList)[0]

    # returns a number from -max to max
    def rndCorrection(self, max):
        if max == 0: return 0.0
        return (random() - 0.5)*2*max

