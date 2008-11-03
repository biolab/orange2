from orngScaleData import *
from copy import copy
from math import sqrt

class orngScaleLinProjData(orngScaleData):
    def __init__(self):
        orngScaleData.__init__(self)
        self.normalizeExamples = 1
        self.anchorData =[]        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.lastAttrIndices = None
        self.anchorDict = {}

    def setAnchors(self, xAnchors, yAnchors, attributes):
        if attributes:
            if xAnchors != None and yAnchors != None:
                self.anchorData = [(xAnchors[i], yAnchors[i], attributes[i]) for i in range(len(attributes))]
            else:
                self.anchorData = self.createAnchors(len(attributes), attributes)

    # create anchors around the circle
    def createAnchors(self, numOfAttr, labels = None):
        xAnchors = self.createXAnchors(numOfAttr)
        yAnchors = self.createYAnchors(numOfAttr)
        if labels:
            return [(xAnchors[i], yAnchors[i], labels[i]) for i in range(numOfAttr)]
        else:
            return [(xAnchors[i], yAnchors[i]) for i in range(numOfAttr)]

    def createXAnchors(self, numOfAttrs):
        if not self.anchorDict.has_key(numOfAttrs):
            self.anchorDict[numOfAttrs] = (numpy.cos(numpy.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs)), numpy.sin(numpy.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs)))
        return self.anchorDict[numOfAttrs][0]

    def createYAnchors(self, numOfAttrs):
        if not self.anchorDict.has_key(numOfAttrs):
            self.anchorDict[numOfAttrs] = (numpy.cos(numpy.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs)), numpy.sin(numpy.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs)))
        return self.anchorDict[numOfAttrs][1]


     # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList, useAnchorData = 0):
        orange.saveTabDelimited(fileName, self.createProjectionAsExampleTable([self.attributeNameIndex[i] for i in attrList], useAnchorData = useAnchorData))

    # for attributes in attrIndices and values of these attributes in values compute point positions
    # this function has more sense in radviz and polyviz methods
    def getProjectedPointPosition(self, attrIndices, values, **settingsDict):
        # load the elements from the settings dict
        useAnchorData = settingsDict.get("useAnchorData")
        XAnchors = settingsDict.get("XAnchors")
        YAnchors = settingsDict.get("YAnchors")
        anchorRadius = settingsDict.get("anchorRadius")
        normalizeExample = settingsDict.get("normalizeExample")

        if attrIndices != self.lastAttrIndices:
            print "getProjectedPointPosition. Warning: Possible bug. The set of attributes is not the same as when computing the whole projection"

        if XAnchors != None and YAnchors != None:
            XAnchors = numpy.array(XAnchors)
            YAnchors = numpy.array(YAnchors)
            if anchorRadius == None: anchorRadius = numpy.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)
        elif useAnchorData and self.anchorData:
            XAnchors = numpy.array([val[0] for val in self.anchorData])
            YAnchors = numpy.array([val[1] for val in self.anchorData])
            if anchorRadius == None: anchorRadius = numpy.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))
            anchorRadius = numpy.ones(len(attrIndices), numpy.float)

        if normalizeExample == 1 or (normalizeExample == None and self.normalizeExamples):
            m = min(values); M = max(values)
            if m < 0.0 or M > 1.0:  # we have to do rescaling of values so that all the values will be in the 0-1 interval
                #print "example values are not in the 0-1 interval"
                values = [max(0.0, min(val, 1.0)) for val in values]
                #m = min(m, 0.0); M = max(M, 1.0); diff = max(M-m, 1e-10)
                #values = [(val-m) / float(diff) for val in values]

            s = sum(numpy.array(values)*anchorRadius)
            if s == 0: return [0.0, 0.0]
            x = self.trueScaleFactor * numpy.dot(XAnchors*anchorRadius, values) / float(s)
            y = self.trueScaleFactor * numpy.dot(YAnchors*anchorRadius, values) / float(s)
        else:
            x = self.trueScaleFactor * numpy.dot(XAnchors, values)
            y = self.trueScaleFactor * numpy.dot(YAnchors, values)

        return [x, y]

    # create the projection of attribute indices given in attrIndices and create an example table with it.
    def createProjectionAsExampleTable(self, attrIndices, **settingsDict):
        if self.dataDomain.classVar:
            domain = settingsDict.get("domain") or orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), orange.EnumVariable(self.dataDomain.classVar.name, values = getVariableValuesSorted(self.dataDomain.classVar))])
        else:
            domain = settingsDict.get("domain") or orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar")])
        data = self.createProjectionAsNumericArray(attrIndices, **settingsDict)
        if data != None:
            return orange.ExampleTable(domain, data)
        else:
            return orange.ExampleTable(domain)


    def createProjectionAsNumericArray(self, attrIndices, **settingsDict):
        # load the elements from the settings dict
        validData = settingsDict.get("validData")
        classList = settingsDict.get("classList")
        sum_i     = settingsDict.get("sum_i")
        XAnchors = settingsDict.get("XAnchors")
        YAnchors = settingsDict.get("YAnchors")
        scaleFactor = settingsDict.get("scaleFactor", 1.0)
        normalize = settingsDict.get("normalize")
        jitterSize = settingsDict.get("jitterSize", 0.0)
        useAnchorData = settingsDict.get("useAnchorData", 0)
        removeMissingData = settingsDict.get("removeMissingData", 1)
        useSubsetData = settingsDict.get("useSubsetData", 0)        # use the data or subsetData?
        #minmaxVals = settingsDict.get("minmaxVals", None)

        # if we want to use anchor data we can get attrIndices from the anchorData
        if useAnchorData and self.anchorData:
            attrIndices = [self.attributeNameIndex[val[2]] for val in self.anchorData]

        if validData == None:
            if useSubsetData: validData = self.getValidSubsetList(attrIndices)
            else:             validData = self.getValidList(attrIndices)
        if sum(validData) == 0:
            return None

        if classList == None and self.dataDomain.classVar:
            if useSubsetData: classList = self.originalSubsetData[self.dataClassIndex]
            else:             classList = self.originalData[self.dataClassIndex]

        # if jitterSize is set below zero we use scaledData that has already jittered data
        if useSubsetData:
            if jitterSize < 0.0: data = self.scaledSubsetData
            else:                data = self.noJitteringScaledSubsetData
        else:
            if jitterSize < 0.0: data = self.scaledData
            else:                data = self.noJitteringScaledData

        selectedData = numpy.take(data, attrIndices, axis = 0)
        if removeMissingData:
            selectedData = numpy.compress(validData, selectedData, axis = 1)
            if classList != None and len(classList) != numpy.shape(selectedData)[1]:
                classList = numpy.compress(validData, classList)

        if useAnchorData and self.anchorData:
            XAnchors = numpy.array([val[0] for val in self.anchorData])
            YAnchors = numpy.array([val[1] for val in self.anchorData])
            r = numpy.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)     # compute the distance of each anchor from the center of the circle
            if normalize == 1 or (normalize == None and self.normalizeExamples):
                XAnchors *= r
                YAnchors *= r
        elif (XAnchors != None and YAnchors != None):
            XAnchors = numpy.array(XAnchors); YAnchors = numpy.array(YAnchors)
            r = numpy.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)     # compute the distance of each anchor from the center of the circle
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))
            r = numpy.ones(len(XAnchors), numpy.float)

        x_positions = numpy.dot(XAnchors, selectedData)
        y_positions = numpy.dot(YAnchors, selectedData)

        if normalize == 1 or (normalize == None and self.normalizeExamples):
            if sum_i == None:
                sum_i = self._getSum_i(selectedData, useAnchorData, r)
            x_positions /= sum_i
            y_positions /= sum_i
            self.trueScaleFactor = scaleFactor
        else:
            if not removeMissingData:
                try:
                    x_validData = numpy.compress(validData, x_positions)
                    y_validData = numpy.compress(validData, y_positions)
                except:
                    print validData
                    print x_positions
                    print numpy.shape(validData)
                    print numpy.shape(x_positions)
            else:
                x_validData = x_positions
                y_validData = y_positions
            
            dist = math.sqrt(max(x_validData*x_validData + y_validData*y_validData)) or 1
            self.trueScaleFactor = scaleFactor / dist

        self.unscaled_x_positions = numpy.array(x_positions)
        self.unscaled_y_positions = numpy.array(y_positions)

        if self.trueScaleFactor != 1.0:
            x_positions *= self.trueScaleFactor
            y_positions *= self.trueScaleFactor

        if jitterSize > 0.0:
            x_positions += numpy.random.uniform(-jitterSize, jitterSize, len(x_positions))
            y_positions += numpy.random.uniform(-jitterSize, jitterSize, len(y_positions))

        self.lastAttrIndices = attrIndices
        if classList != None:
            return numpy.transpose(numpy.array((x_positions, y_positions, classList)))
        else:
            return numpy.transpose(numpy.array((x_positions, y_positions)))


    # ##############################################################
    # function to compute the sum of all values for each element in the data. used to normalize.
    def _getSum_i(self, data, useAnchorData = 0, anchorRadius = None):
        if useAnchorData:
            if anchorRadius == None:
                anchorRadius = numpy.sqrt([a[0]**2+a[1]**2 for a in self.anchorData])
            sum_i = numpy.add.reduce(numpy.transpose(numpy.transpose(data)*anchorRadius))
        else:
            sum_i = numpy.add.reduce(data)
        if len(numpy.nonzero(sum_i)) < len(sum_i):    # test if there are zeros in sum_i
            sum_i += numpy.where(sum_i == 0, 1.0, 0.0)
        return sum_i