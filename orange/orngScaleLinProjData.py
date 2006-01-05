from orngScaleData import *
from copy import copy
from math import sqrt

# build a list (in currList) of different permutations of elements in list of elements
# elements contains a list of indices [1,2..., n]
def buildPermutationIndexList(elements, tempPerm, currList):
    for i in range(len(elements)):
        el =  elements[i]
        elements.remove(el)
        tempPerm.append(el)
        buildPermutationIndexList(elements, tempPerm, currList)

        elements.insert(i, el)
        tempPerm.pop()

    if elements == []:
        temp = copy(tempPerm)
        # in tempPerm we have a permutation. Check if it already exists in the currList
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return
            
        # also try the reverse permutation
        temp.reverse()
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return
        currList[str(tempPerm)] = copy(tempPerm)



class orngScaleLinProjData(orngScaleData):
    def __init__(self):
        orngScaleData.__init__(self)
        self.normalizeExamples = 1
        self.anchorData =[]        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        
    def setAnchors(self, xAnchors, yAnchors, attributes):
        if xAnchors and yAnchors and attributes:
            self.anchorData = [(xAnchors[i], yAnchors[i], attributes[i]) for i in range(len(attributes))]

    # create anchors around the circle
    def createAnchors(self, numOfAttr, labels = None):
        xAnchors = self.createXAnchors(numOfAttr)
        yAnchors = self.createYAnchors(numOfAttr)
        if labels:
            return [(xAnchors[i], yAnchors[i], labels[i]) for i in range(numOfAttr)]
        else:
            return [(xAnchors[i], yAnchors[i]) for i in range(numOfAttr)]
        
    def createXAnchors(self, numOfAttrs):
        return Numeric.cos(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))

    def createYAnchors(self, numOfAttrs):
        return Numeric.sin(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))


     # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList, useAnchorData = 0):
        orange.saveTabDelimited(fileName, self.createProjectionAsExampleTable([self.attributeNameIndex[i] for i in attrList], settingsDict = {"useAnchorData":useAnchorData}))

        
    # for attributes in attrIndices and values of these attributes in values compute point positions
    # this function has more sense in radviz and polyviz methods
    # NOTE: the computed x and y positions are not yet scaled. probably you have to use self.scaleFactor or trueScaleFactor to scale them!!!
    #def getProjectedPointPosition(self, attrIndices, values, useAnchorData = 0, XAnchors = None, YAnchors = None, anchorRadius = None, normalizeExample = None):
    def getProjectedPointPosition(self, attrIndices, values, settingsDict = {}):
        # load the elements from the settings dict
        useAnchorData = settingsDict.get("useAnchorData")
        XAnchors = settingsDict.get("XAnchors")
        YAnchors = settingsDict.get("YAnchors")
        anchorRadius = settingsDict.get("anchorRadius")
        normalizeExample = settingsDict.get("normalizeExample")
        
        if XAnchors and YAnchors:
            XAnchors = Numeric.array(XAnchors)
            YAnchors = Numeric.array(YAnchors)
            if not anchorRadius: anchorRadius = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)
        elif useAnchorData:
            XAnchors = Numeric.array([val[0] for val in self.anchorData])
            YAnchors = Numeric.array([val[1] for val in self.anchorData])
            if not anchorRadius: anchorRadius = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))
            anchorRadius = Numeric.ones(len(attrIndices), Numeric.Float)

        m = min(values); M = max(values)
        if m < 0.0 or M > 1.0:  # we have to do rescaling of values so that all the values will be in the 0-1 interval
            m = min(m, 0.0); M = max(M, 1.0); diff = M-m
            values = [(val-m) / float(diff) for val in values]

        if normalizeExample == 1 or (normalizeExample == None and self.normalizeExamples):
            s = sum(Numeric.array(values)*anchorRadius)
            if s == 0: return [0.0, 0.0]
        else: s = 1

        x = Numeric.matrixmultiply(XAnchors*anchorRadius, values) / float(s)
        y = Numeric.matrixmultiply(YAnchors*anchorRadius, values) / float(s)
        return [x,y]

    # ##############################################################
    # create the projection of attribute indices given in attrIndices and create an example table with it. 
    #def createProjectionAsExampleTable(self, attrIndices, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, domain = None, scaleFactor = 1.0, normalize = None, jitterSize = 0.0, useAnchorData = 0):
    def createProjectionAsExampleTable(self, attrIndices, settingsDict = {}):
        domain = settingsDict.get("domain")
        if not domain: domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        data = self.createProjectionAsNumericArray(attrIndices, settingsDict)
        return orange.ExampleTable(domain, data)
        

    #def createProjectionAsNumericArray(self, attrIndices, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, scaleFactor = 1.0, normalize = None, jitterSize = 0.0, useAnchorData = 0, removeMissingData = 1):
    def createProjectionAsNumericArray(self, attrIndices, settingsDict = {}):

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
        
        # if we want to use anchor data we can get attrIndices from the anchorData
        if useAnchorData:
            attrIndices = [self.attributeNameIndex[val[2]] for val in self.anchorData]

        if not validData: validData = self.getValidList(attrIndices)

        # if jitterSize is set below zero we use scaledData that has already jittered data
        if jitterSize < 0.0: data = self.scaledData
        else:                data = self.noJitteringScaledData

        selectedData = Numeric.take(data, attrIndices)
        if removeMissingData: selectedData = Numeric.compress(validData, selectedData)
        
        if not classList:
            classList = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
            if removeMissingData: classList = Numeric.compress(validData, classList)    

        if useAnchorData:
            XAnchors = Numeric.array([val[0] for val in self.anchorData])
            YAnchors = Numeric.array([val[1] for val in self.anchorData])
            r = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)     # compute the distance of each anchor from the center of the circle
            if normalize == 1 or (normalize == None and self.normalizeExamples):
                XAnchors *= r                                               
                YAnchors *= r
        elif (XAnchors and YAnchors):
            XAnchors = Numeric.array(XAnchors); YAnchors = Numeric.array(YAnchors)
            r = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)     # compute the distance of each anchor from the center of the circle
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))
            r = Numeric.ones(len(XAnchors), Numeric.Float)

        x_positions = Numeric.matrixmultiply(XAnchors, selectedData)
        y_positions = Numeric.matrixmultiply(YAnchors, selectedData)

        if normalize == 1 or (normalize == None and self.normalizeExamples):
            if not sum_i: sum_i = self._getSum_i(selectedData, useAnchorData, r)
            x_positions /= sum_i
            y_positions /= sum_i
            self.trueScaleFactor = scaleFactor
        else:
            if not removeMissingData: x_validData = Numeric.compress(validData, x_positions); y_validData = Numeric.compress(validData, y_positions)
            else:                     x_validData = x_positions; y_validData = y_positions
            self.trueScaleFactor = scaleFactor / math.sqrt(max(x_validData*x_validData + y_validData*y_validData))

        self.unscaled_x_positions, self.unscaled_y_positions = Numeric.array(x_positions), Numeric.array(y_positions)

        if self.trueScaleFactor != 1.0:
            x_positions *= self.trueScaleFactor
            y_positions *= self.trueScaleFactor
    
        if jitterSize > 0.0:
            x_positions += (RandomArray.random(len(x_positions))-0.5)*jitterSize
            y_positions += (RandomArray.random(len(y_positions))-0.5)*jitterSize
        
        return Numeric.transpose(Numeric.array((x_positions, y_positions, classList)))

    
    # ##############################################################
    # function to compute the sum of all values for each element in the data. used to normalize.
    def _getSum_i(self, data, useAnchorData = 0, anchorRadius = None):
        if useAnchorData:
            if not anchorRadius:
                anchorRadius = Numeric.sqrt([a[0]**2+a[1]**2 for a in self.anchorData])
            sum_i = Numeric.add.reduce(Numeric.transpose(Numeric.transpose(data)*anchorRadius))
        else:
            sum_i = Numeric.add.reduce(data)
        if len(Numeric.nonzero(sum_i)) < len(sum_i):    # test if there are zeros in sum_i
            sum_i += Numeric.where(sum_i == 0, 1.0, 0.0)
        return sum_i      