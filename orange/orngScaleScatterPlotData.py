from orngScaleData import *

class orngScaleScatterPlotData(orngScaleData):

    def getOriginalData(self, indices):
        data = self.originalData.take(indices, axis = 0)
        for i, ind in enumerate(indices):
            [minVal, maxVal] = self.attrValues[self.dataDomain[ind].name]
            if self.dataDomain[ind].varType == orange.VarTypes.Discrete:
                data[i] += (self.jitterSize/50.0)*(numpy.random.random(len(self.rawData)) - 0.5)
            elif self.dataDomain[ind].varType == orange.VarTypes.Continuous and self.jitterContinuous:
                data[i] += (self.jitterSize/(50.0*(maxVal-minVal or 1)))*(numpy.random.random(len(self.rawData)) - 0.5)
        return data
    
    def getOriginalSubsetData(self, indices):
        data = self.originalSubsetData.take(indices, axis = 0)
        for i, ind in enumerate(indices):
            [minVal, maxVal] = self.attrValues[self.rawSubsetData.domain[ind].name]
            if self.dataDomain[ind].varType == orange.VarTypes.Discrete:
                data[i] += (self.jitterSize/(50.0*max(1, maxVal)))*(numpy.random.random(len(self.rawSubsetData)) - 0.5)
            elif self.dataDomain[ind].varType == orange.VarTypes.Continuous and self.jitterContinuous:
                data[i] += (self.jitterSize/(50.0*(maxVal-minVal or 1)))*(numpy.random.random(len(self.rawSubsetData)) - 0.5)
        return data

    # create x-y projection of attributes in attrList
    def getXYDataPositions(self, xAttr, yAttr):
        xAttrIndex, yAttrIndex = self.attributeNameIndex[xAttr], self.attributeNameIndex[yAttr]

        xData = self.scaledData[xAttrIndex].copy()
        yData = self.scaledData[yAttrIndex].copy()
        
        if self.dataDomain[xAttrIndex].varType == orange.VarTypes.Discrete: xData = ((xData * 2*len(self.dataDomain[xAttrIndex].values)) - 1.0) / 2.0
        else:  xData = xData * (self.attrValues[xAttr][1] - self.attrValues[xAttr][0]) + float(self.attrValues[xAttr][0])

        if self.dataDomain[yAttrIndex].varType == orange.VarTypes.Discrete: yData = ((yData * 2*len(self.dataDomain[yAttrIndex].values)) - 1.0) / 2.0
        else:  yData = yData * (self.attrValues[yAttr][1] - self.attrValues[yAttr][0]) + float(self.attrValues[yAttr][0])
        return (xData, yData)
    
    
    # create x-y projection of attributes in attrList
    def getXYSubsetDataPositions(self, xAttr, yAttr):
        xAttrIndex, yAttrIndex = self.attributeNameIndex[xAttr], self.attributeNameIndex[yAttr]

        xData = self.scaledSubsetData[xAttrIndex].copy()
        yData = self.scaledSubsetData[yAttrIndex].copy()
        
        if self.dataDomain[xAttrIndex].varType == orange.VarTypes.Discrete: xData = ((xData * 2*len(self.dataDomain[xAttrIndex].values)) - 1.0) / 2.0
        else:  xData = xData * (self.attrValues[xAttr][1] - self.attrValues[xAttr][0]) + float(self.attrValues[xAttr][0])

        if self.dataDomain[yAttrIndex].varType == orange.VarTypes.Discrete: yData = ((yData * 2*len(self.dataDomain[yAttrIndex].values)) - 1.0) / 2.0
        else:  yData = yData * (self.attrValues[yAttr][1] - self.attrValues[yAttr][0]) + float(self.attrValues[yAttr][0])
        return (xData, yData)
    
    

    # for attributes in attrIndices and values of these attributes in values compute point positions
    # this function has more sense in radviz and polyviz methods
    def getProjectedPointPosition(self, attrIndices, values, **settingsDict): # settingsDict has to be because radviz and polyviz have this parameter
        return values


    # create the projection of attribute indices given in attrIndices and create an example table with it.
    #def createProjectionAsExampleTable(self, attrIndices, validData = None, classList = None, domain = None, jitterSize = 0.0):
    def createProjectionAsExampleTable(self, attrIndices, **settingsDict):
        if self.dataHasClass:
            domain = settingsDict.get("domain") or orange.Domain([orange.FloatVariable(self.dataDomain[attrIndices[0]].name), orange.FloatVariable(self.dataDomain[attrIndices[1]].name), orange.EnumVariable(self.dataDomain.classVar.name, values = getVariableValuesSorted(self.dataDomain.classVar))])
        else:
            domain = settingsDict.get("domain") or orange.Domain([orange.FloatVariable(self.dataDomain[attrIndices[0]].name), orange.FloatVariable(self.dataDomain[attrIndices[1]].name)])

        data = self.createProjectionAsNumericArray(attrIndices, **settingsDict)
        if data != None:
            return orange.ExampleTable(domain, data)
        else:
            return orange.ExampleTable(domain)


    def createProjectionAsNumericArray(self, attrIndices, **settingsDict):
        validData = settingsDict.get("validData")
        classList = settingsDict.get("classList")
        jitterSize = settingsDict.get("jitterSize", 0.0)

        if validData == None:
            validData = self.getValidList(attrIndices)
        if sum(validData) == 0:
            return None

        if classList == None and self.dataHasClass:
            classList = self.originalData[self.dataClassIndex]

        xArray = self.noJitteringScaledData[attrIndices[0]]
        yArray = self.noJitteringScaledData[attrIndices[1]]
        if jitterSize > 0.0:
            xArray += (numpy.random.random(len(xArray))-0.5)*jitterSize
            yArray += (numpy.random.random(len(yArray))-0.5)*jitterSize
        if classList != None:
            data = numpy.compress(validData, numpy.array((xArray, yArray, classList)), axis = 1)
        else:
            data = numpy.compress(validData, numpy.array((xArray, yArray)), axis = 1)
        data = numpy.transpose(data)
        return data


    def getOptimalClusters(self, attributeNameOrder, addResultFunct):
        if not self.dataHasClass or self.dataHasContinuousClass:
            return
        
        jitterSize = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation
        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.dataDomain.classVar])

        self.scatterWidget.progressBarInit()  # init again, in case that the attribute ordering took too much time
        startTime = time.time()
        count = len(attributeNameOrder)*(len(attributeNameOrder)-1)/2
        testIndex = 0

        for i in range(len(attributeNameOrder)):
            for j in range(i):
                try:
                    attr1 = self.attributeNameIndex[attributeNameOrder[j]]
                    attr2 = self.attributeNameIndex[attributeNameOrder[i]]
                    testIndex += 1
                    if self.clusterOptimization.isOptimizationCanceled():
                        secs = time.time() - startTime
                        self.clusterOptimization.setStatusBarText("Evaluation stopped (evaluated %d projections in %d min, %d sec)" % (testIndex, secs/60, secs%60))
                        self.scatterWidget.progressBarFinished()
                        return

                    data = self.createProjectionAsExampleTable([attr1, attr2], domain = domain, jitterSize = jitterSize)
                    graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)

                    allValue = 0.0
                    classesDict = {}
                    for key in valueDict.keys():
                        addResultFunct(valueDict[key], closureDict[key], polygonVerticesDict[key], [attributeNameOrder[i], attributeNameOrder[j]], int(graph.objects[polygonVerticesDict[key][0]].getclass()), enlargedClosureDict[key], otherDict[key])
                        classesDict[key] = int(graph.objects[polygonVerticesDict[key][0]].getclass())
                        allValue += valueDict[key]
                    addResultFunct(allValue, closureDict, polygonVerticesDict, [attributeNameOrder[i], attributeNameOrder[j]], classesDict, enlargedClosureDict, otherDict)     # add all the clusters
                    
                    self.clusterOptimization.setStatusBarText("Evaluated %d projections..." % (testIndex))
                    self.scatterWidget.progressBarSet(100.0*testIndex/float(count))
                    del data, graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict, classesDict
                except:
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # print the exception
        
        secs = time.time() - startTime
        self.clusterOptimization.setStatusBarText("Finished evaluation (evaluated %d projections in %d min, %d sec)" % (testIndex, secs/60, secs%60))
        self.scatterWidget.progressBarFinished()

