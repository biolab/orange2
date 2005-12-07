from orngScaleData import *

class orngScaleScatterPlotData(orngScaleData):
                    
    
    # ##############################################################
    # create x-y projection of attributes in attrList
    # ##############################################################
    def createProjection(self, xAttr, yAttr):
        xAttrIndex, yAttrIndex = self.attributeNameIndex[xAttr], self.attributeNameIndex[yAttr]

        xData = self.scaledData[xAttrIndex].copy()
        yData = self.scaledData[yAttrIndex].copy()
        valid = self.getValidList([xAttrIndex, yAttrIndex])

        if self.rawdata.domain[xAttrIndex].varType == orange.VarTypes.Discrete: xData = ((xData * 2*len(self.rawdata.domain[xAttrIndex].values)) - 1.0) / 2.0
        else:  xData = xData * (self.attrValues[xAttr][1] - self.attrValues[xAttr][0]) + float(self.attrValues[xAttr][0])

        if self.rawdata.domain[yAttrIndex].varType == orange.VarTypes.Discrete: yData = ((yData * 2*len(self.rawdata.domain[yAttrIndex].values)) - 1.0) / 2.0
        else:  yData = yData * (self.attrValues[yAttr][1] - self.attrValues[yAttr][0]) + float(self.attrValues[yAttr][0])

        return (xData, yData)


    # for attributes in attrIndices and values of these attributes in values compute point positions
    # function is called from OWClusterOptimization.py
    # this function has more sense in radviz and polyviz methods
    def getProjectedPointPosition(self, attrIndices, values):
        return values


    # ##############################################################
    # create the projection of attribute indices given in attrIndices and create an example table with it. 
    #def createProjectionAsExampleTable(self, attrIndices, validData = None, classList = None, domain = None, jitterSize = 0.0):
    def createProjectionAsExampleTable(self, attrIndices, settingsDict = {}):
        domain = settingsDict.get("domain")
        if not domain: domain = orange.Domain([orange.FloatVariable(self.rawdata.domain[attrIndices[0]].name), orange.FloatVariable(self.rawdata.domain[attrIndices[1]].name), self.rawdata.domain.classVar])
        data = self.createProjectionAsNumericArray(attrIndices, settingsDict)
        return orange.ExampleTable(domain, data)
    

    #def createProjectionAsNumericArray(self, attrIndices, validData = None, classList = None, jitterSize = 0.0):
    def createProjectionAsNumericArray(self, attrIndices, settingsDict = {}):
        validData = settingsDict.get("validData")
        classList = settingsDict.get("classList")
        jitterSize = settingsDict.get("jitterSize", 0.0)
        
        if not validData: validData = self.getValidList(attrIndices)

        if not classList:
            #classIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
            #if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete: classList = (self.noJitteringScaledData[classIndex]*2*len(self.rawdata.domain.classVar.values)- 1 )/2.0  # remove data with missing values and convert floats back to ints
            #else:                                                                classList = self.noJitteringScaledData[classIndex]  # for continuous attribute just add the values
            classList = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]

        xArray = self.noJitteringScaledData[attrIndices[0]]
        yArray = self.noJitteringScaledData[attrIndices[1]]
        if jitterSize > 0.0:
            xArray += (RandomArray.random(len(xArray))-0.5)*jitterSize
            yArray += (RandomArray.random(len(yArray))-0.5)*jitterSize
        data = Numeric.compress(validData, Numeric.array((xArray, yArray, classList)))
        data = Numeric.transpose(data)
        return data


    def getOptimalClusters(self, attributeNameOrder, addResultFunct):
        jitterSize = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation
        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        
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

                    data = self.createProjectionAsExampleTable([attr1, attr2], settingsDict = {"domain": domain, "jitterSize": jitterSize})
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

