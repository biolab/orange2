#
# OWScatterPlotGraph.py
#
# the base for scatterplot

from OWVisGraph import *
import time


###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
###########################################################################################
class OWScatterPlotGraph(OWVisGraph):
    def __init__(self, parent = None, name = None, app = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)

        self.application = app
        self.jitterContinuous = 0
        self.enabledLegend = 0
        self.showFilledSymbols = 1
        self.showAttributeValues = 1

    def enableGraphLegend(self, enable):
        self.enabledLegend = enable

    def setJitterContinuous(self, enable):
        self.jitterContinuous = enable

    def setShowFilledSymbols(self, filled):
        self.showFilledSymbols = filled

    def setShowAttributeValues(self, show):
        self.showAttributeValues = show
        
    ########################################################
    # set new data and scale its values
    def setData(self, data):
        self.rawdata = data
        self.domainDataStat = orange.DomainBasicAttrStat(data)
        self.scaledData = []
        self.attributeNames = []
        
        if data == None: return

        self.attrVariance = []
        for index in range(len(data.domain)):
            attr = data.domain[index]
            self.attributeNames.append(attr.name)
            (scaled, variance)= self.scaleData(data, index, jitteringEnabled = 0)
            self.scaledData.append(scaled)
            self.attrVariance.append(variance)


    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", showColorLegend = 0, statusBar = None):
        self.clear()
        self.tips.removeAll()
        self.enableLegend(0)
        self.statusBar = statusBar
        toolTipList = [xAttr, yAttr]
        if shapeAttr != "": toolTipList.append(shapeAttr)
        if sizeShapeAttr != "": toolTipList.append(sizeShapeAttr)

        (xVarMin, xVarMax) = self.attrVariance[self.attributeNames.index(xAttr)]
        (yVarMin, yVarMax) = self.attrVariance[self.attributeNames.index(yAttr)]
        xVar = xVarMax - xVarMin
        yVar = yVarMax - yVarMin
        
        
        MAX_HUE_VAL = 300           # hue value can go to 360, but at 360 it produces the same color as at 0 so we make the interval shorter
        MIN_SHAPE_SIZE = 10
        MAX_SHAPE_DIFF = 10

        if len(self.scaledData) == 0: self.updateLayout(); return

        if self.rawdata.domain[xAttr].varType == orange.VarTypes.Continuous:
            self.setXlabels(None)
        else:
            self.setXlabels(self.getVariableValuesSorted(self.rawdata, xAttr))
            self.setAxisScale(QwtPlot.xBottom, xVarMin - (self.jitterSize * xVar / 80.0), xVarMax + (self.jitterSize * xVar / 80.0) + showColorLegend * xVar/20, 1)            

        if self.rawdata.domain[yAttr].varType == orange.VarTypes.Continuous: self.setYLlabels(None)
        else:
            self.setYLlabels(self.getVariableValuesSorted(self.rawdata, yAttr))
            self.setAxisScale(QwtPlot.yLeft, yVarMin - (self.jitterSize * yVar / 80.0), yVarMax + (self.jitterSize * yVar / 80.0), 1)

        if self.showXaxisTitle == 1: self.setXaxisTitle(xAttr)
        if self.showYLaxisTitle == 1: self.setYLaxisTitle(yAttr)

        if self.showAttributeValues == 0:
            self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
            self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
            scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
            scaleDraw.setTickLength(1, 1, 0)
            scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
            scaleDraw.setTickLength(1, 1, 0)
        else:
            scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
            scaleDraw.setTickLength(1, 1, 3)
            scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
            scaleDraw.setTickLength(1, 1, 3)
            
        
        colorIndex = -1
        if colorAttr != "" and colorAttr != "(One color)":
            if self.rawdata.domain[colorAttr].varType == orange.VarTypes.Discrete: MAX_HUE_VAL = 360
            colorIndex = self.attributeNames.index(colorAttr)
            (colorData, vals) = self.scaleData(self.rawdata, colorIndex, forColoring = 1)
            

        shapeIndex = -1
        shapeIndices = {}
        if shapeAttr != "" and shapeAttr != "(One shape)" and len(self.rawdata.domain[shapeAttr].values) < 11:
            shapeIndex = self.attributeNames.index(shapeAttr)
            shapeIndices = self.getVariableValueIndices(self.rawdata, shapeAttr)

        sizeShapeIndex = -1
        if sizeShapeAttr != "" and sizeShapeAttr != "(One size)":
            sizeShapeIndex = self.attributeNames.index(sizeShapeAttr)

        shapeList = [QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Diamond, QwtSymbol.Triangle, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.Cross, QwtSymbol.XCross, QwtSymbol.StyleCnt]

        # create hash tables in case of discrete X axis attribute
        attrXIndices = {}
        discreteX = 0
        if self.rawdata.domain[xAttr].varType == orange.VarTypes.Discrete:
            discreteX = 1
            attrXIndices = self.getVariableValueIndices(self.rawdata, xAttr)

        # create hash tables in case of discrete Y axis attribute
        attrYIndices = {}
        discreteY = 0
        if self.rawdata.domain[yAttr].varType == orange.VarTypes.Discrete:
            discreteY = 1
            attrYIndices = self.getVariableValueIndices(self.rawdata, yAttr)

        self.curveKeys = []
        for i in range(len(self.rawdata)):
            if self.rawdata[i][xAttr].isSpecial() == 1: continue
            if self.rawdata[i][yAttr].isSpecial() == 1: continue
            if colorIndex != -1 and self.rawdata[i][colorIndex].isSpecial() == 1: continue
            if shapeIndex != -1 and self.rawdata[i][shapeIndex].isSpecial() == 1: continue
            if sizeShapeIndex != -1 and self.rawdata[i][sizeShapeIndex].isSpecial() == 1: continue
            
            if discreteX == 1:
                x = attrXIndices[self.rawdata[i][xAttr].value] + self.rndCorrection(float(self.jitterSize * xVar) / 100.0)
            elif self.jitterContinuous == 1:
                x = self.rawdata[i][xAttr].value + self.rndCorrection(float(self.jitterSize * xVar) / 100.0)
            else:
                x = self.rawdata[i][xAttr].value

            if discreteY == 1:
                y = attrYIndices[self.rawdata[i][yAttr].value] + self.rndCorrection(float(self.jitterSize * yVar) / 100.0)
            elif self.jitterContinuous == 1:
                y = self.rawdata[i][yAttr].value + self.rndCorrection(float(self.jitterSize * yVar) / 100.0)
            else:
                y = self.rawdata[i][yAttr].value

            newColor = QColor(0,0,0)
            if colorIndex != -1:
                #newColor.setHsv(self.scaledData[colorIndex][i]*MAX_HUE_VAL, 255, 255)
                newColor.setHsv(colorData[i]*MAX_HUE_VAL, 255, 255)
                
            symbol = shapeList[0]
            if shapeIndex != -1:
                symbol = shapeList[shapeIndices[self.rawdata[i][shapeIndex].value]]

            size = self.pointWidth
            if sizeShapeIndex != -1:
                size = MIN_SHAPE_SIZE + round(self.scaledData[sizeShapeIndex][i] * MAX_SHAPE_DIFF)

            newCurveKey = self.insertCurve(str(i))

            symbolBrush = QBrush(QBrush.NoBrush)
            if self.showFilledSymbols == 1:
                symbolBrush = QBrush(newColor)
            newSymbol = QwtSymbol(symbol, symbolBrush, QPen(newColor), QSize(size, size))
            self.setCurveSymbol(newCurveKey, newSymbol)
            self.setCurveData(newCurveKey, [x], [y])
            self.curveKeys.append(newCurveKey)

            ##########
            # we add a tooltip for this point
            r = QRectFloat(x-xVar/100.0, y-yVar/100.0, xVar/50.0, yVar/50.0)
            text= self.getShortExampleText(self.rawdata, self.rawdata[i], toolTipList)
            self.tips.addToolTip(r, text)
            ##########
            

        # show legend if necessary
        if self.enabledLegend == 1:
            if colorIndex != -1 and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Discrete:
                varName = self.rawdata.domain[colorIndex].name
                varValues = self.getVariableValuesSorted(self.rawdata, colorIndex)
                numColors = len(self.rawdata.domain[colorIndex].values)
                for ind in range(numColors):
                    newColor = QColor()
                    newColor.setHsv(float(ind) / float(numColors) * MAX_HUE_VAL, 255, 255)
                    self.addCurve(varName + "=" + varValues[ind], newColor, newColor, self.pointWidth, enableLegend = 1)

            if shapeIndex != -1 and self.rawdata.domain[shapeIndex].varType == orange.VarTypes.Discrete:
                varName = self.rawdata.domain[shapeIndex].name
                varValues = self.getVariableValuesSorted(self.rawdata, shapeIndex)
                for ind in range(len(self.rawdata.domain[shapeIndex].values)):
                    self.addCurve(varName + "=" + varValues[ind], QColor(0,0,0), QColor(0,0,0), self.pointWidth, symbol = shapeList[ind], enableLegend = 1)

            if sizeShapeIndex != -1 and self.rawdata.domain[sizeShapeIndex].varType == orange.VarTypes.Discrete:
                varName = self.rawdata.domain[sizeShapeIndex].name
                varValues = self.getVariableValuesSorted(self.rawdata, sizeShapeIndex)
                for ind in range(len(varValues)):
                    self.addCurve(varName + "=" + varValues[ind], QColor(0,0,0), QColor(0,0,0), MIN_SHAPE_SIZE + round(ind*MAX_SHAPE_DIFF/len(varValues)), enableLegend = 1)


        if colorAttr != "" and showColorLegend == 1:
            for i in range(1000):
                x0 = xVarMax + xVar/100
                x1 = x0 + xVar/20
                y = yVarMin + i*yVar/1000
                newCurveKey = self.insertCurve(str(i))
                newColor = QColor()
                newColor.setHsv(float(i*MAX_HUE_VAL)/1000.0, 255, 255)
                self.setCurvePen(newCurveKey, QPen(newColor))
                self.setCurveData(newCurveKey, [x0,x1], [y,y])
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        
    def getOptimalSeparation(self, attrCount, className, kNeighbours, updateProgress = None):
        if className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Continuous:
            print "incorrect class name for computing optimal ordering"
            return None

        # define lenghts and variables
        dataSize = len(self.rawdata)
        attrCount = len(self.rawdata.domain.attributes)
        classValsCount = len(self.rawdata.domain[className].values)

        fullList = []
        tempValue= 0
        testIndex = 0
        totalTestCount = attrCount * (attrCount-1) / 2
        print "---------------------"
        print "total possibilities: ", str(totalTestCount)

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])

        classValues = list(self.rawdata.domain[className].values)
        classValNum = len(classValues)
        t = time.time()

        

        for x in range(attrCount):
            for y in range(x+1, attrCount):
                testIndex += 1
                if updateProgress != None:
                    updateProgress(testIndex, totalTestCount)
                
                tempValue = 0
                table = orange.ExampleTable(domain)

                for i in range(dataSize):
                    self.application.processEvents(5000)
                    xValue = self.noJitteringScaledData[x][i]
                    yValue = self.noJitteringScaledData[y][i]
                    if xValue == '?' or yValue == '?': continue
                    
                    example = orange.Example(domain, [xValue, yValue, self.rawdata[i][className]])
                    table.append(example)

                """
                exampleDist = orange.ExamplesDistanceConstructor_Euclidean()
                near = orange.FindNearestConstructor_BruteForce(table, distanceConstructor = exampleDist)
                euclidean = orange.ExamplesDistance_Euclidean()
                euclidean.normalizers = [1,1]   # our table has attributes x,y, and class
                for i in range(len(table)):
                    prob = [0]*classValNum
                    neighbours = near(kNeighbours, table[i])
                    for neighbour in neighbours:
                        dist = euclidean(table[i], neighbour)
                        val = math.exp(-(dist*dist))
                        index = classValues.index(neighbour.getclass().value)
                        prob[index] += val

                    # calculate sum for normalization
                    sum = 0
                    for val in prob: sum += val
                    
                    index = classValues.index(table[i].getclass().value)
                    tempValue += float(prob[index])/float(sum)
                """

                # to bo delalo, ko bo popravljen orangov kNNLearner
                classValues = list(self.rawdata.domain[className].values)
                knn = orange.kNNLearner(table, k=kNeighbours)
                for j in range(len(table)):
                    out = knn(table[j], orange.GetProbabilities)
                    index = classValues.index(table[j][2].value)
                    tempValue += out[index]

                print "possibility %6d / %d. Nr. of examples: %4d (Accuracy: %2.2f)" % (testIndex, totalTestCount, len(table), tempValue*100.0/float(len(table)) )

                # save the permutation
                tempList = [self.attributeNames[x], self.attributeNames[y]]
                fullList.append((tempValue*100.0/float(len(table)), len(table), tempList))

        print "------------------------------"
        secs = time.time() - t
        print "Used time: %d min, %d sec" %(secs/60, secs%60)

        return fullList

    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWScatterPlotGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
