#
# OWScatterPlotGraph.py
#
# the base for scatterplot

from OWVisGraph import *
import time
from orngCI import FeatureByCartesianProduct

class QwtPlotCurvePieChart(QwtPlotCurve):
    def __init__(self, parent = None, text = None):
        QwtPlotCurve.__init__(self, parent, text)
        self.color = Qt.black
        self.penColor = Qt.black

    def draw(self, p, xMap, yMap, f, t):
        # save ex settings
        back = p.backgroundMode()
        pen = p.pen()
        brush = p.brush()

        p.setBackgroundMode(Qt.OpaqueMode)
        #p.setBackgroundColor(self.color)
        for i in range(1, self.dataSize()):
            color = QColor()
            if self.dataSize() < len(self.colorHueValues):
                color.setHsv(self.colorHueValues[i-1] * 360, 255, 255) 
            else:
                color.setHsv(float(i-1*360)/float(self.dataSize()-1), 255, 255)
            p.setBrush(QBrush(color))
            p.setPen(QPen(color))

            factor = self.percentOfTotalData * self.percentOfTotalData
            px1 = xMap.transform(self.x(0)-0.1 - 0.5*factor)
            py1 = yMap.transform(self.x(1)-0.1 - 0.5*factor)
            px2 = xMap.transform(self.x(0)+0.1 + 0.5*factor)
            py2 = yMap.transform(self.x(1)+0.1 + 0.5*factor)
            p.drawPie(px1, py1, px2-px1, py2-py1, self.y(i-1)*16*360, (self.y(i)-self.y(i-1))*16*360)

        # restore ex settings
        p.setBackgroundMode(back)
        p.setPen(pen)
        p.setBrush(brush)



###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
###########################################################################################
class OWScatterPlotGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)

        self.jitterContinuous = 0
        self.enabledLegend = 0
        self.showAttributeValues = 1
        self.showDistributions = 1
        self.toolRects = []
        self.tooltipData = []
        self.showManualAxisScale = 0
        self.kNNOptimization = None

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", showColorLegend = 0, statusBar = None, **args):
        self.clear()
        self.tips.removeAll()
        self.enableLegend(0)
        self.removeTooltips()
        self.statusBar = statusBar
        toolTipList = [xAttr, yAttr]
        if shapeAttr != "" and shapeAttr != "(One shape)": toolTipList.append(shapeAttr)
        if sizeShapeAttr != "" and sizeShapeAttr != "(One size)": toolTipList.append(sizeShapeAttr)

        # initial var values
        self.showKNNModel = 0
        self.showCorrect = 1
        self.__dict__.update(args)

        (xVarMin, xVarMax) = self.attrValues[xAttr]
        (yVarMin, yVarMax) = self.attrValues[yAttr]
        xVar = xVarMax - xVarMin
        yVar = yVarMax - yVarMin
        
        MIN_SHAPE_SIZE = 6
        MAX_SHAPE_DIFF = self.pointWidth

        if len(self.scaledData) == 0: self.updateLayout(); return

        if self.rawdata.domain[xAttr].varType == orange.VarTypes.Continuous:
            self.setXlabels(None)
            if self.showManualAxisScale: self.setAxisScale(QwtPlot.xBottom, xVarMin - (self.jitterSize * xVar / 80.0), xVarMax + (self.jitterSize * xVar / 80.0) + showColorLegend * xVar/20, 1)            
        else:
            self.setXlabels(self.getVariableValuesSorted(self.rawdata, xAttr))
            if self.showDistributions == 1: self.setAxisScale(QwtPlot.xBottom, xVarMin - 0.4, xVarMax + 0.4, 1)
            #else: self.setAxisScale(QwtPlot.xBottom, xVarMin - (self.jitterSize * xVar / 50.0), xVarMax + (self.jitterSize * xVar / 50.0) + showColorLegend * xVar/20, 1)
            else: self.setAxisScale(QwtPlot.xBottom, xVarMin - 0.5, xVarMax + +0.5 + showColorLegend * xVar/20, 1)            

        if self.rawdata.domain[yAttr].varType == orange.VarTypes.Continuous:
            self.setYLlabels(None)
            if self.showManualAxisScale: self.setAxisScale(QwtPlot.yLeft, yVarMin - (self.jitterSize * yVar / 80.0), yVarMax + (self.jitterSize * yVar / 80.0), 1)            
        else:
            self.setYLlabels(self.getVariableValuesSorted(self.rawdata, yAttr))
            if self.showDistributions == 1: self.setAxisScale(QwtPlot.yLeft, yVarMin - 0.4, yVarMax + 0.4, 1)
            #else: self.setAxisScale(QwtPlot.yLeft, yVarMin - (self.jitterSize * yVar / 80.0), yVarMax + (self.jitterSize * yVar / 80.0), 1)
            else: self.setAxisScale(QwtPlot.yLeft, yVarMin - 0.5, yVarMax + 0.5, 1)

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
            colorIndex = self.attributeNames.index(colorAttr)
            
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

        #######
        # show the distributions
        if self.showDistributions == 1 and colorIndex != -1 and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Discrete and self.rawdata.domain[xAttr].varType == orange.VarTypes.Discrete and self.rawdata.domain[yAttr].varType == orange.VarTypes.Discrete and not self.showKNNModel:
            (cart, profit) = FeatureByCartesianProduct(self.rawdata, [self.rawdata.domain[xAttr], self.rawdata.domain[yAttr]])
            tempData = self.rawdata.select(list(self.rawdata.domain) + [cart])
            contXY = orange.ContingencyAttrClass(cart, tempData)   # distribution of X attribute
            xValues = self.getVariableValuesSorted(self.rawdata, xAttr)
            yValues = self.getVariableValuesSorted(self.rawdata, yAttr)
            classValuesSorted = self.getVariableValuesSorted(self.rawdata, colorIndex)
            classValues = list(self.rawdata.domain[colorIndex].values)

            sum = 0
            for table in contXY:
                for val in table: sum += val

            for i in range(len(xValues)):
                for j in range(len(yValues)):
                    try: distribution = contXY[str(xValues[i])+'-'+str(yValues[j])]
                    except: continue
                    tempSum = 0
                    for val in distribution: tempSum += val
                    if tempSum == 0: continue

                    tooltipText = "Nr. of examples: <b>%d</b> (%.2f%%) <br>Distribution:" % (tempSum, 100.0*float(tempSum)/float(sum))
                    out = [0.0]
                    key = self.addCurve(QwtPlotCurvePieChart(self), QColor(), QColor(), 0, style = QwtCurve.UserCurve, symbol = QwtSymbol.None)
                    for classVal in classValuesSorted:
                        val = classValues.index(classVal)
                        out += [out[-1] + float(distribution[val])/float(tempSum)]
                        tooltipText += "<br>%s : <b>%d</b> (%.2f%%)" % (classVal, distribution[val], 100.0*distribution[val]/float(tempSum))
                    self.setCurveData(key, [i, j] + [0]*(len(out)-2), out)
                    self.curve(key).colorHueValues = self.colorHueValues
                    self.curve(key).percentOfTotalData = float(tempSum) / float(sum)
                    self.tooltipData.append((tooltipText, i, j))

        # show normal scatterplot with dots
        else:
            # show quality of knn model with only 2 selected attributes
            if self.showKNNModel == 1:
                # variables and domain for the table
                
                shortData = self.rawdata.select([self.rawdata.domain[xAttr], self.rawdata.domain[yAttr], self.rawdata.domain.classVar])
                shortData = orange.Preprocessor_dropMissing(shortData)
                kNNValues = self.kNNOptimization.kNNClassifyData(shortData)
                if self.showCorrect == 1: kNNValues = [1.0 - val for val in kNNValues]
                for j in range(len(kNNValues)):
                    newColor = QColor(55+kNNValues[j]*200, 55+kNNValues[j]*200, 55+kNNValues[j]*200)
                    key = self.addCurve(str(j), newColor, newColor, self.pointWidth, xData = [shortData[j][0].value], yData = [shortData[j][1].value])

            else:
                self.curveKeys = []
                for i in range(len(self.rawdata)):
                    if self.rawdata[i][xAttr].isSpecial() == 1: continue
                    if self.rawdata[i][yAttr].isSpecial() == 1: continue
                    if colorIndex != -1 and self.rawdata[i][colorIndex].isSpecial() == 1: continue
                    if shapeIndex != -1 and self.rawdata[i][shapeIndex].isSpecial() == 1: continue
                    if sizeShapeIndex != -1 and self.rawdata[i][sizeShapeIndex].isSpecial() == 1: continue
                    
                    if discreteX == 1: x = attrXIndices[self.rawdata[i][xAttr].value] + self.rndCorrection(float(self.jitterSize * xVar) / 100.0)
                    else:              x = self.rawdata[i][xAttr].value + self.jitterContinuous * self.rndCorrection(float(self.jitterSize * xVar) / 100.0)

                    if discreteY == 1: y = attrYIndices[self.rawdata[i][yAttr].value] + self.rndCorrection(float(self.jitterSize * yVar) / 100.0)
                    else:              y = self.rawdata[i][yAttr].value + self.jitterContinuous * self.rndCorrection(float(self.jitterSize * yVar) / 100.0)

                    newColor = QColor(0,0,0)
                    if colorIndex != -1: newColor.setHsv(self.coloringScaledData[colorIndex][i]*360, 255, 255)
                        
                    Symbol = shapeList[0]
                    if shapeIndex != -1: Symbol = shapeList[shapeIndices[self.rawdata[i][shapeIndex].value]]

                    size = self.pointWidth
                    if sizeShapeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeShapeIndex][i] * MAX_SHAPE_DIFF)

                    self.addCurve(str(i), newColor, newColor, size, symbol = Symbol, xData = [x], yData = [y])

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
                for ind in range(len(self.rawdata.domain[colorIndex].values)):
                    newColor = QColor()
                    newColor.setHsv(self.colorHueValues[ind] * 360, 255, 255)
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


        if colorAttr != "" and colorAttr != "(One color)" and showColorLegend == 1 and self.showDistributions == 0 and self.rawdata.domain[colorAttr].varType == orange.VarTypes.Continuous:
            x0 = xVarMax + xVar/100
            x1 = x0 + xVar/20
            for i in range(1000):
                y = yVarMin + i*yVar/1000
                newCurveKey = self.insertCurve(str(i))
                newColor = QColor()
                newColor.setHsv(float(i*self.MAX_HUE_VAL)/1000.0, 255, 255)
                self.setCurvePen(newCurveKey, QPen(newColor))
                self.setCurveData(newCurveKey, [x0,x1], [y,y])

            # add markers for min and max value of color attribute
            (colorVarMin, colorVarMax) = self.attrValues[colorAttr]
            self.addMarker("%s = %.3f" % (colorAttr, colorVarMin), x1 + xVar/50, yVarMin + yVar*0.04, Qt.AlignRight)
            self.addMarker("%s = %.3f" % (colorAttr, colorVarMax), x1 + xVar/50, yVarMin + yVar*0.96, Qt.AlignRight)

        
        self.addTooltips()

    # -----------------------------------------------------------
    # -----------------------------------------------------------


    # compute how good is a specific projection with given xAttr and yAttr
    def getProjectionQuality(self, xAttr, yAttr, className):
        dataSize = len(self.rawdata)
        attrCount = len(self.rawdata.domain.attributes)
        classValsCount = len(self.rawdata.domain[className].values)
        xAttrIndex = self.attributeNames.index(xAttr)
        yAttrIndex = self.attributeNames.index(yAttr)
        tempValue= 0

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])
        table = orange.ExampleTable(domain)

        classValues = list(self.rawdata.domain[className].values)
        classValNum = len(classValues)

        for i in range(dataSize):
            xValue = self.noJitteringScaledData[xAttrIndex][i]
            yValue = self.noJitteringScaledData[yAttrIndex][i]
            if xValue == '?' or yValue == '?': continue
            
            example = orange.Example(domain, [xValue, yValue, self.rawdata[i][className]])
            table.append(example)

        accuracy = self.kNNOptimization.kNNComputeAccuracy(table)
        print "kNeighbours = %3.d - Accuracy: %2.2f" % (self.kNNOptimization.kValue, accuracy)
        return accuracy


        
    def getOptimalSeparation(self, attrCount, className, updateProgress = None):
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
                    xValue = self.noJitteringScaledData[x][i]
                    yValue = self.noJitteringScaledData[y][i]
                    if xValue == '?' or yValue == '?': continue
                    
                    example = orange.Example(domain, [xValue, yValue, self.rawdata[i][className]])
                    table.append(example)

                if len(table) < self.kNNOptimization.minExamples: print "possibility %6d / %d. Not enough examples (%d)" % (testIndex, totalTestCount, len(table)); continue

                accuracy = self.kNNOptimization.kNNComputeAccuracy(table)
                if table.domain.classVar.varType == orange.VarTypes.Discrete:
                    print "permutation %6d / %d. Accuracy: %2.2f%%" % (testIndex, totalTestCount, accuracy)
                else:
                    print "permutation %6d / %d. MSE: %2.2f" % (testIndex, totalTestCount, accuracy) 

                # save the permutation
                fullList.append((accuracy, len(table), [self.attributeNames[x], self.attributeNames[y]]))

        print "------------------------------"
        secs = time.time() - t
        print "Used time: %d min, %d sec" %(secs/60, secs%60)

        return fullList


    def addTooltips(self):
        for (text, i, j) in self.tooltipData:
            x_1 = self.transform(QwtPlot.xBottom, i-0.5)
            x_2 = self.transform(QwtPlot.xBottom, i+0.5)
            y_1 = self.transform(QwtPlot.yLeft, j+0.5)
            y_2 = self.transform(QwtPlot.yLeft, j-0.5)
            rect = QRect(x_1, y_1, x_2-x_1, y_2-y_1)
            self.toolRects.append(rect)            
            QToolTip.add(self, rect, text)
            

    def removeTooltips(self):
        for rect in self.toolRects:
            QToolTip.remove(self, rect)
        self.toolRects = []

    def updateLayout(self):
        OWVisGraph.updateLayout(self)
        self.removeTooltips()
        self.addTooltips()

    def onMouseReleased(self, e):
        OWVisGraph.onMouseReleased(self, e)
        self.updateLayout()

        
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWScatterPlotGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
