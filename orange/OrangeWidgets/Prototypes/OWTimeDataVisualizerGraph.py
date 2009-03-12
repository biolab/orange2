from OWGraph import *
from orngScaleScatterPlotData import *

DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

MIN_SHAPE_SIZE = 6

###########################################################################################
##### CLASS : OWTIMEDATAVISUALIZERGRAPH
###########################################################################################
class OWTimeDataVisualizerGraph(OWGraph, orngScaleScatterPlotData):
    def __init__(self, scatterWidget, parent = None, name = "None"):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)
        orngScaleScatterPlotData.__init__(self)
        self.enableGridXB(0)
        self.enableGridYL(0)

        self.drawLines = 1
        self.drawPoints = 0
        self.trackExamples = 1

        self.pointWidth = 2
        self.optimizedDrawing = 0
        self.showAxisScale = 1
        self.showXaxisTitle = 1
        self.showYLaxisTitle = 1
        self.showLegend = 1
        self.tooltipKind = 1
        self.showFilledSymbols = 1
        self.showGrayRects = 1

        self.timeAttr = None
        self.colorAttr = None
        self.attributes = []
        self.shownAttributeIndices = []

        self.scatterWidget = scatterWidget
        self.enableWheelZoom = 1
        self.mouseMoveEventHandler = self.graphOnMouseMoved


    def setData(self, data, subsetData = None, **args):
        OWGraph.setData(self, data)
        orngScaleScatterPlotData.setData(self, data, subsetData, **args)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, **args):
        self.removeDrawingCurves(removeLegendItems = 0)  # my function, that doesn't delete selection curves
        self.verticalLineCurve = None
        self.detachItems(QwtPlotItem.Rtti_PlotMarker)

        self.tips.removeAll()

        if self.noJitteringScaledData == None or len(self.noJitteringScaledData) == 0:
            self.setXaxisTitle(""); self.setYLaxisTitle("")
            return

        (xVarMin, xVarMax) = self.attrValues[self.timeAttr]
        xVar = max(xVarMax - xVarMin, 1e-10)

        timeAttrIndex = self.attributeNameIndex[self.timeAttr]
        colorIndex = self.colorAttr in ["", None, "(Same color)"] and -1 or self.attributeNameIndex[self.colorAttr]

        # if we have some subset data then we show the examples in the data set with full symbols, others with empty
        haveSubsetData = bool(self.rawSubsetData and self.rawData and self.rawSubsetData.domain.checksum() == self.rawData.domain.checksum())
        showContinuousColorLegend = self.showLegend and colorIndex != -1 and self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous

        if args.get("setScale", 0):
            self.setAxisScale(QwtPlot.xBottom, xVarMin, xVarMax + showContinuousColorLegend * 0.05 * xVar, 0)
            self.setYLlabels([self.rawData.domain[ind].name for ind in self.shownAttributeIndices[::-1]])
            self.setAxisScale(QwtPlot.yLeft, -0.5, len(self.shownAttributeIndices)-0.5, 1)

        timeAttrMin, timeAttrMax = self.originalData[timeAttrIndex].min(), self.originalData[timeAttrIndex].max()
        if self.showGrayRects:
            diff = timeAttrMax-timeAttrMin
            m = timeAttrMin - diff
            M = timeAttrMax + diff
            for i in range((len(self.shownAttributeIndices)+1)/2):
                RectangleCurve(QPen(Qt.NoPen), QBrush(QColor(0,0,0, 20)), [m, M, M, m], [2*i-0.5, 2*i-0.5, 2*i+0.5, 2*i+0.5]).attach(self)

        self.setXaxisTitle(self.timeAttr)
        self.setYLaxisTitle("")

        if haveSubsetData:
            subsetTimeValues = [ex[timeAttrIndex].value for ex in self.rawSubsetData]
        else:
            subsetTimeValues = []

        numAttrs = len(self.shownAttributeIndices)

        # ##############################################################
        # no coloring or discrete color index
        if colorIndex == -1 or self.rawData.domain[colorIndex].varType == orange.VarTypes.Discrete:
            if colorIndex != -1:
                classCount = len(colorIndices)
                colorIndices = getVariableValueIndices(self.dataDomain[colorIndex])
                validData = self.getValidList([timeAttrIndex, colorIndex])      # get examples that have valid data for each used attribute
            else:
                classCount = 1
                validData = self.getValidList([timeAttrIndex])      # get examples that have valid data for each used attribute
            self.discPalette.setNumberOfColors(classCount)

            xs = [[[] for a in range(numAttrs)] for i in range(classCount)]
            ys = [[[] for a in range(numAttrs)] for i in range(classCount)]
            index = 0
            wantLines = self.drawLines and not haveSubsetData and colorIndex == -1
            validTimeData = 1
            exTimeVal = -1e30
            for i, ex in enumerate(self.rawData):
                if not validData[i]: continue    # skip if missing value at time attribute
                timeValue = int(ex[timeAttrIndex].value)
                if timeValue in subsetTimeValues: continue
                if colorIndex != -1: index = colorIndices[self.rawData[i][colorIndex].value]
                elif wantLines:
                    if exTimeVal > timeValue:       # check if the time values are always increasing in the data set
                        validTimeData = 0
                    exTimeVal = timeValue
                
                for j, attrInd in enumerate(self.shownAttributeIndices):
                    if not self.validDataArray[attrInd][i]: continue
                    yValue = numAttrs-j-1 -0.4 + 0.8*self.noJitteringScaledData[attrInd][i]
                    xs[index][j].append(timeValue)
                    ys[index][j].append(yValue)
                    self.tips.addToolTip(timeValue, yValue, i)

            # go over data for each class value (different colors)
            for i in range(classCount):
                if colorIndex == -1:
                    color = haveSubsetData and QColor(Qt.lightGray) or QColor(Qt.black) 
                else: 
                    color = QColor(self.discPalette[i]) or QColor(Qt.black)
                color.setAlpha(self.alphaValue)

                # go over each attribute (one line)
                for j in range(len(self.shownAttributeIndices)):
                    if wantLines and validTimeData:
                        self.addCurve("", color, color, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = xs[i][j], yData = ys[i][j])
                    if not (wantLines and validTimeData) or self.drawPoints:
                        self.addCurve("", color, color, self.pointWidth, symbol = self.curveSymbols[0], xData = xs[i][j], yData = ys[i][j], showFilledSymbols = (colorIndex == -1) or not haveSubsetData and self.showFilledSymbols)


            # ##############################################################
            # subset data + discrete color index
            if haveSubsetData:
                if colorIndex != -1:
                    validSubsetData = self.getValidSubsetList([timeAttrIndex, colorIndex])
                else:
                    validSubsetData = self.getValidSubsetList([timeAttrIndex])
                xs = [[] for i in range(classCount)]
                ys = [[] for i in range(classCount)]
                for i, ex in enumerate(self.rawSubsetData):
                    if not validSubsetData[i]: continue    # skip if missing value at time attribute
                    timeValue = int(ex[timeAttrIndex].value)
                    if colorIndex != -1: index = colorIndices[self.rawSubsetData[i][colorIndex].value]

                    for j, attrInd in enumerate(self.shownAttributeIndices):
                        if not self.validSubsetDataArray[attrInd][i]: continue
                        yValue = numAttrs-j-1 - 0.4 + 0.8*self.noJitteringScaledSubsetData[attrInd][i]
                        xs[index].append(timeValue)
                        ys[index].append(yValue)
                        self.tips.addToolTip(timeValue, yValue, -i-1)

                for i in range(classCount):
                    if colorIndex == -1:
                        color = QColor(Qt.black) 
                    else: 
                        color = QColor(self.discPalette[i]) or QColor(Qt.black)
                    color.setAlpha(self.alphaValue)
                    key = self.addCurve("", color, color, self.pointWidth, symbol = self.curveSymbols[0], xData = xs[i], yData = ys[i], showFilledSymbols = 1)

        # ##############################################################
        # continuous color index
        else:
            # ##############################################################
            # main data + continuous color index
            xPointsToAdd = {}
            yPointsToAdd = {}
            validData = self.getValidList([timeAttrIndex, colorIndex])      # get examples that have valid data for each used attribute
            for i, ex in enumerate(self.rawData):
                if not validData[i]: continue    # skip if missing value at time attribute
                timeValue = int(ex[timeAttrIndex].value)
                if timeValue in subsetTimeValues: continue
                color = self.contPalette.getRGB(self.noJitteringScaledData[colorIndex][i])
                x = []
                y = []

                for j, attrInd in enumerate(self.shownAttributeIndices):
                    if not self.validDataArray[attrInd][i]: continue
                    yValue = numAttrs-j-1 -0.4 + 0.8*self.noJitteringScaledData[attrInd][i]
                    x.append(timeValue)
                    y.append(yValue)
                    self.tips.addToolTip(timeValue, yValue, i)
                xPointsToAdd[color] = xPointsToAdd.get(color, []) + x
                yPointsToAdd[color] = yPointsToAdd.get(color, []) + y

            for i, color in enumerate(xPointsToAdd.keys()):
                c = QColor(*color)
                c.setAlpha(self.alphaValue)
                self.addCurve("", c, c, self.pointWidth, symbol = self.curveSymbols[0], xData = xPointsToAdd[color], yData = yPointsToAdd[color], showFilledSymbols = not haveSubsetData and self.showFilledSymbols)

            # ##############################################################
            # subset data + continuous color index

            if haveSubsetData:
                xPointsToAdd, yPointsToAdd = {}, {}
                validSubsetData = self.getValidSubsetList([timeAttrIndex, colorIndex])
                for i, ex in enumerate(self.rawSubsetData):
                    if not validSubsetData[i]: continue    # skip if missing value at time attribute
                    timeValue = int(ex[timeAttrIndex].value)
                    index = colorIndices[self.rawSubsetData[i][colorIndex].value]
                    x, y = [], []

                    for j, attrInd in enumerate(self.shownAttributeIndices):
                        if not self.validSubsetDataArray[attrInd][i]: continue
                        yValue = numAttrs-j-1 - 0.4 + 0.8*self.noJitteringScaledSubsetData[attrInd][i]
                        x.append(timeValue)
                        y.append(yValue)
                    xPointsToAdd[color] = xPointsToAdd.get(color, []) + x
                    yPointsToAdd[color] = yPointsToAdd.get(color, []) + y


                for i, color in enumerate(xPointsToAdd.keys()):
                    c = QColor(*color)
                    c.setAlpha(self.alphaValue)
                    self.addCurve("", c, c, self.pointWidth, symbol = self.curveSymbols[0], xData = xPointsToAdd[color], yData = yPointsToAdd[color], showFilledSymbols = 1)

        # ##############################################################
        # show legend if necessary
        if self.showLegend:
            self.legend().clear()
            if colorIndex != -1:
                if self.rawData.domain[colorIndex].varType == orange.VarTypes.Discrete:
                    varValues = getVariableValuesSorted(self.dataDomain[colorIndex])
                    for ind in range(len(varValues)):
                        self.addCurve(self.rawData.domain[colorIndex].name + "=" + varValues[ind], self.discPalette[ind], self.discPalette[ind], enableLegend = 1)
    
                elif self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous:
                    xVar = (xVarMax - xVarMin)
                    x0 = xVarMax + 0.02 * xVar
                    x1 = x0 + xVar*2.5/100.0
                    count = 200
                    height = numAttrs / float(count)
                    xs = [x0, x1, x1, x0]
    
                    for i in range(count):
                        y = i*numAttrs/float(count) - 0.5
                        col = self.contPalette[i/float(count)]
                        col.setAlpha(self.alphaValue)
                        curve = PolygonCurve(QPen(col), QBrush(col), xs, [y,y, y+height, y+height])
                        curve.attach(self)
    
                    # add markers for min and max value of color attribute
                    (colorVarMin, colorVarMax) = self.attrValues[self.rawData.domain[colorIndex].name]
                    self.addMarker("%s = %%.%df" % (self.rawData.domain[colorIndex].name, self.rawData.domain[colorIndex].numberOfDecimals) % (colorVarMin), x0 - xVar*1./100.0, -0.5  + numAttrs*0.04, Qt.AlignLeft)
                    self.addMarker("%s = %%.%df" % (self.rawData.domain[colorIndex].name, self.rawData.domain[colorIndex].numberOfDecimals) % (colorVarMax), x0 - xVar*1./100.0, numAttrs -0.5 - + numAttrs*0.04, Qt.AlignLeft)

        self.replot()


    # override the default buildTooltip function defined in OWGraph
    def buildTooltip(self, exampleIndex):
        if exampleIndex < 0:
            data = self.rawSubsetData
            index = -exampleIndex - 1
        else:
            data = self.rawData
            index = exampleIndex
            
        if self.tooltipKind == VISIBLE_ATTRIBUTES:
            if self.attributeNameIndex[self.timeAttr] in self.shownAttributeIndices:
                text = self.getExampleTooltipText(data[index], self.shownAttributeIndices)
            else:
                text = self.getExampleTooltipText(data[index], [self.attributeNameIndex[self.timeAttr]] + self.shownAttributeIndices)
        elif self.tooltipKind == ALL_ATTRIBUTES:        
            text = self.getExampleTooltipText(data[index], range(len(data.domain)))
        return text


    # ##############################################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList):
        [xAttr, yAttr] = attrList
        #if not self.rawData: return (None, None, None)
        if not self.rawData: return (None, None)
        if not self.selectionCurveList: return (None, self.rawData)       # if no selections exist

        selIndices, unselIndices = self.getSelectionsAsIndices(attrList)

        selected = self.rawData.selectref(selIndices)
        unselected = self.rawData.selectref(unselIndices)

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None

        return (selected, unselected)

    def staticMouseClick(self, e):
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            if self.tempSelectionCurve: self.tempSelectionCurve.detach()
            self.tempSelectionCurve = None
            canvasPos = self.canvas().mapFrom(self, e.pos())
            x = self.invTransform(QwtPlot.xBottom, canvasPos.x())
            y = self.invTransform(QwtPlot.yLeft, canvasPos.y())
            diffX = (self.axisScaleDiv(QwtPlot.xBottom).hBound() -  self.axisScaleDiv(QwtPlot.xBottom).lBound()) / 2.

            xmin = x - (diffX/2.) * (x - self.axisScaleDiv(QwtPlot.xBottom).lBound()) / diffX
            xmax = x + (diffX/2.) * (self.axisScaleDiv(QwtPlot.xBottom).hBound() - x) / diffX
            ymin = self.axisScaleDiv(QwtPlot.yLeft).hBound() 
            ymax = self.axisScaleDiv(QwtPlot.yLeft).lBound()

            self.zoomStack.append((self.axisScaleDiv(QwtPlot.xBottom).lBound(), self.axisScaleDiv(QwtPlot.xBottom).hBound(), self.axisScaleDiv(QwtPlot.yLeft).lBound(), self.axisScaleDiv(QwtPlot.yLeft).hBound()))
            self.setNewZoom(xmin, xmax, ymax, ymin)

            return 1
        else:
            return 0

    def getSelectionsAsIndices(self, attrList, validData = None):
        [xAttr, yAttr] = attrList
        if not self.rawData: return [], []

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        if validData == None:
            validData = self.getValidList(attrIndices)

        (xArray, yArray) = self.getXYDataPositions(xAttr, yAttr)

        return self.getSelectedPoints(xArray, yArray, validData)


    def onMouseReleased(self, e):
        OWGraph.onMouseReleased(self, e)
        self.updateLayout()

    def graphOnMouseMoved(self, e):
        canvasPos = self.canvas().mapFrom(self, e.pos())
        xFloat = self.invTransform(QwtPlot.xBottom, canvasPos.x())
        if getattr(self, "verticalLineCurve", None) != None:
            self.verticalLineCurve.detach()
            self.verticalLineCurve = None
            
        if self.trackExamples:
            self.verticalLineCurve = self.addCurve("", QColor(Qt.blue), QColor(Qt.blue), 1, xData = [xFloat, xFloat], yData = [-0.5, len(self.shownAttributeIndices)-0.5], style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol)
            self.replot()
        return 0


if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWScatterPlotGraph(None)
    c.show()
    a.exec_()
