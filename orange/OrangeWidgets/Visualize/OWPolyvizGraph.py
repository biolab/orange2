from OWVisGraph import *
from copy import copy, deepcopy
import time
from OWkNNOptimization import *
import math
#import orange

# ####################################################################
# get a list of all different permutations
def getPermutationList(elements, tempPerm, currList, checkReverse):
    for i in range(len(elements)):
        el =  elements[i]
        elements.remove(el)
        tempPerm.append(el)
        getPermutationList(elements, tempPerm, currList, checkReverse)

        elements.insert(i, el)
        tempPerm.pop()

    if elements == []:
        temp = copy(tempPerm)
        # in tempPerm we have a permutation. Check if it already exists in the currList
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return


        if checkReverse == 1:
            # also try the reverse permutation
            temp.reverse()
            for i in range(len(temp)):
                el = temp.pop()
                temp.insert(0, el)
                if str(temp) in currList: return
        currList[str(tempPerm)] = copy(tempPerm)

def fact(i):
        ret = 1
        while i > 1:
            ret = ret*i
            i -= 1
        return ret
    
# return number of combinations where we select "select" from "total"
def combinations(select, total):
    return fact(total)/ (fact(total-select)*fact(select))


SYMBOL = 0
PENCOLOR = 1
BRUSHCOLOR = 2
XANCHORS = 3
YANCHORS = 4

LINE_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

TOOLTIPS_SHOW_DATA = 0
TOOLTIPS_SHOW_SPRINGS = 1

###########################################################################################
##### CLASS : OWPolyvizGRAPH
###########################################################################################
class OWPolyvizGraph(OWVisGraph):
    def __init__(self, polyvizWidget, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.attrLocalValues = {}
        self.lineLength = 2*0.05
        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.startTime = time.time()
        self.enhancedTooltips = 1
        self.kNNOptimization = None
        self.polyvizWidget = polyvizWidget
        self.useDifferentSymbols = 0
        self.useDifferentColors = 1
        self.tooltipKind = 0        # index in ["Show line tooltips", "Show visible attributes", "Show all attributes"]
        self.tooltipValue = 0       # index in ["Tooltips show data values", "Tooltips show spring values"]

        self.dataMap = {}        # each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.validData = []
        self.showLegend = 1

    def setLineLength(self, len):
        self.lineLength = len*0.05

    def createAnchors(self, anchorNum):
        anchors = [[],[]]
        for i in range(anchorNum):
            x = math.cos(2*math.pi * float(i) / float(anchorNum)); strX = "%.5f" % (x)
            y = math.sin(2*math.pi * float(i) / float(anchorNum)); strY = "%.5f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))
        return anchors

    def createXAnchors(self, numOfAttrs):
        return Numeric.cos(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))

    def createYAnchors(self, numOfAttrs):
        return Numeric.sin(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))

    #
    # if we use globalScaling we must also create a copy of localy scaled data
    #
    def setData(self, data):
        # first call the original function to scale data
        OWVisGraph.setData(self, data)
        if data == None: return

        for index in range(len(data.domain)):
            if data.domain[index].varType == orange.VarTypes.Discrete:
                values = [0, len(data.domain[index].values)-1]
            else:
                values = [self.domainDataStat[index].min, self.domainDataStat[index].max]
            self.attrLocalValues[data.domain[index].name] = values


    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, attributeReverse, **args):
        #self.removeCurves()
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        self.removeMarkers()
        self.tips.removeAll()
    
        # initial var values
        self.showKNNModel = 0
        self.showCorrect = 1
        self.__dict__.update(args)

        length = len(labels)
        self.dataMap = {}               # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)
        self.XAnchor = self.createXAnchors(length)
        self.YAnchor = self.createYAnchors(length)
        self.shownAttributes = labels
        polyvizLineCoordsX = []; polyvizLineCoordsY = []    # if class is discrete we will optimize drawing by storing computed values and adding less data curves to plot
            
        # we must have at least 3 attributes to be able to show anything
        if self.scaledData == None or len(labels) < 3: self.updateLayout(); return
        dataSize = len(self.rawdata)
        classIsDiscrete = (self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete)
        classNameIndex = -1

        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)

        if self.showLegend: self.setAxisScale(QwtPlot.xBottom, -1.20, 1.25, 1)
        else:               self.setAxisScale(QwtPlot.xBottom, -1.20, 1.20, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.20, 1.20, 1)

        # store indices to shown attributes
        indices = [self.attributeNames.index(label) for label in labels]

        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)

        classNameIndex = self.attributeNames.index(self.rawdata.domain.classVar.name)
        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:        # if we have a discrete class
            valLen = len(self.rawdata.domain.classVar.values)
            classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)    # we create a hash table of variable values and their indices            
        else:    # if we have a continuous class
            valLen = 0

        # will we show different symbols?        
        useDifferentSymbols = 0
        if self.useDifferentSymbols and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and valLen < len(self.curveSymbols): useDifferentSymbols = 1
        
        dataSize = len(self.rawdata)
        blackColor = QColor(0,0,0)
        curveData = [[0, 0, 0, QwtSymbol.Ellipse, blackColor, blackColor, [], []] for i in range(dataSize)]

        
        # ##########
        # draw text at lines
        for i in range(length):
            # print attribute name
            self.addMarker(labels[i], 0.6*(self.XAnchor[i]+ self.XAnchor[(i+1)%length]), 0.6*(self.YAnchor[i]+ self.YAnchor[(i+1)%length]), Qt.AlignHCenter + Qt.AlignVCenter, bold = 1)

            if self.rawdata.domain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = getVariableValuesSorted(self.rawdata, labels[i])
                count = len(values)
                k = 1.08
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    if attributeReverse[labels[i]] == 0:
                        self.addMarker(values[j], k*(1-pos)*self.XAnchor[i]+k*pos*self.XAnchor[(i+1)%length], k*(1-pos)*self.YAnchor[i]+k*pos*self.YAnchor[(i+1)%length], Qt.AlignHCenter + Qt.AlignVCenter)
                    else:
                        self.addMarker(values[j], k*pos*self.XAnchor[i]+k*(1-pos)*self.XAnchor[(i+1)%length], k*pos*self.YAnchor[i]+k*(1-pos)*self.YAnchor[(i+1)%length], Qt.AlignHCenter + Qt.AlignVCenter)

            else:
                # min and max value
                names = ["%.3f" % (self.attrLocalValues[labels[i]][0]), "%.3f" % (self.attrLocalValues[labels[i]][1])]
                if attributeReverse[labels[i]] == 1: names.reverse()
                self.addMarker(names[0],0.95*self.XAnchor[i]+0.15*self.XAnchor[(i+1)%length], 0.95*self.YAnchor[i]+0.15*self.YAnchor[(i+1)%length], Qt.AlignHCenter + Qt.AlignVCenter)
                self.addMarker(names[1], 0.15*self.XAnchor[i]+0.95*self.XAnchor[(i+1)%length], 0.15*self.YAnchor[i]+0.95*self.YAnchor[(i+1)%length], Qt.AlignHCenter + Qt.AlignVCenter)

        XAnchorPositions = Numeric.zeros([length, dataSize], Numeric.Float)
        YAnchorPositions = Numeric.zeros([length, dataSize], Numeric.Float)
        XAnchor = self.createXAnchors(length)
        YAnchor = self.createYAnchors(length)

        for i in range(length):
            if attributeReverse[labels[i]]:
                Xdata = XAnchor[i] * self.noJitteringScaledData[indices[i]] + XAnchor[(i+1)%length] * (1-self.noJitteringScaledData[indices[i]])
                Ydata = YAnchor[i] * self.noJitteringScaledData[indices[i]] + YAnchor[(i+1)%length] * (1-self.noJitteringScaledData[indices[i]])
            else:
                Xdata = XAnchor[i] * (1-self.noJitteringScaledData[indices[i]]) + XAnchor[(i+1)%length] * self.noJitteringScaledData[indices[i]]
                Ydata = YAnchor[i] * (1-self.noJitteringScaledData[indices[i]]) + YAnchor[(i+1)%length] * self.noJitteringScaledData[indices[i]]
            XAnchorPositions[i] = Xdata
            YAnchorPositions[i] = Ydata

        XAnchorPositions = Numeric.swapaxes(XAnchorPositions, 0,1)
        YAnchorPositions = Numeric.swapaxes(YAnchorPositions, 0,1)
            
        selectedData = Numeric.take(self.scaledData, indices)
        sum_i = Numeric.add.reduce(selectedData)

        # test if there are zeros in sum_i
        if len(Numeric.nonzero(sum_i)) < len(sum_i):
            add = Numeric.where(sum_i == 0, 1.0, 0.0)
            sum_i += add

        x_positions = Numeric.sum(Numeric.swapaxes(XAnchorPositions * Numeric.swapaxes(selectedData, 0,1), 0,1)) * self.scaleFactor / sum_i
        y_positions = Numeric.sum(Numeric.swapaxes(YAnchorPositions * Numeric.swapaxes(selectedData, 0,1), 0,1)) * self.scaleFactor / sum_i
        validData = self.getValidList(indices)      

        for i in range(dataSize):
            if validData[i] == 0: continue                       
            curveData[i][XANCHORS] = XAnchorPositions[i]
            curveData[i][YANCHORS] = YAnchorPositions[i]


        if self.showKNNModel == 1:
            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
            table = orange.ExampleTable(domain)

            # build an example table            
            for i in range(dataSize):
                if validData[i]:
                    table.append(orange.Example(domain, [x_positions[i], y_positions[i], self.rawdata[i].getclass()]))

            kNNValues = self.kNNOptimization.kNNClassifyData(table)
            accuracy = copy(kNNValues)
            measure = self.kNNOptimization.getQualityMeasure()
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                if ((measure == CLASS_ACCURACY or measure == AVERAGE_CORRECT) and self.showCorrect) or (measure == BRIER_SCORE and not self.showCorrect):
                    kNNValues = [1.0 - val for val in kNNValues]
            else:
                if self.showCorrect: kNNValues = [1.0 - val for val in kNNValues]

            # fill and edge color palettes 
            bwColors = ColorPaletteBW(-1, 55, 255)
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous:  classColors = ColorPaletteHSV(-1)
            else:                                                                   classColors = ColorPaletteHSV(len(classValueIndices))
            
            if table.domain.classVar.varType == orange.VarTypes.Continuous: preText = 'Mean square error : '
            else:
                if measure == CLASS_ACCURACY:    preText = "Classification accuracy : "
                elif measure == AVERAGE_CORRECT: preText = "Average correct classification : "
                else:                            preText = "Brier score : "

            for i in range(len(table)):
                fillColor = bwColors.getColor(kNNValues[i])
                edgeColor = classColors.getColor(classValueIndices[table[i].getclass().value])
                key = self.addCurve(str(i), fillColor, edgeColor, self.pointWidth, xData = [table[i][0].value], yData = [table[i][1].value])
                self.addAnchorLine(x_positions[i], y_positions[i], curveData[i][XANCHORS], curveData[i][YANCHORS], fillColor, i, length)

        # CONTINUOUS class 
        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous:
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = QColor(0,0,0)
                if self.useDifferentColors: newColor.setHsv(self.coloringScaledData[classNameIndex][i], 255, 255)
                curveData[i][PENCOLOR] = newColor
                curveData[i][BRUSHCOLOR] = newColor

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveData[i][SYMBOL], xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], curveData[i][XANCHORS], curveData[i][YANCHORS], newColor, i)
                self.addAnchorLine(x_positions[i], y_positions[i], curveData[i][XANCHORS], curveData[i][YANCHORS], newColor, i, length)

        # DISCRETE class + optimize drawing
        elif self.optimizedDrawing:
            pos = [[ [] , [], [],  [], [] ] for i in range(valLen)]
            colors = ColorPaletteHSV(valLen)
            
            for i in range(dataSize):
                if not validData[i]: continue
                pos[classValueIndices[self.rawdata[i].getclass().value]][0].append(x_positions[i])
                pos[classValueIndices[self.rawdata[i].getclass().value]][1].append(y_positions[i])
                pos[classValueIndices[self.rawdata[i].getclass().value]][2].append(i)
                pos[classValueIndices[self.rawdata[i].getclass().value]][3].append(curveData[i][XANCHORS])
                pos[classValueIndices[self.rawdata[i].getclass().value]][4].append(curveData[i][YANCHORS])
                if self.useDifferentColors: self.addAnchorLine(x_positions[i], y_positions[i], curveData[i][XANCHORS], curveData[i][YANCHORS], colors.getColor(classValueIndices[self.rawdata[i].getclass().value]), i, length)
                else: self.addAnchorLine(x_positions[i], y_positions[i], curveData[i][XANCHORS], curveData[i][YANCHORS], QColor(0,0,0), i, length)

            for i in range(valLen):
                newColor = colors[i]
                if not self.useDifferentColors: newColor = QColor(0,0,0)
                
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[i]
                else: curveSymbol = self.curveSymbols[0]

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = pos[i][0], yData = pos[i][1])
                for k in range(len(pos[i][0])):
                    self.addTooltipKey(pos[i][0][k], pos[i][1][k], pos[i][3][k], pos[i][4][k], newColor, pos[i][2][k])

        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            colors = ColorPaletteHSV(valLen)
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = colors[classValueIndices[self.rawdata[i].getclass().value]]
                if not self.useDifferentColors: newColor = QColor(0,0,0)
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[classValueIndices[self.rawdata[i].getclass().value]]
                else: curveSymbol = self.curveSymbols[0]
                self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], curveData[i][XANCHORS], curveData[i][YANCHORS], newColor, i)
                self.addAnchorLine(x_positions[i], y_positions[i], curveData[i][XANCHORS], curveData[i][YANCHORS], newColor, i, length)

        # ##########
        # draw polygon
        xdata = [x for x in self.XAnchor]; xdata.append(xdata[0])
        ydata = [y for y in self.YAnchor]; ydata.append(ydata[0])

        newCurveKey = self.addCurve("polygon", QColor(0,0,0), QColor(0,0,0), 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata, yData = ydata)
        pen = self.curve(newCurveKey).pen(); pen.setWidth(2); self.curve(newCurveKey).setPen(pen)


        #################
        # draw the legend
        if self.showLegend:
            # show legend for discrete class
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                self.addMarker(self.rawdata.domain.classVar.name, 0.87, 1.06, Qt.AlignLeft)
                    
                classVariableValues = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
                classColors = ColorPaletteHSV(len(classVariableValues))
                for index in range(len(classVariableValues)):
                    color = classColors.getColor(index)
                    if not self.useDifferentColors: color = QColor(0,0,0)
                    y = 1.0 - index * 0.05

                    if not self.useDifferentSymbols:  curveSymbol = self.curveSymbols[0]
                    else:                             curveSymbol = self.curveSymbols[index]

                    self.addCurve(str(index), color, color, self.pointWidth, symbol = curveSymbol, xData = [0.95, 0.95], yData = [y, y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft + Qt.AlignHCenter)
            # show legend for continuous class
            else:
                xs = [1.15, 1.20]
                colors = ColorPaletteHSV(-1)
                for i in range(1000):
                    y = -1.0 + i*2.0/1000.0
                    newCurveKey = self.insertCurve(str(i))
                    self.setCurvePen(newCurveKey, QPen(colors.getColor(float(i)/1000.0)))
                    self.setCurveData(newCurveKey, xs, [y,y])

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawdata.domain.classVar.name]
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, minVal), xs[0] - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, maxVal), xs[0] - 0.02, +1.0 - 0.04, Qt.AlignLeft)


    def addAnchorLine(self, x, y, xAnchors, yAnchors, color, index, count):
        for j in range(count):
            dist = EuclDist([x, y], [xAnchors[j] , yAnchors[j]])
            if dist == 0: continue
            kvoc = float(self.lineLength) / dist
            lineX1 = x; lineY1 = y

            # we don't make extrapolation
            if kvoc > 1: lineX2 = lineX1; lineY2 = lineY1
            else:
                lineX2 = (1.0 - kvoc)*xAnchors[j] + kvoc * lineX1
                lineY2 = (1.0 - kvoc)*yAnchors[j] + kvoc * lineY1

            self.addCurve('line' + str(index), color, color, 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = [xAnchors[j], lineX2], yData = [yAnchors[j], lineY2])


    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, xAnchors, yAnchors, color, index):
        dictValue = "%.1f-%.1f"%(x, y)
        if not self.dataMap.has_key(dictValue): self.dataMap[dictValue] = []
        self.dataMap[dictValue].append((x, y, xAnchors, yAnchors, color, index))


    # ##############
    # draw tooltips
    def onMouseMoved(self, e):
        redraw = 0
        if self.tooltipCurveKeys != [] or self.tooltipMarkers != []: redraw = 1
        
        for key in self.tooltipCurveKeys:  self.removeCurve(key)
        for marker in self.tooltipMarkers: self.removeMarker(marker)
        self.tooltipCurveKeys = []
        self.tooltipMarkers = []

        # in case we are drawing a rectangle, we don't draw enhanced tooltips
        # because it would then fail to draw the rectangle
        if self.mouseCurrentlyPressed:
            OWVisGraph.onMouseMoved(self, e)
            if redraw: self.replot()
            return 
            
        xFloat = self.invTransform(QwtPlot.xBottom, e.x())
        yFloat = self.invTransform(QwtPlot.yLeft, e.y())
        dictValue = "%.1f-%.1f"%(xFloat, yFloat)
        if self.dataMap.has_key(dictValue):
            points = self.dataMap[dictValue]
            bestDist = 100.0
            nearestPoint = ()
            for (x_i, y_i, xAnchors, yAnchors, color, index) in points:
                currDist = sqrt((xFloat-x_i)*(xFloat-x_i) + (yFloat-y_i)*(yFloat-y_i))
                if currDist < bestDist:
                    bestDist = currDist
                    nearestPoint = (x_i, y_i, xAnchors, yAnchors, color, index)

            (x_i, y_i, xAnchors, yAnchors, color, index) = nearestPoint
            if self.tooltipKind == LINE_TOOLTIPS and bestDist < 0.05:
                for i in range(len(self.shownAttributes)):
                    
                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x_i, xAnchors[i]], yData = [y_i, yAnchors[i]])
                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = None
                    if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                        marker = self.addMarker(str(self.rawdata[index][self.attributeNames.index(self.shownAttributes[i])]), (x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                        marker = self.addMarker("%.3f" % (self.scaledData[self.attributeNames.index(self.shownAttributes[i])][index]), (x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(12)
                    self.setMarkerFont(marker, font)
                    self.tooltipMarkers.append(marker)
                    
            elif self.tooltipKind == VISIBLE_ATTRIBUTES or self.tooltipKind == ALL_ATTRIBUTES:
                intX = self.transform(QwtPlot.xBottom, x_i)
                intY = self.transform(QwtPlot.yLeft, y_i)
                text = ""
                if self.tooltipKind == VISIBLE_ATTRIBUTES:
                    labels = self.shownAttributes
                else:
                    labels = self.attributeNames

                if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                    text = self.getShortExampleText(self.rawdata, self.rawdata[index], labels)

                elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                    for label in labels: text += "%s = %.3f; " % (label, self.scaledData[self.attributeNames.index(label)][index])

                    # show values of meta attributes
                    if len(self.rawdata.domain.getmetas()) != 0:
                        for m in self.rawdata.domain.getmetas().values():
                            text += "%s = %s; " % (m.name, str(self.rawdata[index][m]))

                        
                self.showTip(intX, intY, text[:-2].replace("; ", "\n"))

        OWVisGraph.onMouseMoved(self, e)
        self.update()
        # -----------------------------------------------------------
        # -----------------------------------------------------------

    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList, attributeReverse):
        (xArray, yArray, validData) = self.createProjection(attrList, attributeReverse)

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
                 
        for i in range(len(self.rawdata)):
            if not validData[i]: continue
            example = orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()])
            table.append(example)
        return self.kNNOptimization.kNNComputeAccuracy(table)


    def generateAttrReverseLists(self, attrList, fullAttribList, tempList):
        if attrList == []: return tempList
        tempList2 = deepcopy(tempList)
        index = fullAttribList.index(attrList[0])
        for list in tempList2: list[index] = 1
        return self.generateAttrReverseLists(attrList[1:], fullAttribList, tempList + tempList2)

   
    # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList, attributeReverse):
        (xArray, yArray, validData) = self.createProjection(attrList, attributeReverse)

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
        validData = self.validData(attrList)
                 
        for i in range(len(self.rawdata)):
            if not validData[i]: continue
            example = orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()])
            table.append(example)

        orange.saveTabDelimited(fileName, table)

    # ####################################
    # create x-y projection of attributes in attrList
    def createProjection(self, attrList, attributeReverse, validData = None, sums = None, scaleFactor = 1.0):
        # store indices to shown attributes
        indices = [self.attributeNames.index(label) for label in attrList]
        length = len(attrList)
        dataSize = len(self.rawdata)

        selectedData = Numeric.take(self.noJitteringScaledData, indices)
        if not sums:
            sums = Numeric.add.reduce(selectedData)

        if not validData:
            validData = self.getValidList(indices)     

        XAnchorPositions = Numeric.zeros([length, dataSize], Numeric.Float)
        YAnchorPositions = Numeric.zeros([length, dataSize], Numeric.Float)
        XAnchor = self.createXAnchors(length)
        YAnchor = self.createYAnchors(length)

        for i in range(length):
            if attributeReverse[attrList[i]]:
                Xdata = XAnchor[i] * selectedData[i] + XAnchor[(i+1)%length] * (1-selectedData[i])
                Ydata = YAnchor[i] * selectedData[i] + YAnchor[(i+1)%length] * (1-selectedData[i])
            else:
                Xdata = XAnchor[i] * (1-selectedData[i]) + XAnchor[(i+1)%length] * selectedData[i]
                Ydata = YAnchor[i] * (1-selectedData[i]) + YAnchor[(i+1)%length] * selectedData[i]
            XAnchorPositions[i] = Xdata
            YAnchorPositions[i] = Ydata

        XAnchorPositions = Numeric.swapaxes(XAnchorPositions, 0,1)
        YAnchorPositions = Numeric.swapaxes(YAnchorPositions, 0,1)
            
        x_positions = Numeric.sum(Numeric.swapaxes(XAnchorPositions * Numeric.swapaxes(selectedData, 0,1), 0,1)) * self.scaleFactor / sums
        y_positions = Numeric.sum(Numeric.swapaxes(YAnchorPositions * Numeric.swapaxes(selectedData, 0,1), 0,1)) * self.scaleFactor / sums

        return (x_positions, y_positions, validData)
    

    # ####################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, attributeReverse):
        if not self.rawdata: return (None, None, None)
        selected = orange.ExampleTable(self.rawdata.domain)
        unselected = orange.ExampleTable(self.rawdata.domain)

        xArray, yArray, validData = self.createProjection(attrList, attributeReverse, scaleFactor = self.scaleFactor)
                 
        for i in range(len(self.rawdata)):
            if not validData[i]: continue
            
            if self.isPointSelected(xArray[i], yArray[i]): selected.append(self.rawdata[i])
            else:                                          unselected.append(self.rawdata[i])

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None
        merged = self.changeClassAttr(selected, unselected)
        return (selected, unselected, merged)


    def createCombinations(self, currCombination, count, attrList, combinations):
        if count > attrList: return combinations

        if count == 0 and len(currCombination) > 2:
            combinations.append(currCombination)
            return combinations

        if len(attrList) == 0: return combinations
        temp = list(currCombination) + [attrList[0][1]]
        temp[0] += attrList[0][0]
        combinations = self.createCombinations(temp, count-1, attrList[1:], combinations)

        combinations = self.createCombinations(currCombination, count, attrList[1:], combinations)
        return combinations


    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    #def getOptimalSeparation(self, attrListLength, attrReverseDict, projections, addResultFunct):
    def getOptimalSeparation(self, attributes, minLength, maxLength, attrReverseDict, addResultFunct):
        dataSize = len(self.rawdata)

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])

        if self.kNNOptimization.getQualityMeasure() == CLASS_ACCURACY: text = "Classification accuracy"
        elif self.kNNOptimization.getQualityMeasure() == AVERAGE_CORRECT: text = "Average correct classification"
        else: text = "Brier score"

        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(minLength, maxLength+1)]

        for z in range(minLength-1, len(attributes)):
            for u in range(minLength-1, maxLength):
                attrListLength = u+1
                
                combinations = self.createCombinations([0.0], u, attributes[:z], [])
                combinations.sort()
                combinations.reverse()

                XAnchors = anchorList[u+1-minLength][0]
                YAnchors = anchorList[u+1-minLength][1]
                
                for attrList in combinations:
                    attrs = attrList[1:] + [attributes[z][1]] # remove the value of this attribute subset
                    indices = [self.attributeNames.index(attr) for attr in attrs]

                    indPermutations = {}
                    getPermutationList(indices, [], indPermutations, attrReverseDict == None)

                    attrReverse = []
                    if attrReverseDict != None: # if we received a dictionary, then we don't reverse attributes
                        temp = [0] * len(self.rawdata.domain)
                        for val in attrReverseDict.keys():
                            temp[self.attributeNames.index(val)] = attrReverseDict[val]
                        attrReverse.append(temp)
                    else:
                        attrReverse = self.generateAttrReverseLists(attrs, self.attributeNames,[[0]*len(self.rawdata.domain)])

                    permutationIndex = 0 # current permutation index
                    totalPermutations = len(indPermutations.values())*len(attrReverse)

                    validData = self.getValidList(indices)                    
                    selectedData = Numeric.take(self.noJitteringScaledData, indices)
                    sum_i = Numeric.add.reduce(selectedData)

                    # test if there are zeros in sum_i
                    if len(Numeric.nonzero(sum_i)) < len(sum_i):
                        add = Numeric.where(sum_i == 0, 1.0, 0.0)
                        sum_i += add
                    
                    count = sum(validData)
                    if count < self.kNNOptimization.minExamples:
                        print "Not enough examples (%s) in example table. Ignoring permutation." % (str(count))
                        self.triedPossibilities += len(indPermutations.keys())
                        if self.polyvizWidget: self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
                        continue                    

                    tempList = []

                    # for every permutation compute how good it separates different classes            
                    for permutation in indPermutations.values():
                        for attrOrder in attrReverse:
                            if self.kNNOptimization.isOptimizationCanceled(): return
                            permutationIndex += 1
                            table = orange.ExampleTable(domain)

                            attrPermutation = [self.attributeNames[val] for val in permutation]
                            attrReverseOrder = {}
                            for i in range(len(attrs)):
                                attrReverseOrder[attrs[i]] = attrOrder[i]
                            x_positions, y_positions, foo = self.createProjection(attrPermutation, attrReverseOrder, validData, sum_i)
    
                            # calculate projections
                            for i in range(dataSize):
                                if not validData[i]: continue                                
                                example = orange.Example(domain, [x_positions[i], y_positions[i], self.rawdata[i].getclass()])
                                table.append(example)

                            accuracy, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                            if table.domain.classVar.varType == orange.VarTypes.Discrete:   print "permutation %6d / %d. %s: %2.2f%%" % (permutationIndex, totalPermutations, text, accuracy)
                            else:                                                           print "permutation %6d / %d. MSE: %2.2f" % (permutationIndex, totalPermutations, accuracy) 
                            
                            # save the permutation
                            tempList.append((accuracy, other_results, len(table), attrPermutation, attrOrder))
                            if not self.kNNOptimization.onlyOnePerSubset:
                                addResultFunct(accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation], attrOrder)

                            self.triedPossibilities += 1
                            self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))

                    if self.kNNOptimization.onlyOnePerSubset:
                        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE: funct = max
                        else: funct = min
                        (acc, other_results, lenTable, attrList, attrOrder) = funct(tempList)
                        addResultFunct(acc, other_results, lenTable, attrList, attrOrder)

    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWPolyvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
