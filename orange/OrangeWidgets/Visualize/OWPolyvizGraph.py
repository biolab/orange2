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
        indices = [self.attributeNameIndex[label] for label in labels]

        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)

        classNameIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
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
            colors = ColorPaletteHSV()
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = QColor(0,0,0)
                if self.useDifferentColors: newColor.setHsv(self.noJitteringScaledData[classNameIndex][i] * colors.maxHueVal, 255, 255)
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
                        marker = self.addMarker(str(self.rawdata[index][self.attributeNameIndex[self.shownAttributes[i]]]), (x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                        marker = self.addMarker("%.3f" % (self.scaledData[self.attributeNameIndex[self.shownAttributes[i]]][index]), (x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
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
                    for label in labels: text += "%s = %.3f; " % (label, self.scaledData[self.attributeNameIndex[label]][index])

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
        return self.kNNOptimization.kNNComputeAccuracy(self.createProjectionAsExampleTable([self.attributeNameIndex[attr] for attr in attrList], [attributeReverse[attr] for attr in attrList]))

    def generateAttrReverseLists(self, attrList, fullAttribList, tempList):
        if attrList == []: return tempList
        tempList2 = deepcopy(tempList)
        index = fullAttribList.index(attrList[0])
        for list in tempList2: list[index] = 1
        return self.generateAttrReverseLists(attrList[1:], fullAttribList, tempList + tempList2)

   
    # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList, attributeReverse):
        orange.saveTabDelimited(fileName, self.createProjectionAsExampleTable([self.attributeNameIndex[i] for i in attrList], [attributeReverse[attr] for attr in attrList]))

    def createProjectionAsExampleTable(self, attrList, attributeReverse, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, domain = None, scaleFactor = 1.0, jitterSize = 0.0):
        if not domain: domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        data = self.createProjectionAsNumericArray(attrList, attributeReverse, validData, classList, sum_i, XAnchors, YAnchors, scaleFactor, jitterSize)
        return orange.ExampleTable(domain, data)

    def createProjectionAsNumericArray(self, attrIndices, attributeReverse, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, scaleFactor = 1.0, jitterSize = 0.0):
        if not validData: validData = self.getValidList(attrIndices)

        selectedData = Numeric.compress(validData, Numeric.take(self.noJitteringScaledData, attrIndices))
        
        if not classList:
            classList = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
            classList = Numeric.compress(validData, classList)
            
        if not sum_i: sum_i = self._getSum_i(selectedData)
        if not XAnchors or not YAnchors:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))

        xanchors = Numeric.zeros(Numeric.shape(selectedData), Numeric.Float)
        yanchors = Numeric.zeros(Numeric.shape(selectedData), Numeric.Float)
        length = len(attrIndices)

        for i in range(length):
            if attributeReverse[i]:
                xanchors[i] = selectedData[i] * XAnchors[i] + (1-selectedData[i]) * XAnchors[(i+1)%length]
                yanchors[i] = selectedData[i] * YAnchors[i] + (1-selectedData[i]) * YAnchors[(i+1)%length]
            else:
                xanchors[i] = (1-selectedData[i]) * XAnchors[i] + selectedData[i] * XAnchors[(i+1)%length]
                yanchors[i] = (1-selectedData[i]) * YAnchors[i] + selectedData[i] * YAnchors[(i+1)%length]

        x_positions = Numeric.sum(Numeric.multiply(xanchors, selectedData)) / sum_i
        y_positions = Numeric.sum(Numeric.multiply(yanchors, selectedData)) / sum_i

        if scaleFactor != 1.0:
            x_positions = x_positions * scaleFactor
            y_positions = y_positions * scaleFactor
        if jitterSize > 0.0:
            x_positions += (RandomArray.random(len(x_positions))-0.5)*jitterSize
            y_positions += (RandomArray.random(len(y_positions))-0.5)*jitterSize
        
        return Numeric.transpose(Numeric.array((x_positions, y_positions, classList)))

    # ##############################################################
    # function to compute the sum of all values for each element in the data. used to normalize.
    def _getSum_i(self, data):
        sum_i = Numeric.add.reduce(data)
        if len(Numeric.nonzero(sum_i)) < len(sum_i):    # test if there are zeros in sum_i
            sum_i += Numeric.where(sum_i == 0, 1.0, 0.0)
        return sum_i 

    # ####################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, attributeReverse):
        if not self.rawdata: return (None, None, None)
        selected = orange.ExampleTable(self.rawdata.domain)
        unselected = orange.ExampleTable(self.rawdata.domain)

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        validData = self.getValidList(attrIndices)
        proj = self.createProjectionAsNumericArray(attrIndices, [attributeReverse[attr] for attr in attrList], validData = validData, scaleFactor = self.scaleFactor)

        missing = 0
        for i in range(len(self.rawdata)):
            if not validData[i]:
                missing += 1
                continue
            
            if self.isPointSelected(proj[i-missing][0], proj[i-missing][1]): selected.append(self.rawdata[i])
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

    def createAttrReverseList(self, attrLen):
        res = [[]]
        for i in range(attrLen):
            res2 = deepcopy(res)
            for l in res: l.append(0)
            for l in res2: l.append(1)
            res += res2
        return res

    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    #def getOptimalSeparation(self, attrListLength, attrReverseDict, projections, addResultFunct):
    def getOptimalSeparation(self, attributes, minLength, maxLength, attrReverseDict, addResultFunct):
        dataSize = len(self.rawdata)
        self.triedPossibilities = 0
        startTime = time.time()
        self.polyvizWidget.progressBarInit()

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        classListFull = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
        allAttrReverse = {}    # dictionary where keys are the number of attributes and the values are dictionaries with all reverse orders for this number of attributes
        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(minLength, maxLength+1)]

        for z in range(minLength-1, len(attributes)):
            for u in range(minLength-1, maxLength):
                attrListLength = u+1
                
                combinations = self.createCombinations([0.0], u, attributes[:z], [])
                combinations.sort()
                combinations.reverse()

                if attrReverseDict == None:
                    if not allAttrReverse.has_key(attrListLength):
                        allAttrReverse[attrListLength] = self.createAttrReverseList(attrListLength)
                    attrReverse = allAttrReverse[attrListLength]

                XAnchors = anchorList[u+1-minLength][0]
                YAnchors = anchorList[u+1-minLength][1]
                
                for attrList in combinations:
                    attrs = attrList[1:] + [attributes[z][1]] # remove the value of this attribute subset
                    indices = [self.attributeNameIndex[attr] for attr in attrs]

                    indPermutations = {}
                    getPermutationList(indices, [], indPermutations, attrReverseDict == None)

                    if attrReverseDict != None: # if we received a dictionary, then we don't reverse attributes 
                        attrReverse = [[attrReverseDict[attr] for attr in attrs]]

                    permutationIndex = 0 # current permutation index
                    totalPermutations = len(indPermutations.values())*len(attrReverse)

                    validData = self.getValidList(indices)
                    classList = Numeric.compress(validData, classListFull)
                    selectedData = Numeric.compress(validData, Numeric.take(self.noJitteringScaledData, indices))
                    sum_i = self._getSum_i(selectedData)

                    if len(selectedData[0]) < self.kNNOptimization.minExamples:
                        print "Not enough examples (%s) in example table. Ignoring permutation." % (str(len(selectedData[0])))
                        self.triedPossibilities += len(indPermutations.keys())
                        self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
                        continue                    

                    tempList = []

                    # for every permutation compute how good it separates different classes            
                    for permutation in indPermutations.values():
                        for attrOrder in attrReverse:
                            if self.kNNOptimization.isOptimizationCanceled():
                                secs = time.time() - startTime
                                self.kNNOptimization.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
                                self.polyvizWidget.progressBarFinished()
                                return
                            permutationIndex += 1

                            table = self.createProjectionAsExampleTable(permutation, attrOrder, validData, classList, sum_i, XAnchors, YAnchors, domain)
                            accuracy, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                        
                            # save the permutation
                            if not self.kNNOptimization.onlyOnePerSubset:
                                addResultFunct(accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation], attrOrder)
                            else:
                                tempList.append((accuracy, other_results, len(table), [self.attributeNames[val] for val in permutation], attrOrder))
                                
                            self.triedPossibilities += 1
                            self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
                            self.kNNOptimization.setStatusBarText("Evaluated %s projections..." % (createStringFromNumber(self.triedPossibilities)))
                            del table

                    if self.kNNOptimization.onlyOnePerSubset:
                        (acc, other_results, lenTable, attrList, attrOrder) = self.kNNOptimization.getMaxFunct()(tempList)
                        addResultFunct(acc, other_results, lenTable, attrList, self.triedPossibilities, attrOrder)

                    del validData, selectedData, classList, sum_i, tempList, indPermutations
                del combinations

        secs = time.time() - startTime
        self.kNNOptimization.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
        self.polyvizWidget.progressBarFinished()

    def optimizeGivenProjection(self, projection, attrReverseList, accuracy, attributes, addResultFunct):
        dataSize = len(self.rawdata)
        classIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
        self.triedPossibilities = 0

        # replace attribute names with indices in domain - faster searching
        attributes = [(val, self.attributeNameIndex[name]) for (val, name) in attributes]
        lenOfAttributes = len(attributes)
        projection = [self.attributeNameIndex[name] for name in projection]

        # variables and domain for the table
        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(3, 50)]
        classListFull = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
        
        optimizedProjection = 1
        while optimizedProjection:
            optimizedProjection = 0
            
            # in the first step try to find a better projection by substituting an existent attribute with a new one
            # in the second step try to find a better projection by adding a new attribute to the circle
            for iteration in range(2):
                if iteration == 1 and optimizedProjection: continue # if we already found a better projection with replacing an attribute then don't try to add a new atribute
                strTotalAtts = createStringFromNumber(lenOfAttributes)
                listOfCanditates = []

                if attrReverseList == None:
                    if not allAttrReverse.has_key(len(projection) + iteration):
                        allAttrReverse[len(projection) + iteration] = self.createAttrReverseList(len(projection) + iteration)
                    attrReverse = allAttrReverse[len(projection) + iteration]
                    
                for (attrIndex, (val, attr)) in enumerate(attributes):
                    if attr in projection: continue

                    if attrReverseList != None:
                        projections = [(copy(projection), copy(attrReverseList)) for i in range(len(projection))]
                    else:
                        rev = [0 for i in range(len(projection))]
                        projections = [(copy(projection), rev) for i in range(len(projection))]
                        
                    if iteration == 0:  # replace one attribute in each projection with attribute attr
                        count = len(projection)
                        for i in range(count): projections[i][0][i] = attr
                    elif iteration == 1:
                        count = len(projection) + 1
                        for i in range(count-1):
                            projections[i][0].insert(i, attr)
                            projections[i][1].insert(i, 0)

                    if attrReverseList != None:
                        projections2 = deepcopy(projections)
                        for i in range(len(projections2)):
                            projections2[i][1][i] = 1 - projections2[i][1][i]
                        projections += projections2

                    if len(anchorList) < count-3: anchorList.append((self.createXAnchors(count), self.createYAnchors(count)))

                    XAnchors = anchorList[count-3][0]
                    YAnchors = anchorList[count-3][1]
                    validData = self.getValidList(projections[0][0])
                    classList = Numeric.compress(validData, classListFull)
                    
                    tempList = []
                    for (testProj, reverse) in projections:
                        if self.kNNOptimization.isOptimizationCanceled(): return

                        table = self.createProjectionAsExampleTable(testProj, reverse, validData, classList, None, XAnchors, YAnchors, domain)
                        acc, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                        
                        # save the permutation
                        tempList.append((acc, other_results, len(table), testProj, reverse))

                        del table
                        self.triedPossibilities += 1
                        qApp.processEvents()        # allow processing of other events
                        if self.kNNOptimization.isOptimizationCanceled(): return
                        

                    # return only the best attribute placements
                    (acc, other_results, lenTable, attrList, reverse) = self.kNNOptimization.getMaxFunct()(tempList)
                    if self.kNNOptimization.getMaxFunct()(acc, accuracy) == acc:
                        addResultFunct(acc, other_results, lenTable, [self.attributeNames[i] for i in attrList], 0, reverse)
                        self.kNNOptimization.setStatusBarText("Found a better projection with accuracy: %2.2f%%" % (acc))
                        optimizedProjection = 1
                        listOfCanditates.append((acc, attrList, reverse))
                    else:
                        self.kNNOptimization.setStatusBarText("Evaluated %s projections (attribute %s/%s). Last accuracy was: %2.2f%%" % (createStringFromNumber(self.triedPossibilities), createStringFromNumber(attrIndex), strTotalAtts, acc))
                        if min(acc, accuracy)/max(acc, accuracy) > 0.98:  # if the found projection is at least 98% as good as the one optimized, add it to the list of projections anyway
                            addResultFunct(acc, other_results, lenTable, [self.attributeNames[i] for i in attrList], 1, reverse)

                    del validData, classList, projections

                # select the best new projection and say this is now our new projection to optimize    
                if len(listOfCanditates) > 0:
                    (accuracy, projection, reverse) = self.kNNOptimization.getMaxFunct()(listOfCanditates)
                    if attrReverseList != None: attrReverseList = reverse
                    self.kNNOptimization.setStatusBarText("Increased accuracy to %2.2f%%" % (accuracy))


    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWPolyvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
