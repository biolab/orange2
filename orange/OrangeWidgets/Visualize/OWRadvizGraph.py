#
# OWRadvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy
import time
from operator import add
from math import *
from OWkNNOptimization import *
import orange



# ####################################################################
# get a list of all different permutations
def getPermutationList(elements, tempPerm, currList):
    for i in range(len(elements)):
        el =  elements[i]
        elements.remove(el)
        tempPerm.append(el)
        getPermutationList(elements, tempPerm, currList)

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
    
# factoriela
def fact(i):
        ret = 1
        for j in range(2, i+1): ret*= j
        return ret
    
# return number of combinations where we select "select" from "total"
def combinations(select, total):
    return fact(total)/ (fact(total-select)*fact(select))


# indices in curveData
SYMBOL = 0
PENCOLOR = 1
BRUSHCOLOR = 2

LINE_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

TOOLTIPS_SHOW_DATA = 0
TOOLTIPS_SHOW_SPRINGS = 1

###########################################################################################
##### CLASS : OWRADVIZGRAPH
###########################################################################################
class OWRadvizGraph(OWVisGraph):
    def __init__(self, radvizWidget, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.startTime = time.time()
        self.p = None
        self.anchorData =[]        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.dataMap = {}        # each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.kNNOptimization = None
        self.radvizWidget = radvizWidget

        self.showLegend = 1
        self.useDifferentSymbols = 0
        self.useDifferentColors = 1
        self.tooltipKind = 0        # index in ["Show line tooltips", "Show visible attributes", "Show all attributes"]
        self.tooltipValue = 0       # index in ["Tooltips show data values", "Tooltips show spring values"]
        self.scaleFactor = 1.0
        self.subsetData = None
        

    # create anchors around the circle
    def createAnchors(self, numOfAttr, labels = None):
        xAnchors = self.createXAnchors(numOfAttr).tolist()
        yAnchors = self.createYAnchors(numOfAttr).tolist()

        if labels:
            return [(xAnchors[i], yAnchors[i], labels[i]) for i in range(numOfAttr)]
        else:
            return [(xAnchors[i], yAnchors[i]) for i in range(numOfAttr)]
        
    def createXAnchors(self, numOfAttrs):
        return Numeric.cos(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))

    def createYAnchors(self, numOfAttrs):
        return Numeric.sin(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))
            
    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels, **args):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        #self.removeCurves()
        self.removeMarkers()

        # initial var values
        self.showKNNModel = 0
        self.showCorrect = 1
        self.__dict__.update(args)

        length = len(labels)
        xs = []
        self.dataMap = {}   # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)

        if self.scaledData == None or len(labels) < 3: self.updateLayout(); return
        
        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
                
        self.setAxisScale(QwtPlot.xBottom, -1.22, 1.22, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)

        # store indices to shown attributes
        indices = [self.attributeNames.index(label) for label in labels]

        self.anchorData = self.createAnchors(length, labels)

        # ##########
        # draw "circle"
        xdata = []; ydata = []
        for i in range(101):
            xdata.append(math.cos(2*math.pi * float(i) / 100.0))
            ydata.append(math.sin(2*math.pi * float(i) / 100.0))
        self.addCurve("circle", QColor(0,0,0), QColor(0,0,0), 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata, yData = ydata)

        # ##########
        # draw dots at anchors
        XAnchors = self.createXAnchors(length)
        YAnchors = self.createYAnchors(length)
        
        self.addCurve("dots", QColor(140,140,140), QColor(140,140,140), 10, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, xData = XAnchors, yData = YAnchors, forceFilledSymbols = 1)

        # ##########
        # draw text at anchors
        for i in range(length):
            self.addMarker(labels[i], XAnchors[i]*1.1, YAnchors[i]*1.04, Qt.AlignHCenter + Qt.AlignVCenter, bold = 1)

        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        # -----------------------------------------------------------
        #  create data curves
        # -----------------------------------------------------------

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

        selectedData = Numeric.take(self.scaledData, indices)
        sum_i = Numeric.add.reduce(selectedData)

        # test if there are zeros in sum_i
        if len(Numeric.nonzero(sum_i)) < len(sum_i):
            add = Numeric.where(sum_i == 0, 1.0, 0.0)
            sum_i += add

        x_positions = Numeric.matrixmultiply(XAnchors, selectedData) * self.scaleFactor / sum_i
        y_positions = Numeric.matrixmultiply(YAnchors, selectedData) * self.scaleFactor / sum_i
        validData = self.getValidList(indices)
            
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

            for j in range(len(table)):
                fillColor = bwColors.getColor(kNNValues[j])
                edgeColor = classColors.getColor(classValueIndices[table[j].getclass().value])
                key = self.addCurve(str(j), fillColor, edgeColor, self.pointWidth, xData = [table[j][0].value], yData = [table[j][1].value])

        # do we have a subset data to show?
        elif self.subsetData:
            showFilled = self.showFilledSymbols
            colors = None
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete: colors = ColorPaletteHSV(valLen)
            for i in range(dataSize):
                if not validData[i]: continue
                if colors:
                    newColor = colors[classValueIndices[self.rawdata[i].getclass().value]]
                else:
                    newColor = QColor()
                    newColor.setHsv(self.coloringScaledData[classNameIndex][i], 255, 255)
                self.showFilledSymbols = 0
                if self.rawdata[i] in self.subsetData: self.showFilledSymbols = 1

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = QwtSymbol.Ellipse, xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], newColor, i)
            self.showFilledSymbols = showFilled                    


        # CONTINUOUS class 
        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous:
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = QColor()
                newColor.setHsv(self.coloringScaledData[classNameIndex][i], 255, 255)

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = QwtSymbol.Ellipse, xData = x_positions[i], yData = y_positions[i])
                self.addTooltipKey(x_positions[i], y_positions[i], newColor, i)

        # DISCRETE class + optimize drawing
        elif self.optimizedDrawing:
            pos = [[ [] , [], [] ] for i in range(valLen)]
            for i in range(dataSize):
                if not validData[i]: continue
                index = classValueIndices[self.rawdata[i].getclass().value]
                pos[index][0].append(x_positions[i])
                pos[index][1].append(y_positions[i])
                pos[index][2].append(i)

            colors = ColorPaletteHSV(valLen)
            for i in range(valLen):
                newColor = colors[i]
                if not self.useDifferentColors: newColor = QColor(0,0,0)
                
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[i]
                else: curveSymbol = self.curveSymbols[0]

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = pos[i][0], yData = pos[i][1])
                for k in range(len(pos[i][0])):
                    self.addTooltipKey(pos[i][0][k], pos[i][1][k], newColor, pos[i][2][k])

        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            colors = ColorPaletteHSV(valLen)
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = colors[classValueIndices[self.rawdata[i].getclass().value]]
                if not self.useDifferentColors: newColor = QColor(0,0,0)
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[classValueIndices[self.rawdata[i].getclass().value]]
                else: curveSymbol = self.curveSymbols[0]
                self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], newColor, i)
                    
        
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

    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, color, index):
        dictValue = "%.1f-%.1f"%(x, y)
        if not self.dataMap.has_key(dictValue): self.dataMap[dictValue] = []
        self.dataMap[dictValue].append((x, y, color, index))


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
            for (x_i, y_i, color, index) in points:
                currDist = sqrt((xFloat-x_i)*(xFloat-x_i) + (yFloat-y_i)*(yFloat-y_i))
                if currDist < bestDist:
                    bestDist = currDist
                    nearestPoint = (x_i, y_i, color, index)

            (x_i, y_i, color, index) = nearestPoint
            if self.tooltipKind == LINE_TOOLTIPS and bestDist < 0.05:
                for (xAnchor,yAnchor,label) in self.anchorData:
                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x_i, xAnchor], yData = [y_i, yAnchor])
                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = None
                    if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                        marker = self.addMarker(str(self.rawdata[index][self.attributeNames.index(label)]), (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                        marker = self.addMarker("%.3f" % (self.scaledData[self.attributeNames.index(label)][index]), (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(12)
                    self.setMarkerFont(marker, font)

                    self.tooltipMarkers.append(marker)
            elif self.tooltipKind == VISIBLE_ATTRIBUTES or self.tooltipKind == ALL_ATTRIBUTES:
                intX = self.transform(QwtPlot.xBottom, x_i)
                intY = self.transform(QwtPlot.yLeft, y_i)
                text = ""
                if self.tooltipKind == VISIBLE_ATTRIBUTES:
                    labels = [s for (xA, yA, s) in self.anchorData]
                else:
                    labels = self.attributeNames

                if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                    text = self.getShortExampleText(self.rawdata, self.rawdata[index], labels)
                elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                    for label in labels: text += "%s = %.3f; " % (label, self.scaledData[self.attributeNames.index(label)][index])
                        
                self.showTip(intX, intY, text[:-2].replace("; ", "\n"))
                
        OWVisGraph.onMouseMoved(self, e)
        self.replot()
 
    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList):
        (xArray, yArray, validData) = self.createProjection(attrList)

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
                 
        for i in range(len(validData)):
            if not validData[i]: continue
            example = orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()])
            table.append(example)
        return self.kNNOptimization.kNNComputeAccuracy(table)

 
    # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList):
        (xArray, yArray, validData) = self.createProjection(attrList)

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
                 
        for i in range(len(validData)):
            if not validData[i]: continue
            example = orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()])
            table.append(example)

        orange.saveTabDelimited(fileName, table)

    # ####################################
    # create x-y projection of attributes in attrList
    def createProjection(self, attrList, scaleFactor = 1.0):
        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)

        # create anchor for every attribute if necessary
        XAnchors = self.createXAnchors(attrListLength)
        YAnchors = self.createYAnchors(attrListLength)

        indices = [self.attributeNames.index(label) for label in attrList]

        selectedData = Numeric.take(self.noJitteringScaledData, indices)
        sum_i = Numeric.add.reduce(selectedData)
        
        # test if there are zeros in sum_i
        if len(Numeric.nonzero(sum_i)) < len(sum_i):
            add = Numeric.where(sum_i == 0, 1.0, 0.0)
            sum_i += add

        x_positions = Numeric.matrixmultiply(XAnchors, selectedData) * self.scaleFactor / sum_i
        y_positions = Numeric.matrixmultiply(YAnchors, selectedData) * self.scaleFactor / sum_i
        validData = self.getValidList(indices)
            
        return (x_positions, y_positions, validData)

    # ####################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList):
        if not self.rawdata: return (None, None, None)
        selected = orange.ExampleTable(self.rawdata.domain)
        unselected = orange.ExampleTable(self.rawdata.domain)

        xArray, yArray, validData = self.createProjection(attrList, scaleFactor = self.scaleFactor)
                 
        for i in range(len(validData)):
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
    def getOptimalSeparation(self, attributes, minLength, maxLength, addResultFunct):
        dataSize = len(self.rawdata)

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])

        if self.kNNOptimization.getQualityMeasure() == CLASS_ACCURACY: text = "Classification accuracy"
        elif self.kNNOptimization.getQualityMeasure() == AVERAGE_CORRECT: text = "Average correct classification"
        else: text = "Brier score"

        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(minLength, maxLength+1)]

        self.radvizWidget.progressBarInit()

        for z in range(minLength-1, len(attributes)):
            for u in range(minLength-1, maxLength):
                combinations = self.createCombinations([0.0], u, attributes[:z], [])
                combinations.sort()
                combinations.reverse()

                XAnchors = anchorList[u+1-minLength][0]
                YAnchors = anchorList[u+1-minLength][1]
                
                for attrList in combinations:
                    attrs = attrList[1:] + [attributes[z][1]] # remove the value of this attribute subset
                    indices = [self.attributeNames.index(attr) for attr in attrs]

                    indPermutations = {}
                    getPermutationList(indices, [], indPermutations)

                    permutationIndex = 0 # current permutation index
                    
                    validData = self.getValidList(indices)
                    selectedData = Numeric.take(self.noJitteringScaledData, indices)
                    sum_i = Numeric.add.reduce(selectedData)

                    # test if there are zeros in sum_i
                    if len(Numeric.nonzero(sum_i)) < len(sum_i):
                        add = Numeric.where(sum_i == 0, 1.0, 0.0)
                        sum_i += add

                    # count total number of valid examples
                    count = sum(validData)
                    if count < self.kNNOptimization.minExamples:
                        print "Not enough examples (%s) in example table. Ignoring permutation." % (str(count))
                        self.triedPossibilities += len(indPermutations.keys())
                        self.radvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
                        continue

                    tempList = []

                    # for every permutation compute how good it separates different classes            
                    for permutation in indPermutations.values():
                        if self.kNNOptimization.isOptimizationCanceled(): return
                        permutationIndex += 1
                        table = orange.ExampleTable(domain)

                        selectedData2 = Numeric.take(self.noJitteringScaledData, permutation)
                        x_positions = Numeric.matrixmultiply(XAnchors, selectedData2) / sum_i
                        y_positions = Numeric.matrixmultiply(YAnchors, selectedData2) / sum_i
                        
                        for i in range(dataSize):
                            if validData[i] == 0: continue
                            table.append([x_positions[i], y_positions[i], self.rawdata[i].getclass()])

                        accuracy, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                        if table.domain.classVar.varType == orange.VarTypes.Discrete:   print "permutation %6d / %d. %s: %2.2f%%" % (permutationIndex, len(indPermutations.values()), text, accuracy)
                        else:                                                           print "permutation %6d / %d. MSE: %2.2f" % (permutationIndex, len(indPermutations.values()), accuracy) 
                        
                        # save the permutation
                        tempList.append((accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation]))
                        if not self.kNNOptimization.onlyOnePerSubset:
                            addResultFunct(accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation])

                        self.triedPossibilities += 1
                        self.radvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))

                    if self.kNNOptimization.onlyOnePerSubset:
                        # return only the best attribute placements
                        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE: funct = max
                        else: funct = min
                        (acc, other_results, lenTable, attrList) = funct(tempList)
                        addResultFunct(acc, other_results, lenTable, attrList)

   

if __name__== "__main__":
    #Draw a simple graph
    import os
    a = QApplication(sys.argv)        
    graph = OWRadvizGraph(None)
    fname = r"..\..\datasets\brown\brown-normalized.tab"
    if os.path.exists(fname):
        table = orange.ExampleTable(fname)
        attrs = [attr.name for attr in table.domain.attributes]
        start = time.time()
        graph.setData(table)
        graph.updateData(attrs)
        print time.time() - start
    a.setMainWidget(graph)
    graph.show()
    a.exec_loop()
