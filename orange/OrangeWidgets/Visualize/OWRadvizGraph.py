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
VALID = 0
XPOS = 1
YPOS = 2
SYMBOL = 3
PENCOLOR = 4
BRUSHCOLOR = 5

RECT_SIZE = 0.01    # size of rectangle


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
        self.anchorData =[]	    # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.dataMap = {}		# each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.showLegend = 1
        self.useDifferentSymbols = 1
        self.enhancedTooltips = 0
        self.kNNOptimization = None
        self.radvizWidget = radvizWidget
        self.optimizeForPrinting = 1        # show class value using different simple empty symbols using black color

    def setEnhancedTooltips(self, enhanced):
        self.enhancedTooltips = enhanced
        self.anchorData = []

    # create anchors around the circle
    def createAnchors(self, numOfAttr, labels = None):
        anchors = []
        for i in range(numOfAttr):
            x = math.cos(2*math.pi * float(i) / float(numOfAttr)); strX = "%.14f" % (x)
            y = math.sin(2*math.pi * float(i) / float(numOfAttr)); strY = "%.14f" % (y)
            if labels:
                anchors.append((float(strX), float(strY), labels[i]))
            else:
                anchors.append((float(strX), float(strY)))
        return anchors
            
    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels, **args):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        #self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()

        # initial var values
        self.showKNNModel = 0
        self.showCorrect = 1
        self.__dict__.update(args)

        length = len(labels)
        xs = []
        self.dataMap = {}   # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)
        indices = []

        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return
        
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
        for label in labels: indices.append(self.attributeNames.index(label))

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
        xArray = [x for (x,y,label) in self.anchorData]
        yArray = [y for (x,y,label) in self.anchorData]
        xArray.append(self.anchorData[0][0])
        yArray.append(self.anchorData[0][1])
        self.addCurve("dots", QColor(140,140,140), QColor(140,140,140), 10, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, xData = xArray, yData = yArray, forceFilledSymbols = 1)

        # ##########
        # draw text at anchors
        for i in range(length):
            self.addMarker(labels[i], self.anchorData[i][0]*1.1, self.anchorData[i][1]*1.04, Qt.AlignHCenter + Qt.AlignVCenter, bold = 1)


        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        # -----------------------------------------------------------
        #  create data curves
        # -----------------------------------------------------------

        classNameIndex = self.attributeNames.index(self.rawdata.domain.classVar.name)
        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:    	# if we have a discrete class
            valLen = len(self.rawdata.domain.classVar.values)
            classValueIndices = self.getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)	# we create a hash table of variable values and their indices            
        else:	# if we have a continuous class
            valLen = 0

        
        # will we show different symbols?        
        useDifferentSymbols = 0
        if self.useDifferentSymbols and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and valLen < len(min(self.curveSymbols, self.curveSymbolsPrinting)): useDifferentSymbols = 1

        dataSize = len(self.rawdata)
        blackColor = QColor(0,0,0)
        curveData = [[0, 0, 0, QwtSymbol.Ellipse, blackColor, blackColor] for i in range(dataSize)]
        validData = self.getValidList(indices)
        
        for i in range(dataSize):
            if validData[i] == 0: continue
            
            sum_i = 0.0
            for j in range(length):
                sum_i += self.scaledData[indices[j]][i]
            if sum_i == 0.0: sum_i = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero

            ##########
            # calculate the position of the data point
            x_i = 0.0; y_i = 0.0
            for j in range(length):
                index = indices[j]
                x_i += self.anchorData[j][0]*(self.scaledData[index][i] / sum_i)
                y_i += self.anchorData[j][1]*(self.scaledData[index][i] / sum_i)

            # scale data according to scale factor
            x_i = x_i * self.scaleFactor
            y_i = y_i * self.scaleFactor
            curveData[i][VALID] = 1
            curveData[i][XPOS] = x_i
            curveData[i][YPOS] = y_i

            
        if self.showKNNModel == 1:
            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
            table = orange.ExampleTable(domain)

            # build an example table            
            for i in range(dataSize):
                if curveData[i][VALID]:
                    table.append(orange.Example(domain, [curveData[i][XPOS], curveData[i][YPOS], self.rawdata[i].getclass()]))

            kNNValues = self.kNNOptimization.kNNClassifyData(table)
            accuracy = copy(kNNValues)
            measure = self.kNNOptimization.getQualityMeasure()
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                if ((measure == CLASS_ACCURACY or measure == AVERAGE_CORRECT) and self.showCorrect) or (measure == BRIER_SCORE and not self.showCorrect):
                    kNNValues = [1.0 - val for val in kNNValues]
            else:
                if self.showCorrect: kNNValues = [1.0 - val for val in kNNValues]
            
            if table.domain.classVar.varType == orange.VarTypes.Continuous: preText = 'Mean square error : '
            else:
                if measure == CLASS_ACCURACY:    preText = "Classification accuracy : "
                elif measure == AVERAGE_CORRECT: preText = "Average correct classification : "
                else:                            preText = "Brier score : "

            for j in range(len(table)):
                newColor = QColor(55+kNNValues[j]*200, 55+kNNValues[j]*200, 55+kNNValues[j]*200)
                if table.domain.classVar.varType == orange.VarTypes.Continuous:
                    dataColor = newColor
                else:
                    dataColor = QColor()
                    dataColor.setHsv(360*self.colorHueValues[classValueIndices[table[j].getclass().value]], 255, 255)
                key = self.addCurve(str(j), newColor, dataColor , self.pointWidth, xData = [table[j][0].value], yData = [table[j][1].value])
                r = QRectFloat(table[j][0].value - RECT_SIZE, table[j][1].value -RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
                self.tips.addToolTip(r, preText + "%.2f "%(accuracy[j]))

        # CONTINUOUS class 
        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous:
            for i in range(dataSize):
                if not curveData[i][VALID]: continue
                newColor = QColor()
                newColor.setHsv(self.coloringScaledData[classNameIndex][i] * 360, 255, 255)
                curveData[i][PENCOLOR] = newColor
                curveData[i][BRUSHCOLOR] = newColor

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveData[i][SYMBOL], xData = curveData[i][XPOS], yData = curveData[i][YPOS])
                self.tips.addToolTip(QRectFloat(curveData[i][XPOS]-RECT_SIZE, curveData[i][YPOS]-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE), self.getShortExampleText(self.rawdata, self.rawdata[i], indices))
                self.addTooltipKey(curveData[i][XPOS], curveData[i][YPOS], newColor, i)

        # DISCRETE class + optimize drawing
        elif self.optimizedDrawing:
            pos = [[ [] , [], [] ] for i in range(valLen)]
            for i in range(dataSize):
                if not curveData[i][VALID]: continue
                pos[classValueIndices[self.rawdata[i].getclass().value]][0].append(curveData[i][XPOS])
                pos[classValueIndices[self.rawdata[i].getclass().value]][1].append(curveData[i][YPOS])
                pos[classValueIndices[self.rawdata[i].getclass().value]][2].append(i)
                self.tips.addToolTip(QRectFloat(curveData[i][XPOS]-RECT_SIZE, curveData[i][YPOS]-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE), self.getShortExampleText(self.rawdata, self.rawdata[i], indices))

            for i in range(valLen):
                newColor = QColor(0,0,0)
                if not self.optimizeForPrinting:
                    if valLen < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[i]*360, 255, 255)
                    else:                                 newColor.setHsv((i*360)/valLen, 255, 255)
                
                if self.useDifferentSymbols:
                    if self.optimizeForPrinting and valLen < len(self.curveSymbolsPrinting): curveSymbol = self.curveSymbolsPrinting[i]
                    elif not self.optimizeForPrinting and valLen < len(self.curveSymbols):   curveSymbol = self.curveSymbols[i]
                    else:                                                                     curveSymbol = self.curveSymbols[0]
                else: curveSymbol = self.curveSymbols[0]

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = pos[i][0], yData = pos[i][1])
                for k in range(len(pos[i][0])):
                    self.addTooltipKey(pos[i][0][k], pos[i][1][k], newColor, pos[i][2][k])

        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            if self.optimizeForPrinting:
                for i in range(dataSize):
                    if not curveData[i][VALID]: continue
                    if self.useDifferentSymbols and valLen < len(self.curveSymbolsPrinting): curveSymbol = self.curveSymbolsPrinting[classValueIndices[self.rawdata[i].getclass().value]]
                    else: curveSymbol = self.curveSymbolsPrinting[0]
                    self.addCurve(str(i), blackColor, blackColor, self.pointWidth, symbol = curveSymbol, xData = [curveData[i][XPOS]], yData = [curveData[i][YPOS]])
                    self.tips.addToolTip(QRectFloat(curveData[i][XPOS]-RECT_SIZE, curveData[i][YPOS]-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE), self.getShortExampleText(self.rawdata, self.rawdata[i], indices))
                    self.addTooltipKey(curveData[i][XPOS], curveData[i][YPOS], blackColor, i)
            else:
                for i in range(dataSize):
                    if not curveData[i][VALID]: continue
                    newColor = QColor(0,0,0)
                    if valLen < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[classValueIndices[self.rawdata[i].getclass().value]]*360, 255, 255)
                    else:                                 newColor.setHsv((classValueIndices[self.rawdata[i].getclass().value]*360)/valLen, 255, 255)
                    if self.useDifferentSymbols and valLen < len(self.curveSymbols): curveSymbol = self.curveSymbols[classValueIndices[self.rawdata[i].getclass().value]]
                    else: curveSymbol = self.curveSymbols[0]
                    self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = [curveData[i][XPOS]], yData = [curveData[i][YPOS]])
                    self.tips.addToolTip(QRectFloat(curveData[i][XPOS]-RECT_SIZE, curveData[i][YPOS]-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE), self.getShortExampleText(self.rawdata, self.rawdata[i], indices))
                    self.addTooltipKey(curveData[i][XPOS], curveData[i][YPOS], newColor, i)
                    
        
        #################
        # draw the legend
        if self.showLegend:
            # show legend for discrete class
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                self.addMarker(self.rawdata.domain.classVar.name, 0.87, 1.06, Qt.AlignLeft)
                    
                classVariableValues = self.getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
                for index in range(len(classVariableValues)):
                    newColor = QColor(0,0,0)
                    if not self.optimizeForPrinting:
                        if valLen < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[index]*360, 255, 255)
                        else:                                 newColor.setHsv((index*360)/valLen, 255, 255)
                                    
                    y = 1.0 - index * 0.05

                    if not self.useDifferentSymbols: curveSymbol = self.curveSymbols[0]
                    elif self.optimizeForPrinting and valLen < len(self.curveSymbolsPrinting): curveSymbol = self.curveSymbolsPrinting[index]
                    elif not self.optimizeForPrinting and valLen < len(self.curveSymbols):   curveSymbol = self.curveSymbols[index]
                    else:                                                                     curveSymbol = self.curveSymbols[0]

                    self.addCurve(str(index), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = [0.95, 0.95], yData = [y, y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft + Qt.AlignHCenter)
            # show legend for continuous class
            else:
                xs = [1.15, 1.20]
                for i in range(1000):
                    y = -1.0 + i*2.0/1000.0
                    newCurveKey = self.insertCurve(str(i))
                    newColor = QColor()
                    newColor.setHsv(float(i*self.MAX_HUE_VAL)/1000.0, 255, 255)
                    self.setCurvePen(newCurveKey, QPen(newColor))
                    self.setCurveData(newCurveKey, xs, [y,y])

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawdata.domain.classVar.name]
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, minVal), x0 - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, maxVal), x0 - 0.02, +1.0 - 0.04, Qt.AlignLeft)

    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, color, index):
        dictValue = "%.1f-%.1f"%(x, y)
        if not self.dataMap.has_key(dictValue): self.dataMap[dictValue] = []
        self.dataMap[dictValue].append((x, y, color, index))


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
            
        x = self.invTransform(QwtPlot.xBottom, e.x())
        y = self.invTransform(QwtPlot.yLeft, e.y())
        dictValue = "%.1f-%.1f"%(x, y)
        if self.dataMap.has_key(dictValue) and self.enhancedTooltips:
            points = self.dataMap[dictValue]
            bestDist = 100.0
            nearestPoint = ()
            for (x_i, y_i, color, index) in points:
                currDist = sqrt((x-x_i)*(x-x_i)+(y-y_i)*(y-y_i))
                if currDist < bestDist:
                    bestDist = currDist
                    nearestPoint = (x_i, y_i, color, index)
           
            if bestDist < 0.05:
                (x_i, y_i, color, index) = nearestPoint
                for (xAnchor,yAnchor,label) in self.anchorData:
                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x_i, xAnchor], yData = [y_i, yAnchor])
                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = self.addMarker(str(self.rawdata[index][self.attributeNames.index(label)].value), (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(12)
                    self.setMarkerFont(marker, font)

                    self.tooltipMarkers.append(marker)
                
        OWVisGraph.onMouseMoved(self, e)
        self.replot()


    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList):
        (xArray, yArray) = self.createProjection(attrList)

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
                 
        for i in range(len(self.rawdata)):
            if xArray[i] == "?": continue
            example = orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()])
            table.append(example)
        return self.kNNOptimization.kNNComputeAccuracy(table)

 
    # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList, validData = None, anchors = None):
        (xArray, yArray) = self.createProjection(attrList, validData, anchors)

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
                 
        for i in range(len(self.rawdata)):
            if xArray[i] == "?": continue
            example = orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()])
            table.append(example)

        orange.saveTabDelimited(fileName, table)

    # ####################################
    # create x-y projection of attributes in attrList
    def createProjection(self, attrList, validData = None, anchors = None, scaleFactor = 1.0):
        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)

        # create anchor for every attribute if necessary
        if anchors == None:
            anchors = self.createAnchors(attrListLength)

        indices = []
        for attr in attrList:
            indices.append(self.attributeNames.index(attr))
        
        if validData == None:
            validData = self.getValidList(indices)

        # store all sums
        sum_i=[]
        for i in range(dataSize):
            if not validData[i]: sum_i.append(1.0); continue

            temp = 0
            for j in range(attrListLength): temp += self.noJitteringScaledData[indices[j]][i]
            if temp == 0.0: temp = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero
            sum_i.append(temp)

        xArray = []
        yArray = []
        for i in range(dataSize):
            if not validData[i]: xArray.append("?"); yArray.append("?"); continue
            
            # calculate projections
            x_i = 0.0; y_i = 0.0
            for j in range(attrListLength):
                x_i = x_i + anchors[j][0]*(self.noJitteringScaledData[indices[j]][i] / sum_i[i])
                y_i = y_i + anchors[j][1]*(self.noJitteringScaledData[indices[j]][i] / sum_i[i])

            xArray.append(x_i*scaleFactor)
            yArray.append(y_i*scaleFactor)
            
        return (xArray, yArray)

    # ####################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList):
        if not self.rawdata: return (None, None, None)
        selected = orange.ExampleTable(self.rawdata.domain)
        unselected = orange.ExampleTable(self.rawdata.domain)

        xArray, yArray = self.createProjection(attrList, scaleFactor = self.scaleFactor)
                 
        for i in range(len(self.rawdata)):
            if xArray[i] == "?": continue
            
            if self.isPointSelected(xArray[i], yArray[i]): selected.append(self.rawdata[i])
            else:                                          unselected.append(self.rawdata[i])

        print len(selected), len(unselected)
        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None
        merged = self.changeClassAttr(selected, unselected)
        return (selected, unselected, merged)
    
    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getOptimalSeparation(self, attrList, printFullOutput = 1):
        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)

        # create anchor for every attribute
        anchors = self.createAnchors(attrListLength)
        
        indices = []
        for attr in attrList:
            indices.append(self.attributeNames.index(attr))

        # create all possible circular permutations of this indices
        if printFullOutput: print "Generating permutations. Please wait..."

        indPermutations = {}
        getPermutationList(indices, [], indPermutations)

        fullList = []
        permutationIndex = 0 # current permutation index
        totalPermutations = len(indPermutations.values())
        if printFullOutput: print "Total permutations: %d" % (totalPermutations)

        if self.totalPossibilities == 0:
            self.totalPossibilities = totalPermutations
            self.triedPossibilities = 0

        validData = self.getValidList(indices)

        ###################
        # print total number of valid examples
        count = 0
        for i in range(dataSize):
            if validData[i] == 1: count+=1
        
        if count < self.kNNOptimization.minExamples:
            print "Nr. of examples: ", str(count)
            print "Not enough examples in example table. Ignoring permutation..."
            print "------------------------------"
            self.triedPossibilities += 1
            if self.radvizWidget: self.radvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
            return []

        # store all sums
        sum_i=[]
        for i in range(dataSize):
            if validData[i] == 0:
                sum_i.append(1.0)
                continue

            temp = 0    
            for j in range(attrListLength):
                temp += self.noJitteringScaledData[indices[j]][i]
            if temp == 0.0: temp = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero
            sum_i.append(temp)

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])

        t = time.time()

        if self.kNNOptimization.getQualityMeasure() == CLASS_ACCURACY: text = "Classification accuracy"
        elif self.kNNOptimization.getQualityMeasure() == AVERAGE_CORRECT: text = "Average correct classification"
        else: text = "Brier score"

        # for every permutation compute how good it separates different classes            
        for permutation in indPermutations.values():
            permutationIndex += 1

            #if progressBar != None: progressBar.setProgress(progressBar.progress()+1)           
            tempPermValue = 0
            table = orange.ExampleTable(domain)
                     
            for i in range(dataSize):
                if validData[i] == 0: continue
                
                # calculate projections
                x_i = 0.0; y_i = 0.0
                for j in range(attrListLength):
                    index = permutation[j]
                    x_i = x_i + anchors[j][0]*(self.noJitteringScaledData[index][i] / sum_i[i])
                    y_i = y_i + anchors[j][1]*(self.noJitteringScaledData[index][i] / sum_i[i])
                
                example = orange.Example(domain, [x_i, y_i, self.rawdata[i].getclass()])
                table.append(example)

            accuracy = self.kNNOptimization.kNNComputeAccuracy(table)
            if table.domain.classVar.varType == orange.VarTypes.Discrete:
                print "permutation %6d / %d. %s: %2.2f%%" % (permutationIndex, totalPermutations, text, accuracy)
            else:
                print "permutation %6d / %d. MSE: %2.2f" % (permutationIndex, totalPermutations, accuracy) 
            
            # save the permutation
            fullList.append((accuracy, len(table), [self.attributeNames[i] for i in permutation]))

            self.triedPossibilities += 1
            if self.radvizWidget: self.radvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))

        if printFullOutput:
            secs = time.time() - t
            print "Used time: %d min, %d sec" %(secs/60, secs%60)
            print "------------------------------"

        if self.kNNOptimization.onlyOnePerSubset:
            # return only the best attribute placements
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE:
                return [max(fullList)]
            else:
                return [min(fullList)]
        else:
            return fullList

    
    # try all possibilities with numOfAttr attributes or less
    # attrList = list of attributes to choose from
    def getOptimalSubsetSeparation(self, attrList, numOfAttr):
        full = []
        self.startTime = time.time()
        for i in range(numOfAttr, 2, -1):
            self.totalPossibilities += combinations(i, len(attrList))*(fact(i-1)/2)

        for i in range(numOfAttr, 2, -1):
            full1 = self.getOptimalExactSeparation(attrList, [], i)
            full = full + full1
            
        return full

    # try all posibilities with exactly numOfAttr attributes
    def getOptimalExactSeparation(self, attrList, subsetList, numOfAttr):
        if attrList == [] or numOfAttr == 0:
            if len(subsetList) < 3 or numOfAttr != 0: return []
            print subsetList
            if self.totalPossibilities > 0 and self.triedPossibilities > 0:
                secs = int(time.time() - self.startTime)
                totalExpectedSecs = int(float(self.totalPossibilities*secs)/float(self.triedPossibilities))
                restSecs = totalExpectedSecs - secs
                #print "Used time: %d:%02d:%02d, Expected remaining time: %d:%02d:%02d (total experiments: %d, rest: %d)" %(secs /3600, (secs-((secs/3600)*3600))/60, secs%60, restSecs /3600, (restSecs-((restSecs/3600)*3600))/60, restSecs%60, self.totalPossibilities, self.totalPossibilities-self.triedPossibilities)
            
            return self.getOptimalSeparation(subsetList, printFullOutput = 0)

        full1 = self.getOptimalExactSeparation(attrList[1:], subsetList, numOfAttr)
        subsetList2 = copy(subsetList)
        subsetList2.insert(0, attrList[0])
        full2 = self.getOptimalExactSeparation(attrList[1:], subsetList2, numOfAttr-1)

        # find max values in booth lists
        full = full1 + full2
        shortList = []
        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE: funct = max
        else: funct = min
        for i in range(min(self.kNNOptimization.resultListLen, len(full))):
            item = funct(full)
            shortList.append(item)
            full.remove(item)

        return shortList

    # for a given list of attribute subsets, compute the most interesting projections
    def getOptimalListSeparation(self, attrSubsetList):
        currList = []
        # find function, that will delete the worst performances from merged lists currList and retList
        # WARNING: funct is here defined to find the WORST projection in a given list. Don't just copy and paste this 2 lines!!!
        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE: funct = min
        else: funct = max
        
        for (acc, attrList) in attrSubsetList:
            retList = self.getOptimalSeparation(attrList, printFullOutput = 0)
            currList += retList
            while (len(currList) > self.kNNOptimization.resultListLen):
                item = funct(currList)
                currList.remove(item)   # remove the projection with the worst accuracy

        return currList

if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWRadvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
