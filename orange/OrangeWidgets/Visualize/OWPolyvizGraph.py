#
# OWPolyvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy, deepcopy
import time
from OWkNNOptimization import *
import math
import orange

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
        self.useDifferentSymbols = 1
        self.optimizeForPrinting = 1

        self.dataMap = {}        # each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.validData = []
        self.statusBar = None
        self.showLegend = 1

    def setEnhancedTooltips(self, enhanced):
        self.enhancedTooltips = enhanced
        self.dataMap = {}

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

    def calculateAttrValuesSum(self, data, datalen, indices, validData):
        if datalen == 0: return []
        
        sum=[]
        for i in range(datalen):
            if validData[i] == 0:
                sum.append(1.0)
                continue
            temp = 0
            for j in range(len(indices)): temp += data[indices[j]][i]
            if temp == 0.0: temp = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero
            sum.append(temp)
        return sum

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
        indices = []
        self.anchorData = []
        polyvizLineCoordsX = []; polyvizLineCoordsY = []    # if class is discrete we will optimize drawing by storing computed values and adding less data curves to plot
            
        # we must have at least 3 attributes to be able to show anything
        if not self.rawdata or len(self.rawdata) == 0 or len(labels) < 3: self.updateLayout(); return
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

        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)

        classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)

        # ##########
        # create anchor for every attribute
        for i in range(length):
            x1 = math.cos(2*math.pi * float(i) / float(length)); strX1 = "%.4f" % (x1)
            y1 = math.sin(2*math.pi * float(i) / float(length)); strY1 = "%.4f" % (y1)
            x2 = math.cos(2*math.pi * float(i+1) / float(length)); strX2 = "%.4f" % (x2)
            y2 = math.sin(2*math.pi * float(i+1) / float(length)); strY2 = "%.4f" % (y2)
            self.anchorData.append((float(strX1), float(strY1), float(strX2), float(strY2), labels[i]))
        
        valLen = 0
        # if we don't want coloring
        if self.showKNNModel:      
            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
            table = orange.ExampleTable(domain)
            
            for i in range(len(labels)):
                polyvizLineCoordsX.append([[]])
                polyvizLineCoordsY.append([[]])                

        # if we have a discrete class
        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            classIsDiscrete = 1
            classNameIndex = self.attributeNames.index(self.rawdata.domain.classVar.name)
            valLen = len(self.rawdata.domain.classVar.values)
            for i in range(len(labels)):
                tempX = []; tempY = []
                for j in range(len(classValueIndices)):
                    tempX.append([]); tempY.append([])
                polyvizLineCoordsX.append(tempX)
                polyvizLineCoordsY.append(tempY)
        else:
            classNameIndex = self.attributeNames.index(self.rawdata.domain.classVar.name)
            
        # ######################
        # compute valid data examples
        self.validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": self.validData[i] = 0
                if classNameIndex >= 0 and self.scaledData[classNameIndex][i] == "?": self.validData[i] = 0


        # ##########
        # draw text at lines
        for i in range(length):
            # print attribute name
            self.addMarker(labels[i], 0.6*(self.anchorData[i][0]+ self.anchorData[i][2]), 0.6*(self.anchorData[i][1]+ self.anchorData[i][3]), Qt.AlignHCenter + Qt.AlignVCenter, bold = 1)

            if self.rawdata.domain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = getVariableValuesSorted(self.rawdata, labels[i])
                count = len(values)
                k = 1.08
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    if attributeReverse[labels[i]] == 0:
                        self.addMarker(values[j], k*(1-pos)*self.anchorData[i][0]+k*pos*self.anchorData[i][2], k*(1-pos)*self.anchorData[i][1]+k*pos*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)
                    else:
                        self.addMarker(values[j], k*pos*self.anchorData[i][0]+k*(1-pos)*self.anchorData[i][2], k*pos*self.anchorData[i][1]+k*(1-pos)*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)

            else:
                # min and max value
                names = ["%.3f" % (self.attrLocalValues[labels[i]][0]), "%.3f" % (self.attrLocalValues[labels[i]][1])]
                if attributeReverse[labels[i]] == 1: names.reverse()
                self.addMarker(names[0],0.95*self.anchorData[i][0]+0.15*self.anchorData[i][2], 0.95*self.anchorData[i][1]+0.15*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)
                self.addMarker(names[1], 0.15*self.anchorData[i][0]+0.95*self.anchorData[i][2], 0.15*self.anchorData[i][1]+0.95*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)


        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y
        contData = []    # list to store color, x and y position of data items in case of continuous class

        sum = self.calculateAttrValuesSum(self.scaledData, len(self.rawdata), indices, self.validData)

        if self.optimizeForPrinting: symbolList = self.curveSymbolsPrinting
        else: symbolList = self.curveSymbols
        
        # ##########
        #  create data curves
        RECT_SIZE = 0.01    # size of tooltip rectangle in percents of graph size
        for i in range(dataSize):
            if self.validData[i] == 0: continue
            
            # #########
            # calculate the position of the data point
            x_i = 0.0; y_i = 0.0
            xDataAnchors = []; yDataAnchors = []
            for j in range(length):
                index = indices[j]
                val = self.noJitteringScaledData[index][i]
                if attributeReverse[labels[j]] == 1: val = 1-val
                xDataAnchor = self.anchorData[j][0]*(1-val) + self.anchorData[j][2]*val
                yDataAnchor = self.anchorData[j][1]*(1-val) + self.anchorData[j][3]*val
                x_i += xDataAnchor * (self.scaledData[index][i] / sum[i])
                y_i += yDataAnchor * (self.scaledData[index][i] / sum[i])
                xDataAnchors.append(xDataAnchor)
                yDataAnchors.append(yDataAnchor)

            # scale data according to scale factor
            x_i = x_i * self.scaleFactor
            y_i = y_i * self.scaleFactor
                

            lineColor = QColor(0,0,0)
            if self.showKNNModel == 1:
                table.append(orange.Example(domain, [x_i, y_i, self.rawdata[i].getclass()]))
            elif classIsDiscrete and self.optimizedDrawing:
                index = classValueIndices[self.rawdata[i].getclass().value]
                curveData[index][0].append(x_i)
                curveData[index][1].append(y_i)
                if not self.optimizeForPrinting: lineColor.setHsv(self.coloringScaledData[classNameIndex][i], 255, 255)
                text= self.getShortExampleText(self.rawdata, self.rawdata[i], indices + [self.rawdata.domain.classVar.name])
                r = QRectFloat(x_i-RECT_SIZE, y_i-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
                self.tips.addToolTip(r, text)
            else:
                if classIsDiscrete and self.useDifferentSymbols and valLen < len(symbolList) : symbol = symbolList[classValueIndices[self.rawdata[i].getclass().value]]
                else: symbol = symbolList[0]
                contData.append([x_i, y_i, self.coloringScaledData[classNameIndex][i], symbol]) # store data for drawing
                if not self.optimizeForPrinting: lineColor.setHsv(self.coloringScaledData[classNameIndex][i], 255, 255)
                text= self.getShortExampleText(self.rawdata, self.rawdata[i], indices + [self.rawdata.domain.classVar.name])
                r = QRectFloat(x_i-RECT_SIZE, y_i-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
                self.tips.addToolTip(r, text)


            # compute the lines from anchors
            for j in range(length):
                dist = EuclDist([x_i, y_i], [xDataAnchors[j] , yDataAnchors[j]])
                if dist == 0: continue
                kvoc = float(self.lineLength) / dist
                lineX1 = x_i; lineY1 = y_i

                # we don't make extrapolation
                if kvoc > 1: lineX2 = lineX1; lineY2 = lineY1
                else:
                    lineX2 = (1.0 - kvoc)*xDataAnchors[j] + kvoc * lineX1
                    lineY2 = (1.0 - kvoc)*yDataAnchors[j] + kvoc * lineY1


                if self.showKNNModel:
                    polyvizLineCoordsX[j][0] += [xDataAnchors[j], lineX2]
                    polyvizLineCoordsY[j][0] += [yDataAnchors[j], lineY2]
                elif classIsDiscrete and self.optimizedDrawing:
                    index = classValueIndices[self.rawdata[i].getclass().value]
                    polyvizLineCoordsX[j][index] += [xDataAnchors[j], lineX2, xDataAnchors[j]]
                    polyvizLineCoordsY[j][index] += [yDataAnchors[j], lineY2, yDataAnchors[j]]
                else:
                    self.addCurve('line' + str(i), lineColor, lineColor, 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = [xDataAnchors[j], lineX2], yData = [yDataAnchors[j], lineY2])

        
            # create a dictionary value so that tooltips will be shown faster
            data = self.rawdata[i]
            dictValue = "%.1f-%.1f"%(x_i, y_i)
            if not self.dataMap.has_key(dictValue):
                self.dataMap[dictValue] = []
            self.dataMap[dictValue].append((x_i, y_i, xDataAnchors, yDataAnchors, lineColor, data))
             

        ###### SHOW KNN MODEL QUALITY/ERROR
        if self.showKNNModel:
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
                r = QRectFloat(table[j][0].value - RECT_SIZE, table[j][1].value -RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
                self.tips.addToolTip(r, preText + "%.2f "%(accuracy[j]))
                for i in range(len(polyvizLineCoordsX)):
                    self.addCurve('line' + str(i), fillColor, edgeColor, 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = polyvizLineCoordsX[i][0][2*j:2*j+2], yData = polyvizLineCoordsY[i][0][2*j:2*j+2])

        ###### ONE COLOR OR DISCRETE CLASS ATTRIBUTE
        elif classIsDiscrete and self.optimizedDrawing:
            colors = ColorPaletteHSV(valLen)
            if self.optimizeForPrinting: colors.colors = [QColor(0,0,0) for i in range(valLen)]
            
            # create data curves for dots
            for i in range(valLen):
                newColor = colors.getColor(i)
                if self.useDifferentSymbols and valLen < len(symbolList): curveSymbol = symbolList[i]
                else: curveSymbol = symbolList[0]
                self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = curveData[i][0], yData = curveData[i][1])
                for j in range(len(labels)):
                    self.addCurve("lines" + str(i), newColor, newColor, 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = polyvizLineCoordsX[j][i], yData = polyvizLineCoordsY[j][i])

        ###### CONTINUOUS CLASS ATTRIBUTE
        else:
            for i in range(len(contData)):
                newColor = QColor();  newColor.setHsv(contData[i][2], 255, 255)
                if classIsDiscrete and self.useDifferentSymbols and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and valLen < len(symbolList): curveSymbol = symbolList[classValueIndices[self.rawdata[i].getclass().value]]
                self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = contData[i][3], xData = [contData[i][0]], yData = [contData[i][1]])

        
        # ##########
        # draw polygon
        xdata = []; ydata = []
        for i in range(len(labels)+1):
            xdata.append(math.cos(2*math.pi * float(i) / float(len(labels))))
            ydata.append(math.sin(2*math.pi * float(i) / float(len(labels))))

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
                    newColor = QColor(0,0,0)
                    if not self.optimizeForPrinting: newColor = classColors.getColor(index)
                    y = 1.0 - index * 0.05
                    if self.useDifferentSymbols and valLen < len(symbolList): curveSymbol = symbolList[index]
                    else: curveSymbol = symbolList[0]
                    self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData= [0.95, 0.95], yData = [y, y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft + Qt.AlignHCenter)

            # show legend for continuous class
            else:
                x0 = 1.20; x1 = 1.24
                classColors = ColorPaletteHSV(-1)
                for i in range(1000):
                    y = -1.0 + i*2.0/1000.0
                    newCurveKey = self.insertCurve(str(i))
                    self.setCurvePen(newCurveKey, QPen(classColors.getColor(float(i)/1000.0)))
                    self.setCurveData(newCurveKey, [x0,x1], [y,y])

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawdata.domain.classVar.name]
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, minVal), x0 - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, maxVal), x0 - 0.02, +1.0 - 0.04, Qt.AlignLeft)


    ##########################
    ## do we want advanced tooltips
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
            dist = 100.0
            nearestPoint = ()
            for (x_i, y_i, xAnchors, yAnchors, color, data) in points:
                if abs(x-x_i)+abs(y-y_i) < dist:
                    dist = abs(x-x_i)+abs(y-y_i)
                    nearestPoint = (x_i, y_i, xAnchors, yAnchors, color, data)
           
            if dist < 0.05:
                x_i = nearestPoint[0]; y_i = nearestPoint[1]; xAnchors = nearestPoint[2]; yAnchors = nearestPoint[3]; color = nearestPoint[4]; data = nearestPoint[5]
                for i in range(len(self.anchorData)):
                    (xAnchor1, yAnchor1, xAnchor2, yAnchor2, label) = self.anchorData[i]

                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, xData = [x_i, xAnchors[i]], yData = [y_i, yAnchors[i]])
                    self.tooltipCurveKeys.append(key)
                    
                    # draw text
                    marker = self.addMarker(str(data[self.attributeNames.index(label)].value),(x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(12)
                    self.setMarkerFont(marker, font)

                    self.tooltipMarkers.append(marker)

        OWVisGraph.onMouseMoved(self, e)
        self.update()
        # -----------------------------------------------------------
        # -----------------------------------------------------------

    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList, attributeReverse):
        (xArray, yArray) = self.createProjection(attrList, attributeReverse)

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
                 
        for i in range(len(self.rawdata)):
            if xArray[i] == "?": continue
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
    def saveProjectionAsTabData(self, fileName, attrList, attributeReverse, validData = None, anchors = None):
        (xArray, yArray) = self.createProjection(attrList, attributeReverse, validData, anchors)

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
                 
        for i in range(len(self.rawdata)):
            if xArray[i] == "?": continue
            example = orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()])
            table.append(example)

        orange.saveTabDelimited(fileName, table)

    # ####################################
    # create x-y projection of attributes in attrList
    def createProjection(self, attrList, attributeReverse, validData = None, anchors = None, scaleFactor = 1.0):
        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)
        classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)

        # create anchor for every attribute if necessary
        if anchors == None:
            anchors = self.createAnchors(attrListLength)

        indices = []
        for attr in attrList:
            indices.append(self.attributeNames.index(attr))
        
        if validData == None:
            validData = self.getValidList(indices)

        # store all sums
        sum = self.calculateAttrValuesSum(self.noJitteringScaledData, len(self.rawdata), indices, validData)

        # calculate projections
        xArray = []
        yArray = []
        for i in range(dataSize):
            if not validData[i]: xArray.append("?"); yArray.append("?"); continue
            
            x_i = 0.0; y_i = 0.0
            for j in range(attrListLength):
                val = self.noJitteringScaledData[indices[j]][i]
                if attributeReverse[attrList[j]] == 1: val = 1-val
                xDataAnchor = anchors[0][j]*(1-val) + anchors[0][(j+1)%attrListLength]*val
                yDataAnchor = anchors[1][j]*(1-val) + anchors[1][(j+1)%attrListLength]*val
                x_i += xDataAnchor * (self.noJitteringScaledData[indices[j]][i] / sum[i])
                y_i += yDataAnchor * (self.noJitteringScaledData[indices[j]][i] / sum[i])
            xArray.append(x_i*scaleFactor)
            yArray.append(y_i*scaleFactor)
               
        return (xArray, yArray)
    

    # ####################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, attributeReverse):
        if not self.rawdata: return (None, None, None)
        selected = orange.ExampleTable(self.rawdata.domain)
        unselected = orange.ExampleTable(self.rawdata.domain)

        xArray, yArray = self.createProjection(attrList, attributeReverse, scaleFactor = self.scaleFactor)
                 
        for i in range(len(self.rawdata)):
            if xArray[i] == "?": continue
            
            if self.isPointSelected(xArray[i], yArray[i]): selected.append(self.rawdata[i])
            else:                                          unselected.append(self.rawdata[i])

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None
        merged = self.changeClassAttr(selected, unselected)
        return (selected, unselected, merged)

    """
    def getOptimalExactSeparation(self, attrList, subsetList, attrReverseDict, numOfAttr, addResultFunct = None):
        if attrList == [] or numOfAttr == 0:
            if len(subsetList) < 3 or numOfAttr != 0: return []
            print subsetList
            return self.getOptimalSeparation(subsetList, attrReverseDict, printFullOutput = 0, addResultFunct = addResultFunct)

        #full1 = self.getOptimalExactSeparation(attrList[1:], subsetList, attrReverseDict, numOfAttr, addResultFunct)
        self.getOptimalExactSeparation(attrList[1:], subsetList, attrReverseDict, numOfAttr, addResultFunct)
        subsetList2 = list(subsetList)
        subsetList2.insert(0, attrList[0])
        #full2 = self.getOptimalExactSeparation(attrList[1:], subsetList2, attrReverseDict, numOfAttr-1, addResultFunct)
        self.getOptimalExactSeparation(attrList[1:], subsetList2, attrReverseDict, numOfAttr-1, addResultFunct)

        # find max values in booth lists
        #full = full1 + full2
        #shortList = []
        #if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE: funct = max
        #else: funct = min
        #for i in range(min(self.kNNOptimization.resultListLen, len(full))):
        #    item = funct(full)
        #    shortList.append(item)
        #    full.remove(item)
        #return shortList

    
    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getOptimalSeparation(self, attrList, attrReverseDict, printFullOutput = 1, addResultFunct = None):
        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)
        classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)

        # create a table of indices that stores the sequence of variable indices        
        indices = [];
        for label in attrList:
            indices.append(self.attributeNames.index(label))

        # if we want global value scaling, we must first compute it
        if self.globalValueScaling == 1:
            selectedGlobScaledData = []
            for i in range(len(self.rawdata.domain)): selectedGlobScaledData.append([])
        
            (minVal, maxVal) =  self.getMinMaxValDomain(self.rawdata, attrList)
            
            for attr in attrList:
                index = self.attributeNames.index(attr)
                scaled, values = self.scaleData(self.rawdata, index, minVal, maxVal, jitteringEnabled = 0)
                selectedGlobScaledData[index] = scaled
        else:
            selectedGlobScaledData = self.noJitteringScaledData

        if printFullOutput:
            print "Generating permutations. Please wait..."

        indPermutations = {}
        getPermutationList(indices, [], indPermutations, attrReverseDict == None)

        attrReverse = []
        if attrReverseDict != None: # if we received a dictionary, then we don't reverse attributes
            temp = [0] * len(self.rawdata.domain)
            for val in attrReverseDict.keys():
                temp[self.attributeNames.index(val)] = attrReverseDict[val]
            attrReverse.append(temp)
        else:
            attrReverse = self.generateAttrReverseLists(attrList, self.attributeNames,[[0]*len(self.rawdata.domain)])

        fullList = []
        permutationIndex = 0 # current permutation index
        totalPermutations = len(indPermutations.values())*len(attrReverse)
        if printFullOutput: print "Total permutations: ", totalPermutations

        if self.totalPossibilities == 0:
            self.totalPossibilities = totalPermutations
            self.triedPossibilities = 0

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])

        # which data items have all values valid
        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(attrListLength):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        count = 0
        for i in range(dataSize):
            if validData[i] == 1: count+=1
        
        if count < self.kNNOptimization.minExamples:
            print "Nr. of examples: ", str(count)
            print "Not enough examples in example table. Ignoring permutation..."
            print "------------------------------"
            self.triedPossibilities += 1
            if self.polyvizWidget: self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
            return []

        # create anchor for every attribute
        anchors = self.createAnchors(attrListLength)
        
        # store all sums
        sum = self.calculateAttrValuesSum(selectedGlobScaledData, len(self.rawdata), indices, validData)

        t = time.time()
        if self.kNNOptimization.getQualityMeasure() == CLASS_ACCURACY: text = "Classification accuracy"
        elif self.kNNOptimization.getQualityMeasure() == AVERAGE_CORRECT: text = "Average correct classification"
        else: text = "Brier score"

        # for every permutation compute how good it separates different classes            
        for permutation in indPermutations.values():
            for attrOrder in attrReverse:
                permutationIndex += 1
                #progressBar.setProgress(progressBar.progress()+1)
                table = orange.ExampleTable(domain)

                # calculate projections
                for i in range(dataSize):
                    if validData[i] == 0: continue
                    
                    x_i = 0.0; y_i = 0.0
                    for j in range(attrListLength):
                        index = permutation[j]
                        val = selectedGlobScaledData[index][i]
                        if attrOrder[index] == 0:
                            xDataAnchor = anchors[0][j]*(1-val) + anchors[0][(j+1)%attrListLength]*val
                            yDataAnchor = anchors[1][j]*(1-val) + anchors[1][(j+1)%attrListLength]*val
                        else:
                            xDataAnchor = anchors[0][j]*val + anchors[0][(j+1)%attrListLength]*(1-val)
                            yDataAnchor = anchors[1][j]*val + anchors[1][(j+1)%attrListLength]*(1-val)
                        x_i += xDataAnchor * (selectedGlobScaledData[index][i] / sum[i])
                        y_i += yDataAnchor * (selectedGlobScaledData[index][i] / sum[i])
                       
                    example = orange.Example(domain, [x_i, y_i, self.rawdata[i].getclass()])
                    table.append(example)

                accuracy = self.kNNOptimization.kNNComputeAccuracy(table)
                if table.domain.classVar.varType == orange.VarTypes.Discrete:
                    print "permutation %6d / %d. %s: %2.2f%%" % (permutationIndex, totalPermutations, text, accuracy)
                else:
                    print "permutation %6d / %d. MSE: %2.2f" % (permutationIndex, totalPermutations, accuracy) 
                
                # save the permutation
                fullList.append((accuracy, len(table), [self.attributeNames[i] for i in permutation], attrOrder))
                if addResultFunct and not self.kNNOptimization.onlyOnePerSubset:
                    addResultFunct(self.rawdata, accuracy, len(table), [self.attributeNames[i] for i in permutation], attrOrder)

                self.triedPossibilities += 1
                if self.polyvizWidget: self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))

        if printFullOutput:
            secs = time.time() - t
            print "Used time: %d min, %d sec" %(secs/60, secs%60)
            print "------------------------------"

        if self.kNNOptimization.onlyOnePerSubset:
            # return only the best attribute placements
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE: funct = max
            else: funct = min
            (acc, lenTable, attrList, attrOrder) = funct(fullList)
            if addResultFunct: addResultFunct(self.rawdata, acc, lenTable, attrList, attrOrder)
            return [(acc, lenTable, attrList, attrOrder)]
        else:
            return fullList
    """

    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getOptimalSeparation(self, attrListLength, attrReverseDict, projections, addResultFunct):
        if projections == []: return []
        
        dataSize = len(self.rawdata)
        anchors = self.createAnchors(attrListLength)
        
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])

        if self.kNNOptimization.getQualityMeasure() == CLASS_ACCURACY: text = "Classification accuracy"
        elif self.kNNOptimization.getQualityMeasure() == AVERAGE_CORRECT: text = "Average correct classification"
        else: text = "Brier score"

        for attrs in projections:
            attrs = attrs[1:]   # remove the value of this attribute subset
            
            indices = [];
            for attr in attrs:
                indices.append(self.attributeNames.index(attr))

            indPermutations = {}
            getPermutationList(indices, [], indPermutations, attrReverseDict == None)

            attrReverse = []
            if attrReverseDict != None: # if we received a dictionary, then we don't reverse attributes
                temp = [0] * len(self.rawdata.domain)
                for val in attrReverseDict.keys():
                    temp[self.attributeNames.index(val)] = attrReverseDict[val]
                attrReverse.append(temp)
            else:
                attrReverse = self.generateAttrReverseLists(attrList, self.attributeNames,[[0]*len(self.rawdata.domain)])

            permutationIndex = 0 # current permutation index
            totalPermutations = len(indPermutations.values())*len(attrReverse)
            
            # which data items have all values valid
            validData = self.getValidList(indices)

            count = sum(validData)
            if count < self.kNNOptimization.minExamples:
                print "Nr. of examples: ", str(count)
                print "Not enough examples in example table. Ignoring permutation..."
                self.triedPossibilities += len(indPermutations.keys())
                if self.polyvizWidget: self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
                continue
            
                        
            # store all sums
            sum_i = self.calculateAttrValuesSum(self.noJitteringScaledData, len(self.rawdata), indices, validData)

            tempList = []

            # for every permutation compute how good it separates different classes            
            for permutation in indPermutations.values():
                for attrOrder in attrReverse:
                    if self.kNNOptimization.isOptimizationCanceled(): return
                    permutationIndex += 1
                    table = orange.ExampleTable(domain)

                    # calculate projections
                    for i in range(dataSize):
                        if validData[i] == 0: continue
                        
                        x_i = 0.0; y_i = 0.0
                        for j in range(attrListLength):
                            index = permutation[j]
                            val = self.noJitteringScaledData[index][i]
                            if attrOrder[index] == 0:
                                xDataAnchor = anchors[0][j]*(1-val) + anchors[0][(j+1)%attrListLength]*val
                                yDataAnchor = anchors[1][j]*(1-val) + anchors[1][(j+1)%attrListLength]*val
                            else:
                                xDataAnchor = anchors[0][j]*val + anchors[0][(j+1)%attrListLength]*(1-val)
                                yDataAnchor = anchors[1][j]*val + anchors[1][(j+1)%attrListLength]*(1-val)
                            x_i += xDataAnchor * (self.noJitteringScaledData[index][i] / sum_i[i])
                            y_i += yDataAnchor * (self.noJitteringScaledData[index][i] / sum_i[i])
                           
                        example = orange.Example(domain, [x_i, y_i, self.rawdata[i].getclass()])
                        table.append(example)

                    accuracy = self.kNNOptimization.kNNComputeAccuracy(table)
                    if table.domain.classVar.varType == orange.VarTypes.Discrete:   print "permutation %6d / %d. %s: %2.2f%%" % (permutationIndex, totalPermutations, text, accuracy)
                    else:                                                           print "permutation %6d / %d. MSE: %2.2f" % (permutationIndex, totalPermutations, accuracy) 
                    
                    # save the permutation
                    tempList.append((accuracy, len(table), [self.attributeNames[i] for i in permutation], attrOrder))
                    if not self.kNNOptimization.onlyOnePerSubset and addResultFunct:
                        addResultFunct(self.rawdata, accuracy, len(table), [self.attributeNames[i] for i in permutation], attrOrder)

                    self.triedPossibilities += 1
                    self.polyvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))

            if self.kNNOptimization.onlyOnePerSubset:
                if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.kNNOptimization.getQualityMeasure() != BRIER_SCORE: funct = max
                else: funct = min
                (acc, lenTable, attrList, attrOrder) = funct(tempList)
                if addResultFunct: addResultFunct(self.rawdata, acc, lenTable, attrList, attrOrder)

    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWPolyvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
