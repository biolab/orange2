#
# OWParallelGraph.py
#

from OWGraph import *
from OWDistributions import *
from orngScaleData import *
from statc import pearsonr
from MLab import mean, std

NO_STATISTICS = 0
MEANS  = 1
MEDIAN = 2

class OWParallelGraph(OWGraph, orngScaleData):
    def __init__(self, parallelDlg, parent = None, name = None):
        OWGraph.__init__(self, parent, name)
        orngScaleData.__init__(self)

        self.parallelDlg = parallelDlg
        self.showDistributions = 0
        self.hidePureExamples = 1
        self.toolInfo = []
        self.toolRects = []
        self.useSplines = 0
        self.showStatistics = 0
        self.lastSelectedKey = 0
        self.enabledLegend = 0
        self.curvePoints = []       # save curve points in form [(y1, y2, ..., yi), (y1, y2, ... yi), ...] - used for sending selected and unselected points
        self.lineTracking = 0
        self.nonDataKeys = []

    def setData(self, data):
        OWGraph.setData(self, data)
        orngScaleData.setData(self, data)
        

    # update shown data. Set attributes, coloring by className ....
    def updateData(self, attributes, targetValue, midLabels = None, startIndex = 0, stopIndex = 0):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        self.removeTooltips()
        self.removeMarkers()

        self.curvePoints = []
        self.nonDataKeys = []

        blackColor = QColor(0, 0, 0)
        
        if self.scaledData == None:  return
        if len(attributes) == 0: return

        if (self.showDistributions == 1 or self.showAttrValues == 1) and self.rawdata.domain[attributes[-1]].varType == orange.VarTypes.Discrete:
            #self.setAxisScale(QwtPlot.xBottom, 0, len(attributes)-0.5, 1)
            #self.setAxisScale(QwtPlot.xBottom, 0, len(attributes)-1, 1)   # changed because of qwtplot's bug. only every second attribute label was shown if -0.5 was used
            self.setAxisScale(QwtPlot.xBottom, startIndex, stopIndex-1, 1)   # changed because of qwtplot's bug. only every second attribute label was shown if -0.5 was used
        else:
            #self.setAxisScale(QwtPlot.xBottom, 0, len(attributes)-1.0, 1)
            self.setAxisScale(QwtPlot.xBottom, startIndex, stopIndex-1, 1)

        if self.showAttrValues: self.setAxisScale(QwtPlot.yLeft, -0.04, 1.04, 1)
        elif midLabels:         self.setAxisScale(QwtPlot.yLeft, 0, 1.04, 1)
        else:                   self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)

##        if (self.rawdata.domain.classVar and self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous and self.enabledLegend) or (len(attributes) and self.rawdata.domain[attributes[-1]].varType == orange.VarTypes.Discrete and (self.showDistributionValues or self.showAttrValues)):
##            self.addCurve("edge", self.canvasBackground(), self.canvasBackground(), 1, QwtCurve.Lines, QwtSymbol.None, xData = [len(attributes),len(attributes)], yData = [1,1])

        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw([self.getAttributeLabel(attr) for attr in attributes]))
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        self.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)  # hide ticks
        self.axisScaleDraw(QwtPlot.xBottom).setOptions(0)           # hide horizontal line representing x axis
        self.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)
        self.axisScaleDraw(QwtPlot.yLeft).setOptions(0) 
        
        self.setAxisMaxMajor(QwtPlot.xBottom, len(attributes))
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)

        classNameIndex = -1
        continuousClass = 0
        if self.rawdata.domain.classVar:
            classNameIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
            continuousClass = self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous

        haveSubsetData = 0
        if self.subsetData and self.rawdata and self.subsetData.domain == self.rawdata.domain:
            haveSubsetData = 1
        
        length = len(attributes)
        indices = [self.attributeNameIndex[label] for label in attributes]
        xs = range(length)
        dataSize = len(self.scaledData[0])
        
        if self.rawdata.domain.classVar and not continuousClass:
            classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)

        # ############################################
        # if self.hidePureExamples == 1 we have to calculate where to stop drawing lines
        # we do this by adding a integer meta attribute, that for each example stores attribute index, where we stop drawing lines
        # ############################################
        lastIndex = indices[-1]
        dataStop = None
        if self.hidePureExamples == 1 and self.rawdata.domain.classVar and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            # add a meta attribute if it doesn't exist yet
            dataStop = dataSize * [lastIndex]  # array where we store index value for each data value where to stop drawing
            metaid = orange.newmetaid()
            self.rawdata.domain.addmeta(metaid, orange.IntVariable("ItemIndex"))
            for i in range(dataSize): self.rawdata[i].setmeta(metaid, i)

            for i in range(length):
                if self.rawdata.domain[indices[i]].varType != orange.VarTypes.Discrete or attributes[i] == self.rawdata.domain.classVar.name: continue

                attr = self.rawdata.domain[indices[i]]                
                for attrVal in attr.values:
                    tempData = self.rawdata.select({attr.name:attrVal})

                    ind = 0
                    while ind < len(tempData):
                        if dataStop[int(tempData[ind].getmeta(metaid))] == lastIndex:
                            val = tempData[ind][classNameIndex]
                            break
                        ind += 1
                        
                    # do all instances belong to the same class?
                    while ind < len(tempData):
                        if dataStop[int(tempData[ind].getmeta(metaid))] != lastIndex: ind += 1; continue
                        if val != tempData[ind][classNameIndex]: break
                        ind += 1

                    # if all examples belong to one class we repair the meta variable values
                    if ind >= len(tempData):
                        for item in tempData:
                            index = int(item.getmeta(metaid))
                            if dataStop[index] == lastIndex:
                                dataStop[index] = indices[i]
            self.rawdata.domain.removemeta(metaid)

        # first create all curves
        curves = [[],[]]

        # ############################################
        # draw the data
        # ############################################
        if not haveSubsetData: subsetReferencesToDraw = []
        else:                  subsetReferencesToDraw = [example.reference() for example in self.subsetData]
        validData = self.getValidList(indices)
        for i in range(dataSize):
            if not validData[i]:
                self.curvePoints.append([]) # add an empty list
                continue
                        
            curve = QwtPlotCurve(self)
            if targetValue != None:
                if self.rawdata[i].getclass().value == targetValue:
                    newColor = self.colorTargetValue
                    curves[1].append(curve)
                else:
                    newColor = self.colorNonTargetValue
                    curves[0].append(curve)
            else:
                if not self.rawdata.domain.classVar: newColor = blackColor
                elif continuousClass:
                    newColor = self.contPalette[self.noJitteringScaledData[classNameIndex][i]]
                else:
                    newColor = self.discPalette[classValueIndices[self.rawdata[i].getclass().value]]

                if haveSubsetData and self.rawdata[i].reference() not in subsetReferencesToDraw:
                    newColor.setHsv(newColor.hsv()[0], 50, newColor.hsv()[2])
                    curves[0].append(curve)
                else:
                    curves[1].append(curve)
                    if subsetReferencesToDraw:
                        subsetReferencesToDraw.remove(self.rawdata[i].reference())
                    
            curve.setPen(QPen(newColor, 1))
            if not dataStop:
                ys = [self.scaledData[index][i] for index in indices]
            else:
                ys = []
                for index in indices:
                    ys.append(self.scaledData[index][i])
                    if index == dataStop[i]: break
            curve.setData(xs, ys)
            self.curvePoints.append(ys)  # save curve points
            if self.useSplines:
                curve.setStyle(QwtCurve.Spline)

        # if we have a data subset that contains examples that don't exist in the original dataset we show them here
        
        if subsetReferencesToDraw != []:
            for i in range(len(self.subsetData)):
                if not self.subsetData[i].reference() in subsetReferencesToDraw: continue
                subsetReferencesToDraw.remove(self.subsetData[i].reference())
                
                # check if has missing values
                if 1 in [self.subsetData[i][ind].isSpecial() for ind in indices]: continue

                curve = QwtPlotCurve(self)
                if targetValue != None:
                    if self.subsetData[i].getclass().value == targetValue:
                        newColor = self.colorTargetValue
                        curves[1].append(curve)
                    else:
                        newColor = self.colorNonTargetValue
                        curves[0].append(curve)
                else:
                    if not self.subsetData.domain.classVar or self.subsetData[i].getclass().isSpecial():
                        newColor = blackColor
                    elif continuousClass:
                        newColor = self.contPalette[self.scaleExampleValue(self.subsetData[i], classNameIndex)]
                    else:
                        newColor = self.discPalette[classValueIndices[self.subsetData[i].getclass().value]]
                    curves[1].append(curve)
                    
                curve.setPen(QPen(newColor, 1))
                ys = [self.scaledData[index][i] for index in indices]
                
                curve.setData(xs, ys)
                if self.useSplines:
                    curve.setStyle(QwtCurve.Spline)
                
        # now add all curves. First add the gray curves (they will be shown in the back) and then the blue (target value) curves (shown in front)
        for curve in curves[0]: self.insertCurve(curve)
        for curve in curves[1]: self.insertCurve(curve)


        # ############################################
        # do we want to show distributions with discrete attributes
        if self.showDistributions and self.rawdata.domain.classVar and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            self.showDistributionValues(targetValue, validData, indices, dataStop)
            
        #curve = RectanglePlotCurve(self, pen = QPen(QColor(0, 0, 0)), brush = QBrush(QBrush.NoBrush))
        #ckey = self.insertCurve(curve)
        #self.setCurveData(ckey, [1,1], [2,2])

        # ############################################
        # draw vertical lines that represent attributes
        for i in range(len(attributes)):
            newCurveKey = self.insertCurve(attributes[i])
            self.nonDataKeys.append(newCurveKey)
            self.setCurveData(newCurveKey, [i,i], [0,1])
            pen = self.curve(newCurveKey).pen(); pen.setWidth(2); self.curve(newCurveKey).setPen(pen)
            if self.showAttrValues == 1:
                attr = self.rawdata.domain[attributes[i]]
                if attr.varType == orange.VarTypes.Continuous:
                    strVal = "%%.%df" % (attr.numberOfDecimals) % (self.attrValues[attr.name][0])
                    mkey1 = self.insertMarker(strVal)
                    self.marker(mkey1).setXValue(i)
                    self.marker(mkey1).setYValue(0.0-0.01)
                    strVal = "%%.%df" % (attr.numberOfDecimals) % (self.attrValues[attr.name][1])
                    mkey2 = self.insertMarker(strVal)
                    self.marker(mkey2).setXValue(i)
                    self.marker(mkey2).setYValue(1.0+0.01)
                    if i == 0:
                        self.marker(mkey1).setLabelAlignment(Qt.AlignRight + Qt.AlignBottom)
                        self.marker(mkey2).setLabelAlignment(Qt.AlignRight + Qt.AlignTop)
                    elif i == len(attributes)-1:
                        self.marker(mkey1).setLabelAlignment(Qt.AlignLeft + Qt.AlignBottom)
                        self.marker(mkey2).setLabelAlignment(Qt.AlignLeft + Qt.AlignTop)
                    else:
                        self.marker(mkey1).setLabelAlignment(Qt.AlignHCenter + Qt.AlignBottom)
                        self.marker(mkey2).setLabelAlignment(Qt.AlignHCenter + Qt.AlignTop)
                elif attr.varType == orange.VarTypes.Discrete:
                    attrVals = getVariableValuesSorted(self.rawdata, attributes[i])
                    valsLen = len(attrVals)
                    for pos in range(len(attrVals)):
                        # show a rectangle behind the marker
                        mkey = self.insertMarker(nonTransparentMarker(QColor(255,255,255), self))
                        self.marker(mkey).setLabel(attrVals[pos])
                        font = self.marker(mkey).font(); font.setBold(1); self.marker(mkey).setFont(font)
                        self.marker(mkey).setXValue(i+0.01)
                        self.marker(mkey).setYValue(float(1+2*pos)/float(2*valsLen))
                        self.marker(mkey).setLabelAlignment(Qt.AlignRight + Qt.AlignVCenter)

        # ##############################################
        # show lines that represent standard deviation or quartiles
        # ##############################################
        if self.showStatistics:
            data = []
            for i in range(length):
                if self.rawdata.domain[indices[i]].varType != orange.VarTypes.Continuous:
                    data.append([()])
                    continue  # only for continuous attributes
                array = Numeric.compress(Numeric.equal(self.validDataArray[indices[i]], 1), self.scaledData[indices[i]])  # remove missing values
                
                if classNameIndex == -1 or continuousClass:    # no class
                    if self.showStatistics == MEANS:
                        m = mean(array)
                        dev = std(array)
                        data.append([(m-dev, m, m+dev)])
                    elif self.showStatistics == MEDIAN:
                        sorted = Numeric.sort(array)
                        data.append([(sorted[int(len(sorted)/4.0)], sorted[int(len(sorted)/2.0)], sorted[int(len(sorted)*0.75)])])
                else:
                    curr = []
                    classValues = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
                    for c in range(len(classValues)):
                        scaledVal = ((classValueIndices[classValues[c]] * 2) + 1) / float(2*len(classValueIndices))
                        nonMissingValues = Numeric.compress(Numeric.equal(self.validDataArray[indices[i]], 1), self.noJitteringScaledData[classNameIndex])  # remove missing values
                        arr_c = Numeric.compress(Numeric.equal(nonMissingValues, scaledVal), array)
                        if len(arr_c) == 0:
                            curr.append(()); continue
                        if self.showStatistics == MEANS:
                            m = mean(arr_c)
                            dev = std(arr_c)
                            curr.append((m-dev, m, m+dev))
                        elif self.showStatistics == MEDIAN:
                            sorted = Numeric.sort(arr_c)
                            curr.append((sorted[int(len(arr_c)/4.0)], sorted[int(len(arr_c)/2.0)], sorted[int(len(arr_c)*0.75)]))
                    data.append(curr)

            # draw vertical lines
            for i in range(len(data)):
                for c in range(len(data[i])):
                    if data[i][c] == (): continue
                    x = i - 0.03*(len(data[i])-1)/2.0 + c*0.03
                    col = self.discPalette[c]
                    self.nonDataKeys.append(self.addCurve("", col, col, 3, QwtCurve.Lines, QwtSymbol.Diamond, xData = [x,x,x], yData = [data[i][c][0], data[i][c][1], data[i][c][2]], lineWidth = 4))
                    self.nonDataKeys.append(self.addCurve("", col, col, 1, QwtCurve.Lines, QwtSymbol.None, xData = [x-0.03, x+0.03], yData = [data[i][c][0], data[i][c][0]], lineWidth = 4))
                    self.nonDataKeys.append(self.addCurve("", col, col, 1, QwtCurve.Lines, QwtSymbol.None, xData = [x-0.03, x+0.03], yData = [data[i][c][1], data[i][c][1]], lineWidth = 4))
                    self.nonDataKeys.append(self.addCurve("", col, col, 1, QwtCurve.Lines, QwtSymbol.None, xData = [x-0.03, x+0.03], yData = [data[i][c][2], data[i][c][2]], lineWidth = 4))

            # draw lines with mean/median values
            classCount = 1
            if classNameIndex == -1 or continuousClass:    classCount = 1 # no class
            else: classCount = len(self.rawdata.domain.classVar.values)
            for c in range(classCount):
                diff = - 0.03*(classCount-1)/2.0 + c*0.03
                ys = []
                xs = []
                for i in range(len(data)):
                    if data[i] != [()]: ys.append(data[i][c][1]); xs.append(i+diff)
                    else:
                        if len(xs) > 1: self.nonDataKeys.append(self.addCurve("", self.discPalette[c], self.discPalette[c], 1, QwtCurve.Lines, QwtSymbol.None, xData = xs, yData = ys, lineWidth = 4))
                        xs = []; ys = []
                self.nonDataKeys.append(self.addCurve("", self.discPalette[c], self.discPalette[c], 1, QwtCurve.Lines, QwtSymbol.None, xData = xs, yData = ys, lineWidth = 4))

        
        # ##################################################
        # show labels in the middle of the axis
        if midLabels:
            for j in range(len(midLabels)):
                mkey = self.insertMarker(midLabels[j])
                self.marker(mkey).setXValue(j+0.5)
                self.marker(mkey).setYValue(1.0)
                self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignTop)

        # show the legend
        if self.enabledLegend == 1 and self.rawdata.domain.classVar:
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                varValues = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
                self.addCurve("<b>" + self.rawdata.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.None, enableLegend = 1)
                for ind in range(len(varValues)):
                    self.addCurve(varValues[ind], self.discPalette[ind], self.discPalette[ind], 15, symbol = QwtSymbol.Rect, enableLegend = 1)
            else:
                l = len(attributes)-1
                xs = [l*1.15, l*1.20, l*1.20, l*1.15]
                count = 200
                for i in range(count):
                    y = i/float(count)
                    col = self.contPalette[y]
                    curve = PolygonCurve(self, QPen(col), QBrush(col))
                    newCurveKey = self.insertCurve(curve)
                    self.nonDataKeys.append(newCurveKey)
                    self.setCurveData(newCurveKey, xs, [y,y, y+height, y+height])

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawdata.domain.classVar.name]
                decimals = self.rawdata.domain.classVar.numberOfDecimals
                self.addMarker("%%.%df" % (decimals) % (minVal), xs[0] - l*0.02, 0.04, Qt.AlignLeft)
                self.addMarker("%%.%df" % (decimals) % (maxVal), xs[0] - l*0.02, 1.0 - 0.04, Qt.AlignLeft)


    # ##########################################
    # SHOW DISTRIBUTION BAR GRAPH
    def showDistributionValues(self, targetValue, validData, indices, dataStop):
        # get index of class         
        classNameIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]

        # create color table            
        count = len(self.rawdata.domain.classVar.values)
        #if count < 1: count = 1.0

        # we create a hash table of possible class values (happens only if we have a discrete class)
        classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)
        classValueSorted  = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)

        self.toolInfo = []        
        for graphAttrIndex in range(len(indices)):
            index = indices[graphAttrIndex]
            if self.rawdata.domain[index].varType != orange.VarTypes.Discrete: continue
            attr = self.rawdata.domain[index]
            attrLen = len(attr.values)
            
            values = []
            totals = [0] * attrLen

            # we create a hash table of variable values and their indices
            variableValueIndices = getVariableValueIndices(self.rawdata, index)
            variableValueSorted = getVariableValuesSorted(self.rawdata, index)
            
            for i in range(count):
                values.append([0] * attrLen)

            stop = indices[:graphAttrIndex]
            for i in range(len(self.rawdata)):
                if dataStop and self.hidePureExamples == 1 and dataStop[i] in stop: continue
                if validData[i] == 0: continue
                # processing for distributions
                attrIndex = variableValueIndices[self.rawdata[i][index].value]
                classIndex = classValueIndices[self.rawdata[i][classNameIndex].value]
                totals[attrIndex] += 1
                values[classIndex][attrIndex] += 1

            # calculate maximum value of all values - needed for scaling
            maximum = 1
            for i in range(len(values)):
                for j in range(len(values[i])):
                    if values[i][j] > maximum: maximum = values[i][j]

            # calculate the sum of totals - needed for tooltips
            sumTotals = 0
            for val in totals: sumTotals += val

            # save info for tooltips
            for i in range(attrLen):
                list= []
                for j in range(count):
                    list.append((classValueSorted[j], values[j][i]))
                list.reverse()
                y_start = float(i+1)/float(attrLen); y_end = float(i)/float(attrLen)
                x_start = float(graphAttrIndex) - 0.45; x_end = float(graphAttrIndex) + 0.45
                item = (self.rawdata.domain[index].name, variableValueSorted[i], totals[i], sumTotals, list, (x_start,x_end), (y_start, y_end))
                self.toolInfo.append(item)


            # create bar curve
            for i in range(count):
                if targetValue != None:
                    if classValueSorted[i] == targetValue: newColor = self.colorTargetValue
                    else: newColor = self.colorNonTargetValue
                else:
                    newColor = self.discPalette[i]
                curve = RectanglePlotCurve(self, pen = QPen(newColor), brush = QBrush(newColor))
                
                xData = []; yData = []
                for j in range(attrLen):
                    width = float(values[i][j]*0.5) / float(maximum)
                    interval = 1.0/float(2*attrLen)
                    yOff = float(1.0 + 2.0*j)/float(2*attrLen)
                    height = 0.7/float(count*attrLen)

                    yLowBott = yOff - float(count*height)/2.0 + i*height
                    xData.append(graphAttrIndex)
                    xData.append(graphAttrIndex + width)
                    yData.append(yLowBott)
                    yData.append(yLowBott + height)

                ckey = self.insertCurve(curve)
                self.nonDataKeys.append(ckey)
                self.setCurveData(ckey, xData, yData)
        self.addTooltips()
        
    def addTooltips(self):
        for i in range(len(self.toolInfo)):
            (name, value, total, sumTotals, lista, (x_start,x_end), (y_start, y_end)) = self.toolInfo[i]
            if total == 0: continue
            tooltipText = "Attribute: <b>%s</b><br>Value: <b>%s</b><br>Total instances: <b>%i</b> (%.1f%%)<br>Class distribution:<br>" % (name, value, total, 100.0*float(total)/float(sumTotals))
            for j in range(len(lista)):
                (val, count) = lista[j]
                tooltipText += "<b>%s</b> : <b>%i</b> (%.1f%%)" % (val, count, 100.0*float(count)/float(total))
                if j != len(lista)-1 : tooltipText += "<br>"
            x_1 = self.transform(QwtPlot.xBottom, x_start)
            x_2 = self.transform(QwtPlot.xBottom, x_end)
            y_1 = self.transform(QwtPlot.yLeft, y_start)
            y_2 = self.transform(QwtPlot.yLeft, y_end)
            rect = QRect(x_1, y_1, x_2-x_1, y_2-y_1)
            self.toolRects.append(rect)            
            QToolTip.add(self, rect, tooltipText)

    def removeTooltips(self):
        for rect in self.toolRects:
            QToolTip.remove(self, rect)
        self.toolRects = []


    # if user clicked between two lines send signal that 
    def staticMouseClick(self, e):
        if self.parallelDlg:
            x1 = int(self.invTransform(QwtPlot.xBottom, e.x()))
            axis = self.axisScaleDraw(QwtPlot.xBottom)
            self.parallelDlg.sendAttributeSelection([str(axis.label(x1)), str(axis.label(x1+1))])

    def updateLayout(self):
        OWGraph.updateLayout(self)
        self.removeTooltips()
        self.addTooltips()

    """
    def updateAxes(self):
        OWGraph.updateAxes()        
        self.removeTooltips()
        self.addTooltips()
    """
    def updateTooltips(self):
        self.removeTooltips()
        self.addTooltips()

    # if we zoomed, we have to update tooltips    
    def onMouseReleased(self, e):
        OWGraph.onMouseReleased(self, e)
        self.updateTooltips()

    def onMouseMoved(self, e):
        if self.mouseCurrentlyPressed:
            OWGraph.onMouseMoved(self, e)
            return
        else:
            (key, foo1, x, y, foo2) = self.closestCurve(e.pos().x(), e.pos().y())
            dist = abs(x-self.invTransform(QwtPlot.xBottom, e.x())) + abs(y-self.invTransform(QwtPlot.yLeft, e.y()))

            if self.lineTracking:
                if (dist >= 0.1 or key != self.lastSelectedKey) and self.lastSelectedKey not in self.nonDataKeys:
                    existingPen = self.curvePen(self.lastSelectedKey)
                    existingPen.setWidth(1)
                    self.setCurvePen(self.lastSelectedKey, existingPen)
                    self.lastSelectedKey = -1

                if dist < 0.1 and key != self.lastSelectedKey and key not in self.nonDataKeys:
                    self.lastSelectedKey = key
                    existingPen = self.curvePen(self.lastSelectedKey)
                    existingPen.setWidth(3)
                    self.setCurvePen(self.lastSelectedKey, existingPen)
                    
            else:
                OWGraph.onMouseMoved(self, e)
                return
            
            OWGraph.onMouseMoved(self, e)
            self.replot()

    def getSelectionsAsExampleTables(self, targetValue = None):
        if not self.rawdata:
            return (None, None)
        if self.selectionCurveKeyList == []:
            return (None, self.rawdata)
        
        selIndices = []
        unselIndices = range(len(self.rawdata))

        for i in range(len(self.curvePoints)):
            for j in range(len(self.curvePoints[i])):
                if self.isPointSelected(j, self.curvePoints[i][j]):
                    if targetValue and targetValue != int(self.rawdata[i].getclass()): continue
                    selIndices.append(i)
                    unselIndices.pop(i)
            
        selected = self.rawdata.getitemsref(selIndices)
        unselected = self.rawdata.getitemsref(unselIndices)
        
        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None

        return (selected, unselected)