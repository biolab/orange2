#
# OWParallelGraph.py
#
from OWGraph import *
from OWDistributions import *
from orngScaleData import *
from statc import pearsonr

NO_STATISTICS = 0
MEANS  = 1
MEDIAN = 2

class OWParallelGraph(OWGraph, orngScaleData):
    def __init__(self, parallelDlg, parent = None, name = None):
        OWGraph.__init__(self, parent, name)
        orngScaleData.__init__(self)

        self.parallelDlg = parallelDlg
        self.showDistributions = 0
        self.toolInfo = []
        self.toolRects = []
        self.useSplines = 0
        self.showStatistics = 0
        self.lastSelectedCurve = None
        self.enabledLegend = 0
        self.curvePoints = []       # save curve points in form [(y1, y2, ..., yi), (y1, y2, ... yi), ...] - used for sending selected and unselected points
        self.lineTracking = 0
        self.nonDataCurves = []
        self.enableGridXB(0)
        self.enableGridYL(0)
        self.domainContingency = None
        self.useAntiAliasing = 0

    def setData(self, data, subsetData = None, **args):
        OWGraph.setData(self, data)
        orngScaleData.setData(self, data, subsetData, **args)
        self.domainContingency = None


    # update shown data. Set attributes, coloring by className ....
    def updateData(self, attributes, midLabels = None, startIndex = 0, stopIndex = 0):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        self.removeTooltips()
        self.removeMarkers()

        self.curvePoints = []
        self.nonDataCurves = []

        blackColor = QColor(0, 0, 0)

        if self.scaledData == None:  return
        if len(attributes) == 0: return

        if (self.showDistributions == 1 or self.showAttrValues == 1) and self.rawData.domain[attributes[-1]].varType == orange.VarTypes.Discrete:
            self.setAxisScale(QwtPlot.xBottom, startIndex, stopIndex-1, 1)   # changed because of qwtplot's bug. only every second attribute label was shown if -0.5 was used
        else:
            self.setAxisScale(QwtPlot.xBottom, startIndex, stopIndex-1, 1)

        if self.showAttrValues: self.setAxisScale(QwtPlot.yLeft, -0.04, 1.04, 1)
        elif midLabels:         self.setAxisScale(QwtPlot.yLeft, 0, 1.04, 1)
        else:                   self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)

        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw([self.getAttributeLabel(attr) for attr in attributes]))
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        self.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Backbone, 0)
        self.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Ticks, 0)
        self.axisScaleDraw(QwtPlot.yLeft).enableComponent(QwtScaleDraw.Backbone, 0)
        self.axisScaleDraw(QwtPlot.yLeft).enableComponent(QwtScaleDraw.Ticks, 0)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(attributes))
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)

        classNameIndex = -1
        continuousClass = 0
        if self.rawData.domain.classVar:
            classNameIndex = self.attributeNameIndex[self.rawData.domain.classVar.name]
            continuousClass = self.rawData.domain.classVar.varType == orange.VarTypes.Continuous

        haveSubsetData = self.rawSubsetData != None

        length = len(attributes)
        indices = [self.attributeNameIndex[label] for label in attributes]
        xs = range(length)
        dataSize = len(self.scaledData[0])

        if self.rawData.domain.classVar and not continuousClass:
            classValueIndices = getVariableValueIndices(self.rawData, self.rawData.domain.classVar.name)

        # first create all curves
        curves = [[],[]]

        # ############################################
        # draw the data
        # ############################################
        subsetReferencesToDraw = haveSubsetData and [self.rawSubsetData[i].reference() for i in self.getValidSubsetIndices(indices)] or []
        validData = self.getValidList(indices)
        xPointsToAdd = {}
        yPointsToAdd = {}

        for i in range(dataSize):
            if not validData[i]:
                self.curvePoints.append([]) # add an empty list
                continue

            curve = QwtPlotCurve("")
            curve.setItemAttribute(QwtPlotItem.Legend, 0)
            if self.useAntiAliasing:
                curve.setRenderHint(QwtPlotItem.RenderAntialiased)

            if not self.rawData.domain.classVar:
                newColor = QColor(blackColor)
            elif continuousClass:
                newColor = QColor(self.contPalette[self.noJitteringScaledData[classNameIndex][i]])
            else:
                newColor = QColor(self.discPalette[classValueIndices[self.rawData[i].getclass().value]])

            # if we have subset data then use alpha2 for main data and alpha for subset data
            if haveSubsetData:
                if self.rawData[i].reference() not in subsetReferencesToDraw:
                    newColor.setAlpha(self.alphaValue2)
                    curves[0].append(curve)
                else:
                    newColor.setAlpha(self.alphaValue)
                    curves[1].append(curve)
                    if subsetReferencesToDraw:
                        subsetReferencesToDraw.remove(self.rawData[i].reference())
            else:
                curves[1].append(curve)

            curve.setPen(QPen(newColor, 1))
            ys = [self.scaledData[index][i] for index in indices]

            curve.setData(xs, ys)
            self.curvePoints.append(ys)  # save curve points
            if self.useSplines:
                curve.setCurveAttribute(QwtPlotCurve.Fitted)

        # if we have a data subset that contains examples that don't exist in the original dataset we show them here
        if subsetReferencesToDraw != []:
            validSubsetData = self.getValidSubsetList(indices)

            for i in range(len(self.rawSubsetData)):
                if not validSubsetData[i]: continue
                if not self.rawSubsetData[i].reference() in subsetReferencesToDraw: continue
                subsetReferencesToDraw.remove(self.rawSubsetData[i].reference())

                curve = QwtPlotCurve("")
                curve.setItemAttribute(QwtPlotItem.Legend, 0)
                if self.useAntiAliasing:
                    curve.setRenderHint(QwtPlotItem.RenderAntialiased)
                if not self.rawSubsetData.domain.classVar or self.rawSubsetData[i].getclass().isSpecial():
                    newColor = QColor(blackColor)
                elif continuousClass:
                    newColor = QColor(self.contPalette[self.scaleExampleValue(self.rawSubsetData[i], classNameIndex)])
                else:
                    newColor = QColor(self.discPalette[classValueIndices[self.rawSubsetData[i].getclass().value]])
                curves[1].append(curve)

                newColor.setAlpha(self.alphaValue)
                curve.setPen(QPen(newColor, 1))
                curve.setData(xs, [self.scaledSubsetData[index][i] for index in indices])

                if self.useSplines:
                    curve.setCurveAttribute(QwtPlotCurve.Fitted)

        # now add all curves. First add the main data curves and then the subset data curves (shown in front)
        for curve in curves[0]: curve.attach(self)
        for curve in curves[1]: curve.attach(self)

        # ############################################
        # do we want to show distributions with discrete attributes
        if self.showDistributions and self.rawData.domain.classVar and self.rawData.domain.classVar.varType == orange.VarTypes.Discrete:
            self.showDistributionValues(validData, indices)

        # ############################################
        # draw vertical lines that represent attributes
        for i in range(len(attributes)):
            self.nonDataCurves.append(self.addCurve("", lineWidth = 2, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [i,i], yData = [0,1]))
            if self.showAttrValues == 1:
                attr = self.rawData.domain[attributes[i]]
                if attr.varType == orange.VarTypes.Continuous:
                    strVal1 = "%%.%df" % (attr.numberOfDecimals) % (self.attrValues[attr.name][0])
                    strVal2 = "%%.%df" % (attr.numberOfDecimals) % (self.attrValues[attr.name][1])
                    align1 = i == 0 and Qt.AlignRight | Qt.AlignBottom or i == len(attributes)-1 and Qt.AlignLeft | Qt.AlignBottom or Qt.AlignHCenter | Qt.AlignBottom
                    align2 = i == 0 and Qt.AlignRight | Qt.AlignTop or i == len(attributes)-1 and Qt.AlignLeft | Qt.AlignTop or Qt.AlignHCenter | Qt.AlignTop
                    self.addMarker(strVal1, i, 0.0-0.01, alignment = align1)
                    self.addMarker(strVal2, i, 1.0+0.01, alignment = align2)

                elif attr.varType == orange.VarTypes.Discrete:
                    attrVals = getVariableValuesSorted(self.rawData, attributes[i])
                    valsLen = len(attrVals)
                    for pos in range(len(attrVals)):
                        # show a rectangle behind the marker
                        self.addMarker(attrVals[pos], i+0.01, float(1+2*pos)/float(2*valsLen), alignment = Qt.AlignRight | Qt.AlignVCenter, bold = 1, brushColor = Qt.white)

        # ##############################################
        # show lines that represent standard deviation or quartiles
        # ##############################################
        if self.showStatistics:
            data = []
            for i in range(length):
                if self.rawData.domain[indices[i]].varType != orange.VarTypes.Continuous:
                    data.append([()])
                    continue  # only for continuous attributes
                array = numpy.compress(numpy.equal(self.validDataArray[indices[i]], 1), self.scaledData[indices[i]])  # remove missing values

                if classNameIndex == -1 or continuousClass:    # no class
                    if self.showStatistics == MEANS:
                        m = array.mean()
                        dev = array.std()
                        data.append([(m-dev, m, m+dev)])
                    elif self.showStatistics == MEDIAN:
                        sorted = numpy.sort(array)
                        if len(sorted) > 0:
                            data.append([(sorted[int(len(sorted)/4.0)], sorted[int(len(sorted)/2.0)], sorted[int(len(sorted)*0.75)])])
                        else:
                            data.append([(0,0,0)])
                else:
                    curr = []
                    classValues = getVariableValuesSorted(self.rawData, self.rawData.domain.classVar.name)
                    for c in range(len(classValues)):
                        scaledVal = ((classValueIndices[classValues[c]] * 2) + 1) / float(2*len(classValueIndices))
                        nonMissingValues = numpy.compress(numpy.equal(self.validDataArray[indices[i]], 1), self.noJitteringScaledData[classNameIndex])  # remove missing values
                        arr_c = numpy.compress(numpy.equal(nonMissingValues, scaledVal), array)
                        if len(arr_c) == 0:
                            curr.append((0,0,0)); continue
                        if self.showStatistics == MEANS:
                            m = arr_c.mean()
                            dev = arr_c.std()
                            curr.append((m-dev, m, m+dev))
                        elif self.showStatistics == MEDIAN:
                            sorted = numpy.sort(arr_c)
                            curr.append((sorted[int(len(arr_c)/4.0)], sorted[int(len(arr_c)/2.0)], sorted[int(len(arr_c)*0.75)]))
                    data.append(curr)

            # draw vertical lines
            for i in range(len(data)):
                for c in range(len(data[i])):
                    if data[i][c] == (): continue
                    x = i - 0.03*(len(data[i])-1)/2.0 + c*0.03
                    col = QColor(self.discPalette[c])
                    col.setAlpha(self.alphaValue2)
                    self.nonDataCurves.append(self.addCurve("", col, col, 3, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [x,x,x], yData = [data[i][c][0], data[i][c][1], data[i][c][2]], lineWidth = 4))
                    self.nonDataCurves.append(self.addCurve("", col, col, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [x-0.03, x+0.03], yData = [data[i][c][0], data[i][c][0]], lineWidth = 4))
                    self.nonDataCurves.append(self.addCurve("", col, col, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [x-0.03, x+0.03], yData = [data[i][c][1], data[i][c][1]], lineWidth = 4))
                    self.nonDataCurves.append(self.addCurve("", col, col, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [x-0.03, x+0.03], yData = [data[i][c][2], data[i][c][2]], lineWidth = 4))

            # draw lines with mean/median values
            classCount = 1
            if classNameIndex == -1 or continuousClass:    classCount = 1 # no class
            else: classCount = len(self.rawData.domain.classVar.values)
            for c in range(classCount):
                diff = - 0.03*(classCount-1)/2.0 + c*0.03
                ys = []
                xs = []
                for i in range(len(data)):
                    if data[i] != [()]: ys.append(data[i][c][1]); xs.append(i+diff)
                    else:
                        if len(xs) > 1:
                            col = QColor(self.discPalette[c])
                            col.setAlpha(self.alphaValue2)
                            self.nonDataCurves.append(self.addCurve("", col, col, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = xs, yData = ys, lineWidth = 4))
                        xs = []; ys = []
                col = QColor(self.discPalette[c])
                col.setAlpha(self.alphaValue2)
                self.nonDataCurves.append(self.addCurve("", col, col, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = xs, yData = ys, lineWidth = 4))


        # ##################################################
        # show labels in the middle of the axis
        if midLabels:
            for j in range(len(midLabels)):
                self.addMarker(midLabels[j], j+0.5, 1.0, alignment = Qt.AlignCenter | Qt.AlignTop)

        # show the legend
        if self.enabledLegend == 1 and self.rawData.domain.classVar:
            if self.rawData.domain.classVar.varType == orange.VarTypes.Discrete:
                varValues = getVariableValuesSorted(self.rawData, self.rawData.domain.classVar.name)
                self.addCurve("<b>" + self.rawData.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.NoSymbol, enableLegend = 1)
                for ind in range(len(varValues)):
                    self.addCurve(varValues[ind], self.discPalette[ind], self.discPalette[ind], 15, symbol = QwtSymbol.Rect, enableLegend = 1)
            else:
                l = len(attributes)-1
                xs = [l*1.15, l*1.20, l*1.20, l*1.15]
                count = 200; height = 1/200.
                for i in range(count):
                    y = i/float(count)
                    col = self.contPalette[y]
                    curve = PolygonCurve(QPen(col), QBrush(col), xData = xs, yData = [y,y, y+height, y+height])
                    curve.attach(self)
                    self.nonDataCurves.append(curve)

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawData.domain.classVar.name]
                decimals = self.rawData.domain.classVar.numberOfDecimals
                self.addMarker("%%.%df" % (decimals) % (minVal), xs[0] - l*0.02, 0.04, Qt.AlignLeft)
                self.addMarker("%%.%df" % (decimals) % (maxVal), xs[0] - l*0.02, 1.0 - 0.04, Qt.AlignLeft)

        self.replot()


    # ##########################################
    # SHOW DISTRIBUTION BAR GRAPH
    def showDistributionValues(self, validData, indices):
        # get index of class
        classNameIndex = self.attributeNameIndex[self.rawData.domain.classVar.name]

        # create color table
        clsCount = len(self.rawData.domain.classVar.values)
        #if clsCount < 1: clsCount = 1.0

        # we create a hash table of possible class values (happens only if we have a discrete class)
        classValueIndices = getVariableValueIndices(self.rawData, self.rawData.domain.classVar.name)
        classValueSorted  = getVariableValuesSorted(self.rawData, self.rawData.domain.classVar.name)
        if self.domainContingency == None:
            self.domainContingency = orange.DomainContingency(self.rawData)

        self.toolInfo = []
        for graphAttrIndex in range(len(indices)):
            index = indices[graphAttrIndex]
            if self.rawData.domain[index].varType != orange.VarTypes.Discrete: continue
            attr = self.rawData.domain[index]
            attrLen = len(attr.values)

            # we create a hash table of variable values and their indices
            variableValueIndices = getVariableValueIndices(self.rawData, index)
            variableValueSorted = getVariableValuesSorted(self.rawData, index)
            attrCont = self.domainContingency[indices[graphAttrIndex]]

            values = []
            totals = []
            for i in range(clsCount):
                values.append([0] * attrLen)

            for attrVal in self.rawData.domain[index].values:
                attrIndex = variableValueIndices[attrVal]
                for clsVal in self.rawData.domain.classVar.values:
                    classIndex = classValueIndices[clsVal]
                    values[classIndex][attrIndex] = attrCont[attrVal][clsVal]
                totals.append(sum(attrCont[attrVal]))

            maximum = max(max(values))    # calculate maximum value of all values - needed for scaling
            if maximum == 0: maximum = 1
            sumTotals = sum(totals)       # calculate the sum of totals - needed for tooltips

            # save info for tooltips
            for i in range(attrLen):
                list= []
                for j in range(clsCount)[::-1]:
                    list.append((classValueSorted[j], values[j][i]))
                y_start = (i+1)/float(attrLen)
                y_end = i/float(attrLen)
                x_start = float(graphAttrIndex) - 0.45
                x_end = float(graphAttrIndex) + 0.45
                item = (self.rawData.domain[index].name, variableValueSorted[i], totals[i], sumTotals, list, (x_start,x_end), (y_start, y_end))
                self.toolInfo.append(item)

            # create bar curve
            for i in range(clsCount):
                newColor = QColor(self.discPalette[i])
                newColor.setAlpha(self.alphaValue2)

                for j in range(attrLen):
                    width = float(values[i][j]*0.5) / float(maximum)
                    interval = 1.0/float(2*attrLen)
                    yOff = float(1.0 + 2.0*j)/float(2*attrLen)
                    height = 0.7/float(clsCount*attrLen)

                    yLowBott = yOff - float(clsCount*height)/2.0 + i*height
                    curve = PolygonCurve(QPen(newColor), QBrush(newColor), xData = [graphAttrIndex, graphAttrIndex + width, graphAttrIndex + width, graphAttrIndex], yData = [yLowBott, yLowBott, yLowBott + height, yLowBott + height])
                    curve.attach(self)
                    self.nonDataCurves.append(curve)
        self.addTooltips()

    def addTooltips(self):
        return
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
        return
        for rect in self.toolRects:
            QToolTip.remove(self, rect)
        self.toolRects = []


    # if user clicked between two lines send signal that
    def staticMouseClick(self, e):
        if self.parallelDlg:
            x1 = int(self.invTransform(QwtPlot.xBottom, e.x()))
            axis = self.axisScaleDraw(QwtPlot.xBottom)
            self.parallelDlg.sendShownAttributes([str(axis.label(x1)), str(axis.label(x1+1))])

    def updateLayout(self):
        OWGraph.updateLayout(self)
        self.updateTooltips()

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
    def mouseReleaseEvent(self, e):
        OWGraph.mouseReleaseEvent(self, e)
        self.updateTooltips()

    def mouseMoveEvent(self, e):
        if self.lineTracking:
            canvasPos = self.canvas().mapFrom(self, e.pos())
            (curve, dist, x, y, index) = self.closestCurve(e.pos().x(), e.pos().y())
            if curve:
                if self.lastSelectedCurve and (dist >= 5 or curve != self.lastSelectedCurve) and self.lastSelectedCurve not in self.nonDataCurves:
                    existingPen = self.lastSelectedCurve.pen()
                    existingPen.setWidth(1)
                    self.lastSelectedCurve.setPen(existingPen)
                    self.lastSelectedCurve = None

                if dist < 5 and curve != self.lastSelectedCurve and curve not in self.nonDataCurves:
                    self.lastSelectedCurve = curve
                    existingPen = curve.pen()
                    existingPen.setWidth(3)
                    curve.setPen(existingPen)
                self.replot()
        OWGraph.mouseMoveEvent(self, e)

    def getSelectionsAsExampleTables(self):
        if not self.rawData:
            return (None, None)
        if self.selectionCurveList == []:
            return (None, self.rawData)

        selIndices = []
        unselIndices = range(len(self.rawData))

        for i in range(len(self.curvePoints))[::-1]:
            for j in range(len(self.curvePoints[i])):
                if self.isPointSelected(j, self.curvePoints[i][j]):
                    selIndices.append(i)
                    unselIndices.pop(i)
                    break

        selIndices.reverse()
        selected = self.rawData.getitemsref(selIndices)
        unselected = self.rawData.getitemsref(unselIndices)

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None

        return (selected, unselected)