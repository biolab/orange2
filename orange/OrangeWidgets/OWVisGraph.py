#
# OWVisGraph.py
#
# extension for the base graph class that is used in all visualization widgets
from OWGraph import *
from Numeric import *
import sys
import math
import orange
import os.path
from qtcanvas import *
from OWTools import *

# ####################################################################
# calculate Euclidean distance between two points
def EuclDist(v1, v2):
    val = 0
    for i in range(len(v1)):
        val += (v1[i]-v2[i])**2
    return sqrt(val)
        

# ####################################################################
# add val to sorted list list. if len > maxLen delete last element
def addToList(list, val, ind, maxLen):
    i = 0
    for i in range(len(list)):
        (val2, ind2) = list[i]
        if val < val2:
            list.insert(i, (val, ind))
            if len(list) > maxLen:
                list.remove(list[maxLen])
            return
    if len(list) < maxLen:
        list.insert(len(list), (val, ind))


class SelectionCurve(QwtPlotCurve):
    def __init__(self, parent, name = ""):
        QwtPlotCurve.__init__(self, parent, name)
        self.pointArrayValid = 0
        self.setStyle(QwtCurve.Lines)
        self.setPen(QPen(QColor(128,128,128), 1, Qt.DotLine))
        self.canvas = QCanvas(2000, 2000)   # we need canvas for QCanvasLine

        
    def addPoint(self, xPoint, yPoint):
        xVals = []
        yVals = []
        for i in range(self.dataSize()):
            xVals.append(self.x(i))
            yVals.append(self.y(i))
        xVals.append(xPoint)
        yVals.append(yPoint)
        self.setData(xVals, yVals)
        self.pointArrayValid = 0        # invalidate the point array

    def replaceLastPoint(self, xPoint, yPoint):
        xVals = []
        yVals = []
        for i in range(self.dataSize()-1):
            xVals.append(self.x(i))
            yVals.append(self.y(i))
        xVals.append(xPoint)
        yVals.append(yPoint)
        self.setData(xVals, yVals)
        self.pointArrayValid = 0        # invalidate the point array

    # is point defined at x,y inside a rectangle defined with this curve
    def isInside(self, x, y):       
        xMap = self.parentPlot().canvasMap(self.xAxis());
        yMap = self.parentPlot().canvasMap(self.yAxis());

        if not self.pointArrayValid:
            self.pointArray = QPointArray(self.dataSize() + 1)
            for i in range(self.dataSize()):
                self.pointArray.setPoint(i, xMap.transform(self.x(i)), yMap.transform(self.y(i)))
            self.pointArray.setPoint(self.dataSize(), xMap.transform(self.x(0)), yMap.transform(self.y(0)))
            self.pointArrayValid = 1

        return QRegion(self.pointArray).contains(QPoint(xMap.transform(x), yMap.transform(y)))

    # test if the line going from before last and last point intersect any lines before
    # if yes, then add the intersection point and remove the outer points
    def closed(self):
        if self.dataSize() < 5: return 0
        #print "all points"
        #for i in range(self.dataSize()):
        #    print "(%.2f,%.2f)" % (self.x(i), self.y(i))
        x1 = self.x(self.dataSize()-3)
        x2 = self.x(self.dataSize()-2)
        y1 = self.y(self.dataSize()-3)
        y2 = self.y(self.dataSize()-2)
        for i in range(self.dataSize()-5, -1, -1):
            """
            X1 = self.parentPlot().transform(QwtPlot.xBottom, self.x(i))
            X2 = self.parentPlot().transform(QwtPlot.xBottom, self.x(i+1))
            Y1 = self.parentPlot().transform(QwtPlot.yLeft, self.y(i))
            Y2 = self.parentPlot().transform(QwtPlot.yLeft, self.y(i+1))
            """
            X1 = self.x(i)
            X2 = self.x(i+1)
            Y1 = self.y(i)
            Y2 = self.y(i+1)
            #print "(%.2f,%.2f),(%.2f,%.2f),(%.2f,%.2f),(%.2f,%.2f)" % (x1, y1, x2, y2, X1, Y1, X2, Y2)
            (intersect, xi, yi) = self.lineIntersection(x1, y1, x2, y2, X1, Y1, X2, Y2)
            if intersect:
                xData = [xi]; yData = [yi]
                for j in range(i+1, self.dataSize()-2): xData.append(self.x(j)); yData.append(self.y(j))
                xData.append(xi); yData.append(yi)
                self.setData(xData, yData)
                return 1
        return 0

    def lineIntersection(self, x1, y1, x2, y2, X1, Y1, X2, Y2):
        if x2-x1 != 0: m1 = (y2-y1)/(x2-x1)
        else:          m1 = 1e+12
        
        if X2-X1 != 0: m2 = (Y2-Y1)/(X2-X1)
        else:          m2 = 1e+12;  

        b1 = -1
        b2 = -1
        c1 = (y1-m1*x1)
        c2 = (Y1-m2*X1)

        det_inv = 1/(m1*b2 - m2*b1)

        xi=((b1*c2 - b2*c1)*det_inv)
        yi=((m2*c1 - m1*c2)*det_inv)

        if xi >= min(x1, x2) and xi <= max(x1,x2) and xi >= min(X1, X2) and xi <= max(X1, X2) and yi >= min(y1,y2) and yi <= max(y1, y2) and yi >= min(Y1, Y2) and yi <= max(Y1, Y2):
            return (1, xi, yi)
        else:
            return (0, xi, yi)



ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3

class OWVisGraph(OWGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)

        self.MAX_HUE_VAL = 280              # max hue value used in coloring continuous data values. because red is at 0 and 360, we shorten the range
        self.rawdata = []                   # input data
        self.scaledData = []                # scaled data to the interval 0-1
        self.noJitteringScaledData = []
        self.coloringScaledData = []
        self.attributeNames = []      # list of attribute names from self.rawdata
        self.domainDataStat = []
        self.optimizedDrawing = 1
        self.pointWidth = 5
        self.jitteringType = 'none'
        self.jitterSize = 10
        self.showFilledSymbols = 1
        self.scaleFactor = 1.0
        self.globalValueScaling = 0         # do we want to scale data globally
        self.setCanvasColor(QColor(Qt.white.name()))
        self.xpos = 0   # we have to initialize values, since we might get onMouseRelease event before onMousePress
        self.ypos = 0
        self.zoomStack = []
        self.zoomState = ()
        self.colorHueValues = [240, 0, 120, 60, 180, 300, 30, 150, 270, 90, 210, 330, 15, 135, 255, 45, 165, 285, 105, 225, 345]
        self.colorHueValues = [float(x)/360.0 for x in self.colorHueValues]
        self.colorNonTargetValue = QColor(200,200,200)
        self.colorTargetValue = QColor(0,0,255)
        self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Triangle, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]

        self.state = ZOOMING
        self.tempSelectionCurve = None
        self.selectionCurveKeyList = []

        self.enableGridX(FALSE)
        self.enableGridY(FALSE)

        self.mouseCurrentlyPressed = 0
        self.mouseCurrentButton = 0
        self.blankClick = 0
        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)
        self.tips = DynamicToolTipFloat()
        self.statusBar = None
        self.canvas().setMouseTracking(1)
        self.connect(self, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.onMouseMoved)
        self.zoomStack = []
        self.connect(self, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.onMousePressed)
        self.connect(self, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)

    def activateZooming(self):
        self.state = ZOOMING

    def activateRectangleSelection(self):
        self.state = SELECT_RECTANGLE

    def activatePolygonSelection(self):
        self.state = SELECT_POLYGON

    def removeLastSelection(self):
        if self.selectionCurveKeyList != []:
            lastCurve = self.selectionCurveKeyList[len(self.selectionCurveKeyList)-1]
            self.removeCurve(lastCurve)
            self.selectionCurveKeyList.remove(lastCurve)
        self.replot()
        

    def removeAllSelections(self):
        while self.selectionCurveKeyList != []:
            curve = self.selectionCurveKeyList[0]
            self.removeCurve(curve)
            self.selectionCurveKeyList.remove(curve)
        self.replot()


    #####################################################################
    #####################################################################
    # set new data and scale its values
    def setData(self, data):
        self.rawdata = data
        self.scaledData = []
        self.noJitteringScaledData = []
        self.coloringScaledData = []
        self.attrValues = {}
        self.attributeNames = []

        if data == None: return
        
        self.domainDataStat = orange.DomainBasicAttrStat(data)
        for attr in data.domain: self.attributeNames.append(attr.name)

        min = -1; max = -1
        if self.globalValueScaling == 1:
            (min, max) = self.getMinMaxValDomain(data, self.attributeNames)

        #
        # scale all data
        # scale all data with no jittering
        # scale all data for coloring
        for index in range(len(data.domain)):
            attr = data.domain[index]
            original = []
            noJittering = []
            coloring = []
            values = []

            # is the attribute discrete
            if attr.varType == orange.VarTypes.Discrete:
                # we create a hash table of variable values and their indices
                variableValueIndices = self.getVariableValueIndices(data, index)
                count = float(len(attr.values))
                values = [0, count-1]
                countx2 = float(2*count)	# we compute this value here, so that we don't have to compute it in the loop
                count100 = float(100.0*count) # same

                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: original.append("?"); noJittering.append("?"); coloring.append("?"); continue
                    val = variableValueIndices[data[i][index].value]
                    noJittering.append( (1.0 + 2.0 * val)/ countx2 )
                    original.append( ((1.0 + 2.0 * val)/ countx2) + ((1+val)/count) * self.rndCorrection(self.jitterSize/count100))
                    if count < len(self.colorHueValues): coloring.append(self.colorHueValues[val])
                    else:                                coloring.append( val / float(count) )

            # is the attribute continuous
            else:
                if self.globalValueScaling == 0:
                    min = self.domainDataStat[index].min
                    max = self.domainDataStat[index].max
                diff = max - min
                values = [min, max]

                max_hue = self.MAX_HUE_VAL / 360.0
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: original.append("?"); coloring.append("?"); continue
                    val = (data[i][attr].value - min) / diff
                    original.append(val)
                    coloring.append(val * max_hue)        # we make color palette smaller, because red is in the begining and ending of hsv
                noJittering = original
                
            self.scaledData.append(original)
            self.noJitteringScaledData.append(noJittering)
            self.coloringScaledData.append(coloring)
            self.attrValues[attr.name] = values

        
    #####################################################################
    #####################################################################


    
    # ####################################################################
    # compute min and max value for a list of attributes 
    def getMinMaxValDomain(self, data, attrList):
        first = TRUE
        min = -1; max = -1
        for attr in attrList:
            if data.domain[attr].varType == orange.VarTypes.Discrete: continue
            (minVal, maxVal) = self.getMinMaxVal(data, attr)
            if first == TRUE:
                min = minVal; max = maxVal
                first = FALSE
            else:
                if minVal < min: min = minVal
                if maxVal > max: max = maxVal
        return (min, max)

    

    # ####################################################################
    # get min and max value of data attribute at index index
    def getMinMaxVal(self, data, index):
        attr = data.domain[index]

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            print "warning. Computing min, max value for discrete attribute."
            return (0, float(len(attr.values))-1)
        else:
            print data.domain[index].name
            return (self.domainDataStat[index].min, self.domainDataStat[index].max)
        
    # ####################################################################
    # scale data at index index to the interval 0 to 1
    # min, max - if booth -1 --> scale to interval 0 to 1, else scale inside interval [min, max]
    # forColoring - if TRUE we don't scale from 0 to 1 but a little less (e.g. [0.2, 0.8]) so that the colours won't overlap
    # jitteringEnabled - jittering enabled or not
    # ####################################################################
    def scaleData(self, data, index, min = -1, max = -1, forColoring = 0, jitteringEnabled = 1):
        attr = data.domain[index]
        temp = []; values = []

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # we create a hash table of variable values and their indices
            variableValueIndices = self.getVariableValueIndices(data, index)
            count = float(len(attr.values))
            values = [0, count-1]
            countx2 = float(2*count)	# we compute this value here, so that we don't have to compute it in the loop
            count100 = float(100.0*count) # same

            if forColoring == 1:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    val = float(variableValueIndices[data[i][index].value]) / float(count)
                    temp.append(val)
            elif jitteringEnabled == 1:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / countx2 + self.rndCorrection(self.jitterSize/count100)
                    temp.append(val)
            else:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / countx2
                    temp.append(val)
                    
        # is the attribute continuous
        else:
            if min == max == -1:
                min = self.domainDataStat[index].min
                max = self.domainDataStat[index].max
            diff = max - min
            values = [min, max]

            if forColoring == 1:
                hue = self.MAX_HUE_VAL /360.0
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    temp.append((data[i][attr].value - min)*hue / diff)        # we make color palette smaller, because red is in the begining and ending of hsv
            else:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    temp.append((data[i][attr].value - min) / diff)
        return (temp, values)

    # ####################################################################
    # scale data with no jittering
    def scaleDataNoJittering(self):
        # we have to create a copy of scaled data, because we don't know if the data in self.scaledData was made with jittering
        self.noJitteringScaledData = []
        for i in range(len(self.rawdata.domain)):
            scaled, vals = self.scaleData(self.rawdata, i, jitteringEnabled = 0)
            self.noJitteringScaledData.append(scaled)


    def rescaleAttributesGlobaly(self, data, attrList, jittering = 1):
        if len(attrList) == 0: return
        # find min, max values
        (Min, Max) = self.getMinMaxValDomain(data, attrList)

        # scale data values inside min and max
        for attr in attrList:
            index = self.attributeNames.index(attr)
            scaled, values = self.scaleData(data, index, Min, Max, jitteringEnabled = jittering)
            self.scaledData[index] = scaled
            self.attrValues[attr] = values

    # get array of 0 and 1 of len = len(self.rawdata). if there is a missing value at any attribute in indices return 0 for that example
    def getValidList(self, indices):
        validData = [1] * len(self.rawdata)
        for i in range(len(self.rawdata)):
            for j in range(len(indices)):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0
        return validData


    # ####################################################################    
    # return a list of sorted values for attribute at index index
    # EXPLANATION: if variable values have values 1,2,3,4,... then their order in orange depends on when they appear first
    # in the data. With this function we get a sorted list of values
    def getVariableValuesSorted(self, data, index):
        if data.domain[index].varType == orange.VarTypes.Continuous:
            print "Invalid index for getVariableValuesSorted"
            return []
        
        values = list(data.domain[index].values)
        intValues = []
        i = 0
        # do all attribute values containt integers?
        try:
            while i < len(values):
                temp = int(values[i])
                intValues.append(temp)
                i += 1
        except: pass

        # if all values were intergers, we first sort them ascendently
        if i == len(values):
            intValues.sort()
            values = intValues
        out = []
        for i in range(len(values)):
            out.append(str(values[i]))

        return out

    # ####################################################################
    # create a dictionary with variable at index index. Keys are variable values, key values are indices (transform from string to int)
    # in case all values are integers, we also sort them
    def getVariableValueIndices(self, data, index):
        if data.domain[index].varType == orange.VarTypes.Continuous:
            print "Invalid index for getVariableValueIndices"
            return {}

        values = self.getVariableValuesSorted(data, index)

        dict = {}
        for i in range(len(values)):
            dict[values[i]] = i
        return dict


    def setScaleFactor(self, num):
        self.scaleFactor = num

    def setJitteringOption(self, jitteringType):
        self.jitteringType = jitteringType

    def setJitterSize(self, size):
        self.jitterSize = size

    def setPointWidth(self, pointWidth):
        self.pointWidth = pointWidth

    def setGlobalValueScaling(self, globalScale):
        self.globalValueScaling = globalScale
        
    def rndCorrection(self, max):
        """
        returns a number from -max to max, self.jitteringType defines which distribution is to be used.
        function is used to plot data points for categorical variables
        """    
        if self.jitteringType == 'none': 
            return 0.0
        elif self.jitteringType  == 'uniform': 
            return (random() - 0.5)*2*max
        elif self.jitteringType  == 'triangle': 
            b = (1 - betavariate(1,1)) ; return choice((-b,b))*max
        elif self.jitteringType  == 'beta': 
            b = (1 - betavariate(1,2)) ; return choice((-b,b))*max
                     

    def addCurve(self, name, brushColor, penColor, size, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, enableLegend = 0, xData = [], yData = [], forceFilledSymbols = 0):
        newCurveKey = self.insertCurve(name)
        if self.showFilledSymbols or forceFilledSymbols:
            newSymbol = QwtSymbol(symbol, QBrush(brushColor), QPen(penColor), QSize(size, size))
        else:
            newSymbol = QwtSymbol(symbol, QBrush(), QPen(penColor), QSize(size, size))
        self.setCurveSymbol(newCurveKey, newSymbol)
        self.setCurveStyle(newCurveKey, style)
        self.setCurvePen(newCurveKey, QPen(penColor))
        self.enableLegend(enableLegend, newCurveKey)
        if xData != [] and yData != []:
            self.setCurveData(newCurveKey, xData, yData)
            
        return newCurveKey

    def addMarker(self, name, x, y, alignment = -1, bold = 0):
        mkey = self.insertMarker(name)
        self.marker(mkey).setXValue(x)
        self.marker(mkey).setYValue(y)
        if alignment != -1:
            self.marker(mkey).setLabelAlignment(alignment)
        if bold:
            font = self.marker(mkey).font(); font.setBold(1); self.marker(mkey).setFont(font)
        return mkey

    # ####################################################################
    # return string with attribute names and their values for example example
    def getExampleText(self, data, example):
        text = ""
        for i in range(len(data.domain)):
            if data.domain[i].varType == orange.VarTypes.Discrete:
                if example[i].isSpecial():
                    text = "%s%s = ?; " % (text, data.domain[i].name)
                else:
                    text = "%s%s = %s; " % (text, data.domain[i].name, str(example[i].value))
            else:
                if example[i].isSpecial():
                    text = "%s%s = ?; " % (text, data.domain[i].name)
                else:
                    text = "%s%s = %.3f; " % (text, data.domain[i].name, example[i].value)
        return text

    # ####################################################################
    # return string with attribute names and their values for example example
    def getShortExampleText(self, data, example, indices):
        text = ""
        for i in range(len(indices)):
            try:
                index = indices[i]
                if data.domain[index].varType == orange.VarTypes.Discrete:
                    if example[index].isSpecial():
                        text = "%s%s = ?; " % (text, data.domain[index].name)
                    else:
                        text = "%s%s = %s; " % (text, data.domain[index].name, str(example[index].value))
                else:
                    if example[i].isSpecial():
                        text = "%s%s = ?; " % (text, data.domain[index].name)
                    else:
                        text = "%s%s = %.3f; " % (text, data.domain[index].name, example[index].value)
            except:
                pass
        return text

    def removeDrawingCurves(self):
        for key in self.curveKeys():
            curve = self.curve(key)
            if not isinstance(curve, SelectionCurve):
                self.removeCurve(key)


    # ###############################################
    # HANDLING MOUSE EVENTS
    # ###############################################
    def onMousePressed(self, e):
        self.mouseCurrentlyPressed = 1
        self.mouseCurrentButton = e.button()
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            self.xpos = e.pos().x(); self.ypos = e.pos().y() # save one edge of rectangle
            self.enableOutline(1)                            # enable drawing a rectangle when the mouse is moved
            self.setOutlinePen(QPen(Qt.black))
            self.setOutlineStyle(Qwt.Rect)      # draw a rectangle
            if self.zoomStack == []:
                self.zoomState = ( self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound(),
                                   self.axisScale(QwtPlot.yLeft).lBound(),   self.axisScale(QwtPlot.yLeft).hBound())

        # ####
        # SELECT RECTANGLE
        elif e.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE:
            self.xpos = e.x(); self.ypos = e.y()
            self.tempSelectionCurve = SelectionCurve(self)
            key = self.insertCurve(self.tempSelectionCurve)
            self.selectionCurveKeyList.append(key)

        # ####
        # SELECT POLYGON
        elif e.button() == Qt.LeftButton and self.state == SELECT_POLYGON:
            x = self.invTransform(QwtPlot.xBottom, e.x())
            y = self.invTransform(QwtPlot.yLeft, e.y())
            if self.tempSelectionCurve == None:
                self.tempSelectionCurve = SelectionCurve(self)
                key = self.insertCurve(self.tempSelectionCurve)
                self.selectionCurveKeyList.append(key)
                self.tempSelectionCurve.addPoint(x, y)
            self.tempSelectionCurve.addPoint(x,y)
            if self.tempSelectionCurve.closed():    # did we intersect an existing line. if yes then close the curve and finish appending lines
                self.tempSelectionCurve = None
                self.replot()
                

        # fake a mouse move to show the cursor position
        self.onMouseMoved(e)
        self.event(e)

    # only needed to show the message in statusbar
    def onMouseMoved(self, e):
        x = e.x(); y = e.y()

        xTransf = self.invTransform(QwtPlot.xBottom, x)
        yTransf = self.invTransform(QwtPlot.yLeft, y)
        if self.statusBar != None:
            text = self.tips.maybeTip(xTransf, yTransf)
            self.statusBar.message(text)

        if self.state == SELECT_RECTANGLE and self.tempSelectionCurve != None:
            xposTransf = self.invTransform(QwtPlot.xBottom, self.xpos)
            yposTransf = self.invTransform(QwtPlot.yLeft, self.ypos)
            xTransf = self.invTransform(QwtPlot.xBottom, e.x())
            yTransf = self.invTransform(QwtPlot.yLeft, e.y())
            xData = [xposTransf, xposTransf, xTransf, xTransf, xposTransf]
            yData = [yposTransf, yTransf, yTransf, yposTransf, yposTransf]
            self.tempSelectionCurve.setData(xData, yData)
            self.replot()

        elif self.state == SELECT_POLYGON and self.tempSelectionCurve != None:
            x = self.invTransform(QwtPlot.xBottom, e.x())
            y = self.invTransform(QwtPlot.yLeft, e.y())
            self.tempSelectionCurve.replaceLastPoint(x,y)
            self.replot()            
            
        self.event(e)


    def onMouseReleased(self, e):
        self.mouseCurrentlyPressed = 0
        self.mouseCurrentButton = 0
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            if self.zoomState == (): return     # this example happens if we clicked outside the graph and released the button inside it
            xmin = min(self.xpos, e.pos().x())
            xmax = max(self.xpos, e.pos().x())
            ymin = min(self.ypos, e.pos().y())
            ymax = ymin + ((xmax-xmin)*self.height())/self.width()  # compute the last value so that the picture remains its w/h ratio
            self.setOutlineStyle(Qwt.Cross)
            xmin = self.invTransform(QwtPlot.xBottom, xmin)
            xmax = self.invTransform(QwtPlot.xBottom, xmax)
            ymin = self.invTransform(QwtPlot.yLeft, ymin)
            ymax = self.invTransform(QwtPlot.yLeft, ymax)
            if xmin == xmax or ymin == ymax: return
            self.blankClick = 0
            self.zoomStack.append(self.zoomState)
            self.zoomState = (xmin, xmax, ymin, ymax)
            self.enableOutline(0)
            self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
            self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
            
        elif Qt.RightButton == e.button() and self.state == ZOOMING:
            if len(self.zoomStack):
                xmin, xmax, ymin, ymax = self.zoomStack.pop()
                self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
                self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
            else:
                self.blankClick = 1 # we just clicked and released the button at the same position. This is used in OWSmartVisualization
                return
            
        # ####
        # SELECT RECTANGLE
        elif e.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE:
            if self.tempSelectionCurve:
                self.tempSelectionCurve = None
       
        self.replot()
        self.event(e)

    # does a point (x,y) lie inside one of the selection rectangles (polygons)
    def isPointSelected(self, x,y):
        for curveKey in self.selectionCurveKeyList:
            curve = self.curve(curveKey)
            if curve.isInside(x,y): return 1
        return 0


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow = OWVisGraph()
    sc = SelectionCurve(ow)
    (intersect, xi, yi) = sc.lineIntersection(-10, -10, 10, 10, -10, 10, 10, -10)
    print intersect, xi, yi
    #a.setMainWidget(ow)
    #ow.show()
    #a.exec_loop()
        
