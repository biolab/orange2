#
# OWVisGraph.py
#
# extension for the base graph class that is used in all visualization widgets
from OWGraph import *
import sys, math, os.path
import orange
import qtcanvas
import Numeric, RandomArray
from OWTools import *

ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3

# ####################################################################    
# return a list of sorted values for attribute at index index
# EXPLANATION: if variable values have values 1,2,3,4,... then their order in orange depends on when they appear first
# in the data. With this function we get a sorted list of values
def getVariableValuesSorted(data, index):
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
def getVariableValueIndices(data, index):
    if data.domain[index].varType == orange.VarTypes.Continuous:
        print "Invalid index for getVariableValueIndices"
        return {}

    values = getVariableValuesSorted(data, index)

    dict = {}
    for i in range(len(values)):
        dict[values[i]] = i
    return dict


class MyQToolTip(QToolTip):
    def __init__(self, parent):
        QToolTip.__init__(self, parent)
        self.rect = None
        self.text = None

    def setRect(self, rect, text):
        self.rect = rect
        self.text = text

    def maybeTip(self, p):
        if self.rect and self.text:
            #print p.x(), p.y(), self.rect.left()
            if self.rect.contains(p):
                #print "tip"
                self.tip(self.rect, text)
        

    
class OWVisGraph(OWGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)

        self.rawdata = None                   # input data
        self.scaledData = None                # scaled data to the interval 0-1
        self.noJitteringScaledData = None
        self.coloringScaledData = None
        self.attributeNames = []      # list of attribute names from self.rawdata
        self.domainDataStat = []
        self.attributeFlipInfo = {}     # dictionary with attrName: 0/1 attribute is flipped or not
        self.optimizedDrawing = 1
        self.pointWidth = 5
        self.jitterSize = 10
        self.jitterContinuous = 0
        self.showFilledSymbols = 1
        self.scaleFactor = 1.0              # used in some visualizations to "stretch" the data - see radviz, polviz
        self.globalValueScaling = 0         # do we want to scale data globally
        self.setCanvasColor(QColor(Qt.white.name()))
        self.xpos = 0   # we have to initialize values, since we might get onMouseRelease event before onMousePress
        self.ypos = 0
        self.zoomStack = []
        self.zoomState = ()
        self.colorNonTargetValue = QColor(200,200,200)
        self.colorTargetValue = QColor(0,0,255)
        self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Triangle, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]
        self.tooltip = MyQToolTip(self)

        self.state = ZOOMING
        self.tempSelectionCurve = None
        self.selectionCurveKeyList = []
        self.autoSendSelectionCallback = None   # callback function to call when we add new selection polygon or rectangle

        self.enableGridX(FALSE)
        self.enableGridY(FALSE)

        self.mouseCurrentlyPressed = 0
        self.mouseCurrentButton = 0
        self.blankClick = 0
        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)
        self.tips = TooltipManager(self)
        self.statusBar = None
        self.canvas().setMouseTracking(1)
        self.connect(self, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.onMouseMoved)
        self.zoomStack = []
        self.connect(self, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.onMousePressed)
        self.connect(self, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)

    def updateZoom(self):
        if self.zoomState == (): return
        (xmin, xmax, ymin, ymax) = self.zoomState
        self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
        self.setAxisScale(QwtPlot.yLeft, ymin, ymax)

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
            self.tempSelectionCurve = None
        self.replot()
        if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection
        

    def removeAllSelections(self, send = 1):
        while self.selectionCurveKeyList != []:
            curve = self.selectionCurveKeyList[0]
            self.removeCurve(curve)
            self.selectionCurveKeyList.remove(curve)
        self.updateZoom()
        self.replot()
        if send and self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection


    #####################################################################
    #####################################################################
    # set new data and scale its values
    def setData(self, data):
        # clear all curves, markers, tips
        self.removeAllSelections(0)  # clear all selections
        self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()
        self.attributeFlipInfo = {}
        self.attrValues = {}
        
        self.rawdata = data

        # reset the fliping information    
        if data != None:
            for attr in data.domain:
                self.attributeFlipInfo[attr.name] = 0
        
        if data == None or len(data) == 0:
            self.scaledData = self.noJitteringScaledData = self.coloringScaledData = self.validDataArray = None
            return
        
        self.scaledData = Numeric.zeros([len(data.domain), len(data)], Numeric.Float)
        self.noJitteringScaledData = Numeric.zeros([len(data.domain), len(data)], Numeric.Float)
        self.coloringScaledData = Numeric.zeros([len(data.domain), len(data)], Numeric.Float)
        self.validDataArray = Numeric.ones([len(data.domain), len(data)])

        self.domainDataStat = orange.DomainBasicAttrStat(data)
        self.attributeNames = [attr.name for attr in data.domain]
        
        min = -1; max = -1
        if self.globalValueScaling == 1:
            (min, max) = self.getMinMaxValDomain(data, self.attributeNames)

        # ################################################
        # scale all data
        # scale all data with no jittering
        # scale all data for coloring
        for index in range(len(data.domain)):
            attr = data.domain[index]

            # is the attribute discrete
            if attr.varType == orange.VarTypes.Discrete:
                # we create a hash table of variable values and their indices
                variableValueIndices = getVariableValueIndices(data, index)
                count = float(len(attr.values))
                colors = ColorPaletteHSV(len(attr.values))
                self.attrValues[attr.name] = [0, len(attr.values)-1]

                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1:
                        self.validDataArray[index][i] = 0
                        continue
                    
                    self.noJitteringScaledData[index][i] = variableValueIndices[data[i][index].value]
                    self.coloringScaledData[index][i] = colors.getHue(int(self.noJitteringScaledData[index][i]))

                self.noJitteringScaledData[index] = (self.noJitteringScaledData[index]*2 + 1) / float(2*count)
                self.scaledData[index] = self.noJitteringScaledData[index] + (self.jitterSize/(50.0*count))*(RandomArray.random(len(data)) - 0.5)
                
            # is the attribute continuous
            else:
                if self.globalValueScaling == 0:
                    min = self.domainDataStat[index].min
                    max = self.domainDataStat[index].max
                diff = float(max - min)
                if diff == 0.0: diff = 1.0    # prevent division by zero
                self.attrValues[attr.name] = [min, max]
                colors = ColorPaletteHSV()

                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1:
                        self.validDataArray[index][i] = 0
                        continue
                    self.noJitteringScaledData[index][i] = data[i][index].value

                self.noJitteringScaledData[index] = (self.noJitteringScaledData[index] - float(min)) / diff
                self.coloringScaledData[index] = self.noJitteringScaledData[index] * colors.maxHueVal
                if self.jitterContinuous:
                    self.scaledData[index] = self.noJitteringScaledData[index] + self.jitterSize/50.0 * (0.5 - RandomArray.random(len(data)))
                    self.scaledData[index] = abs(self.scaledData[index])
                    for i in range(len(data)):
                        if self.scaledData[index][i] > 1.0: self.scaledData[index][i] = 2.0-self.scaledData[index][i]
                else:
                    self.scaledData[index] = self.noJitteringScaledData[index]
                
        
    # ####################################################################
    # ####################################################################

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
            return (self.domainDataStat[index].min, self.domainDataStat[index].max)
        
    # ####################################################################
    # scale data at index index to the interval 0 to 1
    # min, max - if booth -1 --> scale to interval 0 to 1, else scale inside interval [min, max]
    # forColoring - if TRUE we don't scale from 0 to 1 but a little less (e.g. [0.2, 0.8]) so that the colours won't overlap
    # jitteringEnabled - jittering enabled or not
    # ####################################################################
    def scaleData(self, data, index, min = -1, max = -1, forColoring = 0, jitteringEnabled = 1):
        attr = data.domain[index]
        values = []

        arr = Numeric.zeros([len(data)], Numeric.Float)
        
        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # is the attribute discrete
            # we create a hash table of variable values and their indices
            variableValueIndices = getVariableValueIndices(data, index)
            count = float(len(attr.values))
            values = [0, len(attr.values)-1]

            if forColoring:
                colors = ColorPaletteHSV(len(attr.values))
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: continue
                    arr[i] = colors.getHue(variableValueIndices[data[i][index].value])
            else:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: continue
                    arr[i] = variableValueIndices[data[i][index].value]
                    arr = (arr*2 + 1) / float(2*count)
                if jitteringEnabled:
                    arr = arr + 0.5 - (self.jitterSize/(50.0*count))*RandomArray.random(len(data))
            
        # is the attribute continuous
        else:
            if min == max == -1:
                min = self.domainDataStat[index].min
                max = self.domainDataStat[index].max
            values = [min, max]
            diff = max - min
            if diff == 0.0: diff = 1    # prevent division by zero
            
            for i in range(len(data)):
                if data[i][index].isSpecial() == 1: continue
                arr[i] = data[i][index].value
            arr = (arr - min) / diff

            if forColoring == 1:
                colors = ColorPaletteHSV()
                arr = arr * colors.maxHueVal
        return (arr, values)

    def rescaleAttributesGlobaly(self, data, attrList, jittering = 1):
        if len(attrList) == 0: return
        # find min, max values
        (Min, Max) = self.getMinMaxValDomain(data, attrList)

        # scale data values inside min and max
        for attr in attrList:
            if data.domain[attr].varType == orange.VarTypes.Discrete: continue  # don't scale discrete attributes
            index = self.attributeNames.index(attr)
            scaled, values = self.scaleData(data, index, Min, Max, jitteringEnabled = jittering)
            self.scaledData[index] = scaled
            self.attrValues[attr] = values

    def getAttributeLabel(self, attrName):
        if self.attributeFlipInfo[attrName] and self.rawdata.domain[attrName].varType == orange.VarTypes.Continuous: return "-" + attrName
        return attrName

    def flipAttribute(self, attrName):
        if attrName not in self.attributeNames: return 0
        if self.rawdata.domain[attrName].varType == orange.VarTypes.Discrete: return 0
        if self.globalValueScaling: return 0
            
        index = self.attributeNames.index(attrName)
        self.attributeFlipInfo[attrName] = not self.attributeFlipInfo[attrName]
        if self.rawdata.domain[attrName].varType == orange.VarTypes.Continuous:
            self.attrValues[attrName] = [-self.attrValues[attrName][1], -self.attrValues[attrName][0]]
    
        self.scaledData[index] = 1 - self.scaledData[index]
        self.noJitteringScaledData[index] = 1 - self.noJitteringScaledData[index]
        self.coloringScaledData[index] = 1 - self.coloringScaledData[index]
        return 1


    # get array of 0 and 1 of len = len(self.rawdata). if there is a missing value at any attribute in indices return 0 for that example
    def getValidList(self, indices):
        selectedArray = Numeric.take(self.validDataArray, indices)
        arr = Numeric.add.reduce(selectedArray) - len(indices)
        return Numeric.equal(arr, 0)
        
    # returns a number from -max to max
    def rndCorrection(self, max):
        if max == 0: return 0.0
        return (random() - 0.5)*2*max
        
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
            if example[i].isSpecial():
                text += "%s = ?; " % (data.domain[i].name)
            else:
                text += "%s = %s; " % (data.domain[i].name, str(example[i]))
        return text

    # ####################################################################
    # return string with attribute names and their values for example example
    def getShortExampleText(self, data, example, indices):
        text = ""
        try:
            for i in range(len(indices)):
                index = indices[i]
                if example[index].isSpecial():
                    text += "%s = ?; " % (data.domain[index].name)
                else:
                    text += "%s = %s; " % (data.domain[index].name, str(example[index]))
        except:
            text = ""
        return text

    def removeDrawingCurves(self):
        for key in self.curveKeys():
            curve = self.curve(key)
            if not isinstance(curve, SelectionCurve):
                self.removeCurve(key)

    def changeClassAttr(self, selected, unselected):
        classVar = orange.EnumVariable("Selection", values = ["Selected data", "Unselected data"])
        classVar.getValueFrom = lambda ex,what: 0  # orange.Value(classVar, 0)
        if selected:
            domain = orange.Domain(selected.domain.variables + [classVar])
            table = orange.ExampleTable(domain, selected)
            if unselected:
                classVar.getValueFrom = lambda ex,what: 1
                table.extend(unselected)
        elif unselected:
            domain = orange.Domain(unselected.domain.variables + [classVar])
            classVar.getValueFrom = lambda ex,what: 1
            table = orange.ExampleTable(domain, unselected)
        else: table = None
        return table

    # show a tooltip at x,y with text. if the mouse will move for more than 2 pixels it will be removed
    def showTip(self, x, y, text):
        MyQToolTip.tip(self.tooltip, QRect(x+self.canvas().frameGeometry().x()-3, y+self.canvas().frameGeometry().y()-3, 6, 6), text)
       

    # mouse was only pressed and released on the same spot. visualization methods might want to process this event
    def staticMouseClick(self, e):
        pass

    # ###############################################
    # HANDLING MOUSE EVENTS
    # ###############################################
    def onMousePressed(self, e):
        self.mouseCurrentlyPressed = 1
        self.mouseCurrentButton = e.button()
        self.zoomState = (self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound(), self.axisScale(QwtPlot.yLeft).lBound(),   self.axisScale(QwtPlot.yLeft).hBound())
        self.updateZoom()
        self.xpos = e.x()
        self.ypos = e.y()

        # ####
        # ZOOM
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            self.tempSelectionCurve = SelectionCurve(self, pen = Qt.DashLine)
            self.zoomKey = self.insertCurve(self.tempSelectionCurve)
            if self.state == ZOOMING and self.zoomStack == []:
                self.zoomState = ( self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound(), self.axisScale(QwtPlot.yLeft).lBound(),   self.axisScale(QwtPlot.yLeft).hBound())

        # ####
        # SELECT RECTANGLE
        elif e.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE:
            self.tempSelectionCurve = SelectionCurve(self)
            key = self.insertCurve(self.tempSelectionCurve)
            self.selectionCurveKeyList.append(key)

        # ####
        # SELECT POLYGON
        elif e.button() == Qt.LeftButton and self.state == SELECT_POLYGON:
            if self.tempSelectionCurve == None:
                self.tempSelectionCurve = SelectionCurve(self)
                key = self.insertCurve(self.tempSelectionCurve)
                self.selectionCurveKeyList.append(key)
                self.tempSelectionCurve.addPoint(self.invTransform(QwtPlot.xBottom, self.xpos), self.invTransform(QwtPlot.yLeft, self.ypos))
            self.tempSelectionCurve.addPoint(self.invTransform(QwtPlot.xBottom, self.xpos), self.invTransform(QwtPlot.yLeft, self.ypos))

            if self.tempSelectionCurve.closed():    # did we intersect an existing line. if yes then close the curve and finish appending lines
                self.tempSelectionCurve = None
                self.updateZoom()
                self.replot()
                if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection
                

        # fake a mouse move to show the cursor position
        self.onMouseMoved(e)
        self.event(e)

    # only needed to show the message in statusbar
    def onMouseMoved(self, e):
        xFloat = self.invTransform(QwtPlot.xBottom, e.x())
        yFloat = self.invTransform(QwtPlot.yLeft, e.y())

        (text, x, y) = self.tips.maybeTip(xFloat, yFloat)
       
        if self.statusBar != None:
            self.statusBar.message(text)

        if text != "":
            intX = self.transform(QwtPlot.xBottom, x)
            intY = self.transform(QwtPlot.yLeft, y)
            self.showTip(intX, intY, text[:-2].replace("; ", "\n"))

        if self.tempSelectionCurve != None and (self.state == ZOOMING or self.state == SELECT_RECTANGLE):
            x1 = self.invTransform(QwtPlot.xBottom, self.xpos)
            y1 = self.invTransform(QwtPlot.yLeft, self.ypos)
            x2 = xFloat
            y2 = yFloat
            xData = [x1, x1, x2, x2, x1]
            yData = [y1, y2, y2, y1, y1]
            self.tempSelectionCurve.setData(xData, yData)
            self.replot()

        elif self.state == SELECT_POLYGON and self.tempSelectionCurve != None:
            self.tempSelectionCurve.replaceLastPoint(xFloat,yFloat)
            self.replot()            
            
        self.event(e)


    def onMouseReleased(self, e):
        self.mouseCurrentlyPressed = 0
        self.mouseCurrentButton = 0
        staticClick = 0

        if e.button() != Qt.RightButton:
            if self.xpos == e.x() and self.ypos == e.y():
                self.staticMouseClick(e)
                staticClick = 1

        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            if self.zoomState == (): return     # this example happens if we clicked outside the graph and released the button inside it
            xmin = min(self.xpos, e.x());  xmax = max(self.xpos, e.x())
            ymin = min(self.ypos, e.y());  ymax = max(self.ypos, e.y())
            
            self.removeCurve(self.zoomKey)
            self.tempSelectionCurve = None

            if staticClick or (xmax-xmin)+(ymax-ymin) < 4: return

            xmin = self.invTransform(QwtPlot.xBottom, xmin);  xmax = self.invTransform(QwtPlot.xBottom, xmax)
            ymin = self.invTransform(QwtPlot.yLeft, ymin);    ymax = self.invTransform(QwtPlot.yLeft, ymax)
            
            self.blankClick = 0
            self.zoomStack.append(self.zoomState)
            self.zoomState = (xmin, xmax, ymin, ymax)
            self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
            self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
            
            
        elif e.button() == Qt.RightButton and self.state == ZOOMING:
            if len(self.zoomStack):
                xmin, xmax, ymin, ymax = self.zoomStack.pop()
                self.zoomState = (xmin, xmax, ymin, ymax)
                self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
                self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
            else:
                self.blankClick = 1 # we just clicked and released the button at the same position. This is used in OWSmartVisualization
                return

        elif e.button() == Qt.RightButton and self.state == SELECT_RECTANGLE:
            self.removeLastSelection()      # remove the rectangle

        # ####
        # SELECT RECTANGLE
        elif e.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE:
            if self.tempSelectionCurve:
                self.tempSelectionCurve = None
                if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection
       

        elif Qt.RightButton == e.button() and self.state == SELECT_POLYGON:
            if self.tempSelectionCurve:
                self.tempSelectionCurve.removeLastPoint()
                if self.tempSelectionCurve.dataSize() == 0: # remove the temp curve
                    self.tempSelectionCurve = None
                    self.removeLastSelection()
                else:   # set new last point 
                    x = self.invTransform(QwtPlot.xBottom, e.x())
                    y = self.invTransform(QwtPlot.yLeft, e.y())
                    self.tempSelectionCurve.replaceLastPoint(x,y)
                self.replot()
            
        self.replot()
        self.event(e)

    # does a point (x,y) lie inside one of the selection rectangles (polygons)
    def isPointSelected(self, x,y):
        for curveKey in self.selectionCurveKeyList:
            #print "isPointSelected:", x, y
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
        
