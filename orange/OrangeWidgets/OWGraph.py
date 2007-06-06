#
# owGraph.py
#
# the base for all graphs

from qt import *
from OWTools import *
try:
    from qwt import *
except:
    from Qwt4 import *
import qtcanvas, orange, math
from OWGraphTools import *      # color palletes, user defined curves, ...
from OWDlgs import OWChooseImageSizeDlg
from OWBaseWidget import unisetattr, OWBaseWidget

NOTHING = 0
ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3


class OWGraph(QwtPlot):
    def __init__(self, parent = None, name = "None"):
        "Constructs the graph"
        QwtPlot.__init__(self, parent, name)
        self.parentName = name
        self.setWFlags(Qt.WResizeNoErase) #this works like magic.. no flicker during repaint!

        self.setAutoReplot(FALSE)
        self.setAutoLegend(FALSE)
        self.setAxisAutoScale(QwtPlot.xBottom)
        self.setAxisAutoScale(QwtPlot.xTop)
        self.setAxisAutoScale(QwtPlot.yLeft)
        self.setAxisAutoScale(QwtPlot.yRight)

        newFont = QFont('Helvetica', 10, QFont.Bold)
        self.setTitleFont(newFont)
        self.setAxisTitleFont(QwtPlot.xBottom, newFont)
        self.setAxisTitleFont(QwtPlot.xTop, newFont)
        self.setAxisTitleFont(QwtPlot.yLeft, newFont)
        self.setAxisTitleFont(QwtPlot.yRight, newFont)

        newFont = QFont('Helvetica', 9)
        self.setAxisFont(QwtPlot.xBottom, newFont)
        self.setAxisFont(QwtPlot.xTop, newFont)
        self.setAxisFont(QwtPlot.yLeft, newFont)
        self.setAxisFont(QwtPlot.yRight, newFont)
        self.setLegendFont(newFont)

        self.tipLeft = None
        self.tipRight = None
        self.tipBottom = None

        self.showAxisScale = 1
        self.showMainTitle = FALSE
        self.mainTitle = ""
        self.showXaxisTitle = FALSE
        self.XaxisTitle = ""
        self.showYLaxisTitle = FALSE
        self.YLaxisTitle = ""
        self.showYRaxisTitle = FALSE
        self.YRaxisTitle = ""

        self.state = ZOOMING
        self.tooltip = MyQToolTip(self)
        self.zoomKey = None
        self.tempSelectionCurve = None
        self.selectionCurveKeyList = []
        self.selectionChangedCallback = None   # callback function to call when we add new selection polygon or rectangle
        self.legendCurveKeys = []

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
        self.optimizedDrawing = 1
        self.pointWidth = 7
        self.showFilledSymbols = 1
        self.showLegend = 1
        self.scaleFactor = 1.0              # used in some visualizations to "stretch" the data - see radviz, polviz
        self.setCanvasColor(QColor(Qt.white.name()))
        self.xpos = 0   # we have to initialize values, since we might get onMouseRelease event before onMousePress
        self.ypos = 0
        self.colorNonTargetValue = QColor(200,200,200)
        self.colorTargetValue = QColor(0,0,255)
        self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Triangle, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]
        #self.curveSymbols = [QwtSymbol.Triangle, QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]

        # uncomment this if you want to use printer friendly symbols
        #self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.XCross, QwtSymbol.Triangle, QwtSymbol.Cross, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.Rect, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle]
        self.contPalette = ColorPaletteGenerator(numberOfColors = -1)
        self.discPalette = ColorPaletteGenerator()
##        self.currentScale = {}

        if parent:
            if type(parent) > OWBaseWidget:
                parent._guiElements = getattr(parent, "_guiElements", []) + [("qwtPlot", self)]
            elif type(parent.parent()) >= OWBaseWidget:
                parent.parent()._guiElements = getattr(parent.parent(), "_guiElements", []) + [("qwtPlot", self)]


    def __setattr__(self, name, value):
        unisetattr(self, name, value, QwtPlot)

    # call to update dictionary with settings
    def updateSettings(self, **settings):
        self.__dict__.update(settings)

    def saveToFile(self, extraButtons = []):
        sizeDlg = OWChooseImageSizeDlg(self, extraButtons)
        sizeDlg.exec_loop()

    def saveToFileDirect(self, fileName, size = None):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveImage(fileName, size)

##    def setAxisScale(self, axis, min, max, step = 0):
##        current = self.currentScale.get(axis, None)
##        if current and current == (min, max, step): return
##        QwtPlot.setAxisScale(self, axis, min, max, step)
##        self.currentScale[axis] = (min, max, step)

    def setLabels(self, labels, axis):
        if not self.showAxisScale:
            self.setAxisScaleDraw(axis, HiddenScaleDraw())
            self.axisScaleDraw(axis).setTickLength(0, 0, 0)
            self.axisScaleDraw(axis).setOptions(0)
        else:
            self.axisScaleDraw(axis).setTickLength(1, 1, 3)
            self.axisScaleDraw(axis).setOptions(1)

            if (labels <> None):
                self.setAxisScaleDraw(axis, DiscreteAxisScaleDraw(labels))
                self.setAxisScale(axis, 0, len(labels) - 1, 1)
                self.setAxisMaxMinor(axis, 0)
                self.setAxisMaxMajor(axis, len(labels))
            else:
                self.setAxisScaleDraw(axis, QwtScaleDraw())
                self.setAxisAutoScale(axis)
                self.setAxisMaxMinor(axis, 10)
                self.setAxisMaxMajor(axis, 10)
            self.updateToolTips()

    def setYLlabels(self, labels):
        "Sets the Y-axis labels on the left."
        self.setLabels(labels, QwtPlot.yLeft)

    def setYRlabels(self, labels):
        "Sets the Y-axis labels on the right."
        self.setLabels(labels, QwtPlot.yRight)

    def setXlabels(self, labels):
        "Sets the x-axis labels if x-axis discrete."
        "Or leave up to QwtPlot (MaxMajor, MaxMinor) if x-axis continuous."
        self.setLabels(labels, QwtPlot.xBottom)

    def enableXaxis(self, enable):
        self.enableAxis(QwtPlot.xBottom, enable)
        self.repaint()

    def enableYLaxis(self, enable):
        self.enableAxis(QwtPlot.yLeft, enable)
        self.repaint()

    def enableYRaxis(self, enable):
        self.enableAxis(QwtPlot.yRight, enable)
        self.repaint()

    def updateToolTips(self):
        pass

    def setRightTip(self,explain):
        "Sets the tooltip for the right y axis"
        self.tipRight = explain
        self.updateToolTips()

    def setLeftTip(self,explain):
        "Sets the tooltip for the left y axis"
        self.tipLeft = explain
        self.updateToolTips()

    def setBottomTip(self,explain):
        "Sets the tooltip for the left x axis"
        self.tipBottom = explain
        self.updateToolTips()

    def resizeEvent(self, event):
        "Makes sure that the plot resizes"
        self.updateToolTips()
        self.updateLayout()

    def paintEvent(self, qpe):
        """
        Paints the graph.
        Called whenever repaint is needed by the system
        or user explicitly calls repaint()
        """
        for key in self.selectionCurveKeyList:     # the selection curves must set new point array
            self.curve(key).pointArrayValid = 0    # at any change in the graphics otherwise the right examples will not be selected

        QwtPlot.paintEvent(self, qpe) #let the ancestor do its job
        self.replot()

    def setShowMainTitle(self, b):
        self.showMainTitle = b
        if (self.showMainTitle <> 0):
            self.setTitle(self.mainTitle)
        else:
            self.setTitle("")
        self.updateLayout()
        self.repaint()

    def setMainTitle(self, t):
        self.mainTitle = t
        if (self.showMainTitle <> 0):
            self.setTitle(self.mainTitle)
        else:
            self.setTitle("")
        self.updateLayout()
        self.repaint()

    # show or hide axis title. if b = -1 then only update currently set status of the title
    def setShowXaxisTitle(self, b = -1):
        if b == self.showXaxisTitle: return
        if b != -1:
            self.showXaxisTitle = b
        if self.showXaxisTitle:
            self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.xBottom, "")
        self.updateLayout()
        self.repaint()

    def setXaxisTitle(self, title):
        if title == self.XaxisTitle: return
        self.XaxisTitle = title
        if self.showXaxisTitle == 0: return
        self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        self.updateLayout()
        self.repaint()

    # show or hide axis title. if b = -1 then only update currently set status of the title
    def setShowYLaxisTitle(self, b = -1):
        if b == self.showYLaxisTitle: return
        if b != -1:
            self.showYLaxisTitle = b
        if self.showYLaxisTitle:
            self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yLeft, "")
        self.updateLayout()
        self.repaint()

    def setYLaxisTitle(self, title):
        if title == self.YLaxisTitle: return
        self.YLaxisTitle = title
        if self.showYLaxisTitle == 0: return
        self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        self.updateLayout()
        self.repaint()

    # show or hide axis title. if b = -1 then only update currently set status of the title
    def setShowYRaxisTitle(self, b = -1):
        if b == self.showYRaxisTitle: return
        if b != -1:
            self.showYRaxisTitle = b
        if self.showYRaxisTitle:
            self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yRight, "")
        self.updateLayout()
        self.repaint()

    def setYRaxisTitle(self, title):
        if title == self.YRaxisTitle: return
        self.YRaxisTitle = title
        if self.showYRaxisTitle == 0: return
        self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        self.updateLayout()
        self.repaint()

    def enableGridXB(self, b):
        self.setGridXAxis(QwtPlot.xBottom)
        self.enableGridX(b)
        self.repaint()

    def enableGridXT(self, b):
        self.setGridXAxis(QwtPlot.xTop)
        self.enableGridX(b)
        self.repaint()

    def enableGridYR(self, b):
        self.setGridYAxis(QwtPlot.yRight)
        self.enableGridY(b)
        self.repaint()

    def enableGridYL(self, b):
        self.setGridYAxis(QwtPlot.yLeft)
        self.enableGridY(b)
        self.repaint()

    def enableGraphLegend(self, b):
        self.enableLegend(b)
        self.setAutoLegend(b)
        self.repaint()

    def setGridColor(self, c):
        self.setGridPen(QPen(c))
        self.repaint()

    def setCanvasColor(self, c):
        self.setCanvasBackground(c)
        self.repaint()

    # ############################################################
    # functions that were previously in OWVisGraph
    # ############################################################
    def setData(self, data):
        # clear all curves, markers, tips
        self.removeAllSelections(0)  # clear all selections
        self.removeCurves()
        self.legendCurveKeys = []
        if hasattr(self, "oldLegendKeys"):
            self.oldLegendKeys = {}
        self.removeMarkers()
        self.tips.removeAll()
        self.zoomStack = []


    # ####################################################################
    # return string with attribute names and their values for example example
    def getExampleTooltipText(self, data, example, indices = None):
        if not indices: indices = range(len(data.domain.attributes))

        text = "<b>Attributes:</b><br>"
        for index in indices:
            if example[index].isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (data.domain[index].name)
            else:                          text += "&nbsp;"*4 + "%s = %s<br>" % (data.domain[index].name, str(example[index]))

        if data.domain.classVar:
            text += "<hr><b>Class:</b><br>"
            if example.getclass().isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (data.domain.classVar.name)
            else:                              text += "&nbsp;"*4 + "%s = %s<br>" % (data.domain.classVar.name, str(example.getclass().value))

        if len(self.rawdata.domain.getmetas()) != 0:
            text += "<hr><b>Meta attributes:</b><br>"
            # show values of meta attributes
            for key in data.domain.getmetas():
                try: text += "&nbsp;"*4 + "%s = %s<br>" % (data.domain[key].name, str(example[data.domain[key]]))
                except: pass

        return text[:-4]        # remove the last <br>

    def addCurve(self, name, brushColor, penColor, size, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, enableLegend = 0, xData = [], yData = [], showFilledSymbols = None, lineWidth = 1, pen = None):
        newCurveKey = self.insertCurve(name)
        if showFilledSymbols or (showFilledSymbols == None and self.showFilledSymbols):
            newSymbol = QwtSymbol(symbol, QBrush(brushColor), QPen(penColor), QSize(size, size))
        else:
            newSymbol = QwtSymbol(symbol, QBrush(), QPen(penColor), QSize(size, size))
        self.setCurveSymbol(newCurveKey, newSymbol)
        self.setCurveStyle(newCurveKey, style)
        if not pen:
            self.setCurvePen(newCurveKey, QPen(penColor, lineWidth))
        else:
            self.setCurvePen(newCurveKey, pen)
        self.enableLegend(enableLegend, newCurveKey)
        if xData != [] and yData != []:
            self.setCurveData(newCurveKey, xData, yData)

        return newCurveKey

    def addMarker(self, name, x, y, alignment = -1, bold = 0, color = None, size=None):
        mkey = self.insertMarker(name)
        self.marker(mkey).setXValue(x)
        self.marker(mkey).setYValue(y)
        if alignment != -1:
            self.marker(mkey).setLabelAlignment(alignment)
        if bold or size:
            font = self.marker(mkey).font()
            if bold:
                font.setBold(1)
            if size:
                font.setPixelSize(size)
            self.marker(mkey).setFont(font)
##        if color:
##            self.marker(mkey).setLabelColor(color)
        return mkey

    # show a tooltip at x,y with text. if the mouse will move for more than 2 pixels it will be removed
    def showTip(self, x, y, text):
        MyQToolTip.tip(self.tooltip, QRect(x+self.canvas().frameGeometry().x()-3, y+self.canvas().frameGeometry().y()-3, 6, 6), text)

    # mouse was only pressed and released on the same spot. visualization methods might want to process this event
    def staticMouseClick(self, e):
        pass

    def activateZooming(self):
        self.state = ZOOMING
        if self.tempSelectionCurve: self.removeLastSelection()

    def activateRectangleSelection(self):
        self.state = SELECT_RECTANGLE
        if self.tempSelectionCurve: self.removeLastSelection()

    def activatePolygonSelection(self):
        self.state = SELECT_POLYGON
        if self.tempSelectionCurve: self.removeLastSelection()

    def removeDrawingCurves(self, removeLegendItems = 1):
        for key in self.curveKeys():
            if isinstance(self.curve(key), SelectionCurve):
                continue
            if removeLegendItems == 0 and key in self.legendCurveKeys:
                continue
            self.removeCurve(key)

    def removeLastSelection(self):
        if self.selectionCurveKeyList != []:
            lastCurve = self.selectionCurveKeyList.pop()
            self.removeCurve(lastCurve)
            self.tempSelectionCurve = None
            self.replot()
            if self.selectionChangedCallback:
                self.selectionChangedCallback() # do we want to send new selection
            return 1
        else:
            return 0

    def removeAllSelections(self, send = 1):
        for key in self.selectionCurveKeyList:
            self.removeCurve(key)
        self.selectionCurveKeyList = []
        self.replot()
        if send and self.selectionChangedCallback:
            self.selectionChangedCallback()

    def zoomOut(self):
        if len(self.zoomStack):
            newXMin, newXMax, newYMin, newYMax = self.zoomStack.pop()
            self.setNewZoom(self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound(), self.axisScale(QwtPlot.yLeft).lBound(), self.axisScale(QwtPlot.yLeft).hBound(), newXMin, newXMax, newYMin, newYMax)
            return 1
        return 0

    def setNewZoom(self, oldXMin, oldXMax, oldYMin, oldYMax, newXMin, newXMax, newYMin, newYMax):
        #zoomOutCurveKey = self.insertCurve(RectangleCurve(self, brush = None, xData = [oldXMin, oldXMax, oldXMax, oldXMin], yData = [oldYMin, oldYMin, oldYMax, oldYMax]))
        if len(self.curveKeys()) > 2000:    # if too many curves then don't be smooth
            steps = 1
        else:
            steps = 10
        for i in range(1, steps+1):
            midXMin = oldXMin * (steps-i)/float(steps) + newXMin * i/float(steps)
            midXMax = oldXMax * (steps-i)/float(steps) + newXMax * i/float(steps)
            midYMin = oldYMin * (steps-i)/float(steps) + newYMin * i/float(steps)
            midYMax = oldYMax * (steps-i)/float(steps) + newYMax * i/float(steps)
            self.setAxisScale(QwtPlot.yLeft, midYMax, midYMin)
            self.setAxisScale(QwtPlot.xBottom, midXMin, midXMax)

            #if i == steps:
            #    self.removeCurve(zoomOutCurveKey)
            self.replot()


    # ###############################################
    # HANDLING MOUSE EVENTS
    # ###############################################
    def onMousePressed(self, e):
        self.mouseCurrentlyPressed = 1
        self.mouseCurrentButton = e.button()
        self.xpos = e.x()
        self.ypos = e.y()

        # ####
        # ZOOM
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            self.tempSelectionCurve = SelectionCurve(self, pen = Qt.DashLine)
            self.zoomKey = self.insertCurve(self.tempSelectionCurve)

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
                self.replot()
                if self.selectionChangedCallback:
                    self.selectionChangedCallback() # do we want to send new selection

        # fake a mouse move to show the cursor position
        self.onMouseMoved(e)
        self.event(e)

    # only needed to show the message in statusbar
    def onMouseMoved(self, e):
        xFloat = self.invTransform(QwtPlot.xBottom, e.x())
        yFloat = self.invTransform(QwtPlot.yLeft, e.y())

        text = ""
        if not self.mouseCurrentlyPressed:
            (text, x, y) = self.tips.maybeTip(xFloat, yFloat)
            if type(text) == int: text = self.buildTooltip(text)

        if self.statusBar != None:
            self.statusBar.message(text)
        if text != "":
            self.showTip(self.transform(QwtPlot.xBottom, x), self.transform(QwtPlot.yLeft, y), text)

        if self.tempSelectionCurve != None and (self.state == ZOOMING or self.state == SELECT_RECTANGLE):
            x1 = self.invTransform(QwtPlot.xBottom, self.xpos)
            y1 = self.invTransform(QwtPlot.yLeft, self.ypos)
            self.tempSelectionCurve.setData([x1, x1, xFloat, xFloat, x1], [y1, yFloat, yFloat, y1, y1])
            self.replot()

        elif self.state == SELECT_POLYGON and self.tempSelectionCurve != None:
            self.tempSelectionCurve.replaceLastPoint(xFloat,yFloat)
            self.repaint()

        self.event(e)


    def onMouseReleased(self, e):
        if not self.mouseCurrentlyPressed: return   # this might happen if we double clicked the widget titlebar
        self.mouseCurrentlyPressed = 0
        self.mouseCurrentButton = 0
        staticClick = 0

        if e.button() != Qt.RightButton:
            if self.xpos == e.x() and self.ypos == e.y():
                self.staticMouseClick(e)
                staticClick = 1

        if e.button() == Qt.LeftButton:
            if self.state == ZOOMING:
                xmin = min(self.xpos, e.x());  xmax = max(self.xpos, e.x())
                ymin = min(self.ypos, e.y());  ymax = max(self.ypos, e.y())

                if self.zoomKey: self.removeCurve(self.zoomKey)
                self.zoomKey = None
                self.tempSelectionCurve = None

                if staticClick or (xmax-xmin)+(ymax-ymin) < 4:
                    return

                xmin = self.invTransform(QwtPlot.xBottom, xmin);  xmax = self.invTransform(QwtPlot.xBottom, xmax)
                ymin = self.invTransform(QwtPlot.yLeft, ymin);    ymax = self.invTransform(QwtPlot.yLeft, ymax)

                self.blankClick = 0
                self.zoomStack.append((self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound(), self.axisScale(QwtPlot.yLeft).lBound(), self.axisScale(QwtPlot.yLeft).hBound()))
                self.setNewZoom(self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound(), self.axisScale(QwtPlot.yLeft).lBound(), self.axisScale(QwtPlot.yLeft).hBound(), xmin, xmax, ymax, ymin)

            elif self.state == SELECT_RECTANGLE:
                if self.tempSelectionCurve:
                    self.tempSelectionCurve = None
                if self.selectionChangedCallback:
                    self.selectionChangedCallback() # do we want to send new selection

        elif e.button() == Qt.RightButton:
            if self.state == ZOOMING:
                ok = self.zoomOut()
                if not ok:
                    self.removeLastSelection()
                    self.blankClick = 1 # we just clicked and released the button at the same position
                    return

            elif self.state == SELECT_RECTANGLE:
                ok = self.removeLastSelection()      # remove the rectangle
                if not ok: self.zoomOut()

            elif self.state == SELECT_POLYGON:
                if self.tempSelectionCurve:
                    self.tempSelectionCurve.removeLastPoint()
                    if self.tempSelectionCurve.dataSize() == 0: # remove the temp curve
                        self.tempSelectionCurve = None
                        self.removeLastSelection()
                    else:   # set new last point
                        self.tempSelectionCurve.replaceLastPoint(self.invTransform(QwtPlot.xBottom, e.x()), self.invTransform(QwtPlot.yLeft, e.y()))
                    self.replot()
                else:
                    ok = self.removeLastSelection()
                    if not ok: self.zoomOut()

        #self.replot()
        self.event(e)

    # does a point (x,y) lie inside one of the selection rectangles (polygons)
    def isPointSelected(self, x,y):
        for curveKey in self.selectionCurveKeyList:
            if self.curve(curveKey).isInside(x,y): return 1
        return 0

    # return two lists of 0's and 1's whether each point in (xData, yData) is selected or not
    def getSelectedPoints(self, xData, yData, validData):
        import numpy
        total = numpy.zeros(len(xData))
        for curveKey in self.selectionCurveKeyList:
            total += self.curve(curveKey).getSelectedPoints(xData, yData, validData)
        unselected = numpy.equal(total, 0)
        selected = 1 - unselected
        return selected.tolist(), unselected.tolist()

    # set selections
    def setSelections(self, selections):
        for (xs, ys) in selections:
            curve = SelectionCurve(self)
            curve.setData(xs, ys)
            key = self.insertCurve(curve)
            self.selectionCurveKeyList.append(key)

    # get current selections in the form [([xs1, ys1]), (xs2, ys2)]
    def getSelections(self):
        data = []
        for key in self.selectionCurveKeyList:
            curve = self.curve(key)
            data.append(([curve.x(i) for i in range(curve.dataSize())], [curve.y(i) for i in range(curve.dataSize())]))
        return data

    def randomChange(self):
        import random
        if random.randint(0,1) and self.selectionCurveKeyList != []:
            self.removeLastSelection()
        else:
            curve = SelectionCurve(self)
            key = self.insertCurve(curve)
            self.selectionCurveKeyList.append(key)

            xMin = self.axisScale(QwtPlot.xBottom).lBound(); xMax = self.axisScale(QwtPlot.xBottom).hBound()
            yMin = self.axisScale(QwtPlot.yLeft).lBound();   yMax = self.axisScale(QwtPlot.yLeft).hBound()
            x1 = xMin + random.random()* (xMax-xMin); x2 = xMin + random.random()* (xMax-xMin)
            y1 = yMin + random.random()* (yMax-yMin); y2 = yMin + random.random()* (yMax-yMin)

            curve.setData([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])


    # save graph in matplotlib python file
    def saveToMatplotlib(self, fileName, size = QSize(400,400)):
        f = open(fileName, "wt")

        x1 = self.axisScale(QwtPlot.xBottom).lBound(); x2 = self.axisScale(QwtPlot.xBottom).hBound()
        y1 = self.axisScale(QwtPlot.yLeft).lBound();   y2 = self.axisScale(QwtPlot.yLeft).hBound()

        if self.showAxisScale == 0: edgeOffset = 0.01
        else: edgeOffset = 0.08

        f.write("from pylab import *\nfrom matplotlib import font_manager\n\n#possible changes in how the plot looks\n#rcParams['xtick.major.size'] = 0\n#rcParams['ytick.major.size'] = 0\n\n#constants\nx1 = %f; x2 = %f\ny1 = %f; y2 = %f\ndpi = 80\nxsize = %d\nysize = %d\nedgeOffset = %f\n\nfigure(facecolor = 'w', figsize = (xsize/float(dpi), ysize/float(dpi)), dpi = dpi)\nhold(True)\n" % (x1,x2,y1,y2,size.width(), size.height(), edgeOffset))

        linestyles = ['None', "-", "-.", "--", ":", "-", "-"]      # qwt line styles: NoCurve, Lines, Sticks, Steps, Dots, Spline, UserCurve
        markers = ["None", "o", "s", "^", "d", "v", "^", "<", ">", "x", "+"]    # curveSymbols = [None, Ellipse, Rect, Triangle, Diamond, DTriangle, UTriangle, LTriangle, RTriangle, XCross, Cross]

        f.write("#add curves\n")
        for key in self.curveKeys():
            c = self.curve(key)
            xData = [c.x(i) for i in range(c.dataSize())]
            yData = [c.y(i) for i in range(c.dataSize())]
            marker = markers[c.symbol().style()]

            markersize = c.symbol().size().width()
            markeredgecolor = self._getColorFromObject(c.symbol().pen())
            markerfacecolor = self._getColorFromObject(c.symbol().brush())
            color = self._getColorFromObject(c.pen())
            colorB = self._getColorFromObject(c.brush())
            linewidth = c.pen().width()
            if isinstance(c, PolygonCurve):
                x0 = min(xData); x1 = max(xData); diffX = x1-x0
                y0 = min(yData); y1 = max(yData); diffY = y1-y0
                f.write("gca().add_patch(Rectangle((%f, %f), %f, %f, edgecolor=%s, facecolor = %s, linewidth = %d, fill = 1))\n" % (x0,y0,diffX, diffY, color, colorB, linewidth))
            elif isinstance(c, RectangleCurve):        # rectangle curve can contain multiple rects, each has 5 points (the last one is the same as the first)
                for i in range(len(xData))[::5]:
                    x0 = min(xData[i:i+5]); x1 = max(xData[i:i+5]); diffX = x1-x0
                    y0 = min(yData[i:i+5]); y1 = max(yData[i:i+5]); diffY = y1-y0
                    f.write("gca().add_patch(Rectangle((%f, %f), %f, %f, edgecolor=%s, facecolor = %s, linewidth = %d, fill = 1))\n" % (x0,y0,diffX, diffY, color, colorB, linewidth))
            elif c.__class__ == UnconnectedLinesCurve:
                for i in range(len(xData))[::2]:        # multiple unconnected lines
                    f.write("plot(%s, %s, marker = 'None', linestyle = '-', color = %s, linewidth = %d)\n" % (xData[i:i+2], yData[i:i+2], color, linewidth))
            elif c.style() < len(linestyles):
                linestyle = linestyles[c.style()]
                f.write("plot(%s, %s, marker = '%s', linestyle = '%s', markersize = %d, markeredgecolor = %s, markerfacecolor = %s, color = %s, linewidth = %d)\n" % (xData, yData, marker, linestyle, markersize, markeredgecolor, markerfacecolor, color, linewidth))

        f.write("\n# add markers\n")
        for key in self.markerKeys():
            marker = self.marker(key)
            x = marker.xValue()
            y = marker.yValue()
            text = str(marker.label())
            align = marker.labelAlignment()
            xalign = (align & Qt.AlignLeft and "right") or (align & Qt.AlignHCenter and "center") or (align & Qt.AlignRight and "left")
            yalign = (align & Qt.AlignBottom and "top") or (align & Qt.AlignTop and "bottom") or (align & Qt.AlignVCenter and "center")
            vertAlign = (yalign and ", verticalalignment = '%s'" % yalign) or ""
            horAlign = (xalign and ", horizontalalignment = '%s'" % xalign) or ""
            color = (marker.labelColor().red()/255., marker.labelColor().green()/255., marker.labelColor().blue()/255.)
            name = str(marker.font().family())
            weight = marker.font().bold() and "bold" or "normal"
            if marker.__class__ == RotatedMarker: extra = ", rotation = %f" % (marker.rotation)
            else: extra = ""
            f.write("text(%f, %f, '%s'%s%s, color = %s, name = '%s', weight = '%s'%s)\n" % (x, y, text, vertAlign, horAlign, color, name, weight, extra))

        # grid
        f.write("# enable grid\ngrid(%s)\n\n" % (self.grid().xEnabled() and self.grid().yEnabled() and "True" or "False"))

        # axis
        if self.showAxisScale == 0:
            f.write("#hide axis\naxis('off')\naxis([x1, x2, y1, y2])\ngca().set_position([edgeOffset, edgeOffset, 1 - 2*edgeOffset, 1 - 2*edgeOffset])\n")
        else:
            if self.axisScaleDraw(QwtPlot.yLeft).__class__ == DiscreteAxisScaleDraw:
                labels = self.axisScaleDraw(QwtPlot.yLeft).labels
                f.write("yticks(%s, %s)\nlabels = gca().get_yticklabels()\nsetp(labels, rotation=-%.3f) #, weight = 'bold', fontsize=10)\n\n" % (range(len(labels)), labels, self.axisScaleDraw(QwtPlot.yLeft).labelRotation()))
            if self.axisScaleDraw(QwtPlot.xBottom).__class__ == DiscreteAxisScaleDraw:
                labels = self.axisScaleDraw(QwtPlot.xBottom).labels
                f.write("xticks(%s, %s)\nlabels = gca().get_xticklabels()\nsetp(labels, rotation=-%.3f) #, weight = 'bold', fontsize=10)\n\n" % (range(len(labels)), labels, self.axisScaleDraw(QwtPlot.xBottom).labelRotation()))

            f.write("#set axis labels\nxlabel('%s', weight = 'bold')\nylabel('%s', weight = 'bold')\n\n" % (str(self.axisTitle(QwtPlot.xBottom)), str(self.axisTitle(QwtPlot.yLeft))))
            f.write("\naxis([x1, x2, y1, y2])\ngca().set_position([edgeOffset, edgeOffset, 1 - 2*edgeOffset, 1 - 2*edgeOffset])\n#subplots_adjust(left = 0.08, bottom = 0.11, right = 0.98, top = 0.98)\n")

        f.write("\n# possible settings to change\n#axes().set_frame_on(0) #hide the frame\n#axis('off') #hide the axes and labels on them\n\n")

        if self.legend().itemCount() > 0:
            legendItems = []
            for item in self.legend().contentsWidget().children():
                if isinstance(item, QwtLegendButton):
                    text = str(item.title()).replace("<b>", "").replace("</b>", "")
                    if not item.symbol():
                        legendItems.append((text, None, None, None))
                    else:
                        legendItems.append((text, markers[item.symbol().style()], self._getColorFromObject(item.symbol().pen()) , self._getColorFromObject(item.symbol().brush())))
            f.write("""
#functions to show legend below the figure
def drawSomeLegendItems(x, items, itemsPerAxis = 1, yDiff = 0.0):
    axes([x-0.1, .018*itemsPerAxis - yDiff, .2, .018], frameon = 0); axis('off')
    lines = [plot([],[], label = text, marker = marker, markeredgecolor = edgeC, markerfacecolor = faceC) for (text, marker, edgeC, faceC) in items]
    legend(lines, [item[0] for item in items], 'upper center', handlelen = 0.1, numpoints = 1, prop = font_manager.FontProperties(size=11))
    gca().get_legend().draw_frame(False)

def drawLegend(items):
    if not items: return
    maxAttrInLine = 5
    xs = [i/float(min(maxAttrInLine+1, len(items)+1)) for i in range(1, min(maxAttrInLine+1, len(items)+1))]
    if items[0][1] == None: extraLabelForClass = [xs.pop(0), [items.pop(0)]]
    itemsPerAxis = len(items) / len(xs) + (len(items) %% len(xs) != 0)
    if "extraLabelForClass" in dir(): drawSomeLegendItems(extraLabelForClass[0], extraLabelForClass[1], itemsPerAxis, yDiff = 0.004)

    for i, x in enumerate(xs):
        drawSomeLegendItems(x, items[i*itemsPerAxis: min(len(items), (i+1)*itemsPerAxis)], itemsPerAxis)

items = %s
drawLegend(items)\n""" % (str(legendItems)))

        f.write("\nshow()")



    def _getColorFromObject(self, obj):
        if obj.__class__ == QBrush and obj.style() == Qt.NoBrush: return "'none'"
        if obj.__class__ == QPen   and obj.style() == Qt.NoPen: return "'none'"
        col = [obj.color().red(), obj.color().green(), obj.color().blue()];
        col = tuple([v/float(255) for v in col])
        return col

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
            if self.rect.contains(p):
                self.tip(self.rect, self.text)


class RotatedMarker(QwtPlotMarker):
    def __init__(self, parent, label = "", x = 0.0, y = 0.0, rotation = 0):
        QwtPlotMarker.__init__(self, parent)
        self.rotation = rotation
        self.parent = parent
        self.x = x
        self.y = y
        self.setXValue(x)
        self.setYValue(y)
        self.parent = parent

        if rotation != 0: self.setLabel(label + "  ")
        else:             self.setLabel(label)

    def setRotation(self, rotation):
        self.rotation = rotation

    def draw(self, painter, x, y, rect):
        rot = math.radians(self.rotation)

        x2 = x * math.cos(rot) - y * math.sin(rot)
        y2 = x * math.sin(rot) + y * math.cos(rot)

        painter.rotate(-self.rotation)
        QwtPlotMarker.draw(self, painter, x2, y2, rect)
        painter.rotate(self.rotation)

