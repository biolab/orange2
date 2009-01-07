#
# owGraph.py
#
# the base for all graphs

from PyQt4.Qwt5 import *
from OWGraphTools import *      # user defined curves, ...
from OWColorPalette import *      # color palletes, ...
from OWDlgs import OWChooseImageSizeDlg
import orange, math, time
from OWBaseWidget import unisetattr

NOTHING = 0
ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3
PANNING = 4
SELECT = 5

class OWGraph(QwtPlot):
    def __init__(self, parent = None, name = "None", showLegend=1):
        "Constructs the graph"
        QwtPlot.__init__(self, parent)
        self.parentName = name
        #self.setWindowFlags(Qt.WResizeNoErase) #this works like magic.. no flicker during repaint!
        self.setAutoReplot(False)

        self.setAxisAutoScale(QwtPlot.xBottom)
        self.setAxisAutoScale(QwtPlot.xTop)
        self.setAxisAutoScale(QwtPlot.yLeft)
        self.setAxisAutoScale(QwtPlot.yRight)

        self.axisTitleFont = QFont('Helvetica', 10, QFont.Bold)
        text = QwtText("")
        text.setFont(self.axisTitleFont)
        self.setAxisTitle(QwtPlot.xBottom, text)
        self.setAxisTitle(QwtPlot.xTop, text)
        self.setAxisTitle(QwtPlot.yLeft, text)
        self.setAxisTitle(QwtPlot.yRight, text)

        ticksFont = QFont('Helvetica', 9)
        self.setAxisFont(QwtPlot.xBottom, ticksFont)
        self.setAxisFont(QwtPlot.xTop, ticksFont)
        self.setAxisFont(QwtPlot.yLeft, ticksFont)
        self.setAxisFont(QwtPlot.yRight, ticksFont)
        #self.setLegendFont(ticksFont)

        self.tipLeft = None
        self.tipRight = None
        self.tipBottom = None
        self._cursor = Qt.ArrowCursor

        self.showAxisScale = 1
        self.showMainTitle = 0
        self.showXaxisTitle = 0
        self.showYLaxisTitle = 0
        self.showYRaxisTitle = 0
        self.mainTitle = None
        self.XaxisTitle = None
        self.YLaxisTitle = None
        self.YRaxisTitle = None
        self.useAntialiasing = 0

        self.state = ZOOMING
        self.tempSelectionCurve = None
        self.selectionCurveList = []
        self.autoSendSelectionCallback = None   # callback function to call when we add new selection polygon or rectangle
        self.sendSelectionOnUpdate = 0
        self.showLegend = showLegend
        if self.showLegend:
            self.insertLegend(QwtLegend(), QwtPlot.BottomLegend)

        self.gridCurve = QwtPlotGrid()
        #self.gridCurve.attach(self)

        self.mouseCurrentlyPressed = 0
        self.mouseCurrentButton = 0
        self.enableWheelZoom = 0
        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.NoSymbol)
        self.tips = TooltipManager(self)
        self.statusBar = None
        self.canvas().setMouseTracking(1)
        self.setMouseTracking(1)
        self.zoomStack = []
        self.panPosition = None
        self.optimizedDrawing = 1
        self.pointWidth = 5
        self.showFilledSymbols = 1
        self.alphaValue = 255
        self.setCanvasColor(QColor(Qt.white))
        self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Triangle, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]
        #self.curveSymbols = [QwtSymbol.Triangle, QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]

        # uncomment this if you want to use printer friendly symbols
        #self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.XCross, QwtSymbol.Triangle, QwtSymbol.Cross, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.Rect, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle]
        self.contPalette = ColorPaletteGenerator(numberOfColors = -1)
        self.discPalette = ColorPaletteGenerator()

        # when using OWGraph we can define functions that will receive mouse move, press, release events. these functions
        # HAVE TO RETURN whether the signal was handled, or you also want to use default OWGraph handler
        self.mousePressEventHandler = None
        self.mouseMoveEventHandler = None
        self.mouseReleaseEventHandler = None
        self.mouseStaticClickHandler = self.staticMouseClick
        self.enableGridXB(0)
        self.enableGridYL(0)

        #self.updateLayout()

    def setCursor(self, cursor):
        self._cursor = cursor
        self.canvas().setCursor(cursor)

    def __setattr__(self, name, value):
        unisetattr(self, name, value, QwtPlot)

    # call to update dictionary with settings
    def updateSettings(self, **settings):
        self.__dict__.update(settings)

    def saveToFile(self, extraButtons = []):
        sizeDlg = OWChooseImageSizeDlg(self, extraButtons)
        sizeDlg.exec_()

    def saveToFileDirect(self, fileName, size = None):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveImage(fileName, size)


    def setTickLength(self, axis, minor, medium, major):
        self.axisScaleDraw(axis).setTickLength(QwtScaleDiv.MinorTick, minor)
        self.axisScaleDraw(axis).setTickLength(QwtScaleDiv.MediumTick, medium)
        self.axisScaleDraw(axis).setTickLength(QwtScaleDiv.MajorTick, major)


    def setYLlabels(self, labels):
        "Sets the Y-axis labels on the left."
        self.axisScaleDraw(QwtPlot.yLeft).enableComponent(QwtScaleDraw.Backbone, self.showAxisScale)
        self.axisScaleDraw(QwtPlot.yLeft).enableComponent(QwtScaleDraw.Ticks, self.showAxisScale)
        self.axisScaleDraw(QwtPlot.yLeft).enableComponent(QwtScaleDraw.Labels, self.showAxisScale)
        if not self.showAxisScale:
            return

        #self.setTickLength(QwtPlot.yLeft, 1, 1, 3)

        if (labels <> None):
            self.setAxisScaleDraw(QwtPlot.yLeft, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.yLeft, 0, len(labels) - 1, 1)
            self.setAxisMaxMinor(QwtPlot.yLeft, 0)
            self.setAxisMaxMajor(QwtPlot.yLeft, len(labels))
        else:
            self.setAxisScaleDraw(QwtPlot.yLeft, QwtScaleDraw())
            self.setAxisAutoScale(QwtPlot.yLeft)
            self.setAxisMaxMinor(QwtPlot.yLeft, 10)
            self.setAxisMaxMajor(QwtPlot.yLeft, 10)

    def setYRlabels(self, labels):
        "Sets the Y-axis labels on the right."
        self.axisScaleDraw(QwtPlot.yRight).enableComponent(QwtScaleDraw.Backbone, self.showAxisScale)
        self.axisScaleDraw(QwtPlot.yRight).enableComponent(QwtScaleDraw.Ticks, self.showAxisScale)
        self.axisScaleDraw(QwtPlot.yRight).enableComponent(QwtScaleDraw.Labels, self.showAxisScale)
        if not self.showAxisScale:
            return

        if (labels <> None):
            self.setAxisScaleDraw(QwtPlot.yRight, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.yRight, 0, len(labels) - 1, 1)
            self.setAxisMaxMinor(QwtPlot.yRight, 0)
            self.setAxisMaxMajor(QwtPlot.yRight, len(labels))
        else:
            self.setAxisScaleDraw(QwtPlot.yRight, QwtScaleDraw())
            self.setAxisAutoScale(QwtPlot.yRight)
            self.setAxisMaxMinor(QwtPlot.yRight, 10)
            self.setAxisMaxMajor(QwtPlot.yRight, 10)

    def setXlabels(self, labels):
        "Sets the x-axis labels if x-axis discrete."
        "Or leave up to QwtPlot (MaxMajor, MaxMinor) if x-axis continuous."
        self.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Backbone, self.showAxisScale)
        self.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Ticks, self.showAxisScale)
        self.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Labels, self.showAxisScale)
        if not self.showAxisScale:
            return

        if (labels <> None):
            self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.xBottom, 0, len(labels) - 1, 1)
            self.setAxisMaxMinor(QwtPlot.xBottom, 0)
            self.setAxisMaxMajor(QwtPlot.xBottom, len(labels))
        else:
            self.setAxisScaleDraw(QwtPlot.xBottom, QwtScaleDraw())
            self.setAxisAutoScale(QwtPlot.xBottom)
            self.setAxisMaxMinor(QwtPlot.xBottom, 10)
            self.setAxisMaxMajor(QwtPlot.xBottom, 10)

    def enableXaxis(self, enable):
        self.enableAxis(QwtPlot.xBottom, enable)
        self.repaint()

    def enableYLaxis(self, enable):
        self.enableAxis(QwtPlot.yLeft, enable)
        self.repaint()

    def enableYRaxis(self, enable):
        self.enableAxis(QwtPlot.yRight, enable)
        self.repaint()

    def setRightTip(self,explain):
        "Sets the tooltip for the right y axis"
        self.tipRight = explain

    def setLeftTip(self,explain):
        "Sets the tooltip for the left y axis"
        self.tipLeft = explain

    def setBottomTip(self,explain):
        "Sets the tooltip for the left x axis"
        self.tipBottom = explain

    def setShowMainTitle(self, b):
        self.showMainTitle = b
        if self.showMainTitle and self.mainTitle:
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(QwtText())
        self.repaint()

    def setMainTitle(self, t):
        self.mainTitle = t
        if self.showMainTitle and self.mainTitle:
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(QwtText())
        self.repaint()

    def setShowXaxisTitle(self, b = -1):
        if b == self.showXaxisTitle: return
        if b != -1:
            self.showXaxisTitle = b
        if self.showXaxisTitle and self.XaxisTitle:
            self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.xBottom, QwtText())
        self.repaint()

    def setXaxisTitle(self, title):
        if title == self.XaxisTitle: return
        self.XaxisTitle = title
        if self.showXaxisTitle and self.XaxisTitle:
            self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.xBottom, QwtText())
        #self.updateLayout()
        self.repaint()

    def setShowYLaxisTitle(self, b = -1):
        if b == self.showYLaxisTitle: return
        if b != -1:
            self.showYLaxisTitle = b
        if self.showYLaxisTitle and self.YLaxisTitle:
            self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yLeft, QwtText())
        #self.updateLayout()
        self.repaint()

    def setYLaxisTitle(self, title):
        if title == self.YLaxisTitle: return
        self.YLaxisTitle = title
        if self.showYLaxisTitle and self.YLaxisTitle:
            self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yLeft, QwtText())
        #self.updateLayout()
        self.repaint()

    def setShowYRaxisTitle(self, b = -1):
        if b == self.showYRaxisTitle: return
        if b != -1:
            self.showYRaxisTitle = b
        if self.showYRaxisTitle and self.YRaxisTitle:
            self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yRight, QwtText())
        #self.updateLayout()
        self.repaint()

    def setYRaxisTitle(self, title):
        if title == self.YRaxisTitle: return
        self.YRaxisTitle = title
        if self.showYRaxisTitle and self.YRaxisTitle:
            self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yRight, QwtText())
        #self.updateLayout()
        self.repaint()

    def enableGridXB(self, b):
        self.gridCurve.enableX(b)
        self.replot()

    def enableGridYL(self, b):
        self.gridCurve.enableY(b)
        self.replot()

    def setGridColor(self, c):
        self.gridCurve.setPen(QPen(c))
        self.replot()

    def setCanvasColor(self, c):
        self.setCanvasBackground(c)
        self.repaint()

    # ############################################################
    # functions that were previously in OWVisGraph
    # ############################################################
    def setData(self, data):
        # clear all curves, markers, tips
        self.clear()
        self.removeAllSelections(0)  # clear all selections
        self.tips.removeAll()
        self.zoomStack = []

    # ####################################################################
    # return string with attribute names and their values for example example
    def getExampleTooltipText(self, example, indices = None, maxIndices = 20):
        if indices and type(indices[0]) == str:
            indices = [self.attributeNameIndex[i] for i in indices]
        if not indices: 
            indices = range(len(self.dataDomain.attributes))
        
        # don't show the class value twice
        if example.domain.classVar:
            classIndex = self.attributeNameIndex[example.domain.classVar.name]
            while classIndex in indices:
                indices.remove(classIndex)      
      
        text = "<b>Attributes:</b><br>"
        for index in indices[:maxIndices]:
            attr = self.attributeNames[index]
            if attr not in example.domain:  text += "&nbsp;"*4 + "%s = ?<br>" % (attr)
            elif example[attr].isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (attr)
            else:                           text += "&nbsp;"*4 + "%s = %s<br>" % (attr, str(example[attr]))
        if len(indices) > maxIndices:
            text += "&nbsp;"*4 + " ... <br>"

        if example.domain.classVar:
            text = text[:-4]
            text += "<hr><b>Class:</b><br>"
            if example.getclass().isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (example.domain.classVar.name)
            else:                              text += "&nbsp;"*4 + "%s = %s<br>" % (example.domain.classVar.name, str(example.getclass()))

        if len(example.domain.getmetas()) != 0:
            text = text[:-4]
            text += "<hr><b>Meta attributes:</b><br>"
            # show values of meta attributes
            for key in example.domain.getmetas():
                try: text += "&nbsp;"*4 + "%s = %s<br>" % (example.domain[key].name, str(example[key]))
                except: pass
        return text[:-4]        # remove the last <br>

    def addCurve(self, name, brushColor = Qt.black, penColor = Qt.black, size = 5, style = QwtPlotCurve.NoCurve, symbol = QwtSymbol.Ellipse, enableLegend = 0, xData = [], yData = [], showFilledSymbols = None, lineWidth = 1, pen = None, autoScale = 0, antiAlias = None, penAlpha = 255, brushAlpha = 255):
        curve = QwtPlotCurve(name)
        curve.setRenderHint(QwtPlotItem.RenderAntialiased, antiAlias == 1 or self.useAntialiasing)
        curve.setItemAttribute(QwtPlotItem.Legend, enableLegend)
        curve.setItemAttribute(QwtPlotItem.AutoScale, autoScale)
        if penAlpha != 255:
            penColor.setAlpha(penAlpha)
        if brushAlpha != 255:
            brushColor.setAlpha(brushAlpha)

        if showFilledSymbols or (showFilledSymbols == None and self.showFilledSymbols):
            newSymbol = QwtSymbol(symbol, QBrush(brushColor), QPen(penColor), QSize(size, size))
        else:
            newSymbol = QwtSymbol(symbol, QBrush(), QPen(penColor), QSize(size, size))
        curve.setSymbol(newSymbol)
        curve.setStyle(style)
        curve.setPen(pen != None and pen or QPen(penColor, lineWidth))
        if xData != [] and yData != []:
            curve.setData(xData, yData)
        curve.attach(self)
        return curve

    def addMarker(self, name, x, y, alignment = -1, bold = 0, color = None, brushColor = None, size=None, antiAlias = None):
        text = QwtText(name, QwtText.PlainText)
        if color != None:
            text.setColor(color)
            text.setPaintAttribute(QwtText.PaintUsingTextColor, 1)
        if brushColor != None:
            text.setBackgroundBrush(QBrush(brushColor))
        font = text.font()
        if bold:  font.setBold(1)
        if size:  font.setPixelSize(size)
        text.setFont(font)
        text.setPaintAttribute(QwtText.PaintUsingTextFont, 1)
        #if alignment != -1:  text.setRenderFlags(alignment)

        marker = QwtPlotMarker()
        marker.setLabel(text)
        marker.setValue(x,y)
        marker.setRenderHint(QwtPlotItem.RenderAntialiased, antiAlias == 1 or self.useAntialiasing)
        if alignment != -1:
            marker.setLabelAlignment(alignment)
        marker.attach(self)
        return marker

    # show a tooltip at x,y with text. if the mouse will move for more than 2 pixels it will be removed
    def showTip(self, x, y, text):
        QToolTip.showText(self.mapToGlobal(QPoint(x, y)), text, self.canvas(), QRect(x-3,y-3,6,6))

    # mouse was only pressed and released on the same spot. visualization methods might want to process this event
    def staticMouseClick(self, e):
        return 0

    def activateZooming(self):
        self.state = ZOOMING
        if self.tempSelectionCurve: self.removeLastSelection()

    def activateRectangleSelection(self):
        self.state = SELECT_RECTANGLE
        if self.tempSelectionCurve: self.removeLastSelection()

    def activatePolygonSelection(self):
        self.state = SELECT_POLYGON
        if self.tempSelectionCurve: self.removeLastSelection()

    def activatePanning(self):
        self.state = PANNING
        if self.tempSelectionCurve: self.removeLastSelection()

    def activateSelection(self):
        self.state = SELECT

    def removeDrawingCurves(self, removeLegendItems = 1, removeSelectionCurves = 0, removeMarkers = 0):
        for curve in self.itemList():
            if not removeLegendItems and curve.testItemAttribute(QwtPlotItem.Legend):
                continue
            if not removeSelectionCurves and isinstance(curve, SelectionCurve):
                continue
            if not removeMarkers and isinstance(curve, QwtPlotMarker):
                continue
            curve.detach()
        self.gridCurve.attach(self)        # we also removed the grid curve

    def removeMarkers(self):
        self.detachItems(QwtPlotItem.Rtti_PlotMarker)

    def removeLastSelection(self):
        removed = 0
        if self.selectionCurveList != []:
            lastCurve = self.selectionCurveList.pop()
            lastCurve.detach()
            self.tempSelectionCurve = None
            removed = 1
        self.replot()
        if self.autoSendSelectionCallback:
            self.autoSendSelectionCallback() # do we want to send new selection
        return removed

    def removeAllSelections(self, send = 1):
        selectionsExisted = len(self.selectionCurveList) > 0
        self.detachItems(SelectionCurveRtti)
        self.selectionCurveList = []
        if selectionsExisted:
            self.replot()
            if send and self.autoSendSelectionCallback:
                self.autoSendSelectionCallback() # do we want to send new selection

    def zoomOut(self):
        if len(self.zoomStack):
            newXMin, newXMax, newYMin, newYMax = self.zoomStack.pop()
            self.setNewZoom(newXMin, newXMax, newYMin, newYMax)
            return 1
        return 0

    def setNewZoom(self, newXMin, newXMax, newYMin, newYMax):
        oldXMin = self.axisScaleDiv(QwtPlot.xBottom).lBound()
        oldXMax = self.axisScaleDiv(QwtPlot.xBottom).hBound()
        oldYMin = self.axisScaleDiv(QwtPlot.yLeft).lBound()
        oldYMax = self.axisScaleDiv(QwtPlot.yLeft).hBound()
        stepX, stepY = self.axisStepSize(QwtPlot.xBottom), self.axisStepSize(QwtPlot.yLeft)

        steps = 10
        for i in range(1, steps+1):
            midXMin = oldXMin * (steps-i)/float(steps) + newXMin * i/float(steps)
            midXMax = oldXMax * (steps-i)/float(steps) + newXMax * i/float(steps)
            midYMin = oldYMin * (steps-i)/float(steps) + newYMin * i/float(steps)
            midYMax = oldYMax * (steps-i)/float(steps) + newYMax * i/float(steps)
            self.setAxisScale(QwtPlot.xBottom, midXMin, midXMax, stepX)
            self.setAxisScale(QwtPlot.yLeft, midYMin, midYMax, stepY)
            #if i == steps:
            #    self.removeCurve(zoomOutCurveKey)
            t = time.time()
            self.replot()
            if time.time()-t > 0.1:
                self.setAxisScale(QwtPlot.xBottom, newXMin, newXMax, stepX)
                self.setAxisScale(QwtPlot.yLeft, newYMin, newYMax, stepY)
                self.replot()
                break

    def closestMarker(self, intX, intY):
        point = QPoint(intX, intY)
        marker = None
        dist = 1e30
        for curve in self.itemList():
            if isinstance(curve, QwtPlotMarker):
                curvePoint = QPoint(self.transform(QwtPlot.xBottom, curve.xValue()), self.transform(QwtPlot.yLeft, curve.yValue()))
                d = (point - curvePoint).manhattanLength()
                if d < dist:
                    dist = d
                    marker = curve
        return marker, dist


    def closestCurve(self, intX, intY):
        point = QPoint(intX, intY)
        nearestCurve = None
        dist = 10000000000
        index = -1
        for curve in self.itemList():
            if isinstance(curve, QwtPlotCurve) and curve.dataSize() > 0:
                ind, d = curve.closestPoint(point)
                if d < dist:
                    nearestCurve, dist, index = curve, d, ind
        if nearestCurve == None:
            return None, 0, 0, 0, 0
        else:
            return nearestCurve, dist, nearestCurve.x(index), nearestCurve.y(index), index


    # ###############################################
    # HANDLING MOUSE EVENTS
    # ###############################################
    def mousePressEvent(self, e):
        if self.mousePressEventHandler != None:
            handled = self.mousePressEventHandler(e)
            if handled: return
        QwtPlot.mousePressEvent(self, e)
        canvasPos = self.canvas().mapFrom(self, e.pos())
        xFloat = self.invTransform(QwtPlot.xBottom, canvasPos.x())
        yFloat = self.invTransform(QwtPlot.yLeft, canvasPos.y())
        self.xpos = canvasPos.x()
        self.ypos = canvasPos.y()

        self.mouseCurrentlyPressed = 1
        self.mouseCurrentButton = e.button()

        if self.state not in [ZOOMING, PANNING]:
            insideRects = [rect.isInside(xFloat, yFloat) for rect in self.selectionCurveList]
            onEdgeRects = [rect.isOnEdge(xFloat, yFloat) for rect in self.selectionCurveList]

        # ####
        # ZOOM
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            self.tempSelectionCurve = RectangleSelectionCurve(pen = Qt.DashLine)
            self.tempSelectionCurve.attach(self)

        # ####
        # PANNING
        elif e.button() == Qt.LeftButton and self.state == PANNING:
            self.panPosition = e.globalX(), e.globalY()
            self.paniniX = self.axisScaleDiv(QwtPlot.xBottom).lBound(), self.axisScaleDiv(QwtPlot.xBottom).hBound()
            self.paniniY = self.axisScaleDiv(QwtPlot.yLeft).lBound(), self.axisScaleDiv(QwtPlot.yLeft).hBound()

        elif e.button() == Qt.LeftButton and 1 in onEdgeRects and self.tempSelectionCurve == None:
            self.resizingCurve = self.selectionCurveList[onEdgeRects.index(1)]

        # have we pressed the mouse inside one of the selection curves?
        elif e.button() == Qt.LeftButton and 1 in insideRects and self.tempSelectionCurve == None:
            self.movingCurve = self.selectionCurveList[insideRects.index(1)]
            self.movingCurve.mousePosition = (xFloat, yFloat)

        # ####
        # SELECT RECTANGLE
        elif e.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE:
            self.tempSelectionCurve = RectangleSelectionCurve()
            self.tempSelectionCurve.attach(self)
            self.selectionCurveList.append(self.tempSelectionCurve)

        # ####
        # SELECT POLYGON
        elif e.button() == Qt.LeftButton and self.state == SELECT_POLYGON:
            if self.tempSelectionCurve == None:
                self.tempSelectionCurve = SelectionCurve()
                self.tempSelectionCurve.attach(self)
                self.selectionCurveList.append(self.tempSelectionCurve)
                self.tempSelectionCurve.addPoint(self.invTransform(QwtPlot.xBottom, self.xpos), self.invTransform(QwtPlot.yLeft, self.ypos))
            self.tempSelectionCurve.addPoint(self.invTransform(QwtPlot.xBottom, self.xpos), self.invTransform(QwtPlot.yLeft, self.ypos))

            if self.tempSelectionCurve.closed():    # did we intersect an existing line. if yes then close the curve and finish appending lines
                self.tempSelectionCurve = None
                self.replot()
                if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection
            
        


    # only needed to show the message in statusbar
    def mouseMoveEvent(self, e):
        if self.mouseMoveEventHandler != None:
            handled = self.mouseMoveEventHandler(e)
            if handled: return
        QwtPlot.mouseMoveEvent(self, e)
        canvasPos = self.canvas().mapFrom(self, e.pos())
        xFloat = self.invTransform(QwtPlot.xBottom, canvasPos.x())
        yFloat = self.invTransform(QwtPlot.yLeft, canvasPos.y())

        text = ""
        if not self.mouseCurrentlyPressed:
            (text, x, y) = self.tips.maybeTip(xFloat, yFloat)
            if type(text) == int: text = self.buildTooltip(text)

        if self.statusBar != None:
            self.statusBar.showMessage(text)
        if text != "":
            self.showTip(self.transform(QwtPlot.xBottom, x), self.transform(QwtPlot.yLeft, y), text)
        
        if self.tempSelectionCurve != None and (self.state == ZOOMING or self.state == SELECT_RECTANGLE):
            x1 = self.invTransform(QwtPlot.xBottom, self.xpos)
            y1 = self.invTransform(QwtPlot.yLeft, self.ypos)
            self.tempSelectionCurve.setPoints(x1, y1, xFloat, yFloat)
            self.replot()

        elif self.tempSelectionCurve != None and self.state == SELECT_POLYGON:
            self.tempSelectionCurve.replaceLastPoint(xFloat,yFloat)
            self.replot()

        elif hasattr(self, "resizingCurve"):
            self.resizingCurve.updateCurve(xFloat, yFloat)            
            self.replot()
            if self.sendSelectionOnUpdate and self.autoSendSelectionCallback:
                self.autoSendSelectionCallback()
            
        # do we have a selection curve we are currently moving?
        elif hasattr(self, "movingCurve"):
            self.movingCurve.moveBy(xFloat-self.movingCurve.mousePosition[0], yFloat-self.movingCurve.mousePosition[1])
            self.movingCurve.mousePosition = (xFloat, yFloat)
            self.replot()
            if self.sendSelectionOnUpdate and self.autoSendSelectionCallback:
                self.autoSendSelectionCallback() 

        elif self.state == PANNING and self.panPosition:
            if hasattr(self, "paniniX") and hasattr(self, "paniniY"):
                dx = self.invTransform(QwtPlot.xBottom, self.panPosition[0]) - self.invTransform(QwtPlot.xBottom, e.globalX())
                dy = self.invTransform(QwtPlot.yLeft, self.panPosition[1]) - self.invTransform(QwtPlot.yLeft, e.globalY())
                xEnabled, xMin, xMax = getattr(self, "xPanningInfo", (1, self.paniniX[0] + dx, self.paniniX[1] + dx))
                yEnabled, yMin, yMax = getattr(self, "yPanningInfo", (1, self.paniniY[0] + dy, self.paniniY[1] + dy))

                if self.paniniX[0] + dx < xMin:  # if we reached the left edge, don't change the right edge
                    xMax = self.paniniX[1] - (self.paniniX[0] - xMin)
                elif self.paniniX[1] + dx > xMax:   # if we reached the right edge, don't change the left edge
                    xMin = self.paniniX[0] + (xMax - self.paniniX[1])
                else:
                    xMin, xMax = self.paniniX[0] + dx, self.paniniX[1] + dx
                if xEnabled: self.setAxisScale(QwtPlot.xBottom, xMin, xMax, self.axisStepSize(QwtPlot.xBottom))

                if self.paniniY[0] + dy < yMin:  # if we reached the left edge, don't change the right edge
                    yMax = self.paniniY[1] - (self.paniniY[0] - yMin)
                elif self.paniniY[1] + dy > yMax:   # if we reached the right edge, don't change the left edge
                    yMin = self.paniniY[0] + (yMax - self.paniniY[1])
                else:
                    yMin, yMax = self.paniniY[0] + dy, self.paniniY[1] + dy
                if yEnabled: self.setAxisScale(QwtPlot.yLeft, yMin, yMax, self.axisStepSize(QwtPlot.yLeft))

                if xEnabled or yEnabled: self.replot()

        # if we are in the selection state then we perhaps show the cursors to move or resize the selection curves
        if self.state not in [ZOOMING, PANNING] and getattr(self, "resizingCurve", None) == None and self.tempSelectionCurve == None:
            onEdge = [rect.isOnEdge(xFloat, yFloat) for rect in self.selectionCurveList]
            if 1 in onEdge:
                self.canvas().setCursor(self.selectionCurveList[onEdge.index(1)].appropriateCursor)
            # check if we need to change the cursor if we are at some selection box
            elif 1 in [rect.isInside(xFloat, yFloat) for rect in self.selectionCurveList]:
                self.canvas().setCursor(Qt.OpenHandCursor)
            else:
                self.canvas().setCursor(self._cursor)



    def mouseReleaseEvent(self, e):
        if self.mouseReleaseEventHandler != None:
            handled = self.mouseReleaseEventHandler(e)
            if handled: return
        QwtPlot.mouseReleaseEvent(self, e)
        if not self.mouseCurrentlyPressed: return   # this might happen if we double clicked the widget titlebar
        self.mouseCurrentlyPressed = 0
        self.mouseCurrentButton = 0
        self.panPosition = None
        staticClick = 0
        canvasPos = self.canvas().mapFrom(self, e.pos())

        if hasattr(self, "movingCurve"):
            del self.movingCurve
            if self.autoSendSelectionCallback:
                self.autoSendSelectionCallback() # send the new selection
                
        if hasattr(self, "resizingCurve"):
            del self.resizingCurve
            if self.autoSendSelectionCallback:
                self.autoSendSelectionCallback() # send the new selection


        if e.button() == Qt.LeftButton:
            if self.xpos == canvasPos.x() and self.ypos == canvasPos.y():
                handled = self.mouseStaticClickHandler(e)
                if handled: return
                staticClick = 1
                
            if self.state == ZOOMING:
                xmin, xmax = min(self.xpos, canvasPos.x()), max(self.xpos, canvasPos.x())
                ymin, ymax = min(self.ypos, canvasPos.y()), max(self.ypos, canvasPos.y())
                if self.tempSelectionCurve:
                    self.tempSelectionCurve.detach()
                self.tempSelectionCurve = None

                if staticClick or xmax-xmin < 4 or ymax-ymin < 4:
                    x = self.invTransform(QwtPlot.xBottom, canvasPos.x())
                    y = self.invTransform(QwtPlot.yLeft, canvasPos.y())
                    diffX = (self.axisScaleDiv(QwtPlot.xBottom).hBound() -  self.axisScaleDiv(QwtPlot.xBottom).lBound()) / 2.
                    diffY = (self.axisScaleDiv(QwtPlot.yLeft).hBound() -  self.axisScaleDiv(QwtPlot.yLeft).lBound()) / 2.

                    # use this to zoom to the place where the mouse cursor is
                    if diffX:
                        xmin = x - (diffX/2.) * (x - self.axisScaleDiv(QwtPlot.xBottom).lBound()) / diffX
                        xmax = x + (diffX/2.) * (self.axisScaleDiv(QwtPlot.xBottom).hBound() - x) / diffX
                    if diffY:
                        ymin = y + (diffY/2.) * (self.axisScaleDiv(QwtPlot.yLeft).hBound() - y) / diffY
                        ymax = y - (diffY/2.) * (y - self.axisScaleDiv(QwtPlot.yLeft).lBound()) / diffY
                else:
                    xmin = self.invTransform(QwtPlot.xBottom, xmin);  xmax = self.invTransform(QwtPlot.xBottom, xmax)
                    ymin = self.invTransform(QwtPlot.yLeft, ymin);    ymax = self.invTransform(QwtPlot.yLeft, ymax)

                self.zoomStack.append((self.axisScaleDiv(QwtPlot.xBottom).lBound(), self.axisScaleDiv(QwtPlot.xBottom).hBound(), self.axisScaleDiv(QwtPlot.yLeft).lBound(), self.axisScaleDiv(QwtPlot.yLeft).hBound()))
                self.setNewZoom(xmin, xmax, ymax, ymin)

            elif self.state == SELECT_RECTANGLE:
                self.tempSelectionCurve = None
                if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection
            
        elif e.button() == Qt.RightButton:
            if self.state == ZOOMING:
                ok = self.zoomOut()
                if not ok:
                    self.removeLastSelection()
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
                        self.tempSelectionCurve.replaceLastPoint(self.invTransform(QwtPlot.xBottom, canvasPos.x()), self.invTransform(QwtPlot.yLeft, canvasPos.y()))
                    self.replot()
                else:
                    ok = self.removeLastSelection()
                    if not ok: self.zoomOut()



    def wheelEvent(self, e):
        if not self.enableWheelZoom:
            return

        d = -e.delta()/120.

        if getattr(self, "controlPressed", False):
            ys = self.axisScaleDiv(QwtPlot.yLeft)
            yoff = d * (ys.hBound() - ys.lBound()) / 100.
            self.setAxisScale(QwtPlot.yLeft, ys.lBound() + yoff, ys.hBound() + yoff, self.axisStepSize(QwtPlot.yLeft))

        elif getattr(self, "altPressed", False):
            xs = self.axisScaleDiv(QwtPlot.xBottom)
            xoff = d * (xs.hBound() - xs.lBound()) / 100.
            self.setAxisScale(QwtPlot.xBottom, xs.lBound() - xoff, xs.hBound() - xoff, self.axisStepSize(QwtPlot.xBottom))

        else:
            ro, rn = .9**d, 1-.9**d

            pos = self.mapFromGlobal(e.pos())
            ex, ey = pos.x(), pos.y()

            xs = self.axisScaleDiv(QwtPlot.xBottom)
            x = self.invTransform(QwtPlot.xBottom, ex)
            self.setAxisScale(QwtPlot.xBottom, ro*xs.lBound() + rn*x, ro*xs.hBound() + rn*x, self.axisStepSize(QwtPlot.xBottom))

            ys = self.axisScaleDiv(QwtPlot.yLeft)
            y = self.invTransform(QwtPlot.yLeft, ey)
            self.setAxisScale(QwtPlot.yLeft, ro*ys.lBound() + rn*y, ro*ys.hBound() + rn*y, self.axisStepSize(QwtPlot.yLeft))

        self.replot()


    # does a point (x,y) lie inside one of the selection rectangles (polygons)
    def isPointSelected(self, x,y):
        for curve in self.selectionCurveList:
            if curve.isInside(x,y): return 1
        return 0

    # return two lists of 0's and 1's whether each point in (xData, yData) is selected or not
    def getSelectedPoints(self, xData, yData, validData):
        import numpy
        total = numpy.zeros(len(xData))
        for curve in self.selectionCurveList:
            total += curve.getSelectedPoints(xData, yData, validData)
        unselected = numpy.equal(total, 0)
        selected = 1 - unselected
        return selected.tolist(), unselected.tolist()

    # save graph in matplotlib python file
    def saveToMatplotlib(self, fileName, size = QSize(400,400)):
        f = open(fileName, "wt")

        x1 = self.axisScaleDiv(QwtPlot.xBottom).lBound(); x2 = self.axisScaleDiv(QwtPlot.xBottom).hBound()
        y1 = self.axisScaleDiv(QwtPlot.yLeft).lBound();   y2 = self.axisScaleDiv(QwtPlot.yLeft).hBound()

        if self.showAxisScale == 0: edgeOffset = 0.01
        else: edgeOffset = 0.08

        f.write("from pylab import *\nfrom matplotlib import font_manager\n\n#possible changes in how the plot looks\n#rcParams['xtick.major.size'] = 0\n#rcParams['ytick.major.size'] = 0\n\n#constants\nx1 = %f; x2 = %f\ny1 = %f; y2 = %f\ndpi = 80\nxsize = %d\nysize = %d\nedgeOffset = %f\n\nfigure(facecolor = 'w', figsize = (xsize/float(dpi), ysize/float(dpi)), dpi = dpi)\nhold(True)\n" % (x1,x2,y1,y2,size.width(), size.height(), edgeOffset))

        linestyles = ["None", "-", "-.", "--", ":", "-", "-"]      # qwt line styles: NoCurve, Lines, Sticks, Steps, Dots, Spline, UserCurve
        markers = ["None", "o", "s", "^", "d", "v", "^", "<", ">", "x", "+"]    # curveSymbols = [None, Ellipse, Rect, Triangle, Diamond, DTriangle, UTriangle, LTriangle, RTriangle, XCross, Cross]

        f.write("#add curves\n")
        for c in self.itemList():
            if not isinstance(c, QwtPlotCurve): continue
            xData = [c.x(i) for i in range(c.dataSize())]
            yData = [c.y(i) for i in range(c.dataSize())]
            marker = markers[c.symbol().style()+1]

            markersize = c.symbol().size().width()
            markeredgecolor, foo = self._getColorFromObject(c.symbol().pen())
            markerfacecolor, alphaS = self._getColorFromObject(c.symbol().brush())
            colorP, alphaP = self._getColorFromObject(c.pen())
            colorB, alphaB = self._getColorFromObject(c.brush())
            alpha = min(alphaS, alphaP, alphaB)
            linewidth = c.pen().width()
            if c.__class__ == PolygonCurve and len(xData) == 4:
                x0 = min(xData); x1 = max(xData); diffX = x1-x0
                y0 = min(yData); y1 = max(yData); diffY = y1-y0
                f.write("gca().add_patch(Rectangle((%f, %f), %f, %f, edgecolor=%s, facecolor = %s, linewidth = %d, fill = 1, alpha = %.3f))\n" % (x0,y0,diffX, diffY, colorP, colorB, linewidth, alpha))
            elif c.style() < len(linestyles):
                linestyle = linestyles[c.style()]
                f.write("plot(%s, %s, marker = '%s', linestyle = '%s', markersize = %d, markeredgecolor = %s, markerfacecolor = %s, color = %s, linewidth = %d, alpha = %.3f)\n" % (xData, yData, marker, linestyle, markersize, markeredgecolor, markerfacecolor, colorP, linewidth, alpha))

        f.write("\n# add markers\n")
        for marker in self.itemList():
            if not isinstance(marker, QwtPlotMarker): continue
            x = marker.xValue()
            y = marker.yValue()
            text = str(marker.label().text())
            align = marker.labelAlignment()
            xalign = (align & Qt.AlignLeft and "right") or (align & Qt.AlignHCenter and "center") or (align & Qt.AlignRight and "left")
            yalign = (align & Qt.AlignBottom and "top") or (align & Qt.AlignTop and "bottom") or (align & Qt.AlignVCenter and "center")
            vertAlign = (yalign and ", verticalalignment = '%s'" % yalign) or ""
            horAlign = (xalign and ", horizontalalignment = '%s'" % xalign) or ""
            labelColor = marker.label().color()
            color = (labelColor.red()/255., labelColor.green()/255., labelColor.blue()/255.)
            alpha = labelColor.alpha()/255.
            name = str(marker.label().font().family())
            weight = marker.label().font().bold() and "bold" or "normal"
            if marker.__class__ == RotatedMarker: extra = ", rotation = %f" % (marker.rotation)
            else: extra = ""
            f.write("text(%f, %f, '%s'%s%s, color = %s, name = '%s', weight = '%s'%s, alpha = %.3f)\n" % (x, y, text, vertAlign, horAlign, color, name, weight, extra, alpha))

        # grid
        f.write("# enable grid\ngrid(%s)\n\n" % (self.gridCurve.xEnabled() and self.gridCurve.yEnabled() and "True" or "False"))

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

            f.write("#set axis labels\nxlabel('%s', weight = 'bold')\nylabel('%s', weight = 'bold')\n\n" % (str(self.axisTitle(QwtPlot.xBottom).text()), str(self.axisTitle(QwtPlot.yLeft).text())))
            f.write("\naxis([x1, x2, y1, y2])\ngca().set_position([edgeOffset, edgeOffset, 1 - 2*edgeOffset, 1 - 2*edgeOffset])\n#subplots_adjust(left = 0.08, bottom = 0.11, right = 0.98, top = 0.98)\n")

        f.write("\n# possible settings to change\n#axes().set_frame_on(0) #hide the frame\n#axis('off') #hide the axes and labels on them\n\n")


        if self.legend().itemCount() > 0:
            legendItems = []
            for widget in self.legend().legendItems():
                item = self.legend().find(widget)
                text = str(item.title().text()).replace("<b>", "").replace("</b>", "")
                if not item.symbol():
                    legendItems.append((text, None, None, None, None))
                else:
                    penC, penA = self._getColorFromObject(item.symbol().pen())
                    brushC, brushA = self._getColorFromObject(item.symbol().brush())
                    legendItems.append((text, markers[item.symbol().style()+1], penC, brushC, min(brushA, penA)))
            f.write("""
#functions to show legend below the figure
def drawSomeLegendItems(x, items, itemsPerAxis = 1, yDiff = 0.0):
    axes([x-0.1, .018*itemsPerAxis - yDiff, .2, .018], frameon = 0); axis('off')
    lines = [plot([],[], label = text, marker = marker, markeredgecolor = edgeC, markerfacecolor = faceC, alpha = alpha) for (text, marker, edgeC, faceC, alpha) in items]
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
        if isinstance(obj, QBrush) and obj.style() == Qt.NoBrush: return "'none'", 1
        if isinstance(obj, QPen)   and obj.style() == Qt.NoPen: return "'none'", 1
        col = [obj.color().red(), obj.color().green(), obj.color().blue()];
        col = tuple([v/float(255) for v in col])
        return col, obj.color().alpha()/float(255)


