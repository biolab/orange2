#
# owGraph.py
#
# the base for all graphs

from qt import *
from OWTools import *
from qwt import *
from OWGraphTools import *      # color palletes, user defined curves, ...
from OWDlgs import OWChooseImageSizeDlg
import qtcanvas, orange, math
from OWBaseWidget import unisetattr

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
        self.mainTitle = None
        self.showXaxisTitle = FALSE
        self.XaxisTitle = None
        self.showYLaxisTitle = FALSE
        self.YLaxisTitle = None
        self.showYRaxisTitle = FALSE
        self.YRaxisTitle = None


        self.state = ZOOMING
        self.tooltip = MyQToolTip(self)
        self.zoomKey = None
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
        self.optimizedDrawing = 1
        self.pointWidth = 5
        self.showFilledSymbols = 1
        self.showLegend = 1
        self.scaleFactor = 1.0              # used in some visualizations to "stretch" the data - see radviz, polviz
        self.setCanvasColor(QColor(Qt.white.name()))
        self.xpos = 0   # we have to initialize values, since we might get onMouseRelease event before onMousePress
        self.ypos = 0
        self.zoomStack = []
        self.colorNonTargetValue = QColor(200,200,200)
        self.colorTargetValue = QColor(0,0,255)
        self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Triangle, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]
        #self.curveSymbols = [QwtSymbol.Triangle, QwtSymbol.Ellipse, QwtSymbol.Rect, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle, QwtSymbol.XCross, QwtSymbol.Cross]

        # uncomment this if you want to use printer friendly symbols
        #self.curveSymbols = [QwtSymbol.Ellipse, QwtSymbol.XCross, QwtSymbol.Triangle, QwtSymbol.Cross, QwtSymbol.Diamond, QwtSymbol.DTriangle, QwtSymbol.Rect, QwtSymbol.UTriangle, QwtSymbol.LTriangle, QwtSymbol.RTriangle]


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
        
    def setYLlabels(self, labels):
        "Sets the Y-axis labels on the left."
        if not self.showAxisScale:
            self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
            self.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)
            self.axisScaleDraw(QwtPlot.yLeft).setOptions(0) 
            return

        self.axisScaleDraw(QwtPlot.yLeft).setTickLength(1, 1, 3)
        self.axisScaleDraw(QwtPlot.yLeft).setOptions(1)
        
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
        self.updateToolTips()

    def setYRlabels(self, labels):
        "Sets the Y-axis labels on the right."
        if not self.showAxisScale:
            self.setAxisScaleDraw(QwtPlot.yRight, HiddenScaleDraw())
            self.axisScaleDraw(QwtPlot.yRight).setTickLength(0, 0, 0)
            self.axisScaleDraw(QwtPlot.yRight).setOptions(0) 
            return

        self.axisScaleDraw(QwtPlot.yRight).setTickLength(1, 1, 3)
        self.axisScaleDraw(QwtPlot.yRight).setOptions(1)
        
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
        self.updateToolTips

    def setXlabels(self, labels):
        "Sets the x-axis labels if x-axis discrete."
        "Or leave up to QwtPlot (MaxMajor, MaxMinor) if x-axis continuous."
        if not self.showAxisScale:
            self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
            self.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)
            self.axisScaleDraw(QwtPlot.xBottom).setOptions(0) 
            return

        self.axisScaleDraw(QwtPlot.xBottom).setTickLength(1, 1, 3)
        self.axisScaleDraw(QwtPlot.xBottom).setOptions(1)
        
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
        self.updateToolTips()

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
        QwtPlot.paintEvent(self, qpe) #let the ancestor do its job
        self.replot()
 
    def setShowMainTitle(self, b):
        self.showMainTitle = b
        if (self.showMainTitle <> 0):
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(None)
        self.updateLayout()
        self.repaint()

    def setMainTitle(self, t):
        self.mainTitle = t
        if (self.showMainTitle <> 0):
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(None)
        self.updateLayout()
        self.repaint()

    def setShowXaxisTitle(self, b):
        self.showXaxisTitle = b
        if (self.showXaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.xBottom, None)
        self.updateLayout()
        self.repaint()

    def setXaxisTitle(self, title):
        self.XaxisTitle = title
        if (self.showXaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.xBottom, None)
        self.updateLayout()
        self.repaint()

    def setShowYLaxisTitle(self, b):
        self.showYLaxisTitle = b
        if (self.showYLaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yLeft, None)
        self.updateLayout()
        self.repaint()

    def setYLaxisTitle(self, title):
        self.YLaxisTitle = title
        if (self.showYLaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yLeft, None)
        self.updateLayout()
        self.repaint()

    def setShowYRaxisTitle(self, b):
        self.showYRaxisTitle = b
        if (self.showYRaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yRight, None)
        self.updateLayout()
        self.repaint()

    def setYRaxisTitle(self, title):
        self.YRaxisTitle = title
        if (self.showYRaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yRight, None)
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
        self.removeMarkers()
        self.tips.removeAll()


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


    def addCurve(self, name, brushColor, penColor, size, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, enableLegend = 0, xData = [], yData = [], forceFilledSymbols = 0, lineWidth = 1, pen = None):
        newCurveKey = self.insertCurve(name)
        if self.showFilledSymbols or forceFilledSymbols:
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

    def addMarker(self, name, x, y, alignment = -1, bold = 0):
        mkey = self.insertMarker(name)
        self.marker(mkey).setXValue(x)
        self.marker(mkey).setYValue(y)
        if alignment != -1:
            self.marker(mkey).setLabelAlignment(alignment)
        if bold:
            font = self.marker(mkey).font(); font.setBold(1); self.marker(mkey).setFont(font)
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

    def removeDrawingCurves(self):
        for key in self.curveKeys():
            curve = self.curve(key)
            if not isinstance(curve, SelectionCurve):
                self.removeCurve(key)

    def removeLastSelection(self):
        removed = 0
        if self.selectionCurveKeyList != []:
            lastCurve = self.selectionCurveKeyList.pop()
            self.removeCurve(lastCurve)
            self.tempSelectionCurve = None
            removed = 1
        self.replot()
        if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection
        return removed
        
    def removeAllSelections(self, send = 1):
        for key in self.selectionCurveKeyList: self.removeCurve(key)
        self.selectionCurveKeyList = []
        self.replot()
        if send and self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection

    def zoomOut(self):
        if len(self.zoomStack):
            (xmin, xmax, ymin, ymax) = self.zoomStack.pop()
            self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
            self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
            self.replot()
            return 1
        return 0


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
                if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection

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
        
        if self.statusBar != None:  self.statusBar.message(text)
        if text != "": self.showTip(self.transform(QwtPlot.xBottom, x), self.transform(QwtPlot.yLeft, y), text)

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

                if staticClick or (xmax-xmin)+(ymax-ymin) < 4: return

                xmin = self.invTransform(QwtPlot.xBottom, xmin);  xmax = self.invTransform(QwtPlot.xBottom, xmax)
                ymin = self.invTransform(QwtPlot.yLeft, ymin);    ymax = self.invTransform(QwtPlot.yLeft, ymax)
                
                self.blankClick = 0
                self.zoomStack.append((self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound(), self.axisScale(QwtPlot.yLeft).lBound(), self.axisScale(QwtPlot.yLeft).hBound()))
                self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
                self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
                self.replot()

            elif self.state == SELECT_RECTANGLE:
                if self.tempSelectionCurve:
                    self.tempSelectionCurve = None
                if self.autoSendSelectionCallback: self.autoSendSelectionCallback() # do we want to send new selection

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

    # save graph in matplotlib python file
    def saveToMatplotlib(self, fileName, size = QSize(400,400)):
        f = open(fileName, "wt")

        x1 = self.axisScale(QwtPlot.xBottom).lBound(); x2 = self.axisScale(QwtPlot.xBottom).hBound()
        y1 = self.axisScale(QwtPlot.yLeft).lBound();   y2 = self.axisScale(QwtPlot.yLeft).hBound()

        if not (self.axisScaleDraw(QwtPlot.xBottom).options() and self.axisScaleDraw(QwtPlot.yLeft).options()): edgeOffset = 0.01
        else: edgeOffset = 0.08

        f.write("from pylab import *\n\n#constants\nx1 = %f; x2 = %f\ny1 = %f; y2 = %f\ndpi = 80\nxsize = %d\nysize = %d\nedgeOffset = %f\n\nfigure(facecolor = 'w', figsize = (xsize/float(dpi), ysize/float(dpi)), dpi = dpi)\nhold(True)\n" % (x1,x2,y1,y2,size.width(), size.height(), edgeOffset))

        # qwt line styles: NoCurve, Lines, Sticks, Steps, Dots, Spline, UserCurve
        linestyles = ["o", "-", "-.", "--", ":", "-", "-"]

        # curveSymbols = [None, Ellipse, Rect, Triangle, Diamond, DTriangle, UTriangle, LTriangle, RTriangle, XCross, Cross]
        markers = ["None", "o", "s", "^", "d", "v", "^", "<", ">", "x", "+"]
    
        f.write("#add curves\n")
        for key in self.curveKeys():
            c = self.curve(key)
            if c.style() >= len(linestyles): continue   # a user curve case
            
            xData = [c.x(i) for i in range(c.dataSize())]
            yData = [c.y(i) for i in range(c.dataSize())]
            marker = markers[c.symbol().style()]
            linestyle = linestyles[c.style()]
            markersize = c.symbol().size().width()
            markeredgecolor = self._getColorFromObject(c.symbol().pen()) 
            markerfacecolor = self._getColorFromObject(c.symbol().brush())
            color = self._getColorFromObject(c.pen())
            linewidth = c.pen().width()
            #markeredgewidth
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
        if not (self.axisScaleDraw(QwtPlot.xBottom).options() and self.axisScaleDraw(QwtPlot.yLeft).options()):
            f.write("#hide axis\naxis('off')\naxis([x1, x2, y1, y2])\ngca().set_position([edgeOffset, edgeOffset, 1 - 2*edgeOffset, 1 - 2*edgeOffset])\n")
        else:
            if self.axisScaleDraw(QwtPlot.yLeft).__class__ == DiscreteAxisScaleDraw:
                labels = self.axisScaleDraw(QwtPlot.yLeft).labels
                f.write("pos, labels = yticks(%s, %s)\nfor l in labels: l.set_rotation('vertical')\n" % (range(len(labels)), labels))
            if self.axisScaleDraw(QwtPlot.xBottom).__class__ == DiscreteAxisScaleDraw:
                labels = self.axisScaleDraw(QwtPlot.xBottom).labels
                f.write("xticks(%s, %s)\n" % (range(len(labels)), labels))

            f.write("#set axis labels\nxlabel('%s', weight = 'bold')\nylabel('%s', weight = 'bold')\n\n" % (str(self.axisTitle(QwtPlot.xBottom)), str(self.axisTitle(QwtPlot.yLeft))))
            f.write("\naxis([x1, x2, y1, y2])\ngca().set_position([edgeOffset, edgeOffset, 1 - 2*edgeOffset, 1 - 2*edgeOffset])\n")
        
        f.write("show()")
        f.close()

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
        
# ###########################################################
# a class that is able to draw arbitrary polygon curves.
# data points are specified by a standard call to graph.setCurveData(key, xArray, yArray)
# brush and pen can also be set by calls to setPen and setBrush functions
class PolygonCurve(QwtPlotCurve):
    def __init__(self, parent, pen = QPen(Qt.black), brush = QBrush(Qt.white)):
        QwtPlotCurve.__init__(self, parent)
        self.setPen(pen)
        self.setBrush(brush)
        self.Pen = pen
        self.Brush = brush
    """
    def setPen(self, pen):
        self.Pen = pen

    def setBrush(self, brush):
        self.Brush = brush
    """
    # Draws rectangles with the corners taken from the x- and y-arrays.        
    def draw(self, painter, xMap, yMap, start, stop):
        #painter.setPen(self.Pen)
        #painter.setBrush(self.Brush)
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        if stop == -1: stop = self.dataSize()
        start = max(start, 0)
        stop = max(stop, 0)
        array = QPointArray(stop-start)
        for i in range(start, stop):
            array.setPoint(i-start, xMap.transform(self.x(i)), yMap.transform(self.y(i)))

        if stop-start > 2:
            painter.drawPolygon(array)


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
            
