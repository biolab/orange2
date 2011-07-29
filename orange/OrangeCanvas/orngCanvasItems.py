# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    two main objects that are shown in the canvas; Line and Widget
#
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os, sys, math
ERROR = 0
WARNING = 1

def _graphicsEffect(item):
    if hasattr(item, "graphicsEffect"):
        return item.graphicsEffect()
    else:
        return None
    
class TempCanvasLine(QGraphicsPathItem):
    def __init__(self, canvasDlg, canvas):
        QGraphicsLineItem.__init__(self, None, canvas)
        self.setZValue(-10)
        self.canvasDlg = canvasDlg
        self.startWidget = None
        self.endWidget = None
        self.widget = None

        self.setPen(QPen(QColor(180, 180, 180), 3, Qt.SolidLine))
        
        if qVersion() >= "4.6" and canvasDlg.settings["enableCanvasDropShadows"]:
            effect = QGraphicsDropShadowEffect(self.scene())
            effect.setOffset(QPointF(0.0, 0.0))
            effect.setBlurRadius(7)
            self.setGraphicsEffect(effect)        
        
    def setStartWidget(self, widget):
        self.startWidget = widget
        pos = widget.getRightEdgePoint()
        endX, endY = startX, startY = pos.x(), pos.y()
#        self.setLine(pos.x(), pos.y(), pos.x(), pos.y())
        
    def setEndWidget(self, widget):
        self.endWidget = widget
        pos = widget.getLeftEdgePoint()
        endX, endY = startX, startY = pos.x(), pos.y()
#        self.setLine(pos.x(), pos.y(), pos.x(), pos.y())
        
    def updateLinePos(self, newPos):
        if self.startWidget == None and self.endWidget == None:
            return
        
        if self.startWidget != None:
            func = "getDistToLeftEdgePoint"
        else:
            func = "getDistToRightEdgePoint"
        
        schema = self.canvasDlg.schema
        view = schema.canvasView
        
        self.widget = None
        widgets = view.getItemsAtPos(newPos, CanvasWidget)
        if widgets:
            self.widget = widgets[0]
        else:
            dists = [(getattr(w, func)(newPos), w) for w in schema.widgets]
            dists.sort()
            if dists and dists[0][0] < 20:
                self.widget = dists[0][1]
        
        if self.startWidget:
            pos = self.startWidget.getRightEdgePoint()
        else:
            pos = self.endWidget.getLeftEdgePoint()

        if self.widget not in [self.startWidget, self.endWidget]: 
            if self.startWidget == None and self.widget.widgetInfo.outputs:
                newPos = self.widget.getRightEdgePoint()
            elif self.endWidget == None and self.widget.widgetInfo.inputs:
                newPos = self.widget.getLeftEdgePoint()
        
        path = QPainterPath(pos)
        if self.startWidget != None:
            path.cubicTo(pos.x()+60, pos.y(), newPos.x()-60, newPos.y(), newPos.x(),newPos.y())
        else:
            path.cubicTo(pos.x()-60, pos.y(), newPos.x()+60, newPos.y(), newPos.x(),newPos.y())
        self.setPath(path)
#        self.setLine(pos.x(), pos.y(), newPos.x(), newPos.y())
        
    def remove(self):
        self.hide()
        self.startWidget = None
        self.endWidget = None 
        self.scene().removeItem(self)

# #######################################
# # CANVAS LINE
# #######################################
        
class CanvasLine(QGraphicsPathItem):
    def __init__(self, signalManager, canvasDlg, view, outWidget, inWidget, scene, *args):
        self.outWidget = outWidget
        self.inWidget = inWidget
        
        QGraphicsPathItem.__init__(self, None, None)
        self.signalManager = signalManager
        self.canvasDlg = canvasDlg
        self.view = view
        self.setZValue(-10)
        
        self.caption = ""
        self.captionItem = QGraphicsTextItem(self)
        self.captionItem.setDefaultTextColor(QColor(80, 80, 192))
        self.captionItem.setHtml("<center>%s</center>" % self.caption)
        self.captionItem.setAcceptHoverEvents(False)
        self.captionItem.setVisible(bool(self.canvasDlg.settings["showSignalNames"]))
        self.captionItem.setAcceptedMouseButtons(Qt.NoButton)
        
        self.updateTooltip()
        
        # this might seem unnecessary, but the pen size 20 is used for collision detection, when we want to see whether to to show the line menu or not 
        self.setPen(QPen(QColor(200, 200, 200), 20, Qt.SolidLine))
        self.setAcceptHoverEvents(True)
        self.hoverState = False
        if qVersion() >= "4.6" and canvasDlg.settings["enableCanvasDropShadows"]:
            effect = QGraphicsDropShadowEffect(self.scene())
            effect.setOffset(QPointF(0.0, 0.0))
            effect.setBlurRadius(7)
            self.setGraphicsEffect(effect)
        if scene is not None:
            scene.addItem(self)
            
        self._dyn_enabled = False
            
        QObject.connect(self.outWidget.instance, SIGNAL("dynamicLinkEnabledChanged(PyQt_PyObject, bool)"), self.updateDynamicEnableState)

    def remove(self):
        self.hide()
        self.setToolTip("")
        self.outWidget = None
        self.inWidget = None
        self.scene().removeItem(self)
        
    def getEnabled(self):
        signals = self.signalManager.findSignals(self.outWidget.instance, self.inWidget.instance)
        if not signals: return 0
        return int(self.signalManager.isSignalEnabled(self.outWidget.instance, self.inWidget.instance, signals[0][0], signals[0][1]))

    def getSignals(self):
        signals = []
        for link in self.signalManager.getLinks(self.outWidget.instance, self.inWidget.instance):
            signals.append((link.signalNameFrom, link.signalNameTo))
#        for (inWidgetInstance, outName, inName, X) in self.signalManager.links.get(self.outWidget.instance, []):
#            if inWidgetInstance == self.inWidget.instance:
#                signals.append((outName, inName))
        return signals
    
    def isDynamic(self):
        links = self.signalManager.getLinks(self.outWidget.instance, self.inWidget.instance)
        return any(link.dynamic for link in links)
    
    def updateDynamicEnableState(self, link, enabled):
        """ Call when dynamic signal state changes
        """
        links = self.signalManager.getLinks(self.outWidget.instance, self.inWidget.instance)
        if link not in links:
            return
        
        if self.isDynamic():
            if enabled:
                self._dyn_enabled = True
            else:
                self._dyn_enabled = False
        self.update()
        
    
    def updatePainterPath(self):
        p1 = self.outWidget.getRightEdgePoint()
        p2 = self.inWidget.getLeftEdgePoint()
        path = QPainterPath(p1)
        path.cubicTo(p1.x() + 60, p1.y(), p2.x() - 60, p2.y(), p2.x(), p2.y())
        self.setPath(path)
        metrics = QFontMetrics(self.captionItem.font())
        oddLineOffset = -metrics.lineSpacing() / 2 * (len(self.caption.strip().splitlines()) % 2)
        mid = self.path().pointAtPercent(0.5)
        rect = self.captionItem.boundingRect()
        self.captionItem.setPos(mid + QPointF(-rect.width() / 2.0, -rect.height() / 2.0 + oddLineOffset))
        self.update()
        
    def shape(self):
        stroke = QPainterPathStroker()
        stroke.setWidth(6)
        return stroke.createStroke(self.path())
    
    def boundingRect(self):
        rect = QGraphicsPathItem.boundingRect(self)
        if _graphicsEffect(self):
            textRect = self.captionItem.boundingRect() ## Should work without this but for some reason if using graphics effects the text gets clipped
            textRect.moveTo(self.captionItem.pos())
            return rect.united(textRect)
        else:
            return rect

    def paint(self, painter, option, widget = None):
        if self.isDynamic():
            color2 = QColor(Qt.blue if self._dyn_enabled else Qt.red)
            color1 = color2.lighter(150)
        else:
            color1 = QColor(200, 200, 200)
            color2 = QColor(160, 160, 160)
        painter.setPen(QPen(color1, 6 if self.hoverState == True else 4 , self.getEnabled() and Qt.SolidLine or Qt.DashLine, Qt.RoundCap))
        painter.drawPath(self.path())
        painter.setPen(QPen(color2, 2 , self.getEnabled() and Qt.SolidLine or Qt.DashLine, Qt.RoundCap))
        painter.drawPath(self.path())

    def updateTooltip(self):
        if self.inWidget and self.outWidget:
            status = self.getEnabled() == 0 and " (Disabled)" or ""
            string = "<nobr><b>" + self.outWidget.caption + "</b> --> <b>" + self.inWidget.caption + "</b>" + status + "</nobr><hr>Signals:<br>"
            for (outSignal, inSignal) in self.getSignals():
                string += "<nobr> &nbsp; &nbsp; - " + outSignal + " --> " + inSignal + "</nobr><br>"
            string = string[:-4]
            self.setToolTip(string)
    
            # print the text with the signals
            self.caption = "\n".join([outSignal for (outSignal, inSignal) in self.getSignals()])
            self.captionItem.setHtml("<center>%s</center>" % self.caption.replace("\n", "<br/>"))
            self.updatePainterPath()

    def hoverEnterEvent(self, event):
        self.hoverState = True
        self.update()
    
    def hoverLeaveEvent(self, event):
        self.hoverState = False
        self.update()
        

# #######################################
# # CANVAS WIDGET
# #######################################
class CanvasWidget(QGraphicsRectItem):
    def __init__(self, signalManager, scene, view, widgetInfo, defaultPic, canvasDlg, widgetSettings = {}):
        # import widget class and create a class instance
        m = __import__(widgetInfo.fileName)
        self.instance = m.__dict__[widgetInfo.fileName].__new__(m.__dict__[widgetInfo.fileName], _owInfo=canvasDlg.settings["owInfo"],
                                                                _owWarning = canvasDlg.settings["owWarning"], _owError=canvasDlg.settings["owError"],
                                                                _owShowStatus = canvasDlg.settings["owShow"], _useContexts = canvasDlg.settings["useContexts"],
                                                                _category = widgetInfo.category, _settingsFromSchema = widgetSettings)
        self.instance.__init__(signalManager=signalManager)
        self.instance.__dict__["widgetInfo"] = widgetInfo
        self.isProcessing = 0   # is this widget currently processing signals
        self.progressBarShown = 0
        self.progressBarValue = -1
        self.widgetSize = QSizeF(0, 0)
        self.widgetState = {}
        self.caption = widgetInfo.name
        self.selected = False
        self.inLines = []               # list of connected lines on input
        self.outLines = []              # list of connected lines on output
        self.icon = canvasDlg.getWidgetIcon(widgetInfo)
        
        self.instance.setProgressBarHandler(view.progressBarHandler)   # set progress bar event handler
        self.instance.setProcessingHandler(view.processingHandler)
        self.instance.setWidgetStateHandler(self.updateWidgetState)
        self.instance.setEventHandler(canvasDlg.output.widgetEvents)
        self.instance.setWidgetIcon(canvasDlg.getFullWidgetIconName(widgetInfo))
        #self.instance.updateStatusBarState()

        QGraphicsRectItem.__init__(self, None, None)
        self.signalManager = signalManager
        self.widgetInfo = widgetInfo
        self.view = view
        self.canvasDlg = canvasDlg
        canvasPicsDir  = os.path.join(canvasDlg.canvasDir, "icons")
        self.imageLeftEdge = QPixmap(os.path.join(canvasPicsDir,"leftEdge.png"))
        self.imageRightEdge = QPixmap(os.path.join(canvasPicsDir,"rightEdge.png"))
        self.imageLeftEdgeG = QPixmap(os.path.join(canvasPicsDir,"leftEdgeG.png"))
        self.imageRightEdgeG = QPixmap(os.path.join(canvasPicsDir,"rightEdgeG.png"))
        self.imageLeftEdgeR = QPixmap(os.path.join(canvasPicsDir,"leftEdgeR.png"))
        self.imageRightEdgeR = QPixmap(os.path.join(canvasPicsDir,"rightEdgeR.png"))
        self.shownLeftEdge, self.shownRightEdge = self.imageLeftEdge, self.imageRightEdge
        self.imageFrame = QIcon(QPixmap(os.path.join(canvasPicsDir, "frame.png")))
        self.edgeSize = QSizeF(self.imageLeftEdge.size())
        self.resetWidgetSize()
        
        self.oldPos = self.pos()
        
        self.infoIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Info"], self)
        self.warningIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Warning"], self)
        self.errorIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Error"], self)
        self.infoIcon.hide()
        self.warningIcon.hide()
        self.errorIcon.hide()
        
        self.captionItem = QGraphicsTextItem(self)
        self.captionItem.setHtml("<center>%s</center>" % self.caption)
        self.captionItem.document().setTextWidth(min(self.captionItem.document().idealWidth(), 200))
        
        self.captionItem.setPos(-self.captionItem.boundingRect().width()/2.0 + self.widgetSize.width() / 2.0, self.widgetSize.height() + 2)
        self.captionItem.setAcceptHoverEvents(False)
        
        # do we want to restore last position and size of the widget
        if self.canvasDlg.settings["saveWidgetsPosition"]:
            self.instance.restoreWidgetPosition()
            
        self.setAcceptHoverEvents(True)
        self.hoverState = False
        self.setFlags(QGraphicsItem.ItemIsSelectable)# | QGraphicsItem.ItemIsMovable)
        
        if qVersion() >= "4.6" and self.canvasDlg.settings["enableCanvasDropShadows"]:
            effect = QGraphicsDropShadowEffect()
            effect.setOffset(QPointF(1.1, 3.1))
            effect.setBlurRadius(7)
            self.setGraphicsEffect(effect)
            
        if scene is not None:
            scene.addItem(self)

    def resetWidgetSize(self):
        size = self.canvasDlg.schemeIconSizeList[self.canvasDlg.settings['schemeIconSize']]
        self.setRect(0,0, size, size)
        self.widgetSize = QSizeF(size, size)
        self.update()

    # get the list of connected signal names
    def getInConnectedSignalNames(self):
        signals = []
        for line in self.inLines:
            for (outSignal, inSignal) in line.getSignals():
                if inSignal not in signals: signals.append(inSignal)
        return signals

    # get the list of connected signal names
    def getOutConnectedSignalNames(self):
        signals = []
        for line in self.outLines:
            for (outSignal, inSignal) in line.getSignals():
                if outSignal not in signals: signals.append(outSignal)
        return signals

    def remove(self):
        self.hide()
        self.errorIcon.hide()
        self.warningIcon.hide()
        self.infoIcon.hide()

        # save settings
        if (self.instance != None):
            if self.canvasDlg.menuSaveSettings == 1:        # save settings only if checked in the main menu
                try:
                    self.instance.saveSettings()
                except:
                    print "Unable to save settings for %s widget" % (self.instance.captionTitle)
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that it doesn't crash canvas
            self.instance.close()
            self.instance.linksOut.clear()      # this helps python to more quickly delete the unused objects
            self.instance.linksIn.clear()
            self.instance.setProgressBarHandler(None)   # set progress bar event handler
            self.instance.setProcessingHandler(None)
            self.instance.setWidgetStateHandler(None)
            self.instance.setEventHandler(None)
            self.instance.onDeleteWidget()      # this is a cleanup function that can take care of deleting some unused objects
            try:
                import sip
                sip.delete(self.instance)
            except Exception, ex:
                print >> sys.stderr, "Error deleting the widget: \n%s" % str(ex)
            self.instance = None
            
            self.scene().removeItem(self)
                

    def savePosition(self):
        self.oldPos = self.pos()

    def restorePosition(self):
        self.setPos(self.oldPos)

    def updateText(self, text):
        self.caption = str(text)
        self.prepareGeometryChange()
        self.captionItem.setHtml("<center>%s</center>" % self.caption)
        self.captionItem.document().adjustSize()
        self.captionItem.document().setTextWidth(min(self.captionItem.document().idealWidth(), 200))
        
        self.captionItem.setPos(-self.captionItem.boundingRect().width()/2.0 + self.widgetSize.width() / 2.0, self.widgetSize.height() + 2)
        self.updateTooltip()
        self.update()

    def updateWidgetState(self):
        widgetState = self.instance.widgetState

        self.infoIcon.hide()
        self.warningIcon.hide()
        self.errorIcon.hide()

        yPos = - 21 - self.progressBarShown * 20
        iconNum = sum([widgetState.get("Info", {}).values() != [],  widgetState.get("Warning", {}).values() != [], widgetState.get("Error", {}).values() != []])

        if self.canvasDlg.settings["ocShow"]:        # if show icons is enabled in canvas options dialog
            startX = (self.rect().width()/2) - ((iconNum*(self.canvasDlg.widgetIcons["Info"].width()+2))/2)
            off  = 0
            if len(widgetState.get("Info", {}).values()) > 0 and self.canvasDlg.settings["ocInfo"]:
                off  = self.updateWidgetStateIcon(self.infoIcon, startX, yPos, widgetState["Info"])
            if len(widgetState.get("Warning", {}).values()) > 0 and self.canvasDlg.settings["ocWarning"]:
                off += self.updateWidgetStateIcon(self.warningIcon, startX+off, yPos, widgetState["Warning"])
            if len(widgetState.get("Error", {}).values()) > 0 and self.canvasDlg.settings["ocError"]:
                off += self.updateWidgetStateIcon(self.errorIcon, startX+off, yPos, widgetState["Error"])


    def updateWidgetStateIcon(self, icon, x, y, stateDict):
        icon.setPos(x,y)
        icon.show()
        icon.setToolTip(reduce(lambda x,y: x+'<br>'+y, stateDict.values()))
        return icon.pixmap().width() + 3

    # set coordinates of the widget
    def setCoords(self, x, y):
        if self.canvasDlg.settings["snapToGrid"]:
            x = round(x/10)*10
            y = round(y/10)*10
        self.setPos(x, y)
        self.updateWidgetState()
        
    def setPos(self, *args):
        QGraphicsRectItem.setPos(self, *args)
        for line in self.inLines + self.outLines:
            line.updatePainterPath()
            

    # we have to increase the default bounding rect so that we also repaint the name of the widget and input/output boxes
    def boundingRect(self):
        rect = QRectF(QPointF(0, 0), self.widgetSize).adjusted(-11, -6, 11, 6)#.adjusted(-100, -100, 100, 100) #(-10-width, -4, +10+width, +25)
        rect.setTop(rect.top() - 20 - 21) ## Room for progress bar and warning, error, info icons
        if _graphicsEffect(self):
            textRect = self.captionItem.boundingRect() ## Should work without this but for some reason if using graphics effects the text gets clipped
            textRect.moveTo(self.captionItem.pos())
            return rect.united(textRect)
        else:
            return rect 

    # is mouse position inside the left signal channel
    def mouseInsideLeftChannel(self, pos):
        if self.widgetInfo.inputs == []: return False

        boxRect = QRectF(self.x()-self.edgeSize.width(), self.y() + (self.widgetSize.height()-self.edgeSize.height())/2, self.edgeSize.width(), self.edgeSize.height())
        boxRect.adjust(-10,-10,5,10)       # enlarge the rectangle
        if isinstance(pos, QPointF) and boxRect.contains(pos): return True
        elif isinstance(pos, QRectF) and boxRect.intersects(pos): return True
        else: return False

    # is mouse position inside the right signal channel
    def mouseInsideRightChannel(self, pos):
        if self.widgetInfo.outputs == []: return False

        boxRect = QRectF(self.x()+self.widgetSize.width(), self.y() + (self.widgetSize.height()-self.edgeSize.height())/2, self.edgeSize.width(), self.edgeSize.height())
        boxRect.adjust(-5,-10,10,10)       # enlarge the rectangle
        if isinstance(pos, QPointF) and boxRect.contains(pos): return True
        elif isinstance(pos, QRectF) and boxRect.intersects(pos): return True
        else: return False
        
    def canConnect(self, outWidget, inWidget, dynamic=False):
        if outWidget == inWidget:
            return
        
        canConnect = self.signalManager.canConnect(outWidget.instance, inWidget.instance, dynamic=dynamic)
        
#        outputs = [outWidget.instance.getOutputType(output.name) for output in outWidget.widgetInfo.outputs]
#        inputs = [inWidget.instance.getInputType(input.name) for input in inWidget.widgetInfo.inputs]
#        canConnect = 0
#        for outtype in outputs:
#            if any(issubclass(outtype, intype) for intype in inputs):
#                canConnect = 1
#                break
#            elif dynamic and any(issubclass(intype, outtype) for intype in inputs):
#                canConnect = 2
#                break

        if outWidget == self:
            self.shownRightEdge = canConnect and self.imageRightEdgeG or self.imageRightEdgeR
        else:
            self.shownLeftEdge = canConnect and self.imageLeftEdgeG or self.imageLeftEdgeR

    def resetLeftRightEdges(self):
        self.shownLeftEdge = self.imageLeftEdge
        self.shownRightEdge = self.imageRightEdge
    
    
    # we know that the mouse was pressed inside a channel box. We only need to find
    # inside which one it was
    def getEdgePoint(self, pos):
        if self.mouseInsideLeftChannel(pos):
            return self.getLeftEdgePoint()
        elif self.mouseInsideRightChannel(pos):
            return self.getRightEdgePoint()

    def getLeftEdgePoint(self):
        return QPointF(self.x()+2- self.edgeSize.width(), self.y() + self.widgetSize.height()/2)

    def getRightEdgePoint(self):
        return QPointF(self.x()-2+ self.widgetSize.width() + self.edgeSize.width(), self.y() + self.widgetSize.height()/2)

    def getDistToLeftEdgePoint(self, point):
        p = self.getLeftEdgePoint()
        diff = point-p
        return math.sqrt(diff.x()**2 + diff.y()**2)
    
    def getDistToRightEdgePoint(self, point):
        p = self.getRightEdgePoint()
        diff = point-p
        return math.sqrt(diff.x()**2 + diff.y()**2)


    # draw the widget
    def paint(self, painter, option, widget = None):
        if self.isProcessing or self.isSelected() or getattr(self, "invalidPosition", False):
            painter.setPen(QPen(QBrush(QColor(125, 162, 206, 192)), 1, Qt.SolidLine, Qt.RoundCap))
            painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
            painter.drawRoundedRect(-10, -5, self.widgetSize.width()+20, self.widgetSize.height()+10, 5, 5)

        if self.widgetInfo.inputs != []:
            painter.drawPixmap(-self.edgeSize.width()+1, (self.widgetSize.height()-self.edgeSize.height())/2, self.shownLeftEdge)
        if self.widgetInfo.outputs != []:
            painter.drawPixmap(self.widgetSize.width()-2, (self.widgetSize.height()-self.edgeSize.height())/2, self.shownRightEdge)
            
        if self.hoverState:
            color = QColor(125, 162, 206)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(-2, -2, self.widgetSize.width() + 4, self.widgetSize.height() + 4, 5, 5)
            
        painter.drawPixmap(0,0, self.icon.pixmap(self.widgetSize.width(), self.widgetSize.height()))
        
        yPos = -22
        if self.progressBarValue >= 0 and self.progressBarValue <= 100:
            rect = QRectF(0, yPos, self.widgetSize.width(), 16)
            painter.setPen(QPen(QColor(0,0,0)))
            painter.setBrush(QBrush(QColor(255,255,255)))
            painter.drawRect(rect)

            painter.setBrush(QBrush(QColor(0,128,255)))
            painter.drawRect(QRectF(0, yPos, self.widgetSize.width()*self.progressBarValue/100., 16))
            painter.drawText(rect, Qt.AlignCenter, "%d %%" % (self.progressBarValue))


    def addOutLine(self, line):
        self.outLines.append(line)

    def addInLine(self,line):
        self.inLines.append(line)

    def removeLine(self, line):
        if line in self.inLines:
            self.inLines.remove(line)
        elif line in self.outLines:
            self.outLines.remove(line)
        else:
            print "Orange Canvas: Error. Unable to remove line"

        self.updateTooltip()


    def setAllLinesFinished(self, finished):
        for line in self.inLines: line.finished = finished
        for line in self.outLines: line.finished = finished


    def updateTooltip(self):
        string = "<nobr><b>" + self.caption + "</b></nobr><hr>Inputs:<br>"

        if self.widgetInfo.inputs == []: string += "&nbsp; &nbsp; None<br>"
        else:
            for signal in self.widgetInfo.inputs:
                widgets = self.signalManager.getLinkWidgetsIn(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp; &nbsp; - <b>" + signal.name + "</b> (from "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp; &nbsp; - " + signal.name + "</nobr><br>"

        string = string[:-4]
        string += "<hr>Outputs:<br>"
        if self.widgetInfo.outputs == []: string += "&nbsp; &nbsp; None<br>"
        else:
            for signal in self.widgetInfo.outputs:
                widgets = self.signalManager.getLinkWidgetsOut(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp; &nbsp; - <b>" + signal.name + "</b> (to "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp; &nbsp; - " + signal.name + "</nobr><br>"
        string = string[:-4]
        self.setToolTip(string)

    def setProgressBarValue(self, value):
        self.progressBarValue = value
        if value < 0 or value > 100:
            self.updateWidgetState()
        self.update()

    def setProcessing(self, value):
        self.isProcessing = value
        self.update()

    def hoverEnterEvent(self, event):
        self.hoverState = True
        self.update()
        return QGraphicsRectItem.hoverEnterEvent(self, event)
        
    def hoverLeaveEvent(self, event):
        self.hoverState = False
        self.update()
        return QGraphicsRectItem.hoverLeaveEvent(self, event)        

class MyCanvasText(QGraphicsSimpleTextItem):
    def __init__(self, canvas, text, x, y, flags=Qt.AlignLeft, bold=0, show=1):
        QGraphicsSimpleTextItem.__init__(self, text, None, canvas)
        self.setPos(x,y)
        self.setPen(QPen(Qt.black))
        self.flags = flags
        if bold:
            font = self.font();
            font.setBold(1);
            self.setFont(font)
        if show:
            self.show()

    def paint(self, painter, option, widget = None):
        #painter.resetMatrix()
        painter.setPen(self.pen())
        painter.setFont(self.font())

        xOff = 0; yOff = 0
        rect = painter.boundingRect(QRectF(0,0,2000,2000), self.flags, self.text())
        if self.flags & Qt.AlignHCenter: xOff = rect.width()/2.
        elif self.flags & Qt.AlignRight: xOff = rect.width()
        if self.flags & Qt.AlignVCenter: yOff = rect.height()/2.
        elif self.flags & Qt.AlignBottom:yOff = rect.height()
        #painter.drawText(self.pos().x()-xOff, self.pos().y()-yOff, rect.width(), rect.height(), self.flags, self.text())
        painter.drawText(-xOff, -yOff, rect.width(), rect.height(), self.flags, self.text())
        