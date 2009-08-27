# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    signal dialog, canvas options dialog

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from orngCanvasItems import MyCanvasText
import OWGUI, sys, os

# this class is needed by signalDialog to show widgets and lines
class SignalCanvasView(QGraphicsView):
    def __init__(self, dlg, canvasDlg, *args):
        apply(QGraphicsView.__init__,(self,) + args)
        self.dlg = dlg
        self.canvasDlg = canvasDlg
        self.bMouseDown = False
        self.tempLine = None
        self.inWidget = None
        self.outWidget = None
        self.inWidgetIcon = None
        self.outWidgetIcon = None
        self.lines = []
        self.outBoxes = []
        self.inBoxes = []
        self.texts = []
        self.ensureVisible(0,0,1,1)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setRenderHint(QPainter.Antialiasing)

    def addSignalList(self, outWidget, inWidget):
        self.scene().clear()
        outputs, inputs = outWidget.widgetInfo.outputs, inWidget.widgetInfo.inputs
        outIcon, inIcon = self.canvasDlg.getWidgetIcon(outWidget.widgetInfo), self.canvasDlg.getWidgetIcon(inWidget.widgetInfo)
        self.lines = []
        self.outBoxes = []
        self.inBoxes = []
        self.texts = []
        xSpaceBetweenWidgets = 100  # space between widgets
        xWidgetOff = 10             # offset for widget position
        yWidgetOffTop = 10          # offset for widget position
        yWidgetOffBottom = 30       # offset for widget position
        ySignalOff = 10             # space between the top of the widget and first signal
        ySignalSpace = 50           # space between two neighbouring signals
        ySignalSize = 20            # height of the signal box
        xSignalSize = 20            # width of the signal box
        xIconOff = 10
        iconSize = 48

        count = max(len(inputs), len(outputs))
        height = max ((count)*ySignalSpace, 70)

        # calculate needed sizes of boxes to show text
        maxLeft = 0
        for i in range(len(inputs)):
            maxLeft = max(maxLeft, self.getTextWidth("("+inputs[i].name+")", 1))
            maxLeft = max(maxLeft, self.getTextWidth(inputs[i].type))

        maxRight = 0
        for i in range(len(outputs)):
            maxRight = max(maxRight, self.getTextWidth("("+outputs[i].name+")", 1))
            maxRight = max(maxRight, self.getTextWidth(outputs[i].type))

        width = max(maxLeft, maxRight) + 70 # we add 70 to show icons beside signal names

        # show boxes
        pen = QPen(QBrush(QColor(125, 162, 206, 192)), 1, Qt.SolidLine, Qt.RoundCap)
        brush = QBrush(QColor(217, 232, 252, 192))
        self.outWidget = QGraphicsRectItem(xWidgetOff, yWidgetOffTop, width, height, None, self.dlg.canvas)
        self.outWidget.setBrush(brush)
        self.outWidget.setPen(pen)
        self.outWidget.setZValue(-100)

        self.inWidget = QGraphicsRectItem(xWidgetOff + width + xSpaceBetweenWidgets, yWidgetOffTop, width, height, None, self.dlg.canvas)
        self.inWidget.setBrush(brush)
        self.inWidget.setPen(pen)
        self.inWidget.setZValue(-100)
        
        canvasPicsDir  = os.path.join(self.canvasDlg.canvasDir, "icons")
        if os.path.exists(os.path.join(canvasPicsDir, "frame.png")):
            widgetBack = QPixmap(os.path.join(canvasPicsDir, "frame.png"))
        else:
            widgetBack = outWidget.imageFrame

        # if icons -> show them
        if outIcon:
#            frame = QGraphicsPixmapItem(widgetBack, None, self.dlg.canvas)
#            frame.setPos(xWidgetOff + xIconOff, yWidgetOffTop + height/2.0 - frame.pixmap().width()/2.0)
            self.outWidgetIcon = QGraphicsPixmapItem(outIcon.pixmap(iconSize, iconSize), None, self.dlg.canvas)
#            self.outWidgetIcon.setPos(xWidgetOff + xIconOff, yWidgetOffTop + height/2.0 - self.outWidgetIcon.pixmap().width()/2.0)
            self.outWidgetIcon.setPos(xWidgetOff + xIconOff, yWidgetOffTop + xIconOff)
        
        if inIcon:
#            frame = QGraphicsPixmapItem(widgetBack, None, self.dlg.canvas)
#            frame.setPos(xWidgetOff + xSpaceBetweenWidgets + 2*width - xIconOff - frame.pixmap().width(), yWidgetOffTop + height/2.0 - frame.pixmap().width()/2.0)
            self.inWidgetIcon = QGraphicsPixmapItem(inIcon.pixmap(iconSize, iconSize), None, self.dlg.canvas)
#            self.inWidgetIcon.setPos(xWidgetOff + xSpaceBetweenWidgets + 2*width - xIconOff - self.inWidgetIcon.pixmap().width(), yWidgetOffTop + height/2.0 - self.inWidgetIcon.pixmap().width()/2.0)
            self.inWidgetIcon.setPos(xWidgetOff + xSpaceBetweenWidgets + 2*width - xIconOff - self.inWidgetIcon.pixmap().width(), yWidgetOffTop + xIconOff)

        # show signal boxes and text labels
        #signalSpace = (count)*ySignalSpace
        signalSpace = height
        for i in range(len(outputs)):
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(outputs)+1)
            box = QGraphicsRectItem(xWidgetOff + width, y - ySignalSize/2.0, xSignalSize, ySignalSize, None, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.setZValue(200)
            self.outBoxes.append((outputs[i].name, box))

            self.texts.append(MyCanvasText(self.dlg.canvas, outputs[i].name, xWidgetOff + width - 8, y - 7, Qt.AlignRight | Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(MyCanvasText(self.dlg.canvas, outputs[i].type, xWidgetOff + width - 8, y + 7, Qt.AlignRight | Qt.AlignVCenter, bold =0, show=1))

        for i in range(len(inputs)):
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(inputs)+1)
            box = QGraphicsRectItem(xWidgetOff + width + xSpaceBetweenWidgets - xSignalSize, y - ySignalSize/2.0, xSignalSize, ySignalSize, None, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.setZValue(200)
            self.inBoxes.append((inputs[i].name, box))

            self.texts.append(MyCanvasText(self.dlg.canvas, inputs[i].name, xWidgetOff + width + xSpaceBetweenWidgets + 8, y - 7, Qt.AlignLeft | Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(MyCanvasText(self.dlg.canvas, inputs[i].type, xWidgetOff + width + xSpaceBetweenWidgets + 8, y + 7, Qt.AlignLeft | Qt.AlignVCenter, bold =0, show=1))

        self.texts.append(MyCanvasText(self.dlg.canvas, outWidget.caption, xWidgetOff + width/2.0, yWidgetOffTop + height + 5, Qt.AlignHCenter | Qt.AlignTop, bold =1, show=1))
        self.texts.append(MyCanvasText(self.dlg.canvas, inWidget.caption, xWidgetOff + width* 1.5 + xSpaceBetweenWidgets, yWidgetOffTop + height + 5, Qt.AlignHCenter | Qt.AlignTop, bold =1, show=1))

        return (2*xWidgetOff + 2*width + xSpaceBetweenWidgets, yWidgetOffTop + height + yWidgetOffBottom)

    def getTextWidth(self, text, bold = 0):
        temp = QGraphicsSimpleTextItem(text, None, self.dlg.canvas)
        if bold:
            font = temp.font()
            font.setBold(1)
            temp.setFont(font)
        temp.hide()
        return temp.boundingRect().width()

    # ###################################################################
    # mouse button was pressed
    def mousePressEvent(self, ev):
        self.bMouseDown = 1
        point = self.mapToScene(ev.pos())
        activeItem = self.scene().itemAt(QPointF(ev.pos()))
        if type(activeItem) == QGraphicsRectItem and activeItem not in [self.outWidget, self.inWidget]:
            self.tempLine = QGraphicsLineItem(None, self.dlg.canvas)
            self.tempLine.setLine(point.x(), point.y(), point.x(), point.y())
            self.tempLine.setPen(QPen(QColor(200, 200, 200), 1))
            self.tempLine.setZValue(-300)
            
        elif type(activeItem) == QGraphicsLineItem:
            for (line, outName, inName, outBox, inBox) in self.lines:
                if line == activeItem:
                    self.dlg.removeLink(outName, inName)
                    return

    # ###################################################################
    # mouse button was released #########################################
    def mouseMoveEvent(self, ev):
        if self.tempLine:
            curr = self.mapToScene(ev.pos())
            start = self.tempLine.line().p1()
            self.tempLine.setLine(start.x(), start.y(), curr.x(), curr.y())
            self.scene().update()

    # ###################################################################
    # mouse button was released #########################################
    def mouseReleaseEvent(self, ev):
        if self.tempLine:
            activeItem = self.scene().itemAt(QPointF(ev.pos()))
            if type(activeItem) == QGraphicsRectItem:
                activeItem2 = self.scene().itemAt(self.tempLine.line().p1())
                if activeItem.x() < activeItem2.x(): outBox = activeItem; inBox = activeItem2
                else:                                outBox = activeItem2; inBox = activeItem
                outName = None; inName = None
                for (name, box) in self.outBoxes:
                    if box == outBox: outName = name
                for (name, box) in self.inBoxes:
                    if box == inBox: inName = name
                if outName != None and inName != None:
                    self.dlg.addLink(outName, inName)

            self.tempLine.hide()
            self.tempLine = None
            self.scene().update()


    def addLink(self, outName, inName):
        outBox = None; inBox = None
        for (name, box) in self.outBoxes:
            if name == outName: outBox = box
        for (name, box) in self.inBoxes:
            if name == inName : inBox  = box
        if outBox == None or inBox == None:
            print "error adding link. Data = ", outName, inName
            return
        line = QGraphicsLineItem(None, self.dlg.canvas)
        outRect = outBox.rect()
        inRect = inBox.rect()
        line.setLine(outRect.x() + outRect.width()-2, outRect.y() + outRect.height()/2.0, inRect.x()+2, inRect.y() + inRect.height()/2.0)
        line.setPen(QPen(QColor(160, 160, 160), 5))
        line.setZValue(100)
        self.scene().update()
        self.lines.append((line, outName, inName, outBox, inBox))


    def removeLink(self, outName, inName):
        for (line, outN, inN, outBox, inBox) in self.lines:
            if outN == outName and inN == inName:
                line.hide()
                self.lines.remove((line, outN, inN, outBox, inBox))
                self.scene().update()
                return


# #######################################
# # Signal dialog - let the user select active signals between two widgets
# #######################################
class SignalDialog(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg

        self.signals = []
        self._links = []
        self.allSignalsTaken = 0

        # GUI
        self.setWindowTitle('Connect Signals')
        self.setLayout(QVBoxLayout())

        self.canvasGroup = OWGUI.widgetBox(self, 1)
        self.canvas = QGraphicsScene(0,0,1000,1000)
        self.canvasView = SignalCanvasView(self, self.canvasDlg, self.canvas, self.canvasGroup)
        self.canvasGroup.layout().addWidget(self.canvasView)

        buttons = OWGUI.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.buttonHelp = OWGUI.button(buttons, self, "&Help")
        buttons.layout().addStretch(1)
        self.buttonClearAll = OWGUI.button(buttons, self, "Clear &All", callback = self.clearAll)
        self.buttonOk = OWGUI.button(buttons, self, "&OK", callback = self.accept)
        self.buttonOk.setAutoDefault(1)
        self.buttonOk.setDefault(1)
        self.buttonCancel = OWGUI.button(buttons, self, "&Cancel", callback = self.reject)

    def clearAll(self):
        while self._links != []:
            self.removeLink(self._links[0][0], self._links[0][1])

    def setOutInWidgets(self, outWidget, inWidget):
        self.outWidget = outWidget
        self.inWidget = inWidget
        (width, height) = self.canvasView.addSignalList(outWidget, inWidget)
        self.canvas.setSceneRect(0, 0, width, height)
        self.resize(width+50, height+80)

    def countCompatibleConnections(self, outputs, inputs, outInstance, inInstance, outType, inType):
        count = 0
        for outS in outputs:
            if outInstance.getOutputType(outS.name) == None: continue  # ignore if some signals don't exist any more, since we will report error somewhere else
            if not issubclass(outInstance.getOutputType(outS.name), outType): continue
            for inS in inputs:
                if inInstance.getOutputType(inS.name) == None: continue  # ignore if some signals don't exist any more, since we will report error somewhere else
                if not issubclass(inType, inInstance.getInputType(inS.name)): continue
                if issubclass(outInstance.getOutputType(outS.name), inInstance.getInputType(inS.name)): count+= 1

        return count

    def existsABetterLink(self, outSignal, inSignal, outSignals, inSignals):
        existsBetter = 0

        betterOutSignal = None; betterInSignal = None
        for outS in outSignals:
            for inS in inSignals:
                if (outS.name != outSignal.name and outS.name == inSignal.name and outS.type == inSignal.type) or (inS.name != inSignal.name and inS.name == outSignal.name and inS.type == outSignal.type):
                    existsBetter = 1
                    betterOutSignal = outS
                    betterInSignal = inS
        return existsBetter, betterOutSignal, betterInSignal


    def getPossibleConnections(self, outputs, inputs):
        possibleLinks = []
        for outS in outputs:
            outType = self.outWidget.instance.getOutputType(outS.name)
            if outType == None:     #print "Unable to find signal type for signal %s. Check the definition of the widget." % (outS.name)
                continue
            for inS in inputs:
                inType = self.inWidget.instance.getInputType(inS.name)
                if inType == None:
                    continue        #print "Unable to find signal type for signal %s. Check the definition of the widget." % (inS.name)
                if issubclass(outType, inType):
                        possibleLinks.append((outS.name, inS.name))
        return possibleLinks

    def addDefaultLinks(self):
        canConnect = 0
        addedInLinks = []
        addedOutLinks = []
        self.multiplePossibleConnections = 0    # can we connect some signal with more than one widget

        minorInputs = [signal for signal in self.inWidget.widgetInfo.inputs if not signal.default]
        majorInputs = [signal for signal in self.inWidget.widgetInfo.inputs if signal.default]
        minorOutputs = [signal for signal in self.outWidget.widgetInfo.outputs if not signal.default]
        majorOutputs = [signal for signal in self.outWidget.widgetInfo.outputs if signal.default]

        inConnected = self.inWidget.getInConnectedSignalNames()
        outConnected = self.outWidget.getOutConnectedSignalNames()

        # input connections that can be simultaneously connected to multiple outputs are not to be considered as already connected
        for i in inConnected[::-1]:
            if not self.inWidget.instance.signalIsOnlySingleConnection(i):
                inConnected.remove(i)

        for s in majorInputs + minorInputs:
            if not self.inWidget.instance.hasInputName(s.name):
                return -1
        for s in majorOutputs + minorOutputs:
            if not self.outWidget.instance.hasOutputName(s.name):
                return -1

        pl1 = self.getPossibleConnections(majorOutputs, majorInputs)
        pl2 = self.getPossibleConnections(majorOutputs, minorInputs)
        pl3 = self.getPossibleConnections(minorOutputs, majorInputs)
        pl4 = self.getPossibleConnections(minorOutputs, minorInputs)

        all = pl1 + pl2 + pl3 + pl4

        if not all: return 0

        # try to find a link to any inputs that hasn't been previously connected
        self.allSignalsTaken = 1
        for (o,i) in all:
            if i not in inConnected:
                all.remove((o,i))
                all.insert(0, (o,i))
                self.allSignalsTaken = 0       # we found an unconnected link. no need to show the signal dialog
                break
        self.addLink(all[0][0], all[0][1])  # add only the best link

        # there are multiple possible connections if we have in the same priority class more than one possible unconnected link
        for pl in [pl1, pl2, pl3, pl4]:
            #if len(pl) > 1 and sum([i not in inConnected for (o,i) in pl]) > 1: # if we have more than one valid
            if len(pl) > 1:     # if we have more than one valid
                self.multiplePossibleConnections = 1
            if len(pl) > 0:     # when we find a first non-empty list we stop searching
                break
        return len(all) > 0

    def addLink(self, outName, inName):
        if (outName, inName) in self._links: return 1

        # check if correct types
        outType = self.outWidget.instance.getOutputType(outName)
        inType = self.inWidget.instance.getInputType(inName)
        if not issubclass(outType, inType): return 0

        inSignal = None
        inputs = self.inWidget.widgetInfo.inputs
        for i in range(len(inputs)):
            if inputs[i].name == inName: inSignal = inputs[i]

        # if inName is a single signal and connection already exists -> delete it
        for (outN, inN) in self._links:
            if inN == inName and inSignal.single:
                self.removeLink(outN, inN)

        self._links.append((outName, inName))
        self.canvasView.addLink(outName, inName)
        return 1


    def removeLink(self, outName, inName):
        if (outName, inName) in self._links:
            self._links.remove((outName, inName))
            self.canvasView.removeLink(outName, inName)

    def getLinks(self):
        return self._links


class ColorIcon(QToolButton):
    def __init__(self, parent, color):
        QToolButton.__init__(self, parent)
        self.color = color
        self.setMaximumSize(20,20)
        self.connect(self, SIGNAL("clicked()"), self.showColorDialog)
        self.updateColor()

    def updateColor(self):
        pixmap = QPixmap(16,16)
        painter = QPainter()
        painter.begin(pixmap)
        painter.setPen(QPen(self.color))
        painter.setBrush(QBrush(self.color))
        painter.drawRect(0, 0, 16, 16);
        painter.end()
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(16,16))


    def drawButtonLabel(self, painter):
        painter.setBrush(QBrush(self.color))
        painter.setPen(QPen(self.color))
        painter.drawRect(3, 3, self.width()-6, self.height()-6)

    def showColorDialog(self):
        color = QColorDialog.getColor(self.color, self)
        if color.isValid():
            self.color = color
            self.updateColor()
            self.repaint()

# canvas dialog
class CanvasOptionsDlg(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.settings = dict(canvasDlg.settings)        # create a copy of the settings dict. in case we accept the dialog, we update the canvasDlg.settings with this dict
        if sys.platform == "darwin":
            self.setWindowTitle("Preferences")
        else:
            self.setWindowTitle("Canvas Options")
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setSpacing(0)
        self.resize(300,300)
        self.toAdd = []
        self.toRemove = []

        self.tabs = QTabWidget(self)
        GeneralTab = OWGUI.widgetBox(self.tabs, margin = 4)
        ExceptionsTab = OWGUI.widgetBox(self.tabs, margin = 4)
        TabOrderTab = OWGUI.widgetBox(self.tabs, margin = 4)

        self.tabs.addTab(GeneralTab, "General")
        self.tabs.addTab(ExceptionsTab, "Exception handling")
        self.tabs.addTab(TabOrderTab, "Widget tab order")

        # #################################################################
        # GENERAL TAB
        generalBox = OWGUI.widgetBox(GeneralTab, "General Options")
        self.snapToGridCB = OWGUI.checkBox(generalBox, self.settings, "snapToGrid", "Snap widgets to grid", debuggingEnabled = 0)
        self.writeLogFileCB  = OWGUI.checkBox(generalBox, self.settings, "writeLogFile", "Save content of the Output window to a log file", debuggingEnabled = 0)
        self.showSignalNamesCB = OWGUI.checkBox(generalBox, self.settings, "showSignalNames", "Show signal names between widgets", debuggingEnabled = 0)
        self.dontAskBeforeCloseCB= OWGUI.checkBox(generalBox, self.settings, "dontAskBeforeClose", "Don't ask to save schema before closing", debuggingEnabled = 0)
        self.saveWidgetsPositionCB = OWGUI.checkBox(generalBox, self.settings, "saveWidgetsPosition", "Save size and position of widgets", debuggingEnabled = 0)
        self.useContextsCB = OWGUI.checkBox(generalBox, self.settings, "useContexts", "Use context settings")

        validator = QIntValidator(self)
        validator.setRange(0,10000)

        hbox1 = OWGUI.widgetBox(GeneralTab, orientation = "horizontal")
        hbox2 = OWGUI.widgetBox(GeneralTab, orientation = "horizontal")
        canvasDlgSettings = OWGUI.widgetBox(hbox1, "Canvas Dialog Settings")
#        schemeSettings = OWGUI.widgetBox(hbox1, "Scheme Settings") 
         
        self.widthSlider = OWGUI.qwtHSlider(canvasDlgSettings, self.settings, "canvasWidth", minValue = 300, maxValue = 1200, label = "Canvas width:  ", step = 50, precision = " %.0f px", debuggingEnabled = 0)
        self.heightSlider = OWGUI.qwtHSlider(canvasDlgSettings, self.settings, "canvasHeight", minValue = 300, maxValue = 1200, label = "Canvas height:  ", step = 50, precision = " %.0f px", debuggingEnabled = 0)
        OWGUI.separator(canvasDlgSettings)
        
        items = [str(n) for n in QStyleFactory.keys()]
        ind = items.index(self.settings.get("style", "WindowsXP"))
        OWGUI.comboBox(canvasDlgSettings, self.settings, "style", label = "Window style:", orientation = "horizontal", items = [str(n) for n in QStyleFactory.keys()], sendSelectedValue = 1, debuggingEnabled = 0)
        OWGUI.checkBox(canvasDlgSettings, self.settings, "useDefaultPalette", "Use style's standard palette", debuggingEnabled = 0)
        
#        if canvasDlg:
#            selectedWidgetBox = OWGUI.widgetBox(schemeSettings, orientation = "horizontal")
#            self.selectedWidgetIcon = ColorIcon(selectedWidgetBox, canvasDlg.widgetSelectedColor)
#            selectedWidgetBox.layout().addWidget(self.selectedWidgetIcon)
#            selectedWidgetLabel = OWGUI.widgetLabel(selectedWidgetBox, " Selected widget")
# 
#            activeWidgetBox = OWGUI.widgetBox(schemeSettings, orientation = "horizontal")
#            self.activeWidgetIcon = ColorIcon(activeWidgetBox, canvasDlg.widgetActiveColor)
#            activeWidgetBox.layout().addWidget(self.activeWidgetIcon)
#            selectedWidgetLabel = OWGUI.widgetLabel(activeWidgetBox, " Active widget")
# 
#            lineBox = OWGUI.widgetBox(schemeSettings, orientation = "horizontal")
#            self.lineIcon = ColorIcon(lineBox, canvasDlg.lineColor)
#            lineBox.layout().addWidget(self.lineIcon)
#            selectedWidgetLabel = OWGUI.widgetLabel(lineBox, " Lines")
#            
#        OWGUI.separator(schemeSettings)
#        items = ["%d x %d" % (v,v) for v in self.canvasDlg.schemeIconSizeList]
#        val = min(len(items)-1, self.settings['schemeIconSize'])
#        self.schemeIconSizeCombo = OWGUI.comboBoxWithCaption(schemeSettings, self.settings, 'schemeIconSize', "Scheme icon size:", items = items, tooltip = "Set the size of the widget icons on the scheme", debuggingEnabled = 0)
        
        GeneralTab.layout().addStretch(1)

        # #################################################################
        # EXCEPTION TAB
        exceptions = OWGUI.widgetBox(ExceptionsTab, "Exceptions")
        #self.catchExceptionCB = QCheckBox('Catch exceptions', exceptions)
        self.focusOnCatchExceptionCB = OWGUI.checkBox(exceptions, self.settings, "focusOnCatchException", 'Show output window on exception')
        self.printExceptionInStatusBarCB = OWGUI.checkBox(exceptions, self.settings, "printExceptionInStatusBar", 'Print last exception in status bar')

        output = OWGUI.widgetBox(ExceptionsTab, "System output")
        #self.catchOutputCB = QCheckBox('Catch system output', output)
        self.focusOnCatchOutputCB = OWGUI.checkBox(output, self.settings, "focusOnCatchOutput", 'Focus output window on system output')
        self.printOutputInStatusBarCB = OWGUI.checkBox(output, self.settings, "printOutputInStatusBar", 'Print last system output in status bar')

        hboxExc = OWGUI.widgetBox(ExceptionsTab, orientation="horizontal")
        outputCanvas = OWGUI.widgetBox(hboxExc, "Canvas Info Handling")
        outputWidgets = OWGUI.widgetBox(hboxExc, "Widget Info Handling")
        self.ocShow = OWGUI.checkBox(outputCanvas, self.settings, "ocShow", 'Show icon above widget for...')
        self.ocInfo = OWGUI.checkBox(OWGUI.indentedBox(outputCanvas, 10), self.settings, "ocInfo", 'Information')
        self.ocWarning = OWGUI.checkBox(OWGUI.indentedBox(outputCanvas, 10), self.settings, "ocWarning", 'Warnings')
        self.ocError = OWGUI.checkBox(OWGUI.indentedBox(outputCanvas, 10), self.settings, "ocError", 'Errors')

        self.owShow = OWGUI.checkBox(outputWidgets, self.settings, "owShow", 'Show statusbar info for...')
        self.owInfo = OWGUI.checkBox(OWGUI.indentedBox(outputWidgets, 10), self.settings, "owInfo", 'Information')
        self.owWarning = OWGUI.checkBox(OWGUI.indentedBox(outputWidgets, 10), self.settings, "owWarning", 'Warnings')
        self.owError = OWGUI.checkBox(OWGUI.indentedBox(outputWidgets, 10), self.settings, "owError", 'Errors')

        verbosityBox = OWGUI.widgetBox(ExceptionsTab, "Verbosity", orientation = "horizontal")
        verbosityBox.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.verbosityCombo = OWGUI.comboBox(verbosityBox, self.settings, "outputVerbosity", label = "Set level of widget output: ", orientation='horizontal', items=["Small", "Medium", "High"])
        ExceptionsTab.layout().addStretch(1)

        # #################################################################
        # TAB ORDER TAB
        tabOrderBox = OWGUI.widgetBox(TabOrderTab, "Set Order of Widget Categories", orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.tabOrderList = QListWidget(tabOrderBox)
        self.tabOrderList.setAcceptDrops(True)

        tabOrderBox.layout().addWidget(self.tabOrderList)
        self.tabOrderList.setSelectionMode(QListWidget.SingleSelection)
        
        ind = 0
        for (name, show) in self.settings["WidgetTabs"]:
            if self.canvasDlg.widgetRegistry.has_key(name):
                self.tabOrderList.addItem(name)
                self.tabOrderList.item(ind).setCheckState(show and Qt.Checked or Qt.Unchecked)
                ind+=1

        w = OWGUI.widgetBox(tabOrderBox, sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.upButton = OWGUI.button(w, self, "Up", callback = self.moveUp)
        self.downButton = OWGUI.button(w, self, "Down", callback = self.moveDown)
        w.layout().addSpacing(20)
        self.addButton = OWGUI.button(w, self, "Add", callback = self.addCategory)
        self.removeButton = OWGUI.button(w, self, "Remove", callback = self.removeCategory)
        self.removeButton.setEnabled(0)
        w.layout().addStretch(1)

        # OK, Cancel buttons
        hbox = OWGUI.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        hbox.layout().addStretch(1)
        self.okButton = OWGUI.button(hbox, self, "OK", callback = self.accept)
        self.cancelButton = OWGUI.button(hbox, self, "Cancel", callback = self.reject)
        self.connect(self.tabOrderList, SIGNAL("currentRowChanged(int)"), self.enableDisableButtons)

        self.topLayout.addWidget(self.tabs)
        self.topLayout.addWidget(hbox)


    def accept(self):
        self.settings["widgetSelectedColor"] = self.selectedWidgetIcon.color.getRgb()
        self.settings["widgetActiveColor"]   = self.activeWidgetIcon.color.getRgb()
        self.settings["lineColor"]           = self.lineIcon.color.getRgb()
        QDialog.accept(self)
        
        

    # move selected widget category up
    def moveUp(self):
        for i in range(1, self.tabOrderList.count()):
            if self.tabOrderList.item(i).isSelected():
                item = self.tabOrderList.takeItem(i)
                for j in range(self.tabOrderList.count()): self.tabOrderList.item(j).setSelected(0)
                self.tabOrderList.insertItem(i-1, item)
                item.setSelected(1)

    # move selected widget category down
    def moveDown(self):
        for i in range(self.tabOrderList.count()-2,-1,-1):
            if self.tabOrderList.item(i).isSelected():
                item = self.tabOrderList.takeItem(i)
                for j in range(self.tabOrderList.count()): self.tabOrderList.item(j).setSelected(0)
                self.tabOrderList.insertItem(i+1, item)
                item.setSelected(1)

    def enableDisableButtons(self, itemIndex):
        self.upButton.setEnabled(itemIndex > 0)
        self.downButton.setEnabled(itemIndex < self.tabOrderList.count()-1)
        catName = str(self.tabOrderList.currentItem().text())
        if not self.canvasDlg.widgetRegistry.has_key(catName): return
        self.removeButton.setEnabled(os.path.normpath(self.canvasDlg.widgetDir) not in os.path.normpath(self.canvasDlg.widgetRegistry[catName].directory))
        #self.removeButton.setEnabled(1)

    def addCategory(self):
        dir = str(QFileDialog.getExistingDirectory(self, "Select the folder that contains the add-on:"))
        if dir != "":
            if os.path.split(dir)[1] == "widgets":     # register a dir above the dir that contains the widget folder
                dir = os.path.split(dir)[0]
            if os.path.exists(os.path.join(dir, "widgets")):
                name = os.path.split(dir)[1]
                self.toAdd.append((name, dir))
                self.tabOrderList.addItem(name)
                self.tabOrderList.item(self.tabOrderList.count()-1).setCheckState(Qt.Checked)
            else:
                QMessageBox.information( None, "Information", 'The specified folder does not seem to contain an Orange add-on.', QMessageBox.Ok + QMessageBox.Default)
            
        
    def removeCategory(self):
        curCat = str(self.tabOrderList.item(self.tabOrderList.currentRow()).text())
        if QMessageBox.warning(self,'Orange Canvas', "Unregister widget category '%s' from Orange canvas?\nThis will not remove any files." % curCat, QMessageBox.Ok , QMessageBox.Cancel | QMessageBox.Default | QMessageBox.Escape) == QMessageBox.Ok:
            self.toRemove.append((curCat, self.canvasDlg.widgetRegistry[curCat].directory))
            item = self.tabOrderList.takeItem(self.tabOrderList.row(self.tabOrderList.currentItem()))
            #if item: item.setHidden(1)


class KeyEdit(QLineEdit):
    def __init__(self, parent, key, invdict, widget, invInvDict):
        QLineEdit.__init__(self, parent)
        self.setText(key)
        #self.setReadOnly(True)
        self.invdict = invdict
        self.widget = widget
        self.invInvDict = invInvDict

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete or e.key() == Qt.Key_Backspace:
            pressed = "<none>"
            self.setText(pressed)
            prevkey = self.invdict.get(self.widget)
            if prevkey:
                del self.invdict[self.widget]
                del self.invInvDict[prevkey]
            return

        if e.key() not in range(32, 128): # + range(Qt.Key_F1, Qt.Key_F35+1): -- this wouldn't work, see the line below, and also writing to file etc.
            e.ignore()
            return

        pressed = "-".join(filter(None, [e.modifiers() & x and y for x, y in [(Qt.ControlModifier, "Ctrl"), (Qt.AltModifier, "Alt")]]) + [chr(e.key())])

        assigned = self.invInvDict.get(pressed, None)
        if assigned and assigned != self and QMessageBox.question(self, "Confirmation", "'%(pressed)s' is already assigned to '%(assigned)s'. Override?" % {"pressed": pressed, "assigned": assigned.widget.name}, QMessageBox.Yes | QMessageBox.Default, QMessageBox.No | QMessageBox.Escape) == QMessageBox.No:
            return
        
        if assigned:
            assigned.setText("<none>")
            del self.invdict[assigned.widget]
        self.setText(pressed)
        self.invdict[self.widget] = pressed
        self.invInvDict[pressed] = self
        

# widget shortcuts dialog
class WidgetShortcutDlg(QDialog):
    def __init__(self, canvasDlg, *args):
        import orngTabs

        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.setWindowTitle("Widget Shortcuts")
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(10)
        self.resize(700,500)

        self.invDict = dict([(y, x) for x, y in canvasDlg.widgetShortcuts.items()])
        invInvDict = {}

        self.tabs = QTabWidget(self)
        
        extraTabs = [(name, 1) for name in canvasDlg.widgetRegistry.keys() if name not in [tab for (tab, s) in canvasDlg.settings["WidgetTabs"]]]
        for tabName, show in canvasDlg.settings["WidgetTabs"] + extraTabs:
            if not canvasDlg.widgetRegistry.has_key(tabName):
                continue
            scrollArea = QScrollArea()
            self.tabs.addTab(scrollArea, tabName)
            #scrollArea.setWidgetResizable(1)       # you have to use this or set size to wtab manually - otherwise nothing gets shown

            wtab = QWidget(self.tabs)
            scrollArea.setWidget(wtab)

            widgets = [(int(widgetInfo.priority), name, widgetInfo) for (name, widgetInfo) in canvasDlg.widgetRegistry[tabName].items()]
            widgets.sort()
            rows = (len(widgets)+2) / 3
            layout = QGridLayout(wtab)

            for i, (priority, name, widgetInfo) in enumerate(widgets):
                x = i / rows
                y = i % rows

                hlayout = QHBoxLayout()
                mainBox = QWidget(wtab)
                mainBox.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
                mainBox.setLayout(hlayout)
                layout.addWidget(mainBox, y, x, Qt.AlignTop | Qt.AlignLeft)
                label = QLabel(wtab)
                label.setPixmap(canvasDlg.getWidgetIcon(widgetInfo).pixmap(40))
                hlayout.addWidget(label)

                optionsw = QWidget(self)
                optionsw.setLayout(QVBoxLayout())
                hlayout.addWidget(optionsw)
                optionsw.layout().addStretch(1)

                OWGUI.widgetLabel(optionsw, name)
                key = self.invDict.get(widgetInfo, "<none>")
                le = KeyEdit(optionsw, key, self.invDict, widgetInfo, invInvDict)
                optionsw.layout().addWidget(le)
                invInvDict[key] = le
                le.setFixedWidth(60)

            wtab.resize(wtab.sizeHint())

        # OK, Cancel buttons
        hbox = OWGUI.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        hbox.layout().addStretch(1)
        self.okButton = OWGUI.button(hbox, self, "OK", callback = self.accept)
        self.cancelButton = OWGUI.button(hbox, self, "Cancel", callback = self.reject)
        self.okButton.setDefault(True)

        self.layout().addWidget(self.tabs)
        self.layout().addWidget(hbox)


class AboutDlg(QDialog):
    def __init__(self, *args):
        apply(QDialog.__init__,(self,) + args)
        self.topLayout = QVBoxLayout(self)
        self.setWindowFlags(Qt.Popup)
        
        import orngEnviron
        logoImage = QPixmap(os.path.join(orngEnviron.directoryNames["canvasDir"], "icons", "splash.png"))
        logo = OWGUI.widgetLabel(self, "")
        logo.setPixmap(logoImage)
        
        OWGUI.widgetLabel(self, '<p align="center"><h2>Orange</h2></p>') 
        
        try:
            import orange
            version = orange.version.split("(")[0].strip()
            date = orange.version.split(",")[-1].strip(" )")
            OWGUI.widgetLabel(self, '<p align="center">version %s</p>' % (version))
            OWGUI.widgetLabel(self, '<p align="center">(built %s)</p>' % (date))
        except:
            pass
        OWGUI.widgetLabel(self, "" )
        #OWGUI.button(self, self, "Close", callback = self.accept)
        b = QDialogButtonBox(self)
        b.setCenterButtons(1)
        self.layout().addWidget(b)
        butt = b.addButton(QDialogButtonBox.Close)
        self.connect(butt, SIGNAL("clicked()"), self.accept)
        
        

class saveApplicationDlg(QDialog):
    def __init__(self, *args):
        apply(QDialog.__init__,(self,) + args)
        self.setWindowTitle("Set Widget Order")
        self.setLayout(QVBoxLayout())

        listbox = OWGUI.widgetBox(self, 1, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.tabOrderList = QListWidget(listbox)
        self.tabOrderList.setSelectionMode(QListWidget.SingleSelection)
        listbox.layout().addWidget(self.tabOrderList)

        w = OWGUI.widgetBox(listbox, sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.upButton = OWGUI.button(w, self, "Up", callback = self.moveUp)
        self.downButton = OWGUI.button(w, self, "Down", callback = self.moveDown)
        w.layout().addStretch(1)

        hbox = OWGUI.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.button(hbox, self, "Add Separator", callback = self.insertSeparator)
        hbox.layout().addStretch(1)
        OWGUI.button(hbox, self, "&OK", callback = self.accept)
        OWGUI.button(hbox, self, "&Cancel", callback = self.reject)

        self.resize(200,250)

    def accept(self):
        self.shownWidgetList = [str(self.tabOrderList.item(i).text()) for i in range(self.tabOrderList.count())]
        QDialog.accept(self)

    def insertSeparator(self):
        curr = self.tabOrderList.indexFromItem(self.tabOrderList.currentItem()).row()
        self.insertWidgetName("[Separator]", curr)

    def insertWidgetName(self, name, index = -1):
        if index == -1:
            self.tabOrderList.addItem(name)
        else:
            self.tabOrderList.insertItem(index, name)

    # move selected widget category up
    def moveUp(self):
        for i in range(1, self.tabOrderList.count()):
            if self.tabOrderList.item(i).isSelected():
                item = self.tabOrderList.takeItem(i)
                for j in range(self.tabOrderList.count()): self.tabOrderList.item(j).setSelected(0)
                self.tabOrderList.insertItem(i-1, item)
                item.setSelected(1)


    # move selected widget category down
    def moveDown(self):
        for i in range(self.tabOrderList.count()-2,-1,-1):
            if self.tabOrderList.item(i).isSelected():
                item = self.tabOrderList.takeItem(i)
                for j in range(self.tabOrderList.count()): self.tabOrderList.item(j).setSelected(0)
                self.tabOrderList.insertItem(i+1, item)
                item.setSelected(1)


if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    #dlg = saveApplicationDlg(None)
    dlg = AboutDlg(None)
    dlg.show()
    app.exec_()

