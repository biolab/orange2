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
        QDialog.__init__(self, *args)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.canvasDlg = canvasDlg

        self.signals = []
        self._links = []
        self.allSignalsTaken = 0
        self.signalManager = self.canvasDlg.schema.signalManager

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
                if issubclass(inType, outType):
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
    
    def addDefaultLinks(self):
        self.multiplePossibleConnections = 0
        candidates = self.signalManager.proposePossibleLinks(self.outWidget.instance, self.inWidget.instance)
        if candidates:
            ## If there are more candidates with max weights
            maxW = max([w for _,_,w in candidates])
            maxCandidates = [c for c in candidates if c[-1] == maxW]
            if len(maxCandidates) > 1:
                self.multiplePossibleConnections = 1
            best = maxCandidates[0]
            self.addLink(best[0].name, best[1].name) # add the best to the view
            return True
        else:
            return 0
            

    def addLink(self, outName, inName):
        if (outName, inName) in self._links: return 1

        # check if correct types
        outType = self.outWidget.instance.getOutputType(outName)
        inType = self.inWidget.instance.getInputType(inName)
        if not issubclass(outType, inType) and not issubclass(inType, outType): return 0 #TODO check this with signalManager.canConnect

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
#        self.toAdd = []
#        self.toRemove = []

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
        self.enableCanvasDropShadowsCB = OWGUI.checkBox(generalBox, self.settings, "enableCanvasDropShadows", "Enable drop shadows in canvas", debuggingEnabled = 0)
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
         
#        self.widthSlider = OWGUI.qwtHSlider(canvasDlgSettings, self.settings, "canvasWidth", minValue = 300, maxValue = 1200, label = "Canvas width:  ", step = 50, precision = " %.0f px", debuggingEnabled = 0)
#        self.heightSlider = OWGUI.qwtHSlider(canvasDlgSettings, self.settings, "canvasHeight", minValue = 300, maxValue = 1200, label = "Canvas height:  ", step = 50, precision = " %.0f px", debuggingEnabled = 0)
#        OWGUI.separator(canvasDlgSettings)
        
        items = [str(n) for n in QStyleFactory.keys()]
        itemsLower = [s.lower() for s in items]
        ind = itemsLower.index(self.settings.get("style", "Windows").lower())
        self.settings["style"] = items[ind]
        OWGUI.comboBox(canvasDlgSettings, self.settings, "style", label = "Window style:", orientation = "horizontal", items = [str(n) for n in QStyleFactory.keys()], sendSelectedValue = 1, debuggingEnabled = 0)
        OWGUI.checkBox(canvasDlgSettings, self.settings, "useDefaultPalette", "Use style's standard palette", debuggingEnabled = 0)
        
        OWGUI.separator(canvasDlgSettings)
        OWGUI.comboBox(canvasDlgSettings, self.settings, "widgetListType", label="Toolbox style:", orientation="horizontal",
                       items = ["Tool box", "Tree view", "Tree view (no icons)", "Tabs without labels", "Tabs with labels"], debuggingEnabled=0)
        
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
        indent = OWGUI.checkButtonOffsetHint(self.ocShow)
        self.ocInfo = OWGUI.checkBox(OWGUI.indentedBox(outputCanvas, indent), self.settings, "ocInfo", 'Information')
        self.ocWarning = OWGUI.checkBox(OWGUI.indentedBox(outputCanvas, indent), self.settings, "ocWarning", 'Warnings')
        self.ocError = OWGUI.checkBox(OWGUI.indentedBox(outputCanvas, indent), self.settings, "ocError", 'Errors')

        self.owShow = OWGUI.checkBox(outputWidgets, self.settings, "owShow", 'Show statusbar info for...')
        self.owInfo = OWGUI.checkBox(OWGUI.indentedBox(outputWidgets, indent), self.settings, "owInfo", 'Information')
        self.owWarning = OWGUI.checkBox(OWGUI.indentedBox(outputWidgets, indent), self.settings, "owWarning", 'Warnings')
        self.owError = OWGUI.checkBox(OWGUI.indentedBox(outputWidgets, indent), self.settings, "owError", 'Errors')

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
#        w.layout().addSpacing(20)
#        self.addButton = OWGUI.button(w, self, "Add", callback = self.addCategory)
#        self.removeButton = OWGUI.button(w, self, "Remove", callback = self.removeCategory)
#        self.removeButton.setEnabled(0)
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
#        self.settings["widgetSelectedColor"] = self.selectedWidgetIcon.color.getRgb()
#        self.settings["widgetActiveColor"]   = self.activeWidgetIcon.color.getRgb()
#        self.settings["lineColor"]           = self.lineIcon.color.getRgb()
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
#        self.removeButton.setEnabled( all([os.path.normpath(self.canvasDlg.widgetDir) not in os.path.normpath(x.directory) and
#                                           os.path.normpath(self.canvasDlg.addOnsDir) not in os.path.normpath(x.directory)
#                                             for x in self.canvasDlg.widgetRegistry[catName].values()]))
        #self.removeButton.setEnabled(1)

#    def addCategory(self):
#        dir = str(QFileDialog.getExistingDirectory(self, "Select the folder that contains the add-on:"))
#        if dir != "":
#            if os.path.split(dir)[1] == "widgets":     # register a dir above the dir that contains the widget folder
#                dir = os.path.split(dir)[0]
#            if os.path.exists(os.path.join(dir, "widgets")):
#                name = os.path.split(dir)[1]
#                self.toAdd.append((name, dir))
#                self.tabOrderList.addItem(name)
#                self.tabOrderList.item(self.tabOrderList.count()-1).setCheckState(Qt.Checked)
#            else:
#                QMessageBox.information( None, "Information", 'The specified folder does not seem to contain an Orange add-on.', QMessageBox.Ok + QMessageBox.Default)
#            
#        
#    def removeCategory(self):
#        curCat = str(self.tabOrderList.item(self.tabOrderList.currentRow()).text())
#        if QMessageBox.warning(self,'Orange Canvas', "Unregister widget category '%s' from Orange canvas?\nThis will not remove any files." % curCat, QMessageBox.Ok , QMessageBox.Cancel | QMessageBox.Default | QMessageBox.Escape) == QMessageBox.Ok:
#            self.toRemove.append((curCat, self.canvasDlg.widgetRegistry[curCat]))
#            item = self.tabOrderList.takeItem(self.tabOrderList.row(self.tabOrderList.currentItem()))
#            #if item: item.setHidden(1)

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

class AddOnManagerSummary(QDialog):
    def __init__(self, add, remove, *args):
        apply(QDialog.__init__,(self,) + args)
        self.setWindowTitle("Pending Actions")
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setSpacing(10)
        self.resize(200, 0)
        
        OWGUI.label(self, self, "If you confirm, the following actions will take place:")

        self.memo = memo = QTextEdit(self)
        self.layout().addWidget(memo)
        memo.setReadOnly(True)
        memo.setFrameStyle(QFrame.NoFrame)
        pal = QPalette()
        pal.setColor(QPalette.Base, Qt.transparent)
        memo.setPalette(pal)
        memo.setLineWrapMode(QTextEdit.WidgetWidth)
        memo.setWordWrapMode(QTextOption.WordWrap)
        QObject.connect(memo.document().documentLayout(),
                       SIGNAL("documentSizeChanged(const QSizeF &)"),
                        lambda docSize: self.updateMinSize(docSize))
        actions = []
        for addOnId in add:
            if addOnId in remove:
                actions.append("Upgrade %s." % add[addOnId].name)
            elif addOnId.startswith("registered:"):
                actions.append("Register %s." % add[addOnId].name)
            else:
                actions.append("Install %s." % add[addOnId].name)
        for addOnId in remove:
            if not addOnId in add:
                if addOnId.startswith("registered:"):
                    actions.append("Unregister %s." % remove[addOnId].name)
                else:
                    actions.append("Remove %s." % remove[addOnId].name)
        actions.sort()
        memo.setText("\n".join(actions))
        
        self.layout().addStretch(1)
        self.setSizePolicy( QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum) )

        hbox = OWGUI.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        hbox.layout().addStretch(1)
        self.okButton = OWGUI.button(hbox, self, "OK", callback = self.accept)
        self.cancelButton = OWGUI.button(hbox, self, "Cancel", callback = self.reject)
        self.okButton.setDefault(True)

    def updateMinSize(self, documentSize):
        self.memo.update()
        self.memo.setMinimumHeight(min(300, documentSize.height() + 2 * self.memo.frameWidth()))


class AddOnRepositoryData(QDialog):
    def __init__(self, name="", url="", *args):
        apply(QDialog.__init__,(self,) + args)
        self.setWindowTitle("Add-on Repository")
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setSpacing(0)
        
        self.name = name
        self.url = url     
        
        eName = OWGUI.lineEdit(self, self, "name", "Display name:", orientation="horizontal", controlWidth=150)
        eName.parent().layout().addStretch(1)
        eURL = OWGUI.lineEdit(self, self, "url", "URL:", orientation="horizontal", controlWidth=250)
        eURL.parent().layout().addStretch(1)
        self.layout().addSpacing(15)
        hbox = OWGUI.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        hbox.layout().addStretch(1)
        self.okButton = OWGUI.button(hbox, self, "OK", callback = self.accept)
        self.cancelButton = OWGUI.button(hbox, self, "Cancel", callback = self.reject)
        self.okButton.setDefault(True)
        
    def accept(self):
        if self.name.strip() == "":
            QMessageBox.warning(self, "Incorrect Input", "Name cannot be empty")
            return
        if self.url.strip() == "":
            QMessageBox.warning(self, "Incorrect Input", "URL cannot be empty")
            return
        QDialog.accept(self)
        
        
class AddOnManagerDialog(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.setWindowTitle("Add-on Management")
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setSpacing(0)
        self.resize(600,500)
        self.layout().setSizeConstraint(QLayout.SetMinimumSize)
        
        mainBox = OWGUI.widgetBox(self, orientation="vertical", sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        
        # View settings & Search
        
        self.groupByRepo = True
        self.sortInstalledFirst = True
        self.sortSingleLast = True
        self.searchStr = ""

        searchBox = OWGUI.widgetBox(mainBox, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        self.viewBox = viewBox = OWGUI.SmallWidgetLabel(searchBox, pixmap = 1, box = "Grouping and Order", tooltip = "Adjust the order of add-ons in the list")
        cGroupByRepo        = OWGUI.checkBox(viewBox.widget, self, "groupByRepo", "&Group by repository", callback = self.refreshView)
        cSortInstalledFirst = OWGUI.checkBox(viewBox.widget, self, "sortInstalledFirst", "&Installed first", callback = self.refreshView)
        cSortSingleLast     = OWGUI.checkBox(viewBox.widget, self, "sortSingleLast", "&Single widgets last", callback = self.refreshView)

        self.eSearch = self.lineEditSearch(searchBox, self, "searchStr", None, 0, tooltip = "Type in to filter (search) add-ons.", callbackOnType=True, callback=self.searchCallback)
        
        # Repository & Add-on tree
        
        repos = OWGUI.widgetBox(mainBox, "Add-ons", orientation = "horizontal", sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        repos.layout().setSizeConstraint(QLayout.SetMinimumSize)
        self.tree = tree = QTreeWidget(repos)
        self.tree.setMinimumWidth(200)
        self.tree.repoItems = {}
        tree.header().hide()
        repos.layout().addWidget(tree)
        QObject.connect(tree, SIGNAL("itemChanged(QTreeWidgetItem *, int)"), self.cbToggled)
        QObject.connect(tree, SIGNAL("currentItemChanged(QTreeWidgetItem *, QTreeWidgetItem *)"), self.currentItemChanged)

        self.addOnsToAdd = {}
        self.addOnsToRemove = {}
        import orngAddOns
        self.repositories = [repo.clone() for repo in orngAddOns.availableRepositories]
        
        # Bottom info pane
        
        self.infoPane = infoPane = OWGUI.widgetBox(mainBox, orientation="vertical", sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum))
        infoPane.layout().setSizeConstraint(QLayout.SetMinimumSize)

        pVerInstalled = OWGUI.widgetBox(infoPane, orientation="horizontal")
        lblVerInstalled = OWGUI.label(pVerInstalled, self, "Installed version:", 150)
        boldFont = lblVerInstalled.font()
        boldFont.setWeight(QFont.Bold)
        lblVerInstalled.setFont(boldFont)
        self.lblVerInstalledValue = OWGUI.label(pVerInstalled, self, "-")
        self.lblVerInstalledValue.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        pVerInstalled.layout().addSpacing(10)
        self.lblStatus = OWGUI.label(pVerInstalled, self, "")
        self.lblStatus.setFont(boldFont)
        pVerInstalled.layout().addStretch(1)

        pVerAvail = OWGUI.widgetBox(infoPane, orientation="horizontal")
        lblVerAvail = OWGUI.label(pVerAvail, self, "Available version:", 150)
        lblVerAvail.setFont(boldFont)
        self.lblVerAvailValue = OWGUI.label(pVerAvail, self, "")
        pVerAvail.layout().addSpacing(10)
        self.upgradeButton = OWGUI.button(pVerAvail, self, "Upgrade", callback = self.upgrade)
        self.upgradeButton.setFixedHeight(lblVerAvail.height())
        self.donotUpgradeButton = OWGUI.button(pVerAvail, self, "Do not upgrade", callback = self.donotUpgrade)
        self.donotUpgradeButton.setFixedHeight(lblVerAvail.height())
        pVerAvail.layout().addStretch(1)
        
        pInfoBtns = OWGUI.widgetBox(infoPane, orientation="horizontal", sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.webButton = OWGUI.button(pInfoBtns, self, "Open webpage", callback = self.openWebPage)
        self.listWidgetsButton = OWGUI.button(pInfoBtns, self, "List widgets", callback = self.listWidgets)
        pInfoBtns.layout().addStretch(1)

        self.lblDescription = lblDescription = QTextEdit(infoPane)
        infoPane.layout().addWidget(lblDescription)
        lblDescription.setReadOnly(True)
        lblDescription.setFrameStyle(QFrame.NoFrame)
        pal = QPalette()
        pal.setColor(QPalette.Base, Qt.transparent)
        lblDescription.setPalette(pal)
        lblDescription.setLineWrapMode(QTextEdit.WidgetWidth)
        lblDescription.setWordWrapMode(QTextOption.WordWrap)
        QObject.connect(lblDescription.document().documentLayout(),
                        SIGNAL("documentSizeChanged(const QSizeF &)"),
                        lambda docSize: self.updateDescMinSize(docSize))

        # Bottom info pane for registered add-ons
        self.regiInfoPane = regiInfoPane = OWGUI.widgetBox(mainBox, orientation="vertical", sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum))
        regiInfoPane.layout().setSizeConstraint(QLayout.SetMinimumSize)
        self.lblRegisteredAddOnInfo = OWGUI.label(regiInfoPane, self, "-")
        self.lblRegisteredAddOnInfo.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Right panel
        
        rightPanel = OWGUI.widgetBox(repos, orientation = "vertical", sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding))
        rightPanel.layout().setSizeConstraint(QLayout.SetMinimumSize)
        self.addRepoButton = OWGUI.button(rightPanel, self, "Add Repository...", callback = self.addRepo)
        self.editRepoButton = OWGUI.button(rightPanel, self, "Edit Repository...", callback = self.editRepo)
        self.delRepoButton = OWGUI.button(rightPanel, self, "Remove Repository", callback = self.delSelectedRepo)
        self.reloadRepoButton = OWGUI.button(rightPanel, self, "Refresh lists", callback = self.reloadRepos)
        rightPanel.layout().addSpacing(15)
        self.upgradeAllButton = OWGUI.button(rightPanel, self, "Upgrade All", callback = self.upgradeAll)
        rightPanel.layout().addSpacing(15)
        self.registerButton = OWGUI.button(rightPanel, self, "Register Add-on...", callback = self.registerAddOn)
        rightPanel.layout().addStretch(1)
        for btn in rightPanel.children():
            if btn.__class__ is QPushButton:
                btn.setMinimumHeight(btn.height())
        
        # Close button
        hbox = OWGUI.widgetBox(mainBox, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        hbox.layout().addStretch(1)
        self.okButton = OWGUI.button(hbox, self, "OK", callback = self.accept)
        self.cancelButton = OWGUI.button(hbox, self, "Cancel", callback = self.reject)
        self.okButton.setDefault(True)

        self.refreshView()
        
    def updateDescMinSize(self, documentSize):
        self.lblDescription.update()
        self.lblDescription.setMinimumHeight(min(300, documentSize.height() + 2 * self.lblDescription.frameWidth()))
        pass
    
    def accept(self):
        import orngAddOns
        if len(self.addOnsToAdd) + len(self.addOnsToRemove) > 0:
            summary = AddOnManagerSummary(self.addOnsToAdd, self.addOnsToRemove, self)
            if summary.exec_() == QDialog.Rejected:
                return
        orngAddOns.availableRepositories = self.repositories
        orngAddOns.saveRepositories()
        QDialog.accept(self)
        
    def addRepo(self):
        dlg = AddOnRepositoryData()
        while dlg.exec_() == QDialog.Accepted:
            import orngAddOns
            try:
                repo = orngAddOns.OrangeAddOnRepository(dlg.name, dlg.url)   #TODO: This can take some time - inform the user!
                self.repositories.append(repo)
            except Exception, e:
                QMessageBox.critical(self, "Error", "Could not add this repository: %s"%e)
                continue
            break
        self.refreshView()

    def editRepo(self, repo=None):
        if not repo:
            repo = self.getRepoFromItem(self.tree.currentItem())
        if not repo:
            return
        dlg = AddOnRepositoryData(name=repo.name, url=repo.url)
        while dlg.exec_() == QDialog.Accepted:
            import orngAddOns
            try:
                oldname, oldurl = repo.name, repo.url
                repo.name, repo.url = dlg.name, dlg.url
                if oldurl != repo.url:
                    repo.refreshData(force=True)  #TODO: This can take some time - inform the user!
            except Exception, e:
                repo.name, repo.url = oldname, oldurl
                QMessageBox.critical(self, "Error", "Could not load repository %s."%e)
                continue
            break
        self.refreshView()

    def delSelectedRepo(self):
        repo = self.getRepoFromItem(self.tree.currentItem())
        if repo==None:
            return
        # Is it a default repository? We cannot delete it!
        import orngAddOns
        if repo.__class__ is orngAddOns.OrangeDefaultAddOnRepository:
            return
        
        # Are there add-ons selected for installation from this repository? We remove the installation requests.
        for (id, addOn) in self.addOnsToAdd.items():
            if addOn.repository == repo:
                del self.addOnsToAdd[id]
        
        # Remove the repository and refresh tree.
        self.repositories.remove(repo)
        self.refreshView()
    
    def reloadRepos(self):
        # Reload add-on list for all repositories.
        # TODO: This can take some time - show some progress to user!
        for repo in self.repositories:
            try:
                repo.refreshData(force=True)
            except Exception, e:  # Maybe gather all exceptions (for all repositories) and show them in the end?
                QMessageBox.critical(self, "Error", "Could not reload repository '%s': %s." % (repo.name, e))
        # Were any installation-pending add-ons removed from repositories?
        for (id, addOn) in self.addOnsToAdd.items():
            if id in addOn.repository.addOns:
                newObject = [version for version in addOn.repository.addOns[id] if version.version == addOn.version]
                if newObject != []:
                    self.addOnsToAdd[id] = newObject[0]
                    continue
            del self.addOnsToAdd[id]
            if id in self.addOnsToRemove:    # If true, it was a request for upgrade, not installation -- do not remove the installed version!
                del self.addOnsToRemove[id]
        # Finally, refresh the tree on GUI.
        self.refreshView()
            
    def upgradeCandidates(self):
        result = []
        import orngEnviron, orngAddOns
        for item in self.tree.addOnItems:
            id = item.newest.id
            if id.startswith("registered:"): continue
            installedAo = orngAddOns.installedAddOns[id] if id in orngAddOns.installedAddOns else None
            installed = installedAo.version if installedAo else None 
            selected = self.addOnsToAdd[id].version if id in self.addOnsToAdd else None
            if installed:
                if installedAo.directory.startswith(orngEnviron.addOnsDirUser):
                    if installed < item.newest.version:
                        if selected:
                            if selected >= item.newest.version:
                                continue
                        result.append(item.newest)
        return result
    
    def upgradeAll(self):
        for candidate in self.upgradeCandidates():
            self.upgrade(candidate, refresh=False)
        self.refreshInfoPane()
        self.enableDisableButtons()
        
    def upgrade(self, newAddOn=None, refresh=True):
        if not newAddOn:
            newAddOn = self.getAddOnFromItem(self.tree.currentItem())
        if not newAddOn:
            return
        import orngAddOns
        self.addOnsToRemove[newAddOn.id] = orngAddOns.installedAddOns[newAddOn.id]
        self.addOnsToAdd[newAddOn.id] = newAddOn
        if refresh:
            self.refreshInfoPane()
            self.enableDisableButtons()

    def registerAddOn(self):
        dir = str(QFileDialog.getExistingDirectory(self, "Select the folder that contains the add-on:"))
        if dir != "":
            if os.path.split(dir)[1] == "widgets":     # register a dir above the dir that contains the widget folder
                dir = os.path.split(dir)[0]
            if os.path.exists(os.path.join(dir, "widgets")):
                name = os.path.split(dir)[1]
                import orngAddOns
                id = "registered:"+dir
                self.addOnsToAdd[id] = orngAddOns.OrangeRegisteredAddOn(name, dir, systemWide=False)
                self.refreshView(id)
            else:
                QMessageBox.information( None, "Information", 'The specified folder does not seem to contain an Orange add-on.', QMessageBox.Ok + QMessageBox.Default)

    def openWebPage(self):
        addOn = self.getAddOnFromItem(self.tree.currentItem())
        if not addOn: return
        if not addOn.homePage: return
        import webbrowser
        webbrowser.open(addOn.homePage)
        
    def listWidgets(self):
        addOn = self.getAddOnFromItem(self.tree.currentItem())
        if not addOn: return
        import orngAddOns
        if addOn.__class__ is not orngAddOns.OrangeAddOnInRepo: return
        if not addOn.repository.hasWebScript: return
        self.canvasDlg.helpWindow.open("%s/addOnServer.py/%s/doc/widgets/" % (addOn.repository.url, addOn.fileName), modal=True)
        
        
    def donotUpgrade(self, newAddOn=None):
        if not newAddOn:
            newAddOn = self.getAddOnFromItem(self.tree.currentItem())
        if not newAddOn:
            return
        del self.addOnsToAdd[newAddOn.id]
        del self.addOnsToRemove[newAddOn.id]
        self.refreshInfoPane()
        
    def lineEditSearch(self, *args, **props):
        return OWGUI.lineEdit(*args, **props)

    def cbToggled(self, item, column):
        # Not a request from an add-on item in tree?
        if (column != 0) or "disableToggleSignal" not in item.__dict__:
            return
        # Toggle signal currently disabled?
        if item.disableToggleSignal:
            return
        
        addOn = item.newest
        id = addOn.id
        if item.checkState(0) == Qt.Checked:  # Mark for installation (or delete removal request)
            if id not in self.addOnsToAdd:
                if id in self.addOnsToRemove:
                    del self.addOnsToRemove[id]
                else:
                    self.addOnsToAdd[id] = addOn
        else:                                 # Mark for removal (or delete installation request)
            import orngAddOns, orngEnviron
            installedAo = orngAddOns.installedAddOns[id] if id in orngAddOns.installedAddOns else None 
            if installedAo:
                if not installedAo.directory.startswith(orngEnviron.addOnsDirUser):
                    item.disableToggleSignal = True
                    item.setCheckState(0, Qt.Checked)
                    item.disableToggleSignal = False
                    return
            if id in self.addOnsToAdd:
                del self.addOnsToAdd[id]
            elif id not in self.addOnsToRemove:
                import orngAddOns
                if id in orngAddOns.installedAddOns:
                    self.addOnsToRemove[id] = orngAddOns.installedAddOns[id]
                elif id.startswith("registered:"):
                    self.addOnsToRemove[id] = item.newest
        self.resetChecked(id)   # Refresh all checkboxes for this add-on (it might be in multiple repositories!)
        self.refreshInfoPane(item)
        
    def getRepoFromItem(self, item):
        if not item:
            return None
        import orngAddOns
        if hasattr(item, "repository"):
            return item.repository
        else:
            if item.newest.__class__ is not orngAddOns.OrangeAddOnInRepo:
                return None
            return  item.newest.repository
    
    def getAddOnFromItem(self, item):        
        if hasattr(item, "newest"):
            return item.newest
        return None

    def getAddOnIdFromItem(self, item):
        addOn = self.getAddOnFromItem(item)        
        return addOn.id if addOn else None
        
    def refreshInfoPane(self, item=None):
        if not item:
            item = self.tree.currentItem()
        import orngAddOns
        if hasattr(item, "newest"):
            if item.newest.__class__ is not orngAddOns.OrangeRegisteredAddOn:
                import orngAddOns, orngEnviron
                addOn = item.newest
                self.lblDescription.setText(addOn.description.strip() if addOn else "")
                self.lblVerAvailValue.setText(addOn.versionStr)
    
                addOnInstalled = orngAddOns.installedAddOns[addOn.id] if addOn.id in orngAddOns.installedAddOns else None
                addOnToInstall = self.addOnsToAdd[addOn.id] if addOn.id in self.addOnsToAdd else None
                addOnToRemove = self.addOnsToRemove[addOn.id] if addOn.id in self.addOnsToRemove else None
                
                self.lblVerInstalledValue.setText((addOnInstalled.versionStr+("" if addOnInstalled.directory.startswith(orngEnviron.addOnsDirUser) else " (installed system-wide)")) if addOnInstalled else "-")
                self.upgradeButton.setVisible(addOnInstalled!=None and addOnInstalled.version < addOn.version and addOnToInstall!=addOn and addOnInstalled.directory.startswith(orngEnviron.addOnsDirUser))
                self.donotUpgradeButton.setVisible(addOn.id in self.addOnsToRemove and addOnToInstall==addOn)
                self.webButton.setVisible(addOn.homePage != None)
                self.listWidgetsButton.setVisible(len(addOn.widgets) > 0 and addOn.__class__ is orngAddOns.OrangeAddOnInRepo and addOn.repository.hasWebScript)
                
                if addOnToInstall:
                    if addOnToRemove: self.lblStatus.setText("marked for upgrade")
                    else: self.lblStatus.setText("marked for installation")
                elif addOnToRemove: self.lblStatus.setText("marked for removal")
                else: self.lblStatus.setText("")
    
                self.infoPane.setVisible(True)
                self.regiInfoPane.setVisible(False)
            else:
                self.lblRegisteredAddOnInfo.setText("This add-on is registered "+("system-wide." if item.newest.systemWide else "by user."))
                self.infoPane.setVisible(False)
                self.regiInfoPane.setVisible(True)
        else:
            self.infoPane.setVisible(False)
            self.regiInfoPane.setVisible(False)
        
    def enableDisableButtons(self):
        repo = self.getRepoFromItem(self.tree.currentItem())
        import orngAddOns
        self.delRepoButton.setEnabled(repo.__class__ is not orngAddOns.OrangeDefaultAddOnRepository if repo!=None else False)
        self.editRepoButton.setEnabled(repo.__class__ is not orngAddOns.OrangeDefaultAddOnRepository if repo!=None else False)
        self.upgradeAllButton.setEnabled(self.upgradeCandidates() != [])
        
    def currentItemChanged(self, new, previous):
        # Enable/disable buttons
        self.enableDisableButtons()
            
        # Refresh info pane
        self.refreshInfoPane(new)
    
    def resetChecked(self, id):
        import orngAddOns
        value = id in orngAddOns.installedAddOns or id.startswith("registered:")
        value = value and id not in self.addOnsToRemove
        value = value or id in self.addOnsToAdd
        for treeItem in self.tree.addOnItems:
            if treeItem.newest.id == id:
                treeItem.disableToggleSignal = True
                treeItem.setCheckState(0,Qt.Checked if value else Qt.Unchecked);
                treeItem.disableToggleSignal = False

    def addAddOnsToTree(self, repoItem, addOnDict, insertToBeginning=False):
        # Transform dictionary {id->[versions]} list of tuples (newest,[otherVersions])
        if type(addOnDict) is list:
            addOnList = [(ao, []) for ao in addOnDict]
        else:
            addOnList = []
            for id in addOnDict:
                versions = list(addOnDict[id])  # We make a copy, so that we can change it!
                newest = versions[0]
                for v in versions:
                    if v.version > newest.version:
                        newest = v
                versions.remove(newest)
                addOnList.append( (newest, versions) )
        # Sort alphabetically
        addOnList.sort(key=lambda (newest, versions): newest.name)
        # Single-addon packages last
        if self.sortSingleLast:
            addOnList = [(n, v) for (n, v) in addOnList if not n.hasSingleWidget] \
                      + [(n, v) for (n, v) in addOnList if     n.hasSingleWidget]
        # Installed first
        if self.sortInstalledFirst and len(addOnList)>0 and "id" in addOnList[0][0].__dict__:
            import orngAddOns
            addOnList = [(n, v) for (n, v) in addOnList if     n.id in orngAddOns.installedAddOns] \
                      + [(n, v) for (n, v) in addOnList if not n.id in orngAddOns.installedAddOns]
        
        for (i, (newest, versions)) in enumerate(addOnList):
            addOnItem = QTreeWidgetItem(repoItem if not insertToBeginning else None)
            if insertToBeginning:
                if repoItem.__class__ is QTreeWidget:
                    repoItem.insertTopLevelItem(i, addOnItem)
                else:
                    repoItem.insertChild(i, addOnItem)
            addOnItem.disableToggleSignal = True
            addOnItem.setText(0, newest.name)
            if newest.hasSingleWidget():
                italFont = QFont(addOnItem.font(0))
                italFont.setItalic(True)
                addOnItem.setFont(0, italFont)
            addOnItem.setCheckState(0,Qt.Unchecked);
            addOnItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            addOnItem.newest = newest
            addOnItem.otherVersions = versions

            self.tree.addOnItems.append(addOnItem)
            repoItem.addOnItemsDict[newest.id] = addOnItem
            self.resetChecked(newest.id)
            addOnItem.disableToggleSignal = False
            

    def addRepositoryToTree(self, repo):
        repoItem = QTreeWidgetItem(self.tree)
        repoItem.repository = repo
        repoItem.addOnItemsDict = {}
        repoItem.setText(0, repo.name)
        boldFont = QFont(repoItem.font(0))
        boldFont.setWeight(QFont.Bold)
        repoItem.setFont(0, boldFont)
        self.tree.repoItems[repo] = repoItem
        
        addOnsToAdd = {}
        visibleAddOns = repo.searchIndex(self.searchStr)
        for (id, versions) in repo.addOns.items():
            if id in visibleAddOns:
                addOnsToAdd[id] = versions
        self.addAddOnsToTree(repoItem, addOnsToAdd)
        
        return repoItem
            

    def refreshView(self, selectedRegisteredAddOnId=None):
        # Save repository items expanded state
        expandedRepos = set([])
        for (repo, item) in self.tree.repoItems.items():
            if item.isExpanded():
                expandedRepos.add(repo)
        # Save current item selection
        selectedRepository = self.getRepoFromItem(self.tree.currentItem())
        selectedAddOnId = self.getAddOnIdFromItem(self.tree.currentItem())
        #TODO: Save the next repository selection too, in case the current one was deleted

        # Clear the tree
        self.tree.repoItems = {}
        self.tree.addOnItems = []
        self.tree.addOnItemsDict = {}
        self.tree.clear()
        
        # Set button visibility
        self.editRepoButton.setVisible(self.groupByRepo)
        self.delRepoButton.setVisible(self.groupByRepo)
        
        # Add repositories and add-ons
        shownAddOns = set([])
        if self.groupByRepo:
            for repo in self.repositories:
                item = self.addRepositoryToTree(repo)
                shownAddOns = shownAddOns.union(set(repo.addOns).intersection(repo.searchIndex(self.searchStr)))
        else:
            addOns = {}
            for repo in self.repositories:
                for addOnId in repo.addOns:
                    if addOnId in repo.searchIndex(self.searchStr):
                        if addOnId in addOns:
                            addOns[addOnId].extend(repo.addOns[addOnId])
                        else:
                            addOns[addOnId] = list(repo.addOns[addOnId])
            self.addAddOnsToTree(self.tree, addOns)
            shownAddOns = set(addOns)
        
        # Add add-ons that are not present in any repository
        if self.searchStr.strip() == "":   # but we do not need to search among installed add-ons
            import orngAddOns
            onlyInstalledAddOns = {}
            for addOn in orngAddOns.installedAddOns.values():
                if addOn.id not in shownAddOns:
                    onlyInstalledAddOns[addOn.id] = [addOn]
            self.addAddOnsToTree(self.tree, onlyInstalledAddOns, insertToBeginning=True)
            
        # Registered Add-ons
        if orngAddOns.registeredAddOns != [] or any([id.startswith("registered:") for id in self.addOnsToAdd]):
            regiItem = QTreeWidgetItem(self.tree)
            regiItem.repository = None
            regiItem.addOnItemsDict = {}
            regiItem.setText(0, "Registered Add-ons")
            boldFont = QFont(regiItem.font(0))
            boldFont.setWeight(QFont.Bold)
            regiItem.setFont(0, boldFont)
            self.tree.repoItems["Registered Add-ons"] = regiItem
            
            addOnsToAdd = []
            import re
            words = [word for word in re.split(orngAddOns.indexRE, self.searchStr.lower()) if word!=""]
            visibleAddOns = [ao for ao in orngAddOns.registeredAddOns+[ao for ao in self.addOnsToAdd.values() if ao.id.startswith("registered:")] if all([word in ao.name for word in words])]
            self.addAddOnsToTree(regiItem, visibleAddOns)
            if selectedRegisteredAddOnId:
                regiItem.setExpanded(True)
                self.tree.setCurrentItem(regiItem.addOnItemsDict[selectedRegisteredAddOnId])
            
        # Restore repository items expanded state
        if len(expandedRepos)==0:
            self.tree.expandItem(self.tree.topLevelItem(0))
        else:
            for (repo, item) in self.tree.repoItems.items():
                if repo in expandedRepos:
                    item.setExpanded(True)
                    
        # Restore item selection
        if not selectedRegisteredAddOnId:
            select = self.tree.topLevelItem(0)
            search = None
            if selectedRepository in self.tree.repoItems:
                select = self.tree.repoItems[selectedRepository]
                search = select
            elif not selectedRepository:
                search = self.tree
            if selectedAddOnId and search:
                if selectedAddOnId in search.addOnItemsDict:
                    select = search.addOnItemsDict[selectedAddOnId]
            self.tree.setCurrentItem(select)
            
    def searchCallback(self):
        self.refreshView()
    
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
#        self.setWindowFlags(Qt.Popup)       # Commented out, because it makes the window appear in the top-left corner on Linux
        self.setWindowTitle("About Orange")
        
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
        import orngAddOns
        
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
