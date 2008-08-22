# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    signal dialog, canvas options dialog

from PyQt4.QtCore import *
from PyQt4.QtGui import *
#from orngCanvasItems import *
import orngGui

# this class is needed by signalDialog to show widgets and lines
class SignalCanvasView(QGraphicsView):
    def __init__(self, dlg, *args):
        apply(QGraphicsView.__init__,(self,) + args)
        self.dlg = dlg
        self.bMouseDown = False
        self.bLineDragging = False
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
        outputs, inputs = outWidget.widget.getOutputs(), inWidget.widget.getInputs()
        outIconName, inIconName = outWidget.widget.getFullIconName(), inWidget.widget.getFullIconName()
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
        brush = QBrush(QColor(60,150,255))
        self.outWidget = QGraphicsRectItem(xWidgetOff, yWidgetOffTop, width, height, None, self.dlg.canvas)
        self.outWidget.setBrush(brush)
        self.outWidget.setZValue(-100)

        self.inWidget = QGraphicsRectItem(xWidgetOff + width + xSpaceBetweenWidgets, yWidgetOffTop, width, height, None, self.dlg.canvas)
        self.inWidget.setBrush(brush)
        self.inWidget.setZValue(-100)

        # if icons -> show them
        if outIconName:
            frame = QGraphicsPixmapItem(QPixmap(outWidget.imageFrame), None, self.dlg.canvas)
            frame.setPos(xWidgetOff + xIconOff, yWidgetOffTop + height/2.0 - frame.pixmap().width()/2.0)
            self.outWidgetIcon = QGraphicsPixmapItem(QPixmap(outIconName), None, self.dlg.canvas)
            self.outWidgetIcon.setPos(xWidgetOff + xIconOff, yWidgetOffTop + height/2.0 - self.outWidgetIcon.pixmap().width()/2.0)
        if inIconName :
            frame = QGraphicsPixmapItem(QPixmap(inWidget.imageFrame), None, self.dlg.canvas)
            frame.setPos(xWidgetOff + xSpaceBetweenWidgets + 2*width - xIconOff - frame.pixmap().width(), yWidgetOffTop + height/2.0 - frame.pixmap().width()/2.0)
            self.inWidgetIcon = QGraphicsPixmapItem(QPixmap(inIconName), None, self.dlg.canvas)
            self.inWidgetIcon.setPos(xWidgetOff + xSpaceBetweenWidgets + 2*width - xIconOff - self.inWidgetIcon.pixmap().width(), yWidgetOffTop + height/2.0 - self.inWidgetIcon.pixmap().width()/2.0)

        # show signal boxes and text labels
        #signalSpace = (count)*ySignalSpace
        signalSpace = height
        for i in range(len(outputs)):
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(outputs)+1)
            box = QGraphicsRectItem(xWidgetOff + width, y - ySignalSize/2.0, xSignalSize, ySignalSize, None, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.setZValue(200)
            self.outBoxes.append((outputs[i].name, box))

            self.texts.append(orngGui.MyCanvasText(self.dlg.canvas, outputs[i].name, xWidgetOff + width - 5, y - 7, Qt.AlignRight | Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(orngGui.MyCanvasText(self.dlg.canvas, outputs[i].type, xWidgetOff + width - 5, y + 7, Qt.AlignRight | Qt.AlignVCenter, bold =0, show=1))

        for i in range(len(inputs)):
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(inputs)+1)
            box = QGraphicsRectItem(xWidgetOff + width + xSpaceBetweenWidgets - xSignalSize, y - ySignalSize/2.0, xSignalSize, ySignalSize, None, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.setZValue(200)
            self.inBoxes.append((inputs[i].name, box))

            self.texts.append(orngGui.MyCanvasText(self.dlg.canvas, inputs[i].name, xWidgetOff + width + xSpaceBetweenWidgets + 5, y - 7, Qt.AlignLeft | Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(orngGui.MyCanvasText(self.dlg.canvas, inputs[i].type, xWidgetOff + width + xSpaceBetweenWidgets + 5, y + 7, Qt.AlignLeft | Qt.AlignVCenter, bold =0, show=1))

        self.texts.append(orngGui.MyCanvasText(self.dlg.canvas, outWidget.caption, xWidgetOff + width/2.0, yWidgetOffTop + height + 5, Qt.AlignHCenter | Qt.AlignTop, bold =1, show=1))
        self.texts.append(orngGui.MyCanvasText(self.dlg.canvas, inWidget.caption, xWidgetOff + width* 1.5 + xSpaceBetweenWidgets, yWidgetOffTop + height + 5, Qt.AlignHCenter | Qt.AlignTop, bold =1, show=1))

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
            self.bLineDragging = 1
            self.tempLine = QGraphicsLineItem(None, self.dlg.canvas)
            self.tempLine.setLine(point.x(), point.y(), point.x(), point.y())
            self.tempLine.setPen(QPen(QColor(0,255,0), 1))
            self.tempLine.setZValue(-300)
            return
        elif type(activeItem) == QGraphicsLineItem:
            for (line, outName, inName, outBox, inBox) in self.lines:
                if line == activeItem:
                    self.dlg.removeLink(outName, inName)
                    return

    # ###################################################################
    # mouse button was released #########################################
    def mouseMoveEvent(self, ev):
        if self.bLineDragging:
            curr = self.mapToScene(ev.pos())
            start = self.tempLine.line().p1()
            self.tempLine.setLine(start.x(), start.y(), curr.x(), curr.y())
            self.scene().update()

    # ###################################################################
    # mouse button was released #########################################
    def mouseReleaseEvent(self, ev):
        if self.bLineDragging:
            self.bLineDragging = 0
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
        line.setPen(QPen(QColor(0,255,0), 6))
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

        self.canvasGroup = orngGui.widgetBox(self, 1)
        self.canvas = QGraphicsScene(0,0,1000,1000)
        self.canvasView = SignalCanvasView(self, self.canvas, self.canvasGroup)
        self.canvasGroup.layout().addWidget(self.canvasView)

        buttons = orngGui.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.buttonHelp = orngGui.button(buttons, self, "&Help")
        buttons.layout().addStretch(1)
        self.buttonClearAll = orngGui.button(buttons, self, "Clear &All", callback = self.clearAll)
        self.buttonOk = orngGui.button(buttons, self, "&OK", callback = self.accept)
        self.buttonOk.setAutoDefault(1)
        self.buttonOk.setDefault(1)
        self.buttonCancel = orngGui.button(buttons, self, "&Cancel", callback = self.reject)

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

        minorInputs = self.inWidget.widget.getMinorInputs()
        minorOutputs = self.outWidget.widget.getMinorOutputs()
        majorInputs = self.inWidget.widget.getMajorInputs()
        majorOutputs = self.outWidget.widget.getMajorOutputs()

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
        inputs = self.inWidget.widget.getInputs()
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
        self.setWindowTitle("Canvas Options")
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setSpacing(0)
        self.resize(500,500)

        self.removeTabs = []

        self.tabs = QTabWidget(self)
        GeneralTab = orngGui.widgetBox(self.tabs, removeMargin = 0)
        ExceptionsTab = orngGui.widgetBox(self.tabs, removeMargin = 0)
        TabOrderTab = orngGui.widgetBox(self.tabs, removeMargin = 0)

        self.tabs.addTab(GeneralTab, "General")
        self.tabs.addTab(ExceptionsTab, "Exception handling")
        self.tabs.addTab(TabOrderTab, "Widget tab order")

        # #################################################################
        # GENERAL TAB
        generalBox = orngGui.widgetBox(GeneralTab, "General Options")
        self.snapToGridCB = orngGui.checkBox(generalBox, "Snap widgets to grid")
        self.writeLogFileCB  = orngGui.checkBox(generalBox, "Write content of Output window to log file")
        self.showSignalNamesCB = orngGui.checkBox(generalBox, "Show signal names between widgets")
        self.dontAskBeforeCloseCB= orngGui.checkBox(generalBox, "Don't ask to save schema before closing")
        #self.autoSaveSchemasOnCloseCB = orngGui.checkBox(generalBox, "Automatically save temporary schemas on close")
        self.saveWidgetsPositionCB = orngGui.checkBox(generalBox, "Save size and position of widgets")
        self.useContextsCB = orngGui.checkBox(generalBox, "Use context settings")

        validator = QIntValidator(self)
        validator.setRange(0,10000)

#        canvasSizeBox = orngGui.widgetBox(GeneralTab, "Default Size of Orange Canvas")
#        self.widthEdit = orngGui.lineEdit(canvasSizeBox, "Width:  ", orientation='horizontal', validator = validator )
#        self.heightEdit = orngGui.lineEdit(canvasSizeBox, "Height: ", orientation='horizontal', validator = validator)
#
#        stylesBox = orngGui.widgetBox(GeneralTab, "Styles")

        hbox = orngGui.widgetBox(GeneralTab, orientation = "horizontal")
        sizeBox = orngGui.widgetBox(hbox, "Orange Canvas Size")
        looksBox = orngGui.widgetBox(hbox, "Looks")
        self.widthEdit = orngGui.lineEdit(sizeBox, "Canvas width:  ", orientation='horizontal', validator = validator)
        self.heightEdit = orngGui.lineEdit(sizeBox, "Canvas height: ", orientation='horizontal', validator = validator)
        self.stylesCombo = orngGui.comboBox(looksBox, label = "Style:", orientation = "horizontal", items = [str(n) for n in QStyleFactory.keys()])
        self.stylesPalette = orngGui.checkBox(looksBox, "Use style's standard palette")
        
        colorsBox = orngGui.widgetBox(GeneralTab, "Set Colors")
        if canvasDlg:
            selectedWidgetBox = orngGui.widgetBox(colorsBox, orientation = "horizontal")
            self.selectedWidgetIcon = ColorIcon(selectedWidgetBox, canvasDlg.widgetSelectedColor)
            selectedWidgetBox.layout().addWidget(self.selectedWidgetIcon)
            selectedWidgetLabel = orngGui.widgetLabel(selectedWidgetBox, " Selected widget")

            activeWidgetBox = orngGui.widgetBox(colorsBox, orientation = "horizontal")
            self.activeWidgetIcon = ColorIcon(activeWidgetBox, canvasDlg.widgetActiveColor)
            activeWidgetBox.layout().addWidget(self.activeWidgetIcon)
            selectedWidgetLabel = orngGui.widgetLabel(activeWidgetBox, " Active widget")

            lineBox = orngGui.widgetBox(colorsBox, orientation = "horizontal")
            self.lineIcon = ColorIcon(lineBox, canvasDlg.lineColor)
            lineBox.layout().addWidget(self.lineIcon)
            selectedWidgetLabel = orngGui.widgetLabel(lineBox, " Lines")
        GeneralTab.layout().addStretch(1)

        # #################################################################
        # EXCEPTION TAB
        exceptions = orngGui.widgetBox(ExceptionsTab, "Exceptions")
        #self.catchExceptionCB = QCheckBox('Catch exceptions', exceptions)
        self.focusOnCatchExceptionCB = orngGui.checkBox(exceptions, 'Focus output window on catch')
        self.printExceptionInStatusBarCB = orngGui.checkBox(exceptions, 'Print last exception in status bar')

        output = orngGui.widgetBox(ExceptionsTab, "System output")
        #self.catchOutputCB = QCheckBox('Catch system output', output)
        self.focusOnCatchOutputCB = orngGui.checkBox(output, 'Focus output window on system output')
        self.printOutputInStatusBarCB = orngGui.checkBox(output, 'Print last system output in status bar')

        hboxExc = orngGui.widgetBox(ExceptionsTab, orientation="horizontal")
        outputCanvas = orngGui.widgetBox(hboxExc, "Canvas Info Handling")
        outputWidgets = orngGui.widgetBox(hboxExc, "Widget Info Handling")
        self.ocShow = orngGui.checkBox(outputCanvas, 'Show icon above widget for...')
        self.ocInfo = orngGui.checkBox(outputCanvas, 'Information', indent = 10)
        self.ocWarning = orngGui.checkBox(outputCanvas, 'Warnings', indent = 10)
        self.ocError = orngGui.checkBox(outputCanvas, 'Errors', indent = 10)

        self.owShow = orngGui.checkBox(outputWidgets, 'Show statusbar info for...')
        self.owInfo = orngGui.checkBox(outputWidgets, 'Information', indent = 10)
        self.owWarning = orngGui.checkBox(outputWidgets, 'Warnings', indent = 10)
        self.owError = orngGui.checkBox(outputWidgets, 'Errors', indent = 10)

        verbosityBox = orngGui.widgetBox(ExceptionsTab, "Verbosity", orientation = "horizontal")
        verbosityBox.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.verbosityCombo = orngGui.comboBox(verbosityBox, label = "Set level of widget output: ", orientation='horizontal', items=["Small", "Medium", "High"])
        ExceptionsTab.layout().addStretch(1)

        # #################################################################
        # TAB ORDER TAB
        tabOrderBox = orngGui.widgetBox(TabOrderTab, "Set Order of Widget Categories", orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.tabOrderList = QListWidget(tabOrderBox)
        self.tabOrderList.setAcceptDrops(True)

        tabOrderBox.layout().addWidget(self.tabOrderList)
        self.tabOrderList.setSelectionMode(QListWidget.SingleSelection)

        w = orngGui.widgetBox(tabOrderBox, sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.upButton = orngGui.button(w, self, "Up", callback = self.moveUp)
        self.downButton = orngGui.button(w, self, "Down", callback = self.moveDown)
        w.layout().addSpacing(20)
        self.removeButton = orngGui.button(w, self, "Remove", callback = self.removeCategory)
        self.removeButton.setEnabled(0)
        w.layout().addStretch(1)

        # OK, Cancel buttons
        hbox = orngGui.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        hbox.layout().addStretch(1)
        self.okButton = orngGui.button(hbox, self, "OK", callback = self.accept)
        self.cancelButton = orngGui.button(hbox, self, "Cancel", callback = self.reject)
        self.connect(self.tabOrderList, SIGNAL("currentRowChanged(int)"), self.enableDisableButtons)

        self.topLayout.addWidget(self.tabs)
        self.topLayout.addWidget(hbox)


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
        self.removeButton.setEnabled(not self.canvasDlg.tabs.tabDict[str(self.tabOrderList.item(self.tabOrderList.currentRow()).text())].builtIn)

    def removeCategory(self):
        curCat = str(self.tabOrderList.item(self.tabOrderList.currentRow()).text())
        if QMessageBox.warning(self,'Orange Canvas', "Unregister widget category '%s' from Orange canvas? This will not remove any files." % curCat, QMessageBox.Ok , QMessageBox.Cancel | QMessageBox.Default | QMessageBox.Escape) == QMessageBox.Yes:
            self.removeTabs.append(curCat)
            self.tabOrderList.takeItem(self.tabOrderList.currentRow())


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
        if assigned == self:
            return

        if assigned and QMessageBox.question(self, "Confirmation", "'%(pressed)s' is already assigned to '%(assigned)s'. Override?" % {"pressed": pressed, "assigned": assigned.widget.nameKey}, QMessageBox.Yes | QMessageBox.Default, QMessageBox.No | QMessageBox.Escape) == QMessageBox.No:
            return

        self.setText(pressed)
        self.invdict[self.widget] = pressed
        self.invInvDict[pressed] = self
        if assigned:
            assigned.setText("<none>")
            del self.invdict[assigned.widget]

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
        
        for tabName, show in canvasDlg.settings["WidgetTabs"]:
            scrollArea = QScrollArea()
            self.tabs.addTab(scrollArea, tabName)
            #scrollArea.setWidgetResizable(1)       # you have to use this or set size to wtab manually - otherwise nothing gets shown

            wtab = QWidget(self.tabs)
            scrollArea.setWidget(wtab)

            tabWidgets = canvasDlg.tabs.tabDict[tabName].widgets
            widgets = filter(lambda x:x.__class__ == orngTabs.WidgetButton, tabWidgets)
            rows = (len(widgets)+2) / 3
            layout = QGridLayout(wtab)

            for i, w in enumerate(widgets):
                x = i / rows
                y = i % rows

                hlayout = QHBoxLayout()
                mainBox = QWidget(wtab)
                mainBox.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
                mainBox.setLayout(hlayout)
                layout.addWidget(mainBox, y, x, Qt.AlignTop | Qt.AlignLeft)
                label = QLabel(wtab)
                label.setPixmap(w.pixmapWidget.pixmap())
                hlayout.addWidget(label)

                optionsw = QWidget(self)
                optionsw.setLayout(QVBoxLayout())
                hlayout.addWidget(optionsw)
                optionsw.layout().addStretch(1)

                orngGui.widgetLabel(optionsw, w.name)
                key = self.invDict.get(w, "<none>")
                le = KeyEdit(optionsw, key, self.invDict, w, invInvDict)
                optionsw.layout().addWidget(le)
                invInvDict[key] = le
                le.setFixedWidth(60)

            wtab.resize(wtab.sizeHint())

        # OK, Cancel buttons
        hbox = orngGui.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        hbox.layout().addStretch(1)
        self.okButton = orngGui.button(hbox, self, "OK", callback = self.accept)
        self.cancelButton = orngGui.button(hbox, self, "Cancel", callback = self.reject)
        self.okButton.setDefault(True)

        self.layout().addWidget(self.tabs)
        self.layout().addWidget(hbox)


# #######################################
# # Preferences dialog - preferences for signals
# #######################################
class PreferencesDlg(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setSpacing(10)
        self.grid = QGridLayout( 5, 3 )
        self.topLayout.addLayout( self.grid, 10 )

        groupBox  = QGroupBox(self, "Channel_settings")
        groupBox.setTitle("Channel settings")
        self.grid.addWidget(groupBox, 1,1)
        propGrid = QGridLayout(groupBox, 4, 2 )

        cap0 = QLabel("Symbolic channel names:", self)
        cap1 = QLabel("Full name:", groupBox)
        cap2 = QLabel("Priority:", groupBox)
        cap3 = QLabel("Color:", groupBox)
        self.editFullName = QLineEdit(groupBox)
        self.editPriority = QComboBox( False, groupBox, "priority" )
        self.editColor    = QComboBox( False, groupBox, "color" )
        #self.connect( self.editPriority, SIGNAL("activated(int)"), self.comboValueChanged )
        #self.connect( self.editColor, SIGNAL("activated(int)"), self.comboValueChanged )

        propGrid.addWidget(cap1, 0,0, Qt.AlignVCenter|Qt.AlignHCenter)
        propGrid.addWidget(cap2, 1,0, Qt.AlignVCenter|Qt.AlignHCenter)
        propGrid.addWidget(cap3, 2,0, Qt.AlignVCenter|Qt.AlignHCenter)
        propGrid.addWidget(self.editFullName, 0,1, Qt.AlignVCenter)
        propGrid.addWidget(self.editPriority, 1,1, Qt.AlignVCenter)
        propGrid.addWidget(self.editColor, 2,1, Qt.AlignVCenter)

        groupBox.setMinimumSize(180,150)
        groupBox.setMaximumSize(180,150)

        saveButton = QPushButton("Save Changes", groupBox)
        addButton = QPushButton("Add New Channel Name", self)
        removeButton = QPushButton("Remove Selected Name", self)
        closeButton = QPushButton("Close",self)
        self.channelList = QListWidget(self)
        self.channelList.setMinimumHeight(200)
        self.connect( self.channelList, SIGNAL("highlighted(int)"), self.listItemChanged )

        self.grid.addWidget(cap0,0,0, Qt.AlignLeft|Qt.AlignBottom)
        self.grid.addWidget(addButton, 2,1)
        self.grid.addWidget(removeButton, 3,1)
        self.grid.addMultiCellWidget(self.channelList, 1,5,0,0)
        self.grid.addWidget(closeButton, 4,1)
        propGrid.addMultiCellWidget(saveButton, 3,3,0,1)

        saveButton.show()
        addButton.show()
        removeButton.show()
        self.channelList.show()
        closeButton.show()
        self.connect(saveButton, SIGNAL("clicked()"),self.saveChanges)
        self.connect(addButton , SIGNAL("clicked()"),self.addNewSignal)
        self.connect(removeButton, SIGNAL("clicked()"),self.removeSignal)
        self.connect(closeButton, SIGNAL("clicked()"),self.closeClicked)
        self.topLayout.activate()

        self.editColor.addItems(["black", "darkGray", "gray", "lightGray", "red", "green", "blue", "cyan", "magenta", "yellow", "darkRed", "darkGreen", "darkBlue", "darkCyan" , "darkMagenta", "darkYellow"])

        for i in range(20):
            self.editPriority.addItem(str(i+1))

        self.channels = {}
        if self.canvasDlg.settings.has_key("Channels"):
            self.channels = self.canvasDlg.settings["Channels"]

        self.reloadList()

    def listItemChanged(self, index):
        name = str(self.channelList.item(index).text())
        value = self.channels[name]
        items = value.split("::")
        self.editFullName.setText(items[0])

        for i in range(self.editPriority.count()):
            if (str(self.editPriority.text(i)) == items[1]):
                self.editPriority.setCurrentIndex(i)

        for i in range(self.editColor.count()):
            if (str(self.editColor.text(i)) == items[2]):
                self.editColor.setCurrentIndex(i)

    def reloadList(self):
        self.channelList.clear()
        for (key,value) in self.channels.items():
            self.channelList.addItem(key)

    def saveChanges(self):
        index = self.channelList.currentItem()
        if index != -1:
            name = str(self.channelList.item(index).text())
            self.channels[name] = str(self.editFullName.text()) + "::" + str(self.editPriority.currentText()) + "::" + str(self.editColor.currentText())

    def addNewSignal(self):
        (Qstring,ok) = QInputDialog.getText(self, "Add New Channel Name", "Enter new symbolic channel name")
        string = str(Qstring)
        if ok:
            self.editColor.setCurrentIndex(0)
            self.editPriority.setCurrentIndex(0)
            self.editFullName.setText(string)
            self.channels[string] = str(self.editFullName.text()) + "::" + str(self.editPriority.currentText()) + "::" + str(self.editColor.currentText())
            self.reloadList()
            self.selectItem(string)

    def selectItem(self, string):
        for i in range(self.channelList.count()):
            temp = str(self.channelList.item(i).text())
            if temp == string:
                self.channelList.setCurrentIndex(i)
                return

    def removeSignal(self):
        index = self.channelList.currentItem()
        if index != -1:
            tempDict = {}
            symbName = str(self.channelList.item(index).text())

            for key in self.channels.keys():
                if key != symbName:
                    tempDict[key] = self.channels[key]
            self.channels = dict(tempDict)

        self.reloadList()

    def closeClicked(self):
        self.canvasDlg.settings["Channels"] = self.channels
        self.accept()
        return


class saveApplicationDlg(QDialog):
    def __init__(self, *args):
        apply(QDialog.__init__,(self,) + args)
        self.setWindowTitle("Set Widget Order")
        self.setLayout(QVBoxLayout())

        listbox = orngGui.widgetBox(self, 1, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.tabOrderList = QListWidget(listbox)
        self.tabOrderList.setSelectionMode(QListWidget.SingleSelection)
        listbox.layout().addWidget(self.tabOrderList)

        w = orngGui.widgetBox(listbox, sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.upButton = orngGui.button(w, self, "Up", callback = self.moveUp)
        self.downButton = orngGui.button(w, self, "Down", callback = self.moveDown)
        w.layout().addStretch(1)

        hbox = orngGui.widgetBox(self, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        orngGui.button(hbox, self, "Add Separator", callback = self.insertSeparator)
        hbox.layout().addStretch(1)
        orngGui.button(hbox, self, "&OK", callback = self.accept)
        orngGui.button(hbox, self, "&Cancel", callback = self.reject)

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
    dlg = saveApplicationDlg(None)
    dlg.show()
    sys.exit(app.exec_())

