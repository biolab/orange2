# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    signal dialog, canvas options dialog

from qt import *
from qtcanvas import *
from copy import copy
from string import strip
import sys
from orngCanvasItems import *
from qttable import *
#from orngSignalManager import ExampleTable, ExampleTableWithClass
#from orngSignalManager import *

TRUE  = 1
FALSE = 0

class QCanvasIcon(QCanvasRectangle):
    def __init__(self, canvas, fileName):
        QCanvasRectangle.__init__(self,canvas)
        self.pixmap = QPixmap(fileName)
        self.setZ(100)

    def setCenterPos(self, x, y):
        self.x = x - self.pixmap.width()/2.0
        self.y = y - self.pixmap.height()/2.0

    def drawShape(self, painter):
        if self.pixmap:
            painter.drawPixmap(self.x, self.y, self.pixmap)

# this class is needed by signalDialog to show widgets and lines
class SignalCanvasView(QCanvasView):
    def __init__(self, dlg, *args):
        apply(QCanvasView.__init__,(self,) + args)
        self.dlg = dlg
        self.bMouseDown = FALSE
        self.bLineDragging = FALSE
        self.tempLine = None
        self.inWidget = None
        self.outWidget = None
        self.inWidgetIcon = None
        self.outWidgetIcon = None
        self.lines = []
        self.outBoxes = []
        self.inBoxes = []
        self.texts = []

        #self.connect(self, SIGNAL("contentsMoving(int,int)"), self.contentsMoving)

    def addSignalList(self, outName, inName, outputs, inputs, outIconName, inIconName):
        items = self.canvas().allItems()
        for item in items: item.hide()
        self.lines = []; self.outBoxes = []; self.inBoxes = []; self.texts = []
        xSpaceBetweenWidgets = 100  # space between widgets
        xWidgetOff = 10     # offset for widget position
        yWidgetOffTop = 10     # offset for widget position
        yWidgetOffBottom = 30     # offset for widget position
        ySignalOff = 10     # space between the top of the widget and first signal
        ySignalSpace = 50   # space between two neighbouring signals
        ySignalSize = 20    # height of the signal box
        xSignalSize = 20    # width of the signal box
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
        self.outWidget = QCanvasRectangle(xWidgetOff, yWidgetOffTop, width, height, self.dlg.canvas)
        self.outWidget.setBrush(brush)
        self.outWidget.setZ(-100)
        self.outWidget.show()

        self.inWidget = QCanvasRectangle(xWidgetOff + width + xSpaceBetweenWidgets, yWidgetOffTop, width, height, self.dlg.canvas)
        self.inWidget.setBrush(brush)
        self.inWidget.setZ(-100)
        self.inWidget.show()

        # if icons -> show them
        if outIconName:
            self.outWidgetIcon = QCanvasIcon(self.dlg.canvas, outIconName)
            self.outWidgetIcon.setCenterPos(xWidgetOff + xIconOff + self.outWidgetIcon.pixmap.width()/2.0, yWidgetOffTop + height/2.0)
            self.outWidgetIcon.show()
        if inIconName :
            self.inWidgetIcon = QCanvasIcon(self.dlg.canvas, inIconName)
            self.inWidgetIcon.setCenterPos(xWidgetOff + xSpaceBetweenWidgets + 2*width - xIconOff - self.inWidgetIcon.pixmap.width()/2.0, yWidgetOffTop + height/2.0)
            self.inWidgetIcon.show()

        # show signal boxes and text labels
        #signalSpace = (count)*ySignalSpace
        signalSpace = height
        for i in range(len(outputs)):
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(outputs)+1)
            box = QCanvasRectangle(xWidgetOff + width, y - ySignalSize/2.0, xSignalSize, ySignalSize, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.show()
            self.outBoxes.append((outputs[i].name, box))

            self.texts.append(MyCanvasText(self.dlg.canvas, outputs[i].name, xWidgetOff + width - 5, y - 7, Qt.AlignRight + Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(MyCanvasText(self.dlg.canvas, outputs[i].type, xWidgetOff + width - 5, y + 7, Qt.AlignRight + Qt.AlignVCenter, bold =0, show=1))

        for i in range(len(inputs)):
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(inputs)+1)
            box = QCanvasRectangle(xWidgetOff + width + xSpaceBetweenWidgets - xSignalSize, y - ySignalSize/2.0, xSignalSize, ySignalSize, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.show()
            self.inBoxes.append((inputs[i].name, box))

            self.texts.append(MyCanvasText(self.dlg.canvas, inputs[i].name, xWidgetOff + width + xSpaceBetweenWidgets + 5, y - 7, Qt.AlignLeft + Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(MyCanvasText(self.dlg.canvas, inputs[i].type, xWidgetOff + width + xSpaceBetweenWidgets + 5, y + 7, Qt.AlignLeft + Qt.AlignVCenter, bold =0, show=1))

        self.texts.append(MyCanvasText(self.dlg.canvas, outName, xWidgetOff + width/2.0, yWidgetOffTop + height + 5, Qt.AlignHCenter + Qt.AlignTop, bold =1, show=1))
        self.texts.append(MyCanvasText(self.dlg.canvas, inName, xWidgetOff + width* 1.5 + xSpaceBetweenWidgets, yWidgetOffTop + height + 5, Qt.AlignHCenter + Qt.AlignTop, bold =1, show=1))
                
        return (2*xWidgetOff + 2*width + xSpaceBetweenWidgets, yWidgetOffTop + height + yWidgetOffBottom)

    def getTextWidth(self, text, bold = 0):
        temp = QCanvasText(text, self.dlg.canvas)
        if bold:
            font = temp.font()
            font.setBold(1)
            temp.setFont(font)
        rect = temp.boundingRect()
        return rect.width()

    # ###################################################################
    # mouse button was pressed
    def contentsMousePressEvent(self, ev):
        self.bMouseDown = 1
        activeItems = self.canvas().collisions(QRect(ev.pos().x()-1, ev.pos().y()-1,2,2))
       
        if activeItems == []: return
        box = self.findItem(activeItems, QCanvasRectangle)
        if box and box != self.outWidget and box != self.inWidget:
            self.bLineDragging = 1
            self.tempLine = QCanvasLine(self.dlg.canvas)
            self.tempLine.setPoints(ev.pos().x(), ev.pos().y(), ev.pos().x(), ev.pos().y())
            self.tempLine.setPen(QPen(QColor(0,255,0), 1))
            self.tempLine.setZ(-120)
            self.tempLine.show()
            return
        
        line = self.findItem(activeItems, QCanvasLine)
        if line:
            for (Line, outName, inName, outBox, inBox) in self.lines:
                if Line == line:
                    self.dlg.removeLink(outName, inName)
                    return
        
    # ###################################################################
    # mouse button was released #########################################
    def contentsMouseMoveEvent(self, ev):
        if self.bLineDragging:
            start = self.tempLine.startPoint()
            self.tempLine.setPoints(start.x(), start.y(), ev.pos().x(), ev.pos().y())
            self.canvas().update()

    # ###################################################################
    # mouse button was released #########################################
    def contentsMouseReleaseEvent(self, ev):
        if self.bLineDragging:
            self.bLineDragging = 0
            activeItems = self.canvas().collisions(QRect(ev.pos().x()-1, ev.pos().y()-1,2,2))

            box = self.findItem(activeItems, QCanvasRectangle)
            if box:
                startItems = self.canvas().collisions(QRect(self.tempLine.startPoint().x()-1, self.tempLine.startPoint().y()-1,2,2))
                box2 = self.findItem(startItems, QCanvasRectangle)
                if box.x() < box2.x(): outBox = box; inBox = box2
                else:                  outBox = box2; inBox = box
                outName = None; inName = None
                for (name, box) in self.outBoxes:
                    if box == outBox: outName = name
                for (name, box) in self.inBoxes:
                    if box == inBox: inName = name
                if outName != None and inName != None: self.dlg.addLink(outName, inName)
            
            self.tempLine.hide()
            self.tempLine.setCanvas(None)
            self.canvas().update()
        

    def findItem(self, items, wantedType):
        for item in items:
            if isinstance(item, wantedType): return item
        return None

    def addLink(self, outName, inName):
        outBox = None; inBox = None
        for (name, box) in self.outBoxes:
            if name == outName: outBox = box
        for (name, box) in self.inBoxes:
            if name == inName : inBox  = box
        if outBox == None or inBox == None:
            print "error adding link. Data = ", outName, inName
            return
        line = QCanvasLine(self.dlg.canvas)
        line.setPoints(outBox.x() + outBox.width()-2, outBox.y() + outBox.height()/2.0, inBox.x()+2, inBox.y() + inBox.height()/2.0)
        line.setPen(QPen(QColor(0,255,0), 6))
        line.setZ(-120)
        line.show()
        self.canvas().update()
        self.lines.append((line, outName, inName, outBox, inBox))
        

    def removeLink(self, outName, inName):
        for (line, outN, inN, outBox, inBox) in self.lines:
            if outN == outName and inN == inName:
                line.hide()
                line.setCanvas(None)
                self.lines.remove((line, outN, inN, outBox, inBox))
                self.canvas().update()
                return


# #######################################
# # Signal dialog - let the user select active signals between two widgets
# #######################################
class SignalDialog(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.topLayout = QVBoxLayout( self, 10 )
        self.signals = []
        self._links = []

        # GUI
        self.resize(515,286)
        self.setName('Qt Connect Signals')
        self.setCaption(self.tr("Qt Connect Signals"))

        self.grid = QGridLayout( 2, 1 )
        self.topLayout.addLayout( self.grid, 10 )

        self.canvasGroup = QHGroupBox("", self)
        self.canvas = QCanvas(1000,1000)
        self.canvasView = SignalCanvasView(self, self.canvas, self.canvasGroup) 
        self.grid.addWidget(self.canvasGroup, 1,1)
        

        LayoutWidget = QWidget(self,'Layout1')
        LayoutWidget.setGeometry(QRect(20,240,476,33))
        self.grid.addWidget(LayoutWidget, 2,1)
        Layout1 = QHBoxLayout(LayoutWidget)
        Layout1.setSpacing(6)
        Layout1.setMargin(0)

        self.buttonHelp = QPushButton("&Help", LayoutWidget)
        self.buttonHelp.setAutoDefault(1)
        Layout1.addWidget(self.buttonHelp)
        spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout1.addItem(spacer)

        self.buttonClearAll = QPushButton("Clear &All", LayoutWidget)
        Layout1.addWidget(self.buttonClearAll)

        self.buttonOk = QPushButton("&OK", LayoutWidget)
        self.buttonOk.setAutoDefault(1)
        self.buttonOk.setDefault(1)
        Layout1.addWidget(self.buttonOk)

        self.buttonCancel = QPushButton("&Cancel", LayoutWidget)
        self.buttonCancel.setAutoDefault(1)
        Layout1.addWidget(self.buttonCancel)

        self.connect(self.buttonClearAll,SIGNAL('clicked()'),self.clearAll)
        self.connect(self.buttonOk,SIGNAL('clicked()'),self,SLOT('accept()'))
        self.connect(self.buttonCancel,SIGNAL('clicked()'),self,SLOT('reject()'))

    def clearAll(self):
        while self._links != []:
            self.removeLink(self._links[0][0], self._links[0][1])

    def setOutInWidgets(self, outWidget, inWidget):
        self.outWidget = outWidget
        self.inWidget = inWidget
        (width, height) = self.canvasView.addSignalList(outWidget.caption, inWidget.caption, outWidget.widget.getOutputs(), inWidget.widget.getInputs(), outWidget.widget.getFullIconName(), inWidget.widget.getFullIconName())
        self.canvas.resize(width, height)
        self.resize(width+55, height+90)

    def countCompatibleConnections(self, outputs, inputs, outInstance, inInstance, outType, inType):
        count = 0
        for outS in outputs:
            if outInstance.getOutputType(outS.name) == None: continue  # ignore if some signals don't exist any more, since we will dispatch refresh registry somwhere else
            if not issubclass(outInstance.getOutputType(outS.name), outType): continue
            for inS in inputs:
                if inInstance.getOutputType(inS.name) == None: continue  # ignore if some signals don't exist any more, since we will dispatch refresh registry somwhere else
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


    def getPossibleConnections(self, outputs, inputs, outConnected, inConnected):
        possibleLinks = []
        canConnect = 0
        sameType = 0
        for outS in outputs:
            outType = self.outWidget.instance.getOutputType(outS.name)
            if outType == None:
                print "Unable to find signal type for signal %s. Check the definition of the widget." % (outS.name)
                return None                                         # report error
            for inS in inputs:
                inType = self.inWidget.instance.getInputType(inS.name)
                if inType == None:
                    print "Unable to find signal type for signal %s. Check the definition of the widget." % (inS.name)
                    return None                                        # report error
                if issubclass(outType, inType):
                    canConnect = 1
                    if (outS.name in outConnected) or (inS.name in inConnected): continue
                    if issubclass(inType, outType):
                        possibleLinks.insert(sameType, (outS.name, inS.name))
                        sameType += 1
                    else:   possibleLinks.append((outS.name, inS.name))
        return possibleLinks, canConnect, sameType

    def addDefaultLinks(self):
        canConnect = 0
        addedInLinks = []
        addedOutLinks = []
        self.multiplePossibleConnections = 0    # can we connect some signal with more than one widget

        allInputs = self.inWidget.widget.getInputs()
        allOutputs = self.outWidget.widget.getOutputs()
        minorInputs = self.inWidget.widget.getMinorInputs()
        minorOutputs = self.outWidget.widget.getMinorOutputs()
        majorInputs = []
        majorOutputs = []
        for s in allInputs:
            if s not in minorInputs: majorInputs.append(s)

        for s in allOutputs:
            if s not in minorOutputs: majorOutputs.append(s)

        inConnected = self.inWidget.getInConnectedSignalNames()
        outConnected = self.outWidget.getOutConnectedSignalNames()

        """
        # rebuild registry if necessary
        print "before", inConnected
        for s in allInputs:
            print s.name
            if s.name in inConnected:  # if we have already connected signal s and it is not a single type signal, then we don't consider it as connected
                inConnected.remove(s.name)
            if not self.inWidget.instance.hasInputName(s.name): return -1
        for s in allOutputs:
            if not self.outWidget.instance.hasOutputName(s.name): return -1
        print "after", inConnected
        """

        pL1,can1,sameType1 = self.getPossibleConnections(majorOutputs, majorInputs, [], [])
        pL2,can2,sameType2 = self.getPossibleConnections(majorOutputs, minorInputs, [], [v[1] for v in pL1])
        pL3,can3,sameType3 = self.getPossibleConnections(minorOutputs, majorInputs, [], [v[1] for v in pL1+pL2])
        pL4,can4,sameType4 = self.getPossibleConnections(minorOutputs, minorInputs, [], [v[1] for v in pL1+pL2+pL3])

        all = pL1 + pL2 + pL3 + pL4

        # try to find if there are links to any inputs that haven't been previously connected
        for (o,i) in all:
            if not 1 in [i == s for s in inConnected]:
                all.remove((o,i))
                all.insert(0, (o,i))
                break

        if not all: return 0

        self.addLink(all[0][0], all[0][1])  # add only the best link

        same = [sameType1, sameType2, sameType3, sameType4]
        can = [can1, can2, can3, can4]
        self.multiplePossibleConnections = (same[can.index(1)] != 1)
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


class ColorIcon(QPushButton):
    def __init__(self, parent, color):
        QPushButton.__init__(self, parent, "")
        self.color = color
        self.parent = parent
        self.setMaximumSize(20,20)
        self.connect(self, SIGNAL("clicked()"), self.showColorDialog)

    def drawButtonLabel(self, painter):
        painter.setBrush(QBrush(self.color))
        painter.setPen(QPen(self.color))
        painter.drawRect(3, 3, self.width()-6, self.height()-6)

    def showColorDialog(self):
        color = QColorDialog.getColor(self.color, self.parent)        
        if color.isValid():
            self.color = color
            self.repaint()

# canvas dialog
class CanvasOptionsDlg(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.setCaption("Qt Canvas Options")
        #self.controlArea = QVBoxLayout (self)
        self.topLayout = QVBoxLayout( self, 10 )
        self.resize(500,500)

        self.tabs = QTabWidget(self, 'tabWidget')
        GeneralTab = QVGroupBox(self.tabs)
        ExceptionsTab = QVGroupBox(self.tabs)
        TabOrderTab = QVGroupBox(self.tabs)
        self.tabs.insertTab(GeneralTab, "General")
        self.tabs.insertTab(ExceptionsTab, "Exception handling")
        self.tabs.insertTab(TabOrderTab, "Widget tab order")

        # #################################################################
        # GENERAL TAB
        generalBox = QVGroupBox(GeneralTab)
        generalBox.setTitle("General Options")
        self.snapToGridCB = QCheckBox("Snap widgets to grid", generalBox)
        self.useLargeIconsCB = QCheckBox("Show widgets using large icons and text", generalBox)
        self.writeLogFileCB  = QCheckBox("Write content of Output window to log file", generalBox)
        self.showSignalNamesCB = QCheckBox("Show signal names between widgets", generalBox)
        self.verboseCB = QCheckBox("Print extra messages to output (Verbose mode)", generalBox)

        validator = QIntValidator(self)
        validator.setRange(0,10000)

        canvasSizeBox = QVGroupBox(GeneralTab)
        canvasSizeBox.setTitle("Default Size of Orange Canvas")
        widthBox = QHBox(canvasSizeBox)
        widthLabel = QLabel("Width:  ", widthBox)
        self.widthEdit = QLineEdit(widthBox)
        self.widthEdit.setValidator(validator)
        
        heightBox = QHBox(canvasSizeBox)
        heightLabel = QLabel("Height: ", heightBox)
        self.heightEdit = QLineEdit(heightBox)
        self.heightEdit.setValidator(validator)

        colorsBox = QVGroupBox(GeneralTab)
        colorsBox.setTitle("Set Colors For...")

        selectedWidgetBox = QHBox(colorsBox)
        self.selectedWidgetIcon = ColorIcon(selectedWidgetBox, canvasDlg.widgetSelectedColor)
        selectedWidgetLabel = QLabel(" Selected Widget", selectedWidgetBox)

        activeWidgetBox = QHBox(colorsBox)
        self.activeWidgetIcon = ColorIcon(activeWidgetBox, canvasDlg.widgetActiveColor)
        activeWidgetLabel = QLabel(" Active Widget", activeWidgetBox)

        lineBox = QHBox(colorsBox)
        self.lineIcon = ColorIcon(lineBox, canvasDlg.lineColor)
        lineLabel = QLabel(" Lines", lineBox)
        

        # #################################################################
        # EXCEPTION TAB
        exceptions = QVGroupBox("Exceptions", ExceptionsTab)
        #self.catchExceptionCB = QCheckBox('Catch exceptions', exceptions)
        self.focusOnCatchExceptionCB = QCheckBox('Focus output window on catch', exceptions)
        self.printExceptionInStatusBarCB = QCheckBox('Print last exception in status bar', exceptions)
        
        output = QVGroupBox("System output", ExceptionsTab)
        #self.catchOutputCB = QCheckBox('Catch system output', output)
        self.focusOnCatchOutputCB = QCheckBox('Focus output window on system output', output)
        self.printOutputInStatusBarCB = QCheckBox('Print last system output in status bar', output)

        # tab order options
        caption = QLabel("Set order of widget categories", TabOrderTab)
        self.tabOrderList = QListBox(TabOrderTab)
        self.tabOrderList.setSelectionMode(QListBox.Single)
        hbox2 = QHBox(TabOrderTab)
        self.upButton = QPushButton("Up", hbox2)
        self.downButton = QPushButton("Down", hbox2)
        self.connect(self.upButton, SIGNAL("clicked()"), self.moveUp)
        self.connect(self.downButton, SIGNAL("clicked()"), self.moveDown)

        # OK, Cancel buttons
        hbox = QHBox(self)
        self.okButton = QPushButton("OK", hbox)
        self.cancelButton = QPushButton("Cancel", hbox)

        self.topLayout.addWidget(self.tabs)
        self.topLayout.addWidget(hbox)

        self.connect(self.okButton, SIGNAL("clicked()"), self.accept)
        self.connect(self.cancelButton, SIGNAL("clicked()"), self.reject)

    # move selected widget category up
    def moveUp(self):
        for i in range(1, self.tabOrderList.count()):
            if self.tabOrderList.isSelected(i):
                text = self.tabOrderList.text(i)
                self.tabOrderList.removeItem(i)
                self.tabOrderList.insertItem(text, i-1)
                self.tabOrderList.setSelected(i-1, TRUE)

    # move selected widget category down
    def moveDown(self):
        for i in range(self.tabOrderList.count()-2,-1,-1):
            if self.tabOrderList.isSelected(i):
                text = self.tabOrderList.text(i)
                self.tabOrderList.removeItem(i)
                self.tabOrderList.insertItem(text, i+1)
                self.tabOrderList.setSelected(i+1, TRUE)


        
# #######################################
# # Preferences dialog - preferences for signals
# #######################################
class PreferencesDlg(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.topLayout = QVBoxLayout( self, 10 )
        self.grid = QGridLayout( 5, 3 )
        self.topLayout.addLayout( self.grid, 10 )
        
        groupBox  = QGroupBox(self, "Channel_settings")
        groupBox.setTitle("Channel settings")
        self.grid.addWidget(groupBox, 1,1)
        topLayout2 = QVBoxLayout(groupBox, 10 )
        propGrid = QGridLayout(groupBox, 4, 2 )
        topLayout2.addLayout(propGrid, 10)

        cap0 = QLabel("Symbolic channel names:", self)
        cap1 = QLabel("Full name:", groupBox)
        cap2 = QLabel("Priority:", groupBox)
        cap3 = QLabel("Color:", groupBox)
        self.editFullName = QLineEdit(groupBox)
        self.editPriority = QComboBox( FALSE, groupBox, "priority" ) 
        self.editColor    = QComboBox( FALSE, groupBox, "color" )
        #self.connect( self.editPriority, SIGNAL("activated(int)"), self.comboValueChanged )
        #self.connect( self.editColor, SIGNAL("activated(int)"), self.comboValueChanged ) 

        propGrid.addWidget(cap1, 0,0, Qt.AlignVCenter+Qt.AlignHCenter)
        propGrid.addWidget(cap2, 1,0, Qt.AlignVCenter+Qt.AlignHCenter)
        propGrid.addWidget(cap3, 2,0, Qt.AlignVCenter+Qt.AlignHCenter)
        propGrid.addWidget(self.editFullName, 0,1, Qt.AlignVCenter)
        propGrid.addWidget(self.editPriority, 1,1, Qt.AlignVCenter)
        propGrid.addWidget(self.editColor, 2,1, Qt.AlignVCenter)

        groupBox.setMinimumSize(180,150)
        groupBox.setMaximumSize(180,150)
        
        saveButton = QPushButton("Save changes", groupBox)
        addButton = QPushButton("Add new channel name", self)
        removeButton = QPushButton("Remove selected name", self)
        closeButton = QPushButton("Close",self)
        self.channelList = QListBox( self, "channels" )
        self.channelList.setMinimumHeight(200)
        self.connect( self.channelList, SIGNAL("highlighted(int)"), self.listItemChanged ) 

        self.grid.addWidget(cap0,0,0, Qt.AlignLeft+Qt.AlignBottom)
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

        self.editColor.insertItem( "black" )
        self.editColor.insertItem( "darkGray" )
        self.editColor.insertItem( "gray" )
        self.editColor.insertItem( "lightGray" )
        self.editColor.insertItem( "red" )
        self.editColor.insertItem( "green" )
        self.editColor.insertItem( "blue" )
        self.editColor.insertItem( "cyan" )
        self.editColor.insertItem( "magenta" )
        self.editColor.insertItem( "yellow" )
        self.editColor.insertItem( "darkRed" )
        self.editColor.insertItem( "darkGreen" )
        self.editColor.insertItem( "darkBlue" )
        self.editColor.insertItem( "darkCyan" )
        self.editColor.insertItem( "darkMagenta" )
        self.editColor.insertItem( "darkYellow" )

        for i in range(20):
            self.editPriority.insertItem(str(i+1))

        self.channels = {}
        if self.canvasDlg.settings.has_key("Channels"):
            self.channels = self.canvasDlg.settings["Channels"]

        self.reloadList()

    def listItemChanged(self, index):
        name = str(self.channelList.text(index))
        value = self.channels[name]
        items = value.split("::")
        self.editFullName.setText(items[0])

        for i in range(self.editPriority.count()):
            if (str(self.editPriority.text(i)) == items[1]):
                self.editPriority.setCurrentItem(i)

        for i in range(self.editColor.count()):
            if (str(self.editColor.text(i)) == items[2]):
                self.editColor.setCurrentItem(i)

    def reloadList(self):
        self.channelList.clear()
        for (key,value) in self.channels.items():
            self.channelList.insertItem(key)

    def saveChanges(self):
        index = self.channelList.currentItem()
        if index != -1:
            name = str(self.channelList.text(index))
            self.channels[name] = str(self.editFullName.text()) + "::" + str(self.editPriority.currentText()) + "::" + str(self.editColor.currentText())

    def addNewSignal(self):
        (Qstring,ok) = QInputDialog.getText("Qt Add New Channel Name", "Enter new symbolic channel name")
        string = str(Qstring)
        if ok:
            self.editColor.setCurrentItem(0)
            self.editPriority.setCurrentItem(0)
            self.editFullName.setText(string)
            self.channels[string] = str(self.editFullName.text()) + "::" + str(self.editPriority.currentText()) + "::" + str(self.editColor.currentText())
            self.reloadList()
            self.selectItem(string)

    def selectItem(self, string):
        for i in range(self.channelList.count()):
            temp = str(self.channelList.text(i))
            if temp == string:
                self.channelList.setCurrentItem(i)
                return
            
    def removeSignal(self):
        index = self.channelList.currentItem()
        if index != -1:
            tempDict = {}
            symbName = str(self.channelList.text(index))
            
            for key in self.channels.keys():
                if key != symbName:
                    tempDict[key] = self.channels[key]
            self.channels = copy(tempDict)        
            
        self.reloadList()

    def closeClicked(self):
        self.canvasDlg.settings["Channels"] = self.channels
        self.accept()
        return


class saveApplicationDlg(QDialog):
    def __init__(self, *args):
        apply(QDialog.__init__,(self,) + args)
        self.setCaption("Qt Set widget order")
        self.shownWidgetList = []
        self.hiddenWidgetList = []

        self.topLayout = QVBoxLayout( self, 10 )

        self.grid = QGridLayout( 2, 1 )
        self.topLayout.addLayout( self.grid, 10 )

        self.tab = QTable(self)
        self.tab.setSelectionMode(QTable.Single )
        self.tab.setRowMovingEnabled(1)
        self.grid.addWidget(self.tab, 1,1)
        
        self.tab.setNumCols(2)
        self.tabHH = self.tab.horizontalHeader()
        self.tabHH.setLabel(0, 'Show')
        self.tabHH.setLabel(1, 'Widget Name')
        self.tabHH.resizeSection(0, 50)
        self.tabHH.resizeSection(1, 170)

        LayoutWidget = QWidget(self,'Layout1')
        LayoutWidget.setGeometry(QRect(200,240,476,33))
        self.grid.addWidget(LayoutWidget, 2,1)
        Layout1 = QHBoxLayout(LayoutWidget)
        Layout1.setSpacing(6)
        Layout1.setMargin(0)

        self.insertSeparatorButton = QPushButton('Add separator', LayoutWidget)
        self.connect(self.insertSeparatorButton, SIGNAL("clicked()"), self.insertSeparator)
        Layout1.addWidget(self.insertSeparatorButton)

        spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout1.addItem(spacer)

        self.okButton = QPushButton('&OK', LayoutWidget)
        Layout1.addWidget(self.okButton)
        self.connect(self.okButton, SIGNAL("clicked()"), self.accept)

        self.buttonCancel = QPushButton("&Cancel", LayoutWidget)
        self.buttonCancel.setAutoDefault(1)
        Layout1.addWidget(self.buttonCancel)
        self.connect(self.buttonCancel, SIGNAL('clicked()'), self, SLOT('reject()'))

        self.resize(200,250)

    def accept(self):
        self.shownWidgetList = []
        self.hiddenWidgetList = []
        for i in range(self.tab.numRows()):
            if self.tab.cellWidget(i, 0).isChecked():
                self.shownWidgetList.append(self.tab.text(i, 1))
            elif self.tab.text(i,1) != "[Separator]":
                self.hiddenWidgetList.append(self.tab.text(i,1))
        QDialog.accept(self)        
        

    def insertSeparator(self):
        curr = max(0, self.findSelected())
        self.insertWidgetName("[Separator]", curr)


    def insertWidgetName(self, name, index = -1):
        if index == -1: index = self.tab.numRows()
        self.tab.setNumRows(self.tab.numRows()+1)
        for i in range(self.tab.numRows()-1, index-1, -1):
            self.swapCells(i, i+1)
        check = QCheckBox(self.tab)
        check.setChecked(1)
        self.tab.setCellWidget(index, 0, check)
        self.tab.setText(index, 1, name)
        #self.tab.adjustColumn(1)
        
        
    def swapCells(self, row1, row2):
        self.tab.swapCells( row1,0, row2, 0)
        self.tab.swapCells( row1,1, row2, 1)
        self.tab.updateCell(row1,0)
        self.tab.updateCell(row1,1)
        self.tab.updateCell(row2,0)
        self.tab.updateCell(row2,1)

    def findSelected(self):
        for i in range(self.tab.numRows()):
            if self.tab.isSelected(i, 0) or self.tab.isSelected(i, 1): return i
        return -1 


if __name__=="__main__":
    app = QApplication(sys.argv) 
    dlg = CanvasOptionsDlg(app)
    app.setMainWidget(dlg)
    dlg.show()
    app.exec_loop() 

