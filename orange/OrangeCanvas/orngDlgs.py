# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	dialogs 

from qt import *
from qtcanvas import *
from copy import *
from string import strip
import os
import sys
from orngCanvasItems import *

TRUE  = 1
FALSE = 0

import orange

##################
### TO DO: to je treba izbrisati!!!
class ExampleTable(orange.ExampleTable):
    pass

class ExampleTableWithClass(ExampleTable):
    pass


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

    def addSignalList(self, outName, inName, outList, inList, outIconName, inIconName):
        xSpaceBetweenWidgets = 100  # space between widgets
        xWidgetOff = 10     # offset for widget position
        yWidgetOffTop = 10     # offset for widget position
        yWidgetOffBottom = 30     # offset for widget position
        ySignalOff = 10     # space between the top of the widget and first signal
        ySignalSpace = 50   # space between two neighbouring signals
        ySignalSize = 20    # height of the signal box
        xSignalSize = 20    # width of the signal box
        xIconOff = 10
        
        count = max(len(inList), len(outList))
        height = max ((count)*ySignalSpace, 70)
        

        # calculate needed sizes of boxes to show text
        maxLeft = 0
        for i in range(len(inList)):
            maxLeft = max(maxLeft, self.getTextWidth("("+inList[i][0]+")", 1))
            maxLeft = max(maxLeft, self.getTextWidth(inList[i][1]))

        maxRight = 0
        for i in range(len(outList)):
            maxRight = max(maxRight, self.getTextWidth("("+outList[i][0]+")", 1))
            maxRight = max(maxRight, self.getTextWidth(outList[i][1]))

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
        for i in range(len(outList)):
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(outList)+1)
            box = QCanvasRectangle(xWidgetOff + width, y - ySignalSize/2.0, xSignalSize, ySignalSize, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.show()
            self.outBoxes.append((outList[i][0], box))

            self.texts.append(MyCanvasText(self.dlg.canvas, outList[i][0], xWidgetOff + width - 5, y - 7, Qt.AlignRight + Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(MyCanvasText(self.dlg.canvas, outList[i][1], xWidgetOff + width - 5, y + 7, Qt.AlignRight + Qt.AlignVCenter, bold =0, show=1))

        for i in range(len(inList)):
            name = inList[i][0]
            type = inList[i][1]
            y = yWidgetOffTop + ((i+1)*signalSpace)/float(len(inList)+1)
            box = QCanvasRectangle(xWidgetOff + width + xSpaceBetweenWidgets - xSignalSize, y - ySignalSize/2.0, xSignalSize, ySignalSize, self.dlg.canvas)
            box.setBrush(QBrush(QColor(0,0,255)))
            box.show()
            self.inBoxes.append((inList[i][0], box))

            self.texts.append(MyCanvasText(self.dlg.canvas, inList[i][0], xWidgetOff + width + xSpaceBetweenWidgets + 5, y - 7, Qt.AlignLeft + Qt.AlignVCenter, bold =1, show=1))
            self.texts.append(MyCanvasText(self.dlg.canvas, inList[i][1], xWidgetOff + width + xSpaceBetweenWidgets + 5, y + 7, Qt.AlignLeft + Qt.AlignVCenter, bold =0, show=1))

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
        self.inList = []
        self.outList = []
        self._links = []

        # GUI
        self.setName('Qt Set Signals')

        self.resize(515,286)
        self.setCaption(self.tr("Qt Set Signals"))

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

        self.buttonHelp = QPushButton(LayoutWidget,'buttonHelp')
        self.buttonHelp.setText(self.tr("&Help"))
        self.buttonHelp.setAutoDefault(1)
        Layout1.addWidget(self.buttonHelp)
        spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout1.addItem(spacer)

        self.buttonClearAll = QPushButton(LayoutWidget,'ClearAll')
        self.buttonClearAll.setText(self.tr("Clear &All"))
        Layout1.addWidget(self.buttonClearAll)

        self.buttonOk = QPushButton(LayoutWidget,'buttonOk')
        self.buttonOk.setText(self.tr("&OK"))
        self.buttonOk.setAutoDefault(1)
        self.buttonOk.setDefault(1)
        Layout1.addWidget(self.buttonOk)

        self.buttonCancel = QPushButton(LayoutWidget,'buttonCancel')
        self.buttonCancel.setText(self.tr("&Cancel"))
        self.buttonCancel.setAutoDefault(1)
        Layout1.addWidget(self.buttonCancel)

        self.connect(self.buttonClearAll,SIGNAL('clicked()'),self.clearAll)
        self.connect(self.buttonOk,SIGNAL('clicked()'),self,SLOT('accept()'))
        self.connect(self.buttonCancel,SIGNAL('clicked()'),self,SLOT('reject()'))

    def clearAll(self):
        while self._links != []:
            self.removeLink(self._links[0][0], self._links[0][1])

    def addSignalList(self, outName, inName, outList, inList, outIconName = None, inIconName = None):
        self.outWidget = outName
        self.inWidget = inName
        self.inList = inList
        self.outList = outList
        (width, height) = self.canvasView.addSignalList(outName, inName, outList, inList, outIconName, inIconName)
        self.canvas.resize(width, height)
        self.resize(width+50, height+85)

    def addDefaultLinks(self):
        canConnect = 0
        for (outName, outType) in self.outList:
            try:
                eval(outType)
                for (inName, inType, handler, single) in self.inList:
                    try:
                        eval(inType)
                        if issubclass(eval(outType), eval(inType)): canConnect = 1
                        #if outName == inName and issubclass(eval(outType), eval(inType)):
                        if issubclass(eval(outType), eval(inType)):
                            self.addLink(outName, inName)
                    except:
                        print "unknown type: ", inType
            except:
                print "unknown type: ", outType
        return canConnect

    def addLink(self, outName, inName):
        if (outName, inName) in self._links: return

        try:
            # check if correct types
            outType = None; inType = None
            for (name, type, handler, single) in self.inList:
                if name == inName: inType = type
            for (name, type) in self.outList:
                if name == outName: outType = type
            if not issubclass(eval(outType), eval(inType)): return 0
        except:
            "unknown type: ", outType, " or ", inType
            return 0

        # if inName is a single signal and connection already exists -> delete it        
        for (outN, inN) in self._links:
            if inN == inName:
                for (name, type, handler, single) in self.inList:
                    if name == inName and single:
                        for (o, i) in self._links:
                            if i == inName:
                                self.removeLink(o, i)
                                
        self._links.append((outName, inName))
        self.canvasView.addLink(outName, inName)
        return 1

    def removeLink(self, outName, inName):
        if (outName, inName) in self._links:
            self._links.remove((outName, inName))
            self.canvasView.removeLink(outName, inName)

    def getLinks(self):
        return self._links



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

        # general tab options
        self.snapToGridCB = QCheckBox("Snap widgets to grid", GeneralTab)
        self.useLargeIconsCB = QCheckBox("Show widgets using large icons and text", GeneralTab)

        # exception tab options
        exceptions = QVGroupBox("Exceptions", ExceptionsTab)
        self.catchExceptionCB = QCheckBox('Catch exceptions', exceptions)
        self.focusOnCatchExceptionCB = QCheckBox('Focus output window on catch', exceptions)
        self.printExceptionInStatusBarCB = QCheckBox('Print last exception in status bar', exceptions)
        
        output = QVGroupBox("System output", ExceptionsTab)
        self.catchOutputCB = QCheckBox('Catch system output', output)
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
        (Qstring,ok) = QInputDialog.getText("Add New Channel Name", "Enter new symbolic channel name")
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

if __name__=="__main__":
    app = QApplication(sys.argv) 
    #dlg = SignalDialog(app)
    #dlg.addSignalList("outWidget name", "inWidget name", [("Examples", 'ExampleTable'),("Examples", 'ExampleTable'),("Examples", 'ExampleTable'),("Examples", 'ExampleTable'), ("Classified Examples", 'ExampleTableWithClass')],[("Classified Examples", 'ExampleTableWithClass'),("Classified Examples", 'ExampleTableWithClass')], "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png", "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png")
    #dlg.addSignalList("outWidget name", "inWidget name", [("Examples", 'ExampleTable'),("Examples", 'ExampleTable'), ("Classified Examples", 'ExampleTableWithClass')],[("Classified Examples", 'ExampleTableWithClass')], "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png", "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png")
    #dlg.addSignalList("outWidget name", "inWidget name", [("Examples", 'ExampleTable'), ("Classified Examples", 'ExampleTableWithClass')],[("Classified Examples", 'ExampleTableWithClass', None, 1)], "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png", "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png")
    #dlg.addSignalList("outWidget name", "inWidget name", [("Classified Examples", 'ExampleTableWithClass')],[("Classified Examples", 'ExampleTableWithClass')], "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png", "E:/Development/Python23/Lib/site-packages/Orange/OrangeWidgets/icons/SelectAttributes.png")
    dlg = CanvasOptionsDlg(app)
    app.setMainWidget(dlg)
    dlg.show()
    #dlg.addSignals(["data", "cdata", "ddata"], ["test", "ddata", "cdata"])
    app.exec_loop() 

