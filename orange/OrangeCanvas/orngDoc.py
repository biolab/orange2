# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	document class - main operations (save, load, ...)
#
import sys
import os
import string
from qt import *
from qtcanvas import *
from xml.dom.minidom import Document, parse
import orngView
import orngCanvasItems
import orngResources
import orngTabs
from orngDlgs import *
from orngSignalManager import *
import cPickle
TRUE  = 1
FALSE = 0

class SchemaDoc(QMainWindow):
    def __init__(self, canvasDlg, *args):
        apply(QMainWindow.__init__,(self,) + args)
        self.resize(700,500)
        self.showNormal()
        self.canvasDlg = canvasDlg
        self.setCaption("Schema" + str(orngResources.iDocIndex))
        orngResources.iDocIndex = orngResources.iDocIndex + 1
        self.hasChanged = FALSE
        self.canvasDlg.enableSave(FALSE)
        self.setIcon(QPixmap(orngResources.file_new))
        self.canvas = QCanvas(2000,2000)
        self.canvasView = orngView.SchemaView(self, self.canvas, self)
        self.setCentralWidget(self.canvasView)
        self.canvasView.show()
        self.lines = []
        self.widgets = []
        
        # if widget path not registered -> register
        if sys.path.count(canvasDlg.widgetDir) == 0:
            sys.path.append(canvasDlg.widgetDir)

        self.path = os.getcwd()
        self.filename = str(self.caption())
        self.filenameValid = FALSE
 
        
    def closeEvent(self,ce):
        if not self.hasChanged:
            ce.accept()
            self.removeAllWidgets()
            return
        res = QMessageBox.information(self,'Qrange Canvas','Do you want to save changes made to schema?','Yes','No','Cancel',0,1)
        if res == 0:
            self.saveDocument()
            ce.accept()
        elif res == 1:
            ce.accept()
        else:
            ce.ignore()

        self.removeAllWidgets()
 
    def addLine(self, outWidget, inWidget, setSignals = TRUE, enabled = TRUE):
        # check if line already exists
        for line in self.lines:
            if line.inWidget == inWidget and line.outWidget == outWidget:
                QMessageBox.information( None, "Orange Canvas", "This connection already exists.", QMessageBox.Ok + QMessageBox.Default )
                return None

        line = orngCanvasItems.CanvasLine(self.canvasDlg, outWidget, inWidget, self.canvas)
        line.setEnabled(enabled)
        if setSignals:
            dialog = SignalDialog(self.canvasDlg, None, "", TRUE)
            dialog.addSignalList(outWidget.caption, inWidget.caption, outWidget.widget.outList, inWidget.widget.inList, outWidget.widget.iconName, inWidget.widget.iconName)
            canConnect = dialog.addDefaultLinks()
            if not canConnect:
                QMessageBox.information( None, "Orange Canvas", "Selected widgets don't share a common signal type. Unable to connect.", QMessageBox.Ok + QMessageBox.Default )
                line.remove()
                return None

            # if there are multiple choices, how to connect this two widget, then show the dialog
            signals = dialog.getLinks()
            if len(signals) > 1:
                res = dialog.exec_loop()
                if dialog.result() == QDialog.Rejected:
                    line.remove()
                    return None
                
            connected = []
            signalManager.setFreeze(1)
            for (outName, inName) in signals:
                widgets = inWidget.instance.removeExistingSingleLink(inName)
                for widget in  widgets:
                    existingSignals = signalManager.findSignals(widget, inWidget.instance)
                    existingOutName = None
                    for (outN, inN) in existingSignals:
                        if inN == inName: existingOutName = outN
                    self.removeWidgetSignal(widget, inWidget.instance, existingOutName, inName)
                ok = signalManager.addLink(outWidget.instance, inWidget.instance, outName, inName, enabled)
                if ok: connected.append((outName, inName))

            if connected == []:
                print "Error. No connections were maid."
                line.remove()
                return None

            line.setSignals(connected)
            signalManager.setFreeze(0, outWidget.instance)

        self.lines.append(line)
        outWidget.addOutLine(line)
        outWidget.updateTooltip()
        inWidget.addInLine(line)
        inWidget.updateTooltip()
        line.show()
        line.setEnabled(enabled)

        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)
        
        return line


    def resetActiveSignals(self, line, newSignals = None, enabled = 1):
        signals = line.getSignals()
        if newSignals == None:
            dialog = SignalDialog(self.canvasDlg, None, "", TRUE)
            dialog.addSignalList(line.outWidget.caption, line.inWidget.caption, line.outWidget.widget.outList, line.inWidget.widget.inList, line.outWidget.widget.iconName, line.inWidget.widget.iconName)
            for (outName, inName) in signals:
                dialog.addLink(outName, inName)

            # if there are multiple choices, how to connect this two widget, then show the dialog
            res = dialog.exec_loop()
            if dialog.result() == QDialog.Rejected: return line
                
            connected = []
            newSignals = dialog.getLinks()
            
        for (outName, inName) in signals:
            if (outName, inName) not in newSignals:
                widgets = line.inWidget.instance.removeExistingSingleLink(inName)
                for widget in  widgets:
                    self.removeWidgetSignal(widget, line.inWidget.instance, outName, inName)
                signals.remove((outName, inName))
        
        if newSignals == []:
            self.lines.remove(line)
            line.remove()
            return None
    
        connected = []
        signalManager.setFreeze(1)
        for (outName, inName) in newSignals:
            if (outName, inName) not in signals:
                ok = signalManager.addLink(line.outWidget.instance, line.inWidget.instance, outName, inName, enabled)
                if ok: connected.append((outName, inName))
        signalManager.setFreeze(0, line.outWidget.instance)
            
        line.outWidget.updateTooltip()
        line.inWidget.updateTooltip()
        line.setSignals(signals + connected)

        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)

        return line


    # remove line line
    def removeLine1(self, line):
        for (outName, inName) in line.signals:
            signalManager.removeLink(line.outWidget.instance, line.inWidget.instance, outName, inName)   # update SignalManager

        line.inWidget.removeLine(line)
        line.outWidget.removeLine(line)
        line.inWidget.updateTooltip()
        line.outWidget.updateTooltip()
        self.lines.remove(line)
        line.remove()
        #line.repaintLine(self)
        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)        

    # remove line, connecting two widgets
    def removeLine(self, widgetFrom, widgetTo):
        for line in self.lines:
            if line.outWidget.instance == widgetFrom and line.inWidget.instance == widgetTo:
                self.removeLine1(line)
        
    # remove only one signal from connected two widgets. If no signals are left, delete the line
    def removeWidgetSignal(self, widgetFrom, widgetTo, signalNameFrom, signalNameTo):
        signalManager.removeLink(widgetFrom, widgetTo, signalNameFrom, signalNameTo)

        otherSignals = 0
        for (widget, signalFrom, signalTo, enabled) in signalManager.links[widgetFrom]:
            if widget == widgetTo: otherSignals = 1
        if not otherSignals:
            self.removeLine(widgetFrom, widgetTo)

        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)

    def addWidget(self, widget):
        newwidget = orngCanvasItems.CanvasWidget(self.canvas, self.canvasView, widget, self.canvasDlg.defaultPic, self.canvasDlg)
        x = self.canvasView.contentsX() + 10
        for w in self.widgets:
            x = max(w.x() + 90, x)
            x = x/10*10
        y = 150
        newwidget.setCoords(x,y)
        newwidget.setViewPos(self.canvasView.contentsX(), self.canvasView.contentsY())

        list = []
        for item in self.widgets:
            if item.widget.name == widget.name:
                list.append(item.caption)

        i = 2; found = 0
        if len(list) > 0:
            while not found:
                if newwidget.caption + " (" + str(i) + ")" not in list:
                    found = 1
                    newwidget.updateText(newwidget.caption + " (" + str(i) + ")")
                else: i += 1
                
        signalManager.addWidget(newwidget.instance)
        newwidget.show()
        newwidget.updateTooltip()
        self.widgets.append(newwidget)
        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)
        self.canvas.update()    
        return newwidget

    def removeWidget(self, widget):
        signalManager.removeWidget(widget.instance)
        self.widgets.remove(widget)
        widget.remove()

        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)

        while widget.inLines != []: self.removeLine1(widget.inLines[0])
        while widget.outLines != []:  self.removeLine1(widget.outLines[0])

    def removeAllWidgets(self):
        for widget in self.widgets:
            if (widget.instance != None):
                try:
                    code = compile("widget.instance.saveSettings()", ".", "single")
                    exec(code)
                except:
                    pass
            self.widgets.remove(widget)
        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)

    def enableAllLines(self):
        for line in self.lines:
            signalManager.setLinkEnabled(line.outWidget.instance, line.inWidget.instance, 1)
            line.setEnabled(1)
            #line.repaintLine(self.canvasView)
        self.canvas.update()
        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)

    def disableAllLines(self):
        for line in self.lines:
            signalManager.setLinkEnabled(line.outWidget.instance, line.inWidget.instance, 0)
            line.setEnabled(0)
            #line.repaintLine(self.canvasView)
        self.canvas.update()
        self.hasChanged = TRUE
        self.canvasDlg.enableSave(TRUE)

    # return the widget instance that has caption "widgetName"
    def getWidgetByCaption(self, widgetName):
        for widget in self.widgets:
            if (widget.caption == widgetName):
                return widget
        return None
                    
    # return a new widget instance of a widget with name "widgetName"
    def addWidgetByCaption(self, widgetName):
        for widget in self.canvasDlg.tabs.allWidgets:
            if widget.fileName == widgetName:
                return self.addWidget(widget)
        return None
        

    # ###########################################
    # SAVING, LOADING, ....
    # ###########################################
    def saveDocument(self):
        if not self.filenameValid:
            self.saveDocumentAs()
        else:
            self.save()

    def saveDocumentAs(self):
        qname = QFileDialog.getSaveFileName( self.path + "/" + self.filename, "Orange Widget Scripts (*.ows)", self, "", "Save File")
        if qname.isEmpty():
            return
        name = str(qname)
        if name[-4] != ".":
            name = name + ".ows"
        self.path = os.path.dirname(name)
        self.filename = os.path.basename(name)
        self.setCaption(self.filename)
        self.filenameValid = TRUE
        self.save()        

    # save the file            
    def save(self):
        self.hasChanged = FALSE
        self.canvasDlg.enableSave(FALSE)

        # create xml document
        doc = Document()
        schema = doc.createElement("schema")
        widgets = doc.createElement("widgets")
        lines = doc.createElement("channels")
        doc.appendChild(schema)
        schema.appendChild(widgets)
        schema.appendChild(lines)

        #save widgets
        for widget in self.widgets:
            temp = doc.createElement("widget")
            temp.setAttribute("xPos", str(int(widget.x())) )
            temp.setAttribute("yPos", str(int(widget.y())) )
            temp.setAttribute("caption", widget.caption)
            temp.setAttribute("widgetName", widget.widget.fileName)
            widgets.appendChild(temp)

        #save connections
        for line in self.lines:
            temp = doc.createElement("channel")
            temp.setAttribute("outWidgetCaption", line.outWidget.caption)
            temp.setAttribute("inWidgetCaption", line.inWidget.caption)
            temp.setAttribute("enabled", str(line.getEnabled()))
            temp.setAttribute("signals", str(line.getSignals()))
            lines.appendChild(temp)

        xmlText = doc.toprettyxml()
        file = open(self.path + "/" + self.filename, "wt")
        file.write(xmlText)
        file.flush()
        file.close()
        doc.unlink()

        self.saveWidgetSettings(self.path + "/" + self.filename[:-3] + "sav")
        self.canvasDlg.addToRecentMenu(self.path + "/" + self.filename)        

    def saveWidgetSettings(self, filename):
        list = {}
        for widget in self.widgets:
            list[widget.caption] = widget.instance.saveSettingsStr()

        file = open(filename, "w")
        cPickle.dump(list, file)
        file.close()

    def loadWidgetSettings(self, filename):
        if not os.path.exists(filename):
            return

        file = open(filename, "r")
        list = cPickle.load(file)
        for widget in self.widgets:
            str = list[widget.caption]
            widget.instance.loadSettingsStr(str)
            widget.instance.activateLoadedSettings()

        file.close()
                    
    # load a scheme with name "filename"
    def loadDocument(self, filename):
            
        if not os.path.exists(filename):
            self.close()
            QMessageBox.critical(self,'Qrange Canvas','Unable to find file \"'+ filename,  QMessageBox.Ok + QMessageBox.Default)
            return

        # ##################
        #load the data ...
        doc = parse(str(filename))
        schema = doc.firstChild
        widgets = schema.getElementsByTagName("widgets")[0]
        lines = schema.getElementsByTagName("channels")[0]

        # ##################
        #read widgets
        widgetList = widgets.getElementsByTagName("widget")
        for widget in widgetList:
            name = widget.getAttribute("widgetName")
            tempWidget = self.addWidgetByCaption(name)
            if (tempWidget != None):
                xPos = int(widget.getAttribute("xPos"))
                yPos = int(widget.getAttribute("yPos"))
                tempWidget.setCoords(xPos, yPos)
                tempWidget.caption = widget.getAttribute("caption")
            else:
                QMessageBox.information(self,'Qrange Canvas','Unable to find widget \"'+ name + '\"',  QMessageBox.Ok + QMessageBox.Default)

        # ##################
        #read lines                        
        lineList = lines.getElementsByTagName("channel")
        for line in lineList:
            inCaption = line.getAttribute("inWidgetCaption")
            outCaption = line.getAttribute("outWidgetCaption")
            Enabled = int(line.getAttribute("enabled"))
            print Enabled
            signals = line.getAttribute("signals")
            inWidget = self.getWidgetByCaption(inCaption)
            outWidget = self.getWidgetByCaption(outCaption)
            if inWidget == None or outWidget == None:
                print "Unable to create a line due to invalid widget name. Try reinstalling widgets."
                continue

            tempLine = self.addLine(outWidget, inWidget, setSignals = FALSE)
            signalList = eval(signals)
            self.resetActiveSignals(tempLine, signalList, Enabled)
            #tempLine.updateLinePos()
            #tempLine.setRightColors(self.canvasDlg)
            tempLine.setEnabled(Enabled)
            #tempLine.repaintLine(self.canvasView)

        self.canvas.update()
        self.hasChanged = FALSE
        self.canvasDlg.enableSave(FALSE)
        self.path = os.path.dirname(filename)
        self.filename = os.path.basename(filename)
        self.setCaption(self.filename)
        self.filenameValid = TRUE

        self.loadWidgetSettings(self.path + "/" + self.filename[:-3] + "sav")

    # ###########################################
    # save document as application
    # ###########################################
    def saveDocumentAsApp(self, asTabs = 1):
        # get filename
        appName = self.filename
        if len(appName) > 4 and appName[-4] != "." and appName[-3] != ".":
            appName = appName + ".py"
        elif len(appName) > 4 and appName[-4] == '.':
            appName = appName[:-4] + ".py"
        appName = appName.replace(" ", "")
        qname = QFileDialog.getSaveFileName( self.path + "/" + appName, "Orange Scripts (*.py)", self, "", "Save File as Application")
        if qname.isEmpty():
            return
        appName = str(qname)
        if len(appName) > 4 and appName[-4] != "." and appName[-3] != ".":
            appName = appName + ".py"
        (dir, fileName) = os.path.split(appName)
        fileName = fileName[:-3]

        #format string with file content
        t = "    "  # instead of tab
        imports = "import sys\nimport os\nfrom orngSignalManager import *\n"
        instancesT = "# create widget instances\n" +t+t
        instancesB = "# create widget instances\n" +t+t
        tabs = "# add tabs\n"+t+t
        links = "# add widget signals\n"+t+t + "signalManager.setFreeze(1)\n" +t+t
        save = ""
        buttons = "# create widget buttons\n"+t+t
        buttonsConnect = "#connect GUI buttons to show widgets\n"+t+t
        manager = ""
        # add widgets to application as they are topologically sorted
        for instance in signalManager.widgets:
            widget = None
            for i in range(len(self.widgets)):
                if self.widgets[i].instance == instance: widget = self.widgets[i]
            name = widget.caption
            name = name.replace(" ", "_")
            name = name.replace("(", "")
            name = name.replace(")", "")
            imports += "from " + widget.widget.fileName + " import *\n"
            instancesT += "self.ow" + name + " = " + widget.widget.fileName + "(self.tabs)\n"+t+t
            manager += "signalManager.addWidget(self.ow" + name + ")\n" +t+t
            instancesB += "self.ow" + name + " = " + widget.widget.fileName + "()\n"+t+t
            tabs += "self.tabs.insertTab (self.ow" + name + ",\"" + widget.caption + "\")\n"+t+t
            buttons += "owButton" + name + " = QPushButton(\"" + widget.caption + "\", self)\n"+t+t
            buttonsConnect += "self.connect( owButton" + name + ",SIGNAL(\"clicked()\"), self.ow" + name + ".reshow)\n"+t+t
            save += "self.ow" + name + ".saveSettings()\n"+t+t
            
        for line in self.lines:
            if not line.getEnabled(): continue

            outWidgetName = line.outWidget.caption
            outWidgetName = outWidgetName.replace(" ", "_")
            outWidgetName = outWidgetName.replace("(", "")
            outWidgetName = outWidgetName.replace(")", "")
            inWidgetName = line.inWidget.caption
            inWidgetName = inWidgetName.replace(" ", "_")
            inWidgetName = inWidgetName.replace("(", "")
            inWidgetName = inWidgetName.replace(")", "")
    
            for (outName, inName) in line.getSignals():            
                links += "signalManager.addLink( self.ow" + outWidgetName + ", self.ow" + inWidgetName + ", '" + outName + "', '" + inName + "', 1)\n" +t+t
    
        links += "signalManager.setFreeze(0)\n" +t+t
        buttons += "exitButton = QPushButton(\"E&xit\",self)\n"+t+t + "self.connect(exitButton,SIGNAL(\"clicked()\"),a,SLOT(\"quit()\"))\n"+t+t

        classname = os.path.basename(appName)[:-3]
        classname = classname.replace(" ", "_")

        classinit = """
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt %s")
        self.bottom=QHBox(self)""" % (fileName)

        if asTabs == 1:
            classinit += """
        self.tabs = QTabWidget(self, 'tabWidget')
        self.resize(640,480)"""
        

        finish = """
a=QApplication(sys.argv)
ow=""" + classname + """()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()"""

        if save != "":
            save = t+"def exit(self):\n" +t+t+ save

        if asTabs:
            whole = imports + "\n\n" + "class " + classname + "(QVBox):" + classinit + "\n\n"+t+t+ instancesT + "\n"+t+t + manager + "\n"+t+t + tabs + "\n" + t+t + links + "\n\n" + save + "\n\n" + finish
        else:
            whole = imports + "\n\n" + "class " + classname + "(QVBox):" + classinit + "\n\n"+t+t+ instancesB + "\n"+t+t + manager + "\n"+t+t + buttons + "\n" +t+t+ buttonsConnect + "\n" +t+t + links + "\n\n" + save + "\n\n" + finish
        
        #save app
        fileApp = open(appName, "wt")
        fileApp.write(whole)
        fileApp.flush()
        fileApp.close()

if __name__=='__main__': 
	app = QApplication(sys.argv)
	dlg = SchemaDoc()
	app.setMainWidget(dlg)
	dlg.show()
	app.exec_loop() 