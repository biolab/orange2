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
import cPickle
TRUE  = 1
FALSE = 0

class SchemaDoc(QMainWindow):
    def __init__(self, canvasDlg, *args):
        apply(QMainWindow.__init__,(self,) + args)
        self.resize(400,300)
        self.showNormal()
        self.setCaption("Schema" + str(orngResources.iDocIndex))
        orngResources.iDocIndex = orngResources.iDocIndex + 1
        self.hasChanged = FALSE
        self.setIcon(QPixmap(orngResources.file_new))
        self.canvas = QCanvas(2000,2000)
        self.canvasView = orngView.SchemaView(self, self.canvas, self)
        self.setCentralWidget(self.canvasView)
        self.canvasView.show()
        self.lines = []
        self.widgets = []
        self.canvasDlg = canvasDlg
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
 
    def addLine(self, outWidget, inWidget, setSignals = TRUE):
        # check if line already exists
        for line in self.lines:
            if line.inWidget == inWidget and line.outWidget == outWidget:
                QMessageBox.information( None, "Orange Canvas", "This connection already exists.", QMessageBox.Ok + QMessageBox.Default )
                return None
            
        line = orngCanvasItems.CanvasLine(self.canvas)
        line.setInOutWidget(inWidget, outWidget)
        if setSignals == TRUE:
            if line.setActiveSignals(self.canvasDlg) == FALSE:
                line = None
                return None
        self.lines.append(line)
        line.setEnabled(TRUE)
        line.finished = TRUE
        return line

    def addWidget(self, widget):
        newwidget = orngCanvasItems.CanvasWidget(self.canvas, widget, self.canvasDlg.defaultPic, self.canvasDlg)
        x = self.canvasView.contentsX() + 10
        for w in self.widgets:
            x = max(w.x() + 90, x)
            x = x/10*10
        y = 150
        newwidget.move(x,y)

        count = 0
        for item in self.widgets:
            if item.widget.name == widget.name:
                count = count+1

        if count > 0:
            newwidget.caption = newwidget.caption + " (" + str(count+1) + ")"
        newwidget.show()
        newwidget.updateTooltip(self.canvasView)
        self.widgets.append(newwidget)
        self.hasChanged = TRUE
        self.canvas.update()    
        return newwidget

    def removeAllWidgets(self):
        for widget in self.widgets:
            if (widget.instance != None):
                try:
                    code = compile("widget.instance.saveSettings()", ".", "single")
                    exec(code)
                except:
                    pass
            self.widgets.remove(widget)

    def enableAllLines(self):
        for line in self.lines:
            line.setEnabled(TRUE)
            line.repaintLine(self.canvasView)
        self.hasChanged = TRUE

    def disableAllLines(self):
        for line in self.lines:
            line.setEnabled(FALSE)
            line.repaintLine(self.canvasView)
        self.hasChanged = TRUE

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
            temp.setAttribute("inWidgetCaption", line.inWidget.caption)
            temp.setAttribute("outWidgetCaption", line.outWidget.caption)
            temp.setAttribute("enabled", str(line.getEnabled()))
            signals = ""
            for signal in line.signals:
                signals = signals + signal + ","
            signals = signals[:-1]
            temp.setAttribute("signals", signals)
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
        file = open(filename, "w")
        list = {}
        for widget in self.widgets:
            list[widget.caption] = widget.instance.saveSettingsStr()

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

        #load the data ...
        doc = parse(str(filename))
        schema = doc.firstChild
        widgets = schema.getElementsByTagName("widgets")[0]
        lines = schema.getElementsByTagName("channels")[0]

        #read widgets
        widgetList = widgets.getElementsByTagName("widget")
        for widget in widgetList:
            name = widget.getAttribute("widgetName")
            tempWidget = self.addWidgetByCaption(name)
            if (tempWidget != None):
                xPos = int(widget.getAttribute("xPos"))
                yPos = int(widget.getAttribute("yPos"))
                tempWidget.move(xPos, yPos)
                tempWidget.caption = widget.getAttribute("caption")
            else:
                QMessageBox.information(self,'Qrange Canvas','Unable to find widget \"'+ name + '\"',  QMessageBox.Ok + QMessageBox.Default)

        #read lines                        
        lineList = lines.getElementsByTagName("channel")
        for line in lineList:
            inCaption = line.getAttribute("inWidgetCaption")
            outCaption = line.getAttribute("outWidgetCaption")
            enabled = int(line.getAttribute("enabled"))
            signals = line.getAttribute("signals")
            inWidget = self.getWidgetByCaption(inCaption)
            outWidget = self.getWidgetByCaption(outCaption)
            if inWidget != None and outWidget != None:
                tempLine = self.addLine(outWidget, inWidget, FALSE)
                tempLine.show()
                inWidget.inLines.append(tempLine)
                outWidget.outLines.append(tempLine)
                tempLine.signals = signals.split(",")
                tempLine.updateLinePos()
                tempLine.setRightColors(self.canvasDlg)
                tempLine.setEnabled(enabled)
                tempLine.repaintLine(self.canvasView)

        self.hasChanged = FALSE
        self.path = os.path.dirname(filename)
        self.filename = os.path.basename(filename)
        self.setCaption(self.filename)
        self.filenameValid = TRUE

        self.loadWidgetSettings(self.path + "/" + self.filename[:-3] + "sav")

    # ###########################################
    # save document as application
    # ###########################################
    def saveDocumentAsApp(self):
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

        #format string with file content
        t = "    "  # instead of tab
        imports = "import sys\nimport os\n"
        instances = "# create widget instances\n" +t+t
        tabs = "# add tabs\n"+t+t
        links = ""
        save = ""
        for widget in self.widgets:
            name = widget.caption
            name = name.replace(" ", "_")
            name = name.replace("(", "")
            name = name.replace(")", "")
            imports = imports + "from " + widget.widget.fileName + " import *\n"
            instances = instances + "self.ow" + name + " = " + widget.widget.fileName + "(self.tabs)\n"+t+t
            tabs = tabs + "self.tabs.insertTab (self.ow" + name + ",\"" + widget.caption + "\")\n"+t+t
            
            for line in widget.inLines:
                name2 = line.outWidget.caption
                name2 = name2.replace(" ", "_")
                name2 = name2.replace("(", "")
                name2 = name2.replace(")", "")
                for signal in line.signals:
                    links = links + "self.ow" + name + ".link(self.ow" + name2 + ", \"" + signal + "\")\n"+t+t

            save = "self.ow" + name + ".saveSettings()\n"+t+t

        classname = os.path.basename(appName)[:-3]
        classname = classname.replace(" ", "_")
        
        classinit = """
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Orange Widgets Panes")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        self.tabs = QTabWidget(self, 'tabWidget')
        self.bottom=QHBox(self)
        self.resize(640,480)
        exitButton=QPushButton("E&xit",self.bottom)"""

        finish = """
a=QApplication(sys.argv)
ow=""" + classname + """()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('lastWindowClosed()'),ow.exit) 
ow.show()
a.exec_loop()"""

        if save != "":
            save = t+"def exit(self):\n" +t+t+ save
        whole = imports + "\n\n" + "class " + classname + "(QVBox):" + classinit + "\n\n"+t+t+ instances + "\n\n"+t+t + tabs + "\n\n"+t+t + links + "\n\n" + save + "\n\n" + finish
        
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