# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    document class - main operations (save, load, ...)
#
import sys, os, os.path, string, traceback
from qt import *
from qtcanvas import *
from xml.dom.minidom import Document, parse
import orngView
import orngCanvasItems
import orngResources
import orngTabs
from orngDlgs import *
from orngSignalManager import SignalManager
import cPickle

TRUE  = 1
FALSE = 0


class SchemaDoc(QMainWindow):
    def __init__(self, canvasDlg, *args):
        apply(QMainWindow.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.canSave = 0
        self.resize(700,500)
        self.showNormal()
        self.setCaption("Schema " + str(orngResources.iDocIndex))
        orngResources.iDocIndex = orngResources.iDocIndex + 1
        
        self.enableSave(FALSE)
        self.setIcon(QPixmap(orngResources.file_new))
        self.lines = []                         # list of orngCanvasItems.CanvasLine items
        self.widgets = []                       # list of orngCanvasItems.CanvasWidget items
        self.signalManager = SignalManager()    # signal manager to correctly process signals

        self.documentpath = os.getcwd()
        self.documentname = str(self.caption())
        self.applicationpath = os.getcwd()
        self.applicationname = str(self.caption())
        self.documentnameValid = FALSE
        self.canvas = QCanvas(2000,2000)
        self.canvasView = orngView.SchemaView(self, self.canvas, self)
        self.setCentralWidget(self.canvasView)
        self.canvasView.show()
        

    # we are about to close document
    # ask user if he is sure
    def closeEvent(self,ce):
        if not self.canSave:
            print "accept close"
            self.clear()
            ce.accept()
            return
        res = QMessageBox.information(self,'Qrange Canvas','Do you want to save changes made to schema?','Yes','No','Cancel',0,1)
        if res == 0:
            self.saveDocument()
            ce.accept()
            self.clear()
        elif res == 1:
            #print "accept close"
            self.clear()
            ce.accept()
        else:
            #print "ignore close"
            ce.ignore()

    def enableSave(self, enable):
        self.canSave = enable
        self.canvasDlg.enableSave(enable)

    def focusInEvent(self, ev):
        self.canvasDlg.enableSave(self.canSave)

    # add line connecting widgets outWidget and inWidget
    # if necessary ask which signals to connect
    def addLine(self, outWidget, inWidget, enabled = TRUE):
        # check if line already exists
        line = self.getLine(outWidget, inWidget)
        if line:
            self.resetActiveSignals(outWidget, inWidget, None, enabled)
            return
            
        dialog = SignalDialog(self.canvasDlg, None, "", TRUE)
        dialog.setOutInWidgets(outWidget, inWidget)
        connectStatus = dialog.addDefaultLinks()
        if connectStatus == -1:
            #QMessageBox.critical( None, "Orange Canvas", "Error while connecting widgets. Please rebuild  widget registry (menu Options/Rebuild widget registry) because some of the widgets have now different signals.", QMessageBox.Ok + QMessageBox.Default )
            #return
            self.canvasDlg.menuItemRebuildWidgetRegistry()
            connectStatus = dialog.addDefaultLinks()
        
        if connectStatus == 0:
            QMessageBox.information( None, "Orange Canvas", "Selected widgets don't share a common signal type. Unable to connect.", QMessageBox.Ok + QMessageBox.Default )
            return
        elif connectStatus == -1:
            QMessageBox.critical( None, "Orange Canvas", "Error while connecting widgets. Please rebuild  widget registry (menu Options/Rebuild widget registry) and restart Orange Canvas. Some of the widgets have now different signals.", QMessageBox.Ok + QMessageBox.Default )
            return
        

        # if there are multiple choices, how to connect this two widget, then show the dialog
        if len(dialog.getLinks()) > 1 or dialog.multiplePossibleConnections or dialog.getLinks() == []:
            res = dialog.exec_loop()
            if dialog.result() == QDialog.Rejected:
                return
            
        self.signalManager.setFreeze(1)
        signals = dialog.getLinks()
        for (outName, inName) in signals:
            self.addLink(outWidget, inWidget, outName, inName, enabled)
        
        self.signalManager.setFreeze(0, outWidget.instance)

        # if signals were set correctly create the line, update widget tooltips and show the line
        line = self.getLine(outWidget, inWidget)
        if line:
            outWidget.updateTooltip()
            inWidget.updateTooltip()

        self.enableSave(TRUE)
        return line

    # ####################################
    # reset signals of an already created line
    def resetActiveSignals(self, outWidget, inWidget, newSignals = None, enabled = 1):
        #print "<extra>orngDoc.py - resetActiveSignals() - ", outWidget, inWidget, newSignals
        signals = []
        for line in self.lines:
            if line.outWidget == outWidget and line.inWidget == inWidget:
                signals = line.getSignals()
            
        if newSignals == None:
            dialog = SignalDialog(self.canvasDlg, None, "", TRUE)
            dialog.setOutInWidgets(outWidget, inWidget)
            for (outName, inName) in signals:
                #print "<extra>orngDoc.py - SignalDialog.addLink() - adding signal to dialog: ", outName, inName
                dialog.addLink(outName, inName)

            # if there are multiple choices, how to connect this two widget, then show the dialog
            res = dialog.exec_loop()
            if dialog.result() == QDialog.Rejected: return
                
            newSignals = dialog.getLinks()
            
        for (outName, inName) in signals:
            if (outName, inName) not in newSignals:
                self.removeLink(outWidget, inWidget, outName, inName)
                signals.remove((outName, inName))
        
        self.signalManager.setFreeze(1)
        for (outName, inName) in newSignals:
            if (outName, inName) not in signals:
                self.addLink(outWidget, inWidget, outName, inName, enabled)
        self.signalManager.setFreeze(0, outWidget.instance)
            
        outWidget.updateTooltip()
        inWidget.updateTooltip()

        self.enableSave(TRUE)
        

    # #####################################
    # add one link (signal) from outWidget to inWidget. if line doesn't exist yet, we create it
    def addLink(self, outWidget, inWidget, outSignalName, inSignalName, enabled = 1):
        #print "<extra>orngDoc - addLink() - ", outWidget, inWidget, outSignalName, inSignalName
        # in case there already exists link to inSignalName in inWidget that is single, we first delete it
        widgetInstance = inWidget.instance.removeExistingSingleLink(inSignalName)
        if widgetInstance:
            widget = self.findWidgetFromInstance(widgetInstance)
            existingSignals = self.signalManager.findSignals(widgetInstance, inWidget.instance)
            for (outN, inN) in existingSignals:
                if inN == inSignalName:
                    self.removeLink(widget, inWidget, outN, inSignalName)

        # if line does not exist yet, we must create it
        existingSignals = self.signalManager.findSignals(outWidget.instance, inWidget.instance)
        if not existingSignals:
            line = orngCanvasItems.CanvasLine(self.signalManager, self.canvasDlg, self.canvasView, outWidget, inWidget, self.canvas)
            self.lines.append(line)
            line.setEnabled(enabled)
            line.show()
            outWidget.addOutLine(line)
            outWidget.updateTooltip()
            inWidget.addInLine(line)
            inWidget.updateTooltip()
        else:
            line = self.getLine(outWidget, inWidget)

        ok = self.signalManager.addLink(outWidget.instance, inWidget.instance, outSignalName, inSignalName, enabled)
        if not ok: print "orngDoc.addLink - Error. Unable to add link."
        line.setSignals(line.getSignals() + [(outSignalName, inSignalName)])

    # ####################################
    # remove only one signal from connected two widgets. If no signals are left, delete the line
    def removeLink(self, outWidget, inWidget, outSignalName, inSignalName):
        #print "<extra> orngDoc.py - removeLink() - ", outWidget, inWidget, outSignalName, inSignalName
        self.signalManager.removeLink(outWidget.instance, inWidget.instance, outSignalName, inSignalName)
        
        otherSignals = 0
        for (widget, signalFrom, signalTo, enabled) in self.signalManager.links[outWidget.instance]:
            if widget == inWidget.instance: otherSignals = 1
        if not otherSignals:
            self.removeLine(outWidget, inWidget)

        self.enableSave(TRUE)

    # ####################################
    # remove line line
    def removeLine1(self, line):
        for (outName, inName) in line.signals:
            self.signalManager.removeLink(line.outWidget.instance, line.inWidget.instance, outName, inName)   # update SignalManager

        self.lines.remove(line)
        line.inWidget.removeLine(line)
        line.outWidget.removeLine(line)
        line.inWidget.updateTooltip()
        line.outWidget.updateTooltip()
        line.hide()
        line.remove()
        self.enableSave(TRUE)        

    # ####################################
    # remove line, connecting two widgets
    def removeLine(self, outWidget, inWidget):
        #print "<extra> orngDoc.py - removeLine() - ", outWidget, inWidget
        line = self.getLine(outWidget, inWidget)
        if line: self.removeLine1(line)
        
    # ####################################
    # add new widget
    def addWidget(self, widget, x= -1, y=-1, caption = ""):
        qApp.setOverrideCursor(QWidget.waitCursor)
        try:
            newwidget = orngCanvasItems.CanvasWidget(self.signalManager, self.canvas, self.canvasView, widget, self.canvasDlg.defaultPic, self.canvasDlg)
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that it doesn't crash canvas
            qApp.restoreOverrideCursor()
            return None
        qApp.restoreOverrideCursor()
        
        if x==-1 or y==-1:
            x = self.canvasView.contentsX() + 10
            for w in self.widgets:
                x = max(w.x() + 90, x)
                x = x/10*10
            y = 150
        newwidget.setCoords(x,y)
        newwidget.setViewPos(self.canvasView.contentsX(), self.canvasView.contentsY())

        if caption == "": caption = newwidget.caption
        
        if self.getWidgetByCaption(caption):
            i = 2
            while self.getWidgetByCaption(caption + " (" + str(i) + ")"): i+=1
            caption = caption + " (" + str(i) + ")"
        newwidget.updateText(caption)
        newwidget.instance.setCaptionTitle("Qt " + caption)

        self.signalManager.addWidget(newwidget.instance)
        newwidget.show()
        newwidget.updateTooltip()
        self.widgets.append(newwidget)
        self.enableSave(TRUE)
        self.canvas.update()    
        return newwidget

    # ####################################
    # remove widget
    def removeWidget(self, widget):
        if widget.instance:
            widget.instance.saveSettings()
            
        while widget.inLines != []: self.removeLine1(widget.inLines[0])
        while widget.outLines != []:  self.removeLine1(widget.outLines[0])
                
        self.signalManager.removeWidget(widget.instance)
        self.widgets.remove(widget)
        widget.remove()

        self.enableSave(TRUE)

    def clear(self):
        while self.widgets != []:
            self.removeWidget(self.widgets[0])
        self.canvas.update()

    def enableAllLines(self):
        for line in self.lines:
            self.signalManager.setLinkEnabled(line.outWidget.instance, line.inWidget.instance, 1)
            line.setEnabled(1)
            #line.repaintLine(self.canvasView)
        self.canvas.update()
        self.enableSave(TRUE)

    def disableAllLines(self):
        for line in self.lines:
            self.signalManager.setLinkEnabled(line.outWidget.instance, line.inWidget.instance, 0)
            line.setEnabled(0)
            #line.repaintLine(self.canvasView)
        self.canvas.update()
        self.enableSave(TRUE)

    # return a new widget instance of a widget with filename "widgetName"
    def addWidgetByFileName(self, widgetName, x, y, caption):
        for widget in self.canvasDlg.tabs.allWidgets:
            if widget.getFileName() == widgetName:
                return self.addWidget(widget, x, y, caption)
        return None

    # return the widget instance that has caption "widgetName"
    def getWidgetByCaption(self, widgetName):
        for widget in self.widgets:
            if (widget.caption == widgetName):
                return widget
        return None
                    
    def getWidgetCaption(self, widgetInstance):
        for widget in self.widgets:
            if widget.instance == widgetInstance:
                return widget.caption
        print "Error. Invalid widget instance : ", widgetInstance
        return ""

    # ####################################
    # get line from outWidget to inWidget
    def getLine(self, outWidget, inWidget):
        for line in self.lines:
            if line.outWidget == outWidget and line.inWidget == inWidget: return line
        return None

    # ####################################
    # find orngCanvasItems.CanvasWidget from widget instance
    def findWidgetFromInstance(self, widgetInstance):
        for widget in self.widgets:
            if widget.instance == widgetInstance:
                return widget
        return None

    # ###########################################
    # SAVING, LOADING, ....
    # ###########################################
    def saveDocument(self):
        if not self.documentnameValid:
            self.saveDocumentAs()
        else:
            self.save()

    def saveDocumentAs(self):
        qname = QFileDialog.getSaveFileName( os.path.join(self.documentpath, self.documentname), "Orange Widget Scripts (*.ows)", self, "", "Save File")
        name = str(qname)
        if os.path.splitext(name)[0] == "": return
        if os.path.splitext(name)[1] == "": name = name + ".ows"

        (self.documentpath, self.documentname) = os.path.split(name)
        (self.applicationpath, self.applicationname) = os.path.split(name)
        self.applicationname = os.path.splitext(self.applicationname)[0] + ".py"
        self.setCaption(self.documentname)
        self.documentnameValid = TRUE
        self.save()        

    # ####################################
    # save the file            
    def save(self):
        self.enableSave(FALSE)

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
            temp.setAttribute("widgetName", widget.widget.getFileName())
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
        file = open(os.path.join(self.documentpath, self.documentname), "wt")
        file.write(xmlText)
        file.flush()
        file.close()
        doc.unlink()

        self.saveWidgetSettings(os.path.join(self.documentpath, os.path.splitext(self.documentname)[0]) + ".sav")
        self.canvasDlg.addToRecentMenu(os.path.join(self.documentpath, self.documentname))        

    def saveWidgetSettings(self, filename):
        list = {}
        for widget in self.widgets:
            list[widget.caption] = widget.instance.saveSettingsStr()

        file = open(filename, "wt")
        cPickle.dump(list, file)
        file.close()


    # ####################################                    
    # load a scheme with name "filename"
    def loadDocument(self, filename):
        if not os.path.exists(filename):
            self.close()
            QMessageBox.critical(self,'Qrange Canvas','Unable to find file "'+ filename,  QMessageBox.Ok + QMessageBox.Default)
            return

        # ##################
        #load the data ...
        doc = parse(str(filename))
        schema = doc.firstChild
        widgets = schema.getElementsByTagName("widgets")[0]
        lines = schema.getElementsByTagName("channels")[0]

        # #################
        # open settings
        (self.documentpath, self.documentname) = os.path.split(filename)
        (self.applicationpath, self.applicationname) = os.path.split(filename)
        self.applicationname = os.path.splitext(self.applicationname)[0] + ".py"
        settingsFile = os.path.join(self.documentpath, os.path.splitext(self.documentname)[0] + ".sav")

        settingsList = None
        if os.path.exists(settingsFile):
            file = open(settingsFile, "rt")
            settingsList = cPickle.load(file)
            file.close()

        # ##################
        #read widgets
        widgetList = widgets.getElementsByTagName("widget")
        for widget in widgetList:
            name = widget.getAttribute("widgetName")
            tempWidget = self.addWidgetByFileName(name, int(widget.getAttribute("xPos")), int(widget.getAttribute("yPos")), widget.getAttribute("caption"))
            if not tempWidget:
                QMessageBox.information(self,'Qrange Canvas','Unable to create instance of widget \"'+ name + '\"',  QMessageBox.Ok + QMessageBox.Default)
            else:
                if tempWidget.caption in settingsList:
                    tempWidget.instance.loadSettingsStr(settingsList[tempWidget.caption])
                    tempWidget.instance.activateLoadedSettings()

        # ##################
        #read lines                        
        lineList = lines.getElementsByTagName("channel")
        for line in lineList:
            inCaption = line.getAttribute("inWidgetCaption")
            outCaption = line.getAttribute("outWidgetCaption")
            Enabled = int(line.getAttribute("enabled"))
            signals = line.getAttribute("signals")
            inWidget = self.getWidgetByCaption(inCaption)
            outWidget = self.getWidgetByCaption(outCaption)
            if inWidget == None or outWidget == None:
                print "Unable to create a line due to invalid widget name. Try reinstalling widgets."
                continue

            signalList = eval(signals)
            for (outName, inName) in signalList:
                self.addLink(outWidget, inWidget, outName, inName, Enabled)

        self.canvas.update()
        self.enableSave(FALSE)
        
        self.setCaption(self.documentname)
        self.documentnameValid = TRUE
        self.signalManager.processNewSignals(self.widgets[0].instance)
    

    # ###########################################
    # save document as application
    # ###########################################
    def saveDocumentAsApp(self, asTabs = 1):
        # get filename
        appName = os.path.splitext(self.applicationname)[0] + ".py"
        qname = QFileDialog.getSaveFileName( os.path.join(self.applicationpath, appName) , "Orange Scripts (*.py)", self, "", "Save File as Application")
        if qname.isEmpty(): return

        saveDlg = saveApplicationDlg(None, "", TRUE)

        # add widget captions
        for instance in self.signalManager.widgets:
            widget = None
            for i in range(len(self.widgets)):
                if self.widgets[i].instance == instance: saveDlg.insertWidgetName(self.widgets[i].caption)

        res = saveDlg.exec_loop()
        if saveDlg.result() == QDialog.Rejected:
            return

        shownWidgetList = saveDlg.shownWidgetList
        hiddenWidgetList = saveDlg.hiddenWidgetList
        
        (self.applicationpath, self.applicationname) = os.path.split(str(qname))
        fileName = os.path.splitext(self.applicationname)[0]
        if os.path.splitext(self.applicationname)[1] != ".py": self.applicationname += ".py"

        #format string with file content
        t = "    "  # instead of tab
        imports = "import sys, os, cPickle, orange\nfrom orngSignalManager import *\n"
        imports += """
widgetDir = os.path.join(os.path.split(orange.__file__)[0], "OrangeWidgets")
if os.path.exists(widgetDir):
        for name in os.listdir(widgetDir):
            fullName = os.path.join(widgetDir, name)
            if os.path.isdir(fullName): sys.path.append(fullName)\n\n"""
        
        captions = "# set widget captions\n" +t+t
        instancesT = "# create widget instances\n" +t+t
        instancesB = "# create widget instances\n" +t+t
        tabs = "# add tabs\n"+t+t
        links = "#load settings before we connect widgets\n" +t+t+ "self.loadSettings()\n\n" +t+t + "# add widget signals\n"+t+t + "signalManager.setFreeze(1)\n" +t+t
        buttons = "# create widget buttons\n"+t+t
        buttonsConnect = "#connect GUI buttons to show widgets\n"+t+t
        manager = ""
        loadSett = ""
        saveSett = ""
        progressHandlers = ""

        sepCount = 1
        # gui for shown widgets
        for widgetName in shownWidgetList:
            if widgetName != "[Separator]":
                widget = None
                for i in range(len(self.widgets)):
                    if self.widgets[i].caption == widgetName: widget = self.widgets[i]

                name = widget.caption
                name = name.replace(" ", "_")
                name = name.replace("(", "")
                name = name.replace(")", "")
                imports += "from %s import *\n" % (widget.widget.getFileName())
                instancesT += "self.ow%s = %s (self.tabs)\n" % (name, widget.widget.getFileName())+t+t
                instancesB += "self.ow%s = %s()\n" %(name, widget.widget.getFileName()) +t+t
                captions  += "self.ow%s.setCaptionTitle('Qt %s')\n" %(name, widget.caption) +t+t
                manager += "signalManager.addWidget(self.ow%s)\n" %(name) +t+t
                tabs += """self.tabs.insertTab (self.ow%s, "%s")\n""" % (name , widget.caption) +t+t
                buttons += """owButton%s = QPushButton("%s", self)\n""" % (name, widget.caption) +t+t
                buttonsConnect += """self.connect(owButton%s ,SIGNAL("clicked()"), self.ow%s.reshow)\n""" % (name, name) +t+t
                progressHandlers += "self.ow%s.progressBarSetHandler(self.progressHandler)\n" % (name) +t+t
                loadSett += """self.ow%s.loadSettingsStr(strSettings["%s"])\n""" % (name, widget.caption) +t+t
                loadSett += """self.ow%s.activateLoadedSettings()\n""" % (name) +t+t
                saveSett += """strSettings["%s"] = self.ow%s.saveSettingsStr()\n""" % (widget.caption, name) +t+t
            else:
                buttons += "frameSpace%s = QFrame(self);  frameSpace%s.setMinimumHeight(10); frameSpace%s.setMaximumHeight(10)\n" % (str(sepCount), str(sepCount), str(sepCount)) +t+t
                sepCount += 1

        instancesT += "\n" +t+t + "# create instances of hidden widgets\n" +t+t
        instancesB += "\n" +t+t + "# create instances of hidden widgets\n" +t+t
        
        # gui for hidden widgets
        for widgetName in hiddenWidgetList:
            widget = None
            for i in range(len(self.widgets)):
                if self.widgets[i].caption == widgetName: widget = self.widgets[i]

            name = widget.caption
            name = name.replace(" ", "_")
            name = name.replace("(", "")
            name = name.replace(")", "")
            imports += "from %s import *\n" % (widget.widget.getFileName())
            instancesT += "self.ow%s = %s (self.tabs)\n" % (name, widget.widget.getFileName())+t+t
            manager += "signalManager.addWidget(self.ow%s)\n" %(name) +t+t
            instancesB += "self.ow%s = %s()\n" %(name, widget.widget.getFileName()) +t+t
            tabs += """self.tabs.insertTab (self.ow%s, "%s")\n""" % (name , widget.caption) +t+t
            progressHandlers += "self.ow%s.progressBarSetHandler(self.progressHandler)\n" % (name) +t+t
            loadSett += """self.ow%s.loadSettingsStr(strSettings["%s"])\n""" % (name, widget.caption) +t+t
            loadSett += """self.ow%s.activateLoadedSettings()\n""" % (name) +t+t
            saveSett += """strSettings["%s"] = self.ow%s.saveSettingsStr()\n""" % (widget.caption, name) +t+t
        
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
        buttons += "frameSpace = QFrame(self);  frameSpace.setMinimumHeight(20); frameSpace.setMaximumHeight(20)\n"+t+t
        buttons += "exitButton = QPushButton(\"E&xit\",self)\n"+t+t + "self.connect(exitButton,SIGNAL(\"clicked()\"), application, SLOT(\"quit()\"))\n"+t+t

        classname = os.path.basename(appName)[:-3]
        classname = classname.replace(" ", "_")

        classinit = """
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt %s")""" % (fileName)

        if asTabs == 1:
            classinit += """
        self.tabs = QTabWidget(self, 'tabWidget')
        self.resize(800,600)"""

        progress = """
        statusBar = QStatusBar(self)
        self.caption = QLabel('', statusBar)
        self.caption.setMaximumWidth(200)
        self.caption.hide()
        self.progress = QProgressBar(100, statusBar)
        self.progress.setMaximumWidth(100)
        self.progress.hide()
        self.progress.setCenterIndicator(1)
        statusBar.addWidget(self.caption, 1)
        statusBar.addWidget(self.progress, 1)"""
        """
        else:
            progress =
        self.caption = QLabel('', self)
        self.caption.hide()
        self.progress = QProgressBar(100, self)
        self.progress.hide()
        self.progress.setCenterIndicator(1)
        """

        handlerFunct = """
    def progressHandler(self, widget, val):
        if val < 0:
            self.caption.setText("<nobr>Processing: <b>" + str(widget.captionTitle) + "</b></nobr>")
            self.caption.show()
            self.progress.setProgress(0)
            self.progress.show()
        elif val >100:
            self.caption.hide()
            self.progress.hide()
        else:
            self.progress.setProgress(val)
            self.update()"""    

        loadSettings = """
        
    def loadSettings(self):
        try:
            file = open("%s", "r")
        except:
            return
        strSettings = cPickle.load(file)
        file.close()
        """ % (fileName + ".sav")
        loadSettings += loadSett

        saveSettings = """
        
    def saveSettings(self):
        strSettings = {}
        """ + saveSett + "\n" + t+t + """file = open("%s", "w")
        cPickle.dump(strSettings, file)
        file.close()
        """ % (fileName + ".sav")
        
                
        finish = """
application = QApplication(sys.argv)
ow = """ + classname + """()
application.setMainWidget(ow)
ow.loadSettings()
ow.show()
application.exec_loop()
ow.saveSettings()
"""

        #if save != "":
        #    save = t+"def exit(self):\n" +t+t+ save

        if asTabs:
            whole = imports + "\n\n" + "class " + classname + "(QVBox):" + classinit + "\n\n"+t+t+ instancesT + progressHandlers + "\n"+t+t + progress + "\n" +t+t + manager + "\n"+t+t + tabs + "\n" + t+t + links + "\n" + handlerFunct + "\n\n" + loadSettings + saveSettings + "\n\n" + finish
        else:
            whole = imports + "\n\n" + "class " + classname + "(QVBox):" + classinit + "\n\n"+t+t+ instancesB + "\n\n"+t+t+ captions + "\n"+t+t+ progressHandlers + "\n"+t+t + manager + "\n"+t+t + buttons + "\n" + progress + "\n" +t+t+  buttonsConnect + "\n" +t+t + links + "\n\n" + handlerFunct + "\n\n" + loadSettings + saveSettings + "\n\n" + finish
        
        #save app
        fileApp = open(os.path.join(self.applicationpath, self.applicationname), "wt")
        fileApp.write(whole)
        fileApp.flush()
        fileApp.close()

        # save settings
        self.saveWidgetSettings(os.path.join(self.applicationpath, fileName) + ".sav")
        
        

if __name__=='__main__': 
    app = QApplication(sys.argv)
    dlg = SchemaDoc()
    app.setMainWidget(dlg)
    dlg.show()
    app.exec_loop() 