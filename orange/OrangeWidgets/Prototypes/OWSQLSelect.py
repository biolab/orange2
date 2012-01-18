"""
<name>SQL Select</name>
<description>Reads data from an SQL select statement.</description>
<icon>icons/SQLSelect.png</icon>
<contact>Gasper Fele-Zorz (polz(@at@)fri.uni-lj.si)</contact>
<priority>10</priority>
"""

#
# OWSQLSelect.py
# Based on the file widget
#

from OWWidget import *

import orngSQL
import OWGUI, os.path

class OWSubSQLSelect(OWWidget):
    allSQLSelectWidgets = []
    settingsList=["recentConnections", "lastQuery"]
    def __init__(self, parent=None, signalManager = None, name = "SQLSelect"):
        OWWidget.__init__(self, parent, signalManager, name)
        OWSubSQLSelect.allSQLSelectWidgets.append(self)

    def destroy(self, destroyWindow, destroySubWindows):
        OWSubSQLSelect.allSQLSelectWidgets.remove(self)
        OWWidget.destroy(self, destroyWindow, destroySubWindows)

    def activateLoadedSettings(self):
        # print "activating", self.recentQueries, ", ",self.recentConnections
        self.query = self.lastQuery
        self.setConnectionList()

    def selectConnection(self, n):
        if n < len(self.recentConnections):
            name = self.recentConnections[n]
            self.recentConnections.remove(name)
            self.recentConnections.insert(0, name)
        if len(self.recentConnections) > 0:
            self.setConnectionList()
            self.connectDB(self.recentConnections[0])

    def setInfo(self, info):
        for (i, s) in enumerate(info):
            self.info[i].setText(s)

    def setMeta(self):
        domain = self.sqlReader.domain
        s = "Attrs:\n    " + "\n    ".join([str(i) for i in domain.attributes]) + "\n" + "Class:" + str(domain.classVar)
        self.domainLabel.setText(s)
        # for i in domain.getmetas():
            # self.propertyCheckBoxes[i].set()

    # checks whether any file widget knows of any variable from the current domain
    def attributesOverlap(self, domain):
        for fw in OWSubFile.allFileWidgets:
            if fw != self and getattr(fw, "dataDomain", None):
                for var in domain:
                    if var in fw.dataDomain:
                        return True
        return False

    # Execute a query, create data from it and send it over the data channel
    def executeQuery(self, query = None, throughReload = 0, DK=None, DC=None):
        if query is None:
            query = str(self.queryTextEdit.toPlainText())
        try:
            self.sqlReader.execute(query)
        except Exception, e:
            self.setInfo(('Query failed:', str(e)))
        self.send("Data", self.sqlReader.data())
        self.setInfo(('Query returned', 'Read ' + str(len(self.sqlReader.data())) + ' examples!'))
        self.send("Feature Definitions", self.sqlReader.domain)
        self.setMeta()
        self.lastQuery = query
    
    def connectDB(self, connectString = None):
        if connectString is None:
            connectString = str(self.connectString)
        if connectString in self.recentConnections: self.recentConnections.remove(connectString)
        self.recentConnections.insert(0, connectString)
        print connectString
        self.sqlReader.connect(connectString)

class OWSQLSelect(OWSubSQLSelect):
    def __init__(self,parent=None, signalManager = None):
        OWSubSQLSelect.__init__(self, parent, signalManager, "SQL select")
        self.sqlReader = orngSQL.SQLReader()
        self.inputs = []
        self.outputs = [("Data", ExampleTable), ("Feature Definitions", orange.Domain)]

        #set default settings
        self.domain = None
        self.recentConnections=["(none)"]
        self.queryFile = None
        self.query = ''
        self.lastQuery = None
        self.loadSettings()
        if self.lastQuery is not None:
            self.query = self.lastQuery
        self.connectString = self.recentConnections[0]
        self.connectBox = OWGUI.widgetBox(self.controlArea, "Database")

        self.connectLineEdit = OWGUI.lineEdit(self.connectBox, self, 'connectString', callback = self.connectDB)
        self.connectCombo = OWGUI.comboBox(self.connectBox, self, 'connectString', items = self.recentConnections, callback = self.selectConnection)
        button = OWGUI.button(self.connectBox, self, 'connect', callback = self.connectDB, disabled = 0)
        #query
        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)

        self.textBox = OWGUI.widgetBox(self, 'SQL select')
        self.splitCanvas.addWidget(self.textBox)
        self.queryTextEdit = QPlainTextEdit(self.query, self)
        self.textBox.layout().addWidget(self.queryTextEdit)

        self.selectBox = OWGUI.widgetBox(self.controlArea, "Select statement")
        # self.selectSubmitBox = QHGroupBox("", self.selectBox)
        # self.queryTextEdit.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        # self.queryTextEdit.setMinimumWidth(300)
        # self.connect(self.queryTextEdit, SIGNAL('returnPressed()'), self.executeQuery)
        OWGUI.button(self.selectBox, self, "Open...", callback=self.openScript)
        OWGUI.button(self.selectBox, self, "Save...", callback=self.saveScript)
        self.selectBox.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        button = OWGUI.button(self.selectBox, self, 'execute!', callback = self.executeQuery, disabled=0)
        self.domainBox = OWGUI.widgetBox(self.controlArea, "Domain")
        self.domainLabel = OWGUI.label(self.domainBox, self, '')
        # info
        self.infoBox = OWGUI.widgetBox(self.controlArea, "Info")
        self.info = []
        self.info.append(OWGUI.label(self.infoBox, self, 'No data loaded.'))
        self.info.append(OWGUI.label(self.infoBox, self, ''))
        self.resize(300,300)

    # set the query combo box
    def setConnectionList(self):
        self.connectCombo.clear()
        if not self.recentConnections:
            self.connectCombo.insertItem("(none)")
        else:
            self.connectLineEdit.setText(self.recentConnections[0])
        for connection in self.recentConnections:
            self.connectCombo.insertItem(connection)
        self.connectCombo.updateGeometry()
    
    def openScript(self, filename=None):
        if self.queryFile is None:
            self.queryFile = ''
        if filename == None:
            self.queryFile = str(QFileDialog.getOpenFileName(self, 'Open SQL file', self.queryFile, 'SQL files (*.sql)\nAll files(*.*)'))    
        else:
            self.queryFile = filename
            
        if self.queryFile == "": return
            
        f = open(self.queryFile, 'r')
        self.queryTextEdit.setPlainText(f.read())
        f.close()
    
    def saveScript(self):
        if self.queryFile is None:
            self.queryFile = ''
        self.queryFile = QFileDialog.getSaveFileName(self, 'Save SQL file', self.queryFile, 'SQL files (*.sql)\nAll files(*.*)')
        
        if self.queryFile:
            fn = ""
            head, tail = os.path.splitext(str(self.queryFile))
            if not tail:
                fn = head + ".sql"
            else:
                fn = str(self.queryFile)
            f = open(fn, 'w')
            f.write(self.queryTextEdit.toPlainText())
            f.close()

if __name__ == "__main__":
    a=QApplication(sys.argv)
    ows=OWSQLSelect()
    ows.activateLoadedSettings()
    a.setMainWidget(ows)
    ows.show()
    a.exec_loop()
    ows.saveSettings()
