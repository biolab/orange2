"""
<name>SQL Select</name>
<description>Reads data from an SQL select statement.</description>
<icon>icons/Sql.png</icon>
<contact>Gasper Fele-Zorz (polz(@at@)fri.uni-lj.si)</contact>
<priority>10</priority>
"""

#
# OWSQLSelect.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
import orngSQL
import OWGUI, string, os.path

class OWSubSQLSelect(OWWidget):
    allSQLSelectWidgets = []
    settingsList=["recentQueries", "recentConnections"]
    def __init__(self, parent=None, signalManager = None, name = "File"):
        OWWidget.__init__(self, parent, signalManager, name)
        OWSubSQLSelect.allSQLSelectWidgets.append(self)

    def destroy(self, destroyWindow, destroySubWindows):
        OWSubSQLSelect.allSQLSelectWidgets.remove(self)
        OWWidget.destroy(self, destroyWindow, destroySubWindows)

    def activateLoadedSettings(self):
        # print "activating", self.recentQueries, ", ",self.recentConnections
        self.setQueryList()
        self.setConnectionList()

    def selectConnection(self, n):
        if n < len(self.recentConnections):
            name = self.recentConnections[n]
            self.recentConnections.remove(name)
            self.recentConnections.insert(0, name)
        if len(self.recentConnections) > 0:
            self.setConnectionList()
            self.connectDB(self.recentConnections[0])

    def selectQuery(self, n):
        if n < len(self.recentQueries):
            name = self.recentQueries[n]
            self.recentQueries.remove(name)
            self.recentQueries.insert(0, name)
        if len(self.recentQueries) > 0:
            self.setQueryList()
            # self.executeQuery(self.recentQueries[0])

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
            query = str(self.queryTextEdit.text())
        if query in self.recentQueries: self.recentQueries.remove(query)
        self.recentQueries.insert(0, query)
        try:
            self.sqlReader.execute(query)
        except Exception, e:
            self.setInfo(('Query failed:', str(e)))
        self.send("Examples", self.sqlReader.data())
        self.setInfo(('Query returned', 'Read ' + str(len(self.sqlReader.data())) + ' examples!'))
        self.send("Attribute Definitions", self.sqlReader.domain)
        self.setMeta()
    
    def connectDB(self, connStr = None):
        if connStr is None:
            connStr = str(self.connectLineEdit.text())
        if connStr in self.recentConnections: self.recentConnections.remove(connStr)
        self.recentConnections.insert(0, connStr)
        print connStr
        self.sqlReader.connect(connStr)

class OWSQLSelect(OWSubSQLSelect):
    def __init__(self,parent=None, signalManager = None):
        OWSubSQLSelect.__init__(self, parent, signalManager, "SQL")
        self.sqlReader = orngSQL.SQLReader()
        self.inputs = []
        self.outputs = [("Examples", ExampleTable), ("Attribute Definitions", orange.Domain)]

        #set default settings
        self.domain = None
        self.recentConnections=["(none)"]
        self.recentQueries=["(none)"]
        #get settings from the ini file, if they exist
        self.loadSettings()
        #GUI
        #database connect
        self.connectBox = QHGroupBox("Database", self.controlArea)

        self.connectLineEdit = QLineEdit(self.connectBox)
        self.connectLineEdit.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred))
        self.connectLineEdit.setMinimumWidth(150)
        # self.connect(self.connectLineEdit, SIGNAL('returnPressed()'), self.connectDB)
        self.connectCombo = QComboBox(self.connectBox)
        self.connect(self.connectCombo, SIGNAL('activated(int)'), self.selectConnection)
        button = OWGUI.button(self.connectBox, self, 'connect', callback = self.connectDB, disabled = 0)
        #query
        self.selectBox = QVGroupBox("Select statement", self.controlArea)
        # self.selectSubmitBox = QHGroupBox("", self.selectBox)
        self.queryTextEdit = QTextEdit(self.selectBox)
        # self.queryTextEdit.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        self.queryTextEdit.setMinimumWidth(300)
        #self.connect(self.queryTextEdit, SIGNAL('returnPressed()'), self.executeQuery)
        self.selectHBox = QHBox(self.selectBox)
        self.queryCombo=QComboBox(self.selectHBox)
        self.queryCombo.setMaximumWidth(300)
        #self.queryCombo.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        #self.queryCombo.setMaximumWidth(400)
        self.connect(self.queryCombo, SIGNAL('activated(int)'), self.selectQuery)
        self.selectBox.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        button = OWGUI.button(self.selectHBox, self, 'execute!', callback = self.executeQuery, disabled=0)
        self.domainBox = QHGroupBox("Domain", self.controlArea)
        self.domainLabel = QLabel('', self.domainBox)
        # info
        self.infoBox = QVGroupBox("Info", self.controlArea)
        self.info = []
        self.info.append(QLabel('No data loaded.', self.infoBox))
        self.info.append(QLabel('', self.infoBox))
        self.resize(300,300)

    # set the query combo box
    def setQueryList(self):
        self.queryCombo.clear()
        if not self.recentQueries:
            self.queryCombo.insertItem("(none)")
        else:
            self.queryTextEdit.setText(self.recentQueries[0])
        for query in self.recentQueries:
            self.queryCombo.insertItem(query)
        #self.filecombo.adjustSize() #doesn't work properly :(
        self.queryCombo.updateGeometry()
    def setConnectionList(self):
        self.connectCombo.clear()
        if not self.recentConnections:
            self.connectCombo.insertItem("(none)")
        else:
            self.connectLineEdit.setText(self.recentConnections[0])
        for connection in self.recentConnections:
            self.connectCombo.insertItem(connection)
        self.connectCombo.updateGeometry()

if __name__ == "__main__":
    a=QApplication(sys.argv)
    ows=OWSQLSelect()
    ows.activateLoadedSettings()
    a.setMainWidget(ows)
    ows.show()
    a.exec_loop()
    ows.saveSettings()
