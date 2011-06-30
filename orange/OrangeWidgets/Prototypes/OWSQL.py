"""<name>SQL</name>
<description>Constructs Data Using SQL Queries</description>
<icon>icons/SQL.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
import orngSQL, orngEnviron
from urllib import quote

class OWSQL(OWWidget):
    settingsList = ["host", "username", "password", "showPassword", "database", "query", "library", "lastDir", "databaseType"]
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Compare Examples", wantMainArea=0)
        self.outputs = [("Examples", ExampleTable, Default)]

        self.host = "localhost"
        self.username = "root"
        self.password = self.database = self.query = ""
        self.showPassword = False
        self.library = []
        self.lastDir = orngEnviron.defaultReportsDir
        self.databaseType = 0
        self.resize(600, 300)
        self.loadSettings()
        
        self.error = ""
        self.databaseTypes = ["MySQL", "Postgres"]
                
        self.currentName = ""
        b1 = OWGUI.widgetBox(self.controlArea, box="Server", orientation=0)
        OWGUI.comboBox(b1, self, "databaseType", label="Type", items = self.databaseTypes)
        OWGUI.lineEdit(b1, self, "host", label="Host")
        OWGUI.lineEdit(b1, self, "username", label="Username")
        self.lePwd = OWGUI.lineEdit(b1, self, "password", label="Password")
        OWGUI.lineEdit(b1, self, "database", label="Database")
        self.showPasswordChanged()
       
        b1 = OWGUI.widgetBox(self.controlArea, box="Query")
        self.queryEdit = QTextEdit() # don't put the string here, it's interpreted as HTML
        self.queryEdit.setText(self.query)
        b1.layout().addWidget(self.queryEdit)
        b2 = OWGUI.widgetBox(b1, orientation=0)
        errorIconName = os.path.join(orngEnviron.canvasDir, "icons", "error.png")
        self.errorIcon = OWGUI.widgetLabel(b2, "")
        self.errorIcon.setPixmap(QPixmap(errorIconName))
        self.errorIcon.setVisible(False)
        OWGUI.label(b2, self, "%(error)s")
        OWGUI.rubber(b2)
        OWGUI.button(b2, self, "Save to Library", callback=self.saveToLibrary)
        OWGUI.separator(b2, width=40)
        OWGUI.button(b2, self, "Execute", callback=self.executeQuery)

        OWGUI.separator(b1)
        OWGUI.widgetLabel(b1, "Library")
        self.libraryList = QListWidget(self)
        for name, query in self.library:
            li = QListWidgetItem(name)
            li.setToolTip(query)
            self.libraryList.addItem(li)
        b1.layout().addWidget(self.libraryList)
        b2 = OWGUI.widgetBox(b1, orientation=0)
        OWGUI.button(b2, self, "Load Query", callback=self.loadFromLibrary)
        OWGUI.button(b2, self, "Add to Query", callback=self.addFromLibrary)
        OWGUI.button(b2, self, "Remove from library", callback=self.removeFromLibrary)
        OWGUI.rubber(b2)
        OWGUI.button(b2, self, "Save Library to File", callback=self.saveToFile)
        

    def showPasswordChanged(self):
        self.lePwd.setEchoMode(QLineEdit.Normal if self.showPassword else QLineEdit.Password)

    def executeQuery(self):
        self.query = str(self.queryEdit.toPlainText()).strip()
        try:
            sqlReader = orngSQL.SQLReader("%s://%s:%s@%s/%s" % 
                                          (self.databaseTypes[self.databaseType].lower(),
                                           quote(self.username), quote(self.password), quote(self.host), quote(self.database)))
            sqlReader.execute(self.query)
            data = sqlReader.data()
            self.error = ""
            self.errorIcon.setVisible(False)
        except Exception, d:
            for m in reversed(d.args):
                if isinstance(m, (str, unicode)):
                    self.error = "Error: " + m
                    break
            else:
                self.error = "Error: " + str(d)
            self.errorIcon.setVisible(True)
            data = None
        self.send("Examples", data)
        
    def saveToLibrary(self):
        ci = self.libraryList.currentItem()
        name, ok = QInputDialog.getText(self, "Query name", "Query name", QLineEdit.Normal, ci and ci.text() or "")
        if not ok:
            return
        name = str(name)
        query = str(self.queryEdit.toPlainText()).strip()
        for row, (sname, squery) in enumerate(self.library):
            if name == sname:
                self.library[row] = (name, query)
                ci.setToolTip(query)
                break
        else:
            self.library.append((name, query))
            li = QListWidgetItem(name)
            li.setToolTip(query)
            self.libraryList.addItem(li)
    
    def loadFromLibrary(self):
        cr = self.libraryList.currentRow()
        if cr > -1:
            self.queryEdit.setText(self.library[cr][1])
    
    def addFromLibrary(self):
        cr = self.libraryList.currentRow()
        if cr > -1:
            self.queryEdit.setText(self.queryEdit.toPlainText() + "\n\n" + self.library[cr][1])
    
    def removeFromLibrary(self):
        cr = self.libraryList.currentRow()
        if cr > -1:
            del self.library[cr]
            self.libraryList.takeItem(cr)
    
    def saveToFile(self):
        fname = str(QFileDialog.getSaveFileName(self, "File name", self.lastDir, "SQL File (*.sql)\nAll files (*.*)"))
        if not fname:
            return
        self.lastDir = os.path.split(fname)[0]
        file(fname, "wt").write("\n\n".join("-- %s\n%s" % l for l in self.library))