"""
<name>Learner</name>
<description>Learner is a widget that allows experienced Python and Orange users to define a script with a
learning object. This way, using a powerful combination of Python and Orange, one can define sophisticated
learners and used them in other Orange Widgets.</description>
<category>Classification</category>
<icon>icons/Learner.png</icon>
"""

from OData import *
from OWWidget import *
from OWGUI import *
import os.path
from operator import add


class OWLearnerLearner:
    def __init__(self, function, namespace, parent):
        self.function = function
        self.namespace = namespace
        self.parent = parent
        
    def __call__(self, examples, weightID=0):
        self.namespace["__EXAMPLES"] = examples
        self.namespace["__WEIGHTID"] = weightID
        self.namespace["orange"] = orange
        try:
            res = eval("apply(%s, (__EXAMPLES, __WEIGHTID))" % self.function, self.namespace)
        except Exception, txt:
            QMessageBox("OWLearner: Error in Script", str(txt), QMessageBox.Warning,
                        QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self.parent).show()
            res = None
        del self.namespace["__EXAMPLES"]
        del self.namespace["__WEIGHTID"]
        return res
        
class OWLearner(OWWidget):
    settingsList = ["name", "lastPath", "functionName", "script"]

    def __init__(self, parent=None, name='Learner'):
        OWWidget.__init__(self,
        parent,
        name,
        """Learner is a widget that allows experienced
Python and Orange users to define a script with a
learning object. This way, using a powerful combination
of Python and Orange, one can define sophisticated
learners and used them in other Orange Widgets.""",
        FALSE,
        FALSE)
        
        self.callbackDeposit = []

        self.addInput("cdata")
        self.addInput("pp")
        self.addOutput("learner")
        self.addOutput("classifier")

        # Settings
        self.name = 'Learner'
        self.lastPath = '.'
        self.functionName = ""
        self.script = ""
        self.loadSettings()

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        # self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
        
        # GUI
        # name

        self.nameBox = QVGroupBox(self.controlArea)
        self.nameBox.setTitle('Learner/Classifier Name')
        QToolTip.add(self.nameBox,"Name to be used by other widgets to identify your learner/classifier.")
        lineEditOnly(self.nameBox, self, '', 'name')
        QWidget(self.controlArea).setFixedSize(0, 16)
        
        # Load/Save script
        self.boxFile = QVGroupBox(self.controlArea)
        self.boxFile.setTitle('Script File')
        self.btnLoad = QPushButton("&Load", self.boxFile)
        self.btnSave = QPushButton("&Save", self.boxFile)
        self.btnSaveAs = QPushButton("Save As...", self.boxFile)

        # Name of learning function/class
        self.boxLearner = QVGroupBox(self.controlArea)
        self.boxLearner.setTitle('Learner')
        self.cmbLearner = QComboBox(self.boxLearner)
        self.btnRefresh = QPushButton("&Refresh", self.boxLearner)
        QWidget(self.controlArea).setFixedSize(0, 16)

        # apply button
        self.applyBtn = QPushButton("&Apply", self.controlArea)

        self.layout = QVBoxLayout(self.mainArea)
        self.tedScript = QMultiLineEdit(self.mainArea)
        self.tedScript.setFont(QFont("Courier", 10))
        self.tedScript.setText(QString(self.script))
        self.layout.add(self.tedScript)

        self.refresh()

        self.connect(self.btnLoad, SIGNAL("clicked()"), self.load)
        self.connect(self.btnSave, SIGNAL("clicked()"), self.save)
        self.connect(self.btnSaveAs, SIGNAL("clicked()"), self.saveAs)
        self.connect(self.btnRefresh, SIGNAL("clicked()"), self.refresh)
        self.connect(self.applyBtn, SIGNAL("clicked()"), self.setLearner)
        self.connect(self.cmbLearner, SIGNAL("highlighted ( const QString & )"), ValueCallback(self, "functionName", str))
        self.connect(self.tedScript, SIGNAL("textChanged()"), self.scriptChanged)
        self.resize(500,450)

    # main part:         

    def load(self):
        fname = QFileDialog.getOpenFileName(self.lastPath, "*.py")
        if not fname:
            return

        fname = str(fname)

        try:        
            f = open(fname, "rt")
        except IOError:
            QMessageBox("OWLearner Error: File Not Found", "File %s not found" % fname, QMessageBox.Warning,
                        QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
            return

        self.fname = fname            
        self.lastPath = os.path.dirname(self.fname)
        self.tedScript.setText(reduce(add, f.readlines()))
        self.refresh()
        f.close()

    def save(self):
        if not self.fname:
            self.saveAs()

        try:            
            f = open(self.fname, "wt")
        except IOError:
            self.saveAs()
            return

        f.write(str(self.tedScript.getText()))
        f.close()

    def saveAs(self):
        fname = QFileDialog.getSaveFileName(self.lastPath)
        if not fname:
            return

        fname = str(fname)
        try:
            f = open(self.fname, "wt")
        except IOError:
            QMessageBox("", "Cannot open '%s' for writing" % fname, QMessageBox.Warning,
                        QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self)
            return
        
        f.write(str(self.tedScript.getText()))
        f.close()

    def refresh(self):
        s=str(self.tedScript.text())
        r = {}
        exec(str(self.tedScript.text()), r)
        fn = self.functionName
        self.cmbLearner.clear()
        for key, item in r.items():
            if callable(item):
                self.cmbLearner.insertItem(key)
                if key==fn:
                    self.cmbLearner.setCurrentItem(self.cmbLearner.count()-1)
        self.functionName = str(self.cmbLearner.currentText())


    def scriptChanged(self):
        self.script = str(self.tedScript.text())
        
    def setLearner(self):
        t = str(self.tedScript.text())
        if t and self.functionName:
            r = {}
            exec(str(t), r)
            self.learner = OWLearnerLearner(self.functionName, r, self)
            self.learner.name = self.name                                                                               
            self.send("learner", self.learner)
            if self.data:
                self.classifier = self.learner(self.data)
                #self.classifier.name = self.name
                self.send("classifier", self.classifier)
                print self.classifier
    
    # slots: handle input signals        
        
    def cdata(self,data):
        self.data=data.table
        self.setLearner()

    def pp():
        pass
        # include preprocessing!!!

    # signal processing

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWLearner()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('test')
    od = OrangeData(dataset)
    ow.cdata(od)

    ow.show()
    a.exec_loop()
    ow.saveSettings()