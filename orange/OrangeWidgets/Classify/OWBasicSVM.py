"""
<name>SVM</name>
<description>Basic SVM</description>
<category>Classification</category>
<icon>icons/BasicSVM.png</icon>
<priority>100</priority>
"""

from OWWidget import *
from orngSVM import *
import orngLR_Jakulin

##############################################################################

class OWBasicSVM(OWWidget):
    settingsList = ["kernelMethod", "polyD", "polyG", "polyC", "rbfG", "sigmaG", "sigmaC", "nu", "p", "c", "eps", "shrinking", "nuOptionEnabled", "methodIsOneClass"]
    
    def __init__(self, parent=None, name='Support Vector Machine'):
        OWWidget.__init__(self,
        parent,
        name,
        """Basic SVM widget can either \nconstruct a learner, or,
if given a data set, a classifier. \nIt can also be combined with
preprocessors to filter/change the data.
""",
        FALSE,
        FALSE)

        self.inputs = [("Examples", ExampleTable, self.cdata, 1)]
        self.outputs = [("Learner", orange.Learner), ("Classifier", orange.Classifier), ("Support Vectors", ExampleTable)]

        self.kernelMethod = 0                
        self.data = None
        self.polyD = 3
        self.polyG = 0.0
        self.polyC = 0.0
        self.rbfG = 0.0
        self.sigmaG = 0.0
        self.sigmaC = 0.0
        self.nu = 0.5
        self.p = 0.5
        self.c = 1.0
        self.eps = 1e-3
        self.shrinking = 1
        self.nuOptionEnabled = 0
        self.methodIsOneClass = 0
        
        # Settings
        self.name = "SVM Learner"
        self.loadSettings()

        self.controlArea.setSpacing(5)
        
        # GUI
        self.nameBox = QVGroupBox(self.controlArea)
        self.nameBox.setTitle('Learner/Classifier Name')
        QToolTip.add(self.nameBox,"Name to be used by other widgets to identify your learner/classifier.")
        self.nameEdt = QLineEdit(self.nameBox)
        self.nameEdt.setText(self.name)

        self.kernelCaption = QLabel("Select kernel method:", self.controlArea)
        self.KernelTabs = QTabWidget(self.controlArea, 'kernel')

        # linear kernel: x.y
        # polynomial kernel: (g*x.y+c)^d
        # RBF (default): e^(-g(x-y).(x-y))
        # sigmoid: tanh(g*x.y+c)

        # kernel tabs
        self.linearTab = QVBox(self)
        self.polyTab = QVBox(self)
        self.rbfTab = QVBox(self)
        self.sigmaTab = QVBox(self)
        self.KernelTabs.insertTab(self.linearTab, "Linear")
        self.KernelTabs.insertTab(self.polyTab, "Polynomial")
        self.KernelTabs.insertTab(self.rbfTab, "RBF")
        self.KernelTabs.insertTab(self.sigmaTab, "Sigmoid")
        self.KernelTabs.setCurrentPage(self.kernelMethod)

        self.linearCaption = QLabel("Kernel Function: x.y", self.linearTab)
        self.polyCaption = QLabel("Kernel Function: (g*x.y+c)^d", self.polyTab)
        self.rbfCaption = QLabel("Kernel Function: e^(-g(x-y).(x-y))", self.rbfTab)
        self.sigmaCaption = QLabel("Kernel Function: tanh(g*x.y+c)", self.sigmaTab)
        
        polyGBox = QHBox(self.polyTab)
        self.polyGCaption = QLabel("g:", polyGBox)
        self.polyGValue = QLineEdit(str(self.polyG), polyGBox)

        polyCBox = QHBox(self.polyTab)
        self.polyCCaption = QLabel("c:", polyCBox)
        self.polyCValue = QLineEdit(str(self.polyC), polyCBox)

        polyDBox = QHBox(self.polyTab)
        self.polyDCaption = QLabel("d:", polyDBox)
        self.polyDValue = QLineEdit(str(self.polyD), polyDBox)

        rbfGBox = QHBox(self.rbfTab)
        self.rbfGCaption = QLabel("g:", rbfGBox)
        self.rbfGValue = QLineEdit(str(self.rbfG), rbfGBox)

        sigmaGBox = QHBox(self.sigmaTab)
        self.sigmaGCaption = QLabel("g:", sigmaGBox)
        self.sigmaGValue = QLineEdit(str(self.sigmaG), sigmaGBox)

        sigmaCBox = QHBox(self.sigmaTab)
        self.sigmaCCaption = QLabel("c:", sigmaCBox)
        self.sigmaCValue = QLineEdit(str(self.sigmaC), sigmaCBox)        
        

        # method tabs
        self.methodOptions = QVButtonGroup("Method Options", self.controlArea)
        self.methodButtons = QHButtonGroup("Method", self.methodOptions)
        self.methodButtons.setExclusive(TRUE)
        self.methodClassifier = QRadioButton("Supervised", self.methodButtons)
        self.methodOneClass = QRadioButton("One Class", self.methodButtons)
        if self.methodIsOneClass: self.methodButtons.setButton(1)
        else:                     self.methodButtons.setButton(0)

        NuHBox = QHBox(self.methodOptions)
        self.nuOption = QCheckBox("nu:", NuHBox)
        self.nuOption.setChecked(self.nuOptionEnabled)
        self.nuValue  = QLineEdit(str(self.nu), NuHBox)
        self.nuValid = QDoubleValidator(self.controlArea)
        self.nuValid.setRange(0,1,3)
        self.nuValue.setValidator(self.nuValid)

        cHBox = QHBox(self.methodOptions)
        self.cCaption = QLabel('C:', cHBox)
        self.cValue = QLineEdit(str(self.c), cHBox)
        
        pHBox = QHBox(self.methodOptions)
        self.pCaption = QLabel("p:", pHBox)
        self.pValue = QLineEdit(str(self.p), pHBox)
        self.pValid = QDoubleValidator(self.controlArea)
        self.pValid.setBottom(0)
        self.pValue.setValidator(self.pValid)

        epsHBox = QHBox(self.methodOptions)
        self.epsCaption = QLabel("eps:", epsHBox)
        self.epsValue = QLineEdit(str(self.eps), epsHBox)
        self.epsValid = QDoubleValidator(self.controlArea)
        self.epsValid.setBottom(1e-6)
        self.epsValue.setValidator(self.epsValid)

        self.shrinkingCB = QCheckBox("Shrinking", self.methodOptions)
        self.shrinkingCB.setChecked(self.shrinking)

        # apply button
        self.applyButton = QPushButton("&Apply", self.controlArea)
        self.connect(self.applyButton, SIGNAL("clicked()"), self.applySettings)

        # create learner with this settings
        self.applySettings()
        self.resize(self.controlArea.width()+140, self.controlArea.height()+10)
        

    def applySettings(self):
        self.updateSettings()
        self.learner = BasicSVMLearner()
        self.learner.name = self.name

        pageIndex = self.KernelTabs.currentPageIndex()
        if pageIndex == 0:  # linear
            self.learner.kernel = 0
            self.learner.for_nomogram = 1

        elif pageIndex == 1:    # polynomial
            self.learner.kernel = 1
            self.learner.degree = int(str(self.polyDValue.text()))
            self.learner.gamma = float(str(self.polyGValue.text()))
            self.learner.coef0 = float(str(self.polyCValue.text()))

        elif pageIndex == 2:
            self.learner.kernel = 2     # rbf
            self.learner.gamma = float(str(self.rbfGValue.text()))

        else:
            self.learner.kernel = 3     # sigmoid
            self.learner.gamma = float(str(self.sigmaGValue.text()))
            self.learner.coef0 = float(str(self.sigmaCValue.text()))

        if self.methodOneClass.isOn():
            self.learner.type = -3
        else:
            self.learner.type = -1
        if self.nuOption.isChecked():
            self.learner.type = -2
            self.learner.nu = float(str(self.nuValue.text()))

        self.learner.C = float(str(self.cValue.text()))
        self.learner.p = float(str(self.pValue.text()))
        self.learner.eps = float(str(self.epsValue.text()))
        self.learner.shrinking = self.shrinkingCB.isChecked()
            
        self.send("Learner", self.learner)
        if self.data <> None:
            try: 
                self.classifier = orngLR_Jakulin.MarginMetaLearner(self.learner,folds = 1)(self.data)
                self.classifier.name = self.name
                self.classifier.domain = self.data.domain
                self.classifier.data = self.data
                self.send("Classifier", self.classifier)
                vectors = self.data.getitemsref(self.classifier.classifier.model["SVi"])
                self.send("Support Vectors", vectors)
            except Exception, (errValue):
                self.classifier = None
                self.error("SVM error:"+ str(errValue))
#                QMessageBox("SVM error:", str(errValue), QMessageBox.Warning,
#                            QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
                return                
            except:
                self.classifier = None
                self.error("SVM error:"+ "Unidentified error!")
#                QMessageBox("SVM error:", "Unidentified error!", QMessageBox.Warning,
#                            QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
                return                
        self.error()
        
    def cdata(self,data):
        self.data=data
        self.applySettings()

    def pp():
        pass
        # include preprocessing!!!

    def updateSettings(self):
        self.polyD = int(str(self.polyDValue.text()))
        self.polyG = float(str(self.polyGValue.text()))
        self.polyC = float(str(self.polyCValue.text()))
        self.rbfG = float(str(self.rbfGValue.text()))
        self.sigmaG = float(str(self.sigmaGValue.text()))
        self.sigmaC = float(str(self.sigmaCValue.text()))
        self.nu = float(str(self.nuValue.text()))
        self.p = float(str(self.pValue.text()))
        self.c = float(str(self.cValue.text()))
        self.eps = float(str(self.epsValue.text()))
        self.shrinking = self.shrinkingCB.isChecked()
        self.nuOptionEnabled = self.nuOption.isChecked()
        
        if self.methodButtons.selected() == self.methodOneClass:
            self.methodIsOneClass = 1
        else:
            self.methodIsOneClass = 0

    def saveSettings(self, file = None):
        self.updateSettings()
        OWBaseWidget.saveSettings(self, file)
        
    def saveSettingsStr(self):
        self.updateSettings()
        OWBaseWidget.saveSettingsStr(self)

        
##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWBasicSVM()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()
    ow.saveSettings()