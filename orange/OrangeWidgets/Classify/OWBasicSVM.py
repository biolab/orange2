"""
<name>SVM</name>
<description>Basic SVM</description>
<icon>icons/BasicSVM.png</icon>
<priority>100</priority>
"""

from OWWidget import *
from orngSVM import *
import orngLR_Jakulin
import OWGUI

##############################################################################

class OWBasicSVM(OWWidget):
    settingsList = ["kernel", "gamma", "coef0", "degree", "C", "p", "eps", "nu", "shrinking", "useNu"]
    
    def __init__(self, parent=None, signalManager = None, name='Support Vector Machine'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Examples", ExampleTable, self.cdata)]
        self.outputs = [("Learner", orange.Learner), ("Classifier", orange.Classifier), ("Support Vectors", ExampleTable)]

        self.name = "SVM Learner"
        self.kernel = 2
        self.gamma = 0.0
        self.coef0 = 0.0
        self.degree = 3
        self.C = 1.0
        self.p = 0.5
        self.eps = 1e-3
        self.nu = 0.5
        self.useNu = 0
        self.shrinking = 1
        self.data = None
        
        self.loadSettings()

#        self.controlArea.setSpacing(5)

        validDouble = QDoubleValidator(self.controlArea)
        
        # GUI
        namebox = OWGUI.widgetBox(self.controlArea, "Learner/Classifier name")
        OWGUI.lineEdit(namebox, self, "name")

        bgKernel = QVButtonGroup("Kernel", self.controlArea)
        self.kernelradio = OWGUI.radioButtonsInBox(bgKernel, self, "kernel", btnLabels=["Linear,   x.y", "Polynomial,   (g*x.y+c)^d", "RBF,   exp(-g*(x-y).(x-y))", "Sigmoid,   tanh(g*x.y+c)"], callback=self.changeKernel)
        OWGUI.separator(self.kernelradio)
        self.gcd = OWGUI.widgetBox(bgKernel, orientation="horizontal")
        self.leg = OWGUI.lineEdit(self.gcd, self, "gamma", "g: ", orientation="horizontal", validator = validDouble)
        self.led = OWGUI.lineEdit(self.gcd, self, "coef0", "  c: ", orientation="horizontal", validator = validDouble)
        self.lec = OWGUI.lineEdit(self.gcd, self, "degree", "  d: ", orientation="horizontal", validator = validDouble)
        self.changeKernel()

##        bgKernel = QHButtonGroup("Kernel", self.controlArea)
##        buttons = OWGUI.widgetBox(bgKernel)
##        self.kernelradio = OWGUI.radioButtonsInBox(buttons, self, "kernel", btnLabels=["Linear,   x.y", "Polynomial,   (g*x.y+c)^d", "RBF,   exp(-g*(x-y).(x-y))", "Sigmoid,   tanh(g*x.y+c)"], callback=self.changeKernel)
##        gcd = OWGUI.widgetBox(bgKernel)
##        self.leg = OWGUI.lineEdit(gcd, self, "gamma", "g: ", orientation="horizontal", validator = validDouble)
##        self.led = OWGUI.lineEdit(gcd, self, "coef0", "  c: ", orientation="horizontal", validator = validDouble)
##        self.lec = OWGUI.lineEdit(gcd, self, "degree", "  d: ", orientation="horizontal", validator = validDouble)
##        self.changeKernel()
##
        labwidth = 120
        methodOptions = QVButtonGroup("Options", self.controlArea)
        self.leC = OWGUI.lineEdit(methodOptions, self, "C", "Model complexity (C): ", orientation="horizontal", validator = validDouble, labelWidth = labwidth)

        pValid = QDoubleValidator(self.controlArea)
        pValid.setBottom(0)
        self.lep = OWGUI.lineEdit(methodOptions, self, "p", "Tolerance (p): ", orientation="horizontal", validator = pValid, labelWidth = labwidth)

        epsValid = QDoubleValidator(self.controlArea)
        epsValid.setBottom(1e-6)
        self.leeps = OWGUI.lineEdit(methodOptions, self, "eps", "Numeric precision (eps): ", orientation="horizontal", validator = epsValid, labelWidth = labwidth)

        OWGUI.separator(methodOptions)
        OWGUI.checkBox(methodOptions, self, "shrinking", "Shrinking")
        self.cbNu = OWGUI.checkBox(methodOptions, self, "useNu", "Limit the number of support vectors")
        nuValid = QDoubleValidator(self.controlArea)
        nuValid.setRange(0,1,3)
        self.leNu = OWGUI.lineEdit(methodOptions, self, "nu", "      Fraction (nu): ", orientation="horizontal", validator = nuValid, labelWidth = labwidth)
        self.cbNu.disables = [self.leNu]
        self.leNu.setDisabled(not self.useNu)

# This should not be a settin, it should be change according to the 'data' signal
#        OWGUI.checkBox(methodOptions, self, "oneClass", "One class ...")

        OWGUI.separator(self.controlArea)
        self.applyButton = QPushButton("&Apply", self.controlArea)
        self.connect(self.applyButton, SIGNAL("clicked()"), self.applySettings)

        # create learner with this settings
        self.applySettings()
        self.resize(self.controlArea.width()+140, self.controlArea.height()+10)
        

    enabledMethod = [[], [0, 1, 2], [0], [0, 1]]
    def changeKernel(self):
        em = self.enabledMethod[self.kernel]
        for i, c in enumerate([self.leg, self.led, self.lec]):
            c.setDisabled(i not in em)

    def applySettings(self):
        self.learner = BasicSVMLearner()
        for attr in ("name", "kernel", "degree", "gamma", "coef0", "C", "p", "eps", "nu", "shrinking"):
            setattr(self.learner, attr, getattr(self, attr))
        self.learner.for_nomogram = 1

        ### XXX What to do with this?!
        ### Should we check whether domain has the classVar?
##        if self.methodOneClass.isOn():
##            self.learner.type = -3
        if self.useNu:
            self.learner.type = -2
        else:
            self.learner.type = -1

        self.send("Learner", self.learner)
        self.sendData()

    def sendData(self):
        # should not renew the classifier here (if settings are not applied,
        #the classifier should not change)
        if self.data:
            self.classifier = orngLR_Jakulin.MarginMetaLearner(self.learner, folds = 1)(self.data)
            self.classifier.name = self.name
            self.classifier.domain = self.data.domain
            self.classifier.data = self.data
            self.send("Classifier", self.classifier)
            
            vectors = self.data.getitemsref(self.classifier.classifier.model["SVi"])
            self.send("Support Vectors", vectors)

    def cdata(self,data):
        self.data = data
        self.sendData()

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