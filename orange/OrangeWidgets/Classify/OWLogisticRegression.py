"""
<name>Logistic Regression</name>
<description>LogisticRegression widget can eithet construct a Logistic learner, or (if given a data set)
a Logistic regression classifier. </description>
<category>Classification</category>
<icon>icons/LogisticRegression.png</icon>
<priority>15</priority>
"""

from OData import *
from OWWidget import *
from orngLR import *
import OWGUI

##############################################################################

class OWLogisticRegression(OWWidget):
    settingsList = ["removeSingular", "univariate", "name", "stepwiseLR", "addCrit", "removeCrit", "numAttr", "zeroPoint", "imputation"]

    def __init__ (self, parent=None, name = "Logistic regression"):    
        # tu pridejo vse nastavitve
        # nastavitve: doloci = TODO
        OWWidget.__init__(self,
        parent,
        name,
        """LogisticRegression widget can eithet construct a Logistic learner, or (if given a data set)
a Logistic regression classifier. \nIt can induce a logistic regression classifier on all
attriubte, but it can also perform stepwise logistic regression. \nIt can also be combined with
preprocessors to filter/change the data.
""",
        FALSE,
        FALSE)

        self.imputationMethodsStr = ["Classification/Regression trees", "Average values", "Minimal value", "Maximal value", "None (skip examples)"]
        self.imputationMethods = [orange.ImputerConstructor_model(), orange.ImputerConstructor_average(), orange.ImputerConstructor_minimal(), orange.ImputerConstructor_maximal(), None]
        # inputs / outputs
        #self.addInput("cdata")
        #self.addInput("pp")
        self.inputs = [("Examples", ExampleTable, self.cdata, 1)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("Attributes", list)]
        #self.addOutput("learner")
        #self.addOutput("classifier")
        #self.addOutput("lrClassifier")
        #self.addOutput("attributes")

        self.callbackDeposit = []
        
        # Settings
        self.name = "Logistic regression"
        self.removeSingular = 1
        self.univariate = 0
        self.stepwiseLR = 0
        self.addCrit = 0.1
        self.removeCrit = 0.1
        self.numAttr = -1
        self.zeroPoint = 0
        self.imputation = 1

        self.data = None
        self.preprocessor = None


        self.loadSettings()


        
        #name
        self.nameBox = QVGroupBox(self.controlArea)
        self.nameBox.setTitle('Learner/Classifier Name')
        QToolTip.add(self.nameBox,"Name to be used by other widgets to identify your learner/classifier.")

        self.nameEdt = QLineEdit(self.nameBox)
        self.nameEdt.setText(self.name)

        #parameters
        
        # remove singularity
        self.removeSingularCB = QCheckBox("Remove singular attributes", self.controlArea)
        QToolTip.add(self.removeSingularCB, "Remove attributes that cause singularity and constants")
        self.connect(self.removeSingularCB, SIGNAL("clicked()"), self.setRemoveSingular)

        # use univariate logistic regression
        self.univariateCB = QCheckBox("Univariate logistic regression", self.controlArea)
        QToolTip.add(self.univariateCB, "Fit univariate logistic regression.")
        self.connect(self.univariateCB, SIGNAL("clicked()"), self.setUnivariate)
        self.univariateCB.setDisabled(True)

        # get 0-point betas ?
        self.zeroCB = QCheckBox("Calculate 0-point for nomograms", self.controlArea)
        QToolTip.add(self.zeroCB, "Basic logistic regression does not compute prior contribution of each attribute to class \
                                   If nomograms are used to visualize logistic regression model, this could be very helpful.")
        self.connect(self.zeroCB, SIGNAL("clicked()"), self.setZeroPoint)
        self.zeroCB.setDisabled(True)

        self.imputationCombo = OWGUI.comboBox(self.controlArea, self, "imputation", items=self.imputationMethodsStr)
        
        # stepwise logistic regression
        self.swBox = QVGroupBox(self.controlArea)
        self.swBox.setTitle('Stepwise logistic regression')

        self.use_swlr_CB = QCheckBox("Apply stepwise logistic regression", self.swBox)
        QToolTip.add(self.use_swlr_CB, "Use only attributes selected by stepwise logistic regression.")
        self.connect(self.use_swlr_CB, SIGNAL("clicked()"), self.setStepwiseLR)


        self.acBox = QHBox(self.swBox)
        self.labAdd = QLabel('Add criteria:           ', self.acBox)
        QToolTip.add(self.labAdd, "Requested significance of attribute to be added in common model.")
        self.addEdt = QLineEdit(self.acBox)

        self.rcBox = QHBox(self.swBox)
        self.labDel = QLabel('Remove criteria:    ', self.rcBox)
        QToolTip.add(self.labDel, "Requested significance of attribute to be removed from common model.")
        self.removeEdt = QLineEdit(self.rcBox)

        self.nBox = QHBox(self.swBox)
        self.labAttr = QLabel('Num. of attributes: ', self.nBox)
        QToolTip.add(self.labAttr, "Maximum number of selected attributes. Algorithm stops when it selects specified number of attributes.\n Use -1 for infinity.")
        self.attrEdt = QLineEdit(self.nBox)

        self.addEdt.setText(str(self.addCrit))
        self.removeEdt.setText(str(self.removeCrit))
        self.attrEdt.setText(str(self.numAttr))
        

        self.refreshControls()
        QWidget(self.controlArea).setFixedSize(0, 8)
        # apply button
        self.applyBtn = QPushButton("&Apply", self.controlArea)
        self.connect(self.applyBtn, SIGNAL("clicked()"), self.setLearner)        
        
        self.resize(150,100)

        #signals
        self.connect(self.applyBtn, SIGNAL("clicked()"), self.setLearner)
        self.connect(self.nameEdt, SIGNAL("textChanged(const QString &)"), self.setName)
        self.connect(self.addEdt, SIGNAL("textChanged(const QString &)"), self.setAddCrit)
        self.connect(self.removeEdt, SIGNAL("textChanged(const QString &)"), self.setRemoveCrit)
        self.connect(self.attrEdt, SIGNAL("textChanged(const QString &)"), self.setNumAttr)

        #connect controls to appropriate functions
        self.activateLoadedSettings()

        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
                                            
    def setLearner(self):
        imputer = self.imputationMethods[self.imputation]
        removeMissing = not imputer

        if self.univariate:
            self.learner = Univariate_LogRegLearner()
        else:            
            self.learner = LogRegLearner(removeSingular = self.removeSingular, imputer = imputer, removeMissing = removeMissing)
            if self.stepwiseLR:
                self.learner.stepwiseLR = 1
                self.learner.addCrit = float(self.addCrit)
                self.learner.removeCrit = float(self.removeCrit)
                self.learner.numAttr = float(self.numAttr)

        self.learner.name = self.name
        self.send("Learner", self.learner)        

        if self.data:
            if self.zeroPoint:
                self.classifier, betas_ap = LogRegLearner_getPriors(self.data)
                self.classifier.betas_ap = betas_ap
            else:
                try:
                    self.classifier = self.learner(self.data)
                except orange.KernelException, (errValue):
                    self.classifier = None
                    QMessageBox("LogRegFitter error:", str(errValue), QMessageBox.Warning,
                                QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
                    return
            self.classifier.betas_ap = None
                    
            self.classifier.name = self.name
            self.send("Classifier", self.classifier)

    def activateLoadedSettings(self):
        self.removeSingularCB.setChecked(self.removeSingular)
        self.univariateCB.setChecked(self.univariate)
        self.use_swlr_CB.setChecked(self.stepwiseLR)
        self.addEdt = str(self.addCrit)
        self.removeEdt = str(self.removeCrit)
        self.attrEdt = self.numAttr
        self.zeroCB.setChecked(self.zeroPoint)
        self.refreshControls()
        
        
    def cdata(self,data):
        self.data=data
        self.setLearner()

    def pp():
        pass
        # include preprocessing!!!

    def refreshControls(self):
        self.acBox.setEnabled(self.use_swlr_CB.isOn())
        self.rcBox.setEnabled(self.use_swlr_CB.isOn())
        self.nBox.setEnabled(self.use_swlr_CB.isOn())
        
    def setName(self, value):
        self.name = str(value)

    def setRemoveSingular(self):
        self.removeSingular = self.removeSingularCB.isOn()

    def setUnivariate(self):
        self.univariate = self.univariateCB.isOn()

    def setZeroPoint(self):
        self.zeroPoint = self.zeroCB.isOn()
        
    def setStepwiseLR(self):
        self.stepwiseLR = self.use_swlr_CB.isOn()
        self.refreshControls()

    def setAddCrit(self, value):
        self.addCrit = str(value)
        
    def setRemoveCrit(self, value):
        self.removeCrit = str(value)

    def setNumAttr(self, value):
        self.numAttr = str(value)

        
    # ostanejo se razne nastavitve posameznih vrednosti v nastavitvah, ki pa jih je s
    # sploh potrebno se izumtr

    # Possible attribute setting :
    #    * stepwise logistic regression (addCrit, deleteCrit, number_of_attributes)
    #    * remove singular attributes

    # adding a trace windows would work perhaps --> better option
    # processing and a "slide" window at the bottom ? --> this couldn't work

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWLogisticRegression()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable(r'C:\Documents and Settings\janez\Desktop\crush\crush.tab')
    ow.cdata(dataset)

    ow.show()
    a.exec_loop()
    ow.saveSettings() 
