"""
<name>Logistic Regression</name>
<description>Logistic regression learner/classifier.</description>
<icon>icons/LogisticRegression.svg</icon>
<contact>Martin Mozina (martin.mozina(@at@)fri.uni-lj.si)</contact>
<priority>15</priority>
"""
from OWWidget import *
from orngLR import *
import OWGUI

from orngWrap import PreprocessedLearner

class OWLogisticRegression(OWWidget):
    settingsList = ["univariate", "name", "stepwiseLR", "addCrit", "removeCrit", "numAttr", "zeroPoint", "imputation", "limitNumAttr"]

    def __init__ (self, parent=None, signalManager = None, name = "Logistic regression"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.inputs = [("Data", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner), ("Classifier", orange.Classifier), ("Features", list)]

        from orngTree import TreeLearner
        imputeByModel = orange.ImputerConstructor_model()
        imputeByModel.learnerDiscrete = TreeLearner(measure = "infoGain", minSubset = 50)
        imputeByModel.learnerContinuous = TreeLearner(measure = "retis", minSubset = 50)
        self.imputationMethods = [imputeByModel, orange.ImputerConstructor_average(), orange.ImputerConstructor_minimal(), orange.ImputerConstructor_maximal(), None]
        self.imputationMethodsStr = ["Classification/Regression trees", "Average values", "Minimal value", "Maximal value", "None (skip examples)"]

        self.name = "Logistic regression"
        self.univariate = 0
        self.stepwiseLR = 0
        self.addCrit = 10
        self.removeCrit = 10
        self.numAttr = 10
        self.limitNumAttr = False
        self.zeroPoint = 0
        self.imputation = 1

        self.data = None
        self.preprocessor = None

        self.loadSettings()

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Attribute selection")

        stepwiseCb = OWGUI.checkBox(box, self, "stepwiseLR", "Stepwise attribute selection")
        ibox = OWGUI.indentedBox(box, sep=OWGUI.checkButtonOffsetHint(stepwiseCb))
        form = QFormLayout(
            spacing=8, fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            labelAlignment=Qt.AlignLeft, formAlignment=Qt.AlignLeft
        )
        ibox.layout().addLayout(form)

        addCritSpin = OWGUI.spin(
            ibox, self, "addCrit", 1, 50,
            tooltip="Requested significance for adding an attribute"
        )

        addCritSpin.setSuffix(" %")

        form.addRow("Add threshold", addCritSpin)

        remCritSpin = OWGUI.spin(
            ibox, self, "removeCrit", 1, 50,
            tooltip="Requested significance for removing an attribute"
        )
        remCritSpin.setSuffix(" %")

        form.addRow("Remove threshold", remCritSpin)

        # Need to wrap the check box in a layout to force vertical centering
        limitBox = OWGUI.widgetBox(ibox, "")
        limitCb = OWGUI.checkBox(
            limitBox, self, "limitNumAttr", "Limit number of attributes to",
        )

        limitAttSpin = OWGUI.spin(
            ibox, self, "numAttr", 1, 100,
            tooltip="Maximum number of attributes. Algorithm stops when it " +
                    "selects specified number of attributes."
        )

        limitCb.disables += [limitAttSpin]
        limitCb.makeConsistent()

        form.addRow(limitBox, limitAttSpin)

        stepwiseCb.disables += [ibox]
        stepwiseCb.makeConsistent()

        OWGUI.separator(self.controlArea)

        self.imputationCombo = OWGUI.comboBox(self.controlArea, self, "imputation", box="Imputation of unknown values", items=self.imputationMethodsStr)
        OWGUI.separator(self.controlArea)

        applyButton = OWGUI.button(self.controlArea, self, "&Apply", callback=self.applyLearner, default=True)

        OWGUI.rubber(self.controlArea)
        #self.adjustSize()

        self.applyLearner()

    def sendReport(self):
        if self.stepwiseLR:
            step = "add at %i%%, remove at %i%%" % (self.addCrit, self.removeCrit)
            if self.limitNumAttr:
                step += "; allow up to %i attributes" % self.numAttr
        else:
            step = "No"
        self.reportSettings("Learning parameters",
                            [("Stepwise attribute selection", step),
                             ("Imputation of unknown values", self.imputationMethodsStr[self.imputation])])
        self.reportData(self.data)
        

    def applyLearner(self):
        imputer = self.imputationMethods[self.imputation]
        removeMissing = not imputer

        if self.univariate:
            self.learner = Univariate_LogRegLearner()
        else:
            self.learner = LogRegLearner(removeSingular = True, imputer = imputer, removeMissing = removeMissing,
                                         stepwiseLR = self.stepwiseLR, addCrit = self.addCrit/100., removeCrit = self.removeCrit/100.,
                                         numAttr = self.limitNumAttr and float(self.numAttr) or -1.0)

        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)
        self.learner.name = self.name
        self.send("Learner", self.learner)
        self.applyData()

    def applyData(self):
        classifier = None

        if self.data:
            if self.zeroPoint:
                classifier, betas_ap = LogRegLearner_getPriors(self.data)
                self.error()
                classifier.setattr("betas_ap", betas_ap)
            else:
                try:
                    classifier = self.learner(self.data)
                    self.error()
                except orange.KernelException, (errValue):
                    self.error("LogRegFitter error:"+ str(errValue))

            if classifier:
                classifier.setattr("data", self.data)
                classifier.setattr("betas_ap", None)
                classifier.name = self.name

        self.send("Classifier", classifier)


    def setData(self, data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete, checkMissing=True) and data or None
        self.applyData()
        
    def setPreprocessor(self, pp):
        self.preprocessor = pp
        self.applyLearner()


if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWLogisticRegression()

    #dataset = orange.ExampleTable(r'..\..\doc\datasets\heart_disease')
    #ow.setData(dataset)

    ow.show()
    a.exec_()
    ow.saveSettings()
