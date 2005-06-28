"""
<name>Naive Bayes</name>
<description>Naive Bayesian learner/classifier.</description>
<icon>icons/NaiveBayes.png</icon>
<priority>10</priority>
"""
#
# OWDataTable.py
#
# wishes:
# ignore attributes, filter examples by attribute values, do
# all sorts of preprocessing (including discretization) on the table,
# output a new table and export it in variety of formats.

from OWWidget import *
import OWGUI, orange

##############################################################################

class OWNaiveBayes(OWWidget):
    settingsList = ["m", "name", "probEstimation", "condProbEstimation", "condProbContEstimation", "adjustThreshold", "windowProportion"]
    
    def __init__(self, parent=None, signalManager = None, name='NaiveBayes'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("Naive Bayesian Classifier", orange.BayesClassifier)]
                
        # Settings
        self.m = 2.0                        # m for probability estimation
        self.name = 'Naive Bayes'           # name of the classifier/learner
        self.probEstimation = 0             # relative frequency
        self.condProbEstimation = 0         # relative frequency
        self.condProbContEstimation = 4     # relative frequency
        self.adjustThreshold = 0            # adjust threshold (for binary classes)
        self.windowProportion = 0.5         # Percentage of instances taken in loess learning

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.loadSettings()

        self.m_estimator = orange.ProbabilityEstimatorConstructor_m()        
        self.estMethods=[("Relative Frequency", orange.ProbabilityEstimatorConstructor_relative()),
                         ("Laplace", orange.ProbabilityEstimatorConstructor_Laplace()),
                         ("m-Estimate", self.m_estimator)]
        self.condEstMethods=[("", None),
                             ("Relative Frequency", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_relative())),
                             ("Laplace", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_Laplace())),
                             ("m-Estimate", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=self.m_estimator))]
        self.condEstContMethods=[("", None),
                             ("Relative Frequency", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_relative())),
                             ("Laplace", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_Laplace())),
                             ("m-Estimate", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=self.m_estimator)),
                             ("LOESS", orange.ConditionalProbabilityEstimatorConstructor_loess())]

        # GUI
        # name
        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        # parameters
        box = QVGroupBox(self.controlArea)
        box.setTitle('Probability Estimation')

        width = 123
        itms = [e[0] for e in self.estMethods]
        OWGUI.comboBox(box, self, 'probEstimation', items=itms, label='Unconditional:', labelWidth=width, orientation='horizontal',
                       tooltip='Method to estimate unconditional probability.', callback=self.refreshControls)
        itms = [e[0] for e in self.condEstMethods]
        self.est2 = OWGUI.comboBox(box, self, 'condProbEstimation', items=itms, label='Conditional (discrete):', labelWidth=width, orientation='horizontal',
                                   tooltip='Conditional probability estimation method used for discrete attributes.', callback=self.refreshControls)

        itms = [e[0] for e in self.condEstContMethods]
        self.est3 = OWGUI.comboBox(box, self, 'condProbContEstimation', items=itms, label='Conditional (continuous):', labelWidth=width, orientation='horizontal',
                                   tooltip='Conditional probability estimation method used for continuous attributes.', callback=self.refreshControls)
        self.est3.setEnabled(True)

        OWGUI.separator(box)

        mValid = QDoubleValidator(self.controlArea)
        mValid.setRange(0,10000,1)
        self.mwidget = OWGUI.lineEdit(box, self, 'm', label='Parameter for m-estimate:', labelWidth=width, orientation='horizontal', box=None, tooltip=None, callback=None, valueType = str, validator = mValid)

        self.refreshControls()

#        contBox = QVGroupBox(self.controlArea)
#        contBox.setTitle("Continuous variables")

        kernelSizeValid = QDoubleValidator(self.controlArea)
        kernelSizeValid.setRange(0,1,3)
        OWGUI.lineEdit(box, self, 'windowProportion', label = 'Size of LOESS window:',
                       labelWidth=width, orientation='horizontal', box=None, tooltip='Proportion of examples used for local learning in loess. Use 0 to learn from few local instances (3) and 1 to learn from all in the data set (this kind of learning is not local anymore).',
                       callback=None, valueType = str, validator = kernelSizeValid)
        
        OWGUI.separator(self.controlArea)

        OWGUI.checkBox(self.controlArea, self, "adjustThreshold", "Adjust threshold (for binary classes)", box = "Threshold")
        
        OWGUI.separator(self.controlArea)
        self.applyBtn = OWGUI.button(self.controlArea, self, "&Apply", callback=self.setLearner)
        
        self.resize(150,100)
        self.setLearner()                   # this just sets the learner, no data yet

    def activateLoadedSettings(self):
        self.setLearner()

    # setup the bayesian learner
    def setLearner(self):
        if float(self.m) < 0:
            self.error("Parameter m Out of Bounds: Parameter m should be positive!")
#            QMessageBox.information(self.controlArea, "Parameter m Out of Bounds",
#                                    "Parameter m should be positive!", QMessageBox.Ok)
            return
        if float(self.windowProportion) < 0 or float(self.windowProportion) > 1:
            self.error("Parameter windowProportion (for LOESS) Out of Bounds: Parameter should be between 0.0 and 1.0!")
            return
        self.learner = orange.BayesLearner()
        self.learner.name = self.name
        # set the probability estimation
        self.m_estimator.m = float(self.m)
        self.learner.estimatorConstructor = self.estMethods[self.probEstimation][1]
        if self.condProbEstimation:
            self.learner.conditionalEstimatorConstructor = self.condEstMethods[self.condProbEstimation][1]
        if self.condProbContEstimation:
            self.learner.conditionalEstimatorConstructorContinuous = self.condEstContMethods[self.condProbContEstimation][1]
            if hasattr(self.learner.conditionalEstimatorConstructorContinuous, "windowProportion"):
                setattr(self.learner.conditionalEstimatorConstructorContinuous, "windowProportion", float(self.windowProportion))
            
                
        for attr, cons in ( ("estimatorConstructor", self.estMethods[self.probEstimation][1]),
                            ("conditionalEstimatorConstructor", self.condEstMethods[self.condProbEstimation][1]),
                            ("conditionalEstimatorConstructorContinuous", self.condEstContMethods[self.condProbContEstimation][1])):
            if cons:
                setattr(self.learner, attr, cons)
                if hasattr(cons, "m"):
                    setattr(cons, "m", float(self.m))

        self.learner.adjustThreshold = self.adjustThreshold
        
        self.send("Learner", self.learner)
        if self.data <> None:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.setattr("data", self.data)
            except orange.KernelException, (errValue):
                self.classifier = None
                self.error("Naive Bayes error:"+str(errValue))
#                QMessageBox("Naive Bayes error:", str(errValue), QMessageBox.Warning,
#                            QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
                return            
            self.classifier.name = self.name
            self.send("Classifier", self.classifier)
            self.send("Naive Bayesian Classifier", self.classifier)
        self.error()

    # handles input signal
    def cdata(self,data):
        self.data = data
        if data:
            self.setLearner()
        else:
            self.learner = None
            self.classifier = None
            self.send("Classifier", self.classifier)
            self.send("Naive Bayesian Classifier", self.classifier)

    # signal processing
    def refreshControls(self):
        self.mwidget.box.setEnabled(self.probEstimation==2 or self.condProbEstimation==3 or self.condProbContEstimation==3)

        self.est2.changeItem(QString("same (%s)" % self.estMethods[self.probEstimation][0]), 0)
        if self.est2.currentItem(): # is est3 set to same?
          self.est3.changeItem(QString("same (%s)" % self.condEstMethods[self.condProbEstimation][0]), 0)
        else:
          self.est3.changeItem(QString("same (%s)" % self.estMethods[self.probEstimation][0]), 0)

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNaiveBayes()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
