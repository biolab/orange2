"""
<name>Naive Bayes</name>
<description>NaiveBayes widget can either construct a Naive Bayesian learner, or 
(if given a data set) a Naive Bayesian classifier.</description>
<category>Classification</category>
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

from OData import *
from OWWidget import *

##############################################################################

class OWNaiveBayes(OWWidget):
    settingsList = ["m", "name", "probEstimation", "condProbEstimation", "condProbContEstimation"]
    
    def __init__(self, parent=None, name='NaiveBayes'):
        OWWidget.__init__(self,
        parent,
        name,
        """NaiveBayes widget can either \nconstruct a Naive Bayesian learner, or,
if given a data set, a Naive Bayesian classifier. \nIt can also be combined with
preprocessors to filter/change the data.
""",
        FALSE,
        FALSE)

        # TODO: dodaj ps
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("Naive Bayesian Classifier", orange.BayesClassifier)]
                
        #self.addInput("cdata")
        #self.addInput("pp")
        #self.addOutput("learner")
        #self.addOutput("classifier")
        #self.addOutput("nbClassifier")

        # Settings
        self.m = 2.0                        # m for probability estimation
        self.name = 'Naive Bayes'           # name of the classifier/learner
        self.probEstimation = 0             # relative frequency
        self.condProbEstimation = 0         # relative frequency
        self.condProbContEstimation = 0     # relative frequency

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.loadSettings()
        
        self.estMethods=[("Relative Frequency", orange.ProbabilityEstimatorConstructor_relative()),
                         ("Laplace", orange.ProbabilityEstimatorConstructor_Laplace()),
                         ("m-Estimate", orange.ProbabilityEstimatorConstructor_m())]
        self.condEstMethods=[("", None),
                             ("Relative Frequency", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_relative())),
                             ("Laplace", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_Laplace())),
                             ("m-Estimate", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_m()))]
        self.condEstContMethods=[("", None),
                             ("Relative Frequency", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_relative())),
                             ("Laplace", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_Laplace())),
                             ("m-Estimate", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_m())),
                             ("LOESS", orange.ConditionalProbabilityEstimatorConstructor_loess())]

        # GUI
        # name
        self.nameBox = QVGroupBox(self.controlArea)
        self.nameBox.setTitle('Learner/Classifier Name')
        QToolTip.add(self.nameBox,"Name to be used by other widgets to identify your learner/classifier.")

        self.nameEdt = QLineEdit(self.nameBox)
        self.nameEdt.setText(self.name)

        QWidget(self.controlArea).setFixedSize(0, 8)
        # parameters
        self.parBox = QVGroupBox(self.controlArea)
        self.parBox.setTitle('Probability Estimation')

        self.labu = QLabel(self.parBox)
        self.labu.setText('Unconditional probabilities ')

        self.estBox1 = QHBox(self.parBox)
        self.lab1 = QLabel(self.estBox1)
        self.lab1.setText('')

        self.est1 = QComboBox(self.estBox1)
        for e in self.estMethods:
            self.est1.insertItem(e[0])
        self.est1.setCurrentItem(self.probEstimation)

        QWidget(self.parBox).setFixedSize(0, 8)

        self.labc = QLabel(self.parBox)
        self.labc.setText('Conditional Probabilities ')
        
        self.estBox2 = QHBox(self.parBox)
        self.lab2 = QLabel(self.estBox2)
        self.lab2.setText('discrete: ')

        self.est2 = QComboBox(self.estBox2)
        for e in self.condEstMethods:
            self.est2.insertItem(e[0])
        self.est2.setCurrentItem(self.condProbEstimation)
        
        self.estBox3 = QHBox(self.parBox)
        self.lab3 = QLabel(self.estBox3)
        self.lab3.setText('continuous: ')

        self.est3 = QComboBox(self.estBox3)
        for e in self.condEstMethods:
            self.est3.insertItem(e[0])
        self.est3.setCurrentItem(self.condProbContEstimation)

        QWidget(self.parBox).setFixedSize(0, 8)

        self.mBox = QHBox(self.parBox)
        self.lab2 = QLabel(self.mBox)
        self.lab2.setText('              m: ')
        
        self.mEdt = QLineEdit(self.mBox)
        self.mEdt.setText(str(self.m))
        self.mValid = QDoubleValidator(self.controlArea)
        self.mValid.setRange(0,10000,1)
        self.mEdt.setValidator(self.mValid)

        self.refreshControls()

        #self.mBox.setDisabled(self.m<>2)        
                
        QWidget(self.controlArea).setFixedSize(0, 8)
        # apply button
        self.applyBtn = QPushButton("&Apply", self.controlArea)
        #self.applyBtn.setFixedSize(70,22)
        
        # signals
        self.connect(self.applyBtn, SIGNAL("clicked()"), self.setLearner)
        self.connect(self.mEdt, SIGNAL("textChanged(const QString &)"), self.setM)
        self.connect(self.nameEdt, SIGNAL("textChanged(const QString &)"), self.setName)
        self.connect(self.est1, SIGNAL("activated(int)"), self.setEst1Method)
        self.connect(self.est2, SIGNAL("activated(int)"), self.setEst2Method)
        self.connect(self.est3, SIGNAL("activated(int)"), self.setEst3Method)
        
        self.resize(150,100)

        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet

    # main part:         

    def setLearner(self):
        if self.m < 0:
            QMessageBox.information(self.controlArea, "Parameter m Out of Bounds",
                                    "Parameter m should be positive!", QMessageBox.Ok)
            return
        
        self.learner = orange.BayesLearner()

        for attr, cons in ( ("estimatorConstructor", self.estMethods[self.probEstimation][1]),
                            ("conditionalEstimatorConstructor", self.condEstMethods[self.condProbEstimation][1]),
                            ("conditionalEstimatorConstructorContinuous", self.condEstContMethods[self.condProbContEstimation][1])):
            if cons:
                setattr(self.learner, attr, cons)
                if hasattr(cons, "m"):
                    setattr(cons, "m", self.m)
            
        self.learner.name = self.name
        self.send("Learner", self.learner)
        if self.data <> None:
            self.classifier = self.learner(self.data)
            self.classifier.name = self.name

            self.send("Classifier", self.classifier)
            self.send("Naive Bayesian Classifier", self.classifier)

    # slots: handle input signals        
        
    def cdata(self,data):
        self.data=data
        self.setLearner()

    def pp():
        pass
        # include preprocessing!!!

    # signal processing

    def setM(self, value):
        if str(value) <> '-' and len(str(value))>0:
            self.m = float(str(value))
        
    def setName(self, value):
        self.name = str(value)

    def refreshControls(self):
        self.mBox.setEnabled(self.probEstimation==2 or self.condProbEstimation==3 or self.condProbContEstimation==3)

        self.est2.changeItem(QString("same (%s)" % self.est1.currentText()), 0)
        if self.est2.currentItem():
          self.est3.changeItem(QString("same (%s)" % self.est2.currentText()), 0)
        else:
          self.est3.changeItem(QString("same (%s)" % self.est1.currentText()), 0)
        
    def setEst1Method(self, value):
        self.probEstimation = value
        self.refreshControls()

    def setEst2Method(self, value):
        self.condProbEstimation = value
        self.refreshControls()

    def setEst3Method(self, value):
        self.condProbContEstimation = value
        self.refreshControls()
        
##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNaiveBayes()
    a.setMainWidget(ow)

    #dataset = orange.ExampleTable('test')
    #od = OrangeData(dataset)
    #ow.cdata(od)

    ow.show()
    a.exec_loop()
    ow.saveSettings()