"""
<name>Logistic Regression</name>
<description>LogisticRegression widget can eithet construct a Logistic learner, or (if given a data set)
a Logistic regression classifier. </description>
<category>Classification</category>
<icon>Unknown.png</icon>
<priority>15</priority>
"""

from OData import *
from OWWidget import *
from orngLR import *

##############################################################################

class OWLogisticRegression(OWWidget):
    settingsList = ["removeSingular", "univariate", "name", "stepwiseLR", "addCrit", "removeCrit", "numAttr"]

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

        # inputs / outputs
        #self.addInput("cdata")
        #self.addInput("pp")
        self.inputs = [("Examples", ExampleTable, self.cdata, 1)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("lrClassifier", orange.LogisticClassifier),("Attributes", list)]
        #self.addOutput("learner")
        #self.addOutput("classifier")
        #self.addOutput("lrClassifier")
        #self.addOutput("attributes")
        
        # Settings
        self.name = "Logistic regression"
        self.removeSingular = 0
        self.univariate = 0
        self.stepwiseLR = 0
        self.addCrit = 0.2
        self.removeCrit = 0.3
        self.numAttr = -1

        self.data = None
        self.preprocessor = None

        self.loadSettings()

        print "remove Singular = "+str(self.removeSingular)        

        
        #name
        self.nameBox = QVGroupBox(self.controlArea)
        self.nameBox.setTitle('Learner/Classifier Name')
        QToolTip.add(self.nameBox,"Name to be used by other widgets to identify your learner/classifier.")

        self.nameEdt = QLineEdit(self.nameBox)
        self.nameEdt.setText(self.name)

        #parameters
        
        # remove singularity
        self.sBox = QVGroupBox(self.controlArea)
        self.removeSingularCB = QCheckBox("Remove singular attributes", self.sBox)
        QToolTip.add(self.removeSingularCB, "Remove attributes causing singularity and constants?")
        self.connect(self.removeSingularCB, SIGNAL("clicked()"), self.setRemoveSingular)

        # use univariate logistic regression
        self.uBox = QVGroupBox(self.controlArea)
        self.univariateCB = QCheckBox("univariate logistic regression", self.uBox)
        QToolTip.add(self.univariateCB, "Fit univariate logistic regression.")
        self.connect(self.univariateCB, SIGNAL("clicked()"), self.setUnivariate)

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
        # perform stepwise lr
        self.newData = self.data
        print "stepWise"
        print self.stepwiseLR
        print "data = " + str(self.data)
        self.Attributes = None
        if self.stepwiseLR and self.data <> None:
            print "grem not" + str(self.addCrit)
            # kako najlažje Qstring  prrevedes =' fdfsdflksdfksd v  floatt
            self.attributes = StepWiseFSS(self.data, addCrit = float(str(self.addCrit)), deleteCrit = float(str(self.removeCrit)))
            print self.attributes
            self.newData = self.newData.select(orange.Domain(self.attributes, self.newData.domain.classVar))
        if self.univariate:
            self.learner = Univariate_LogRegLearner()
        else:            
            self.learner = LogisticLearner(removeSingular = self.removeSingular)
        self.learner.name = self.name
        self.send("Learner", self.learner)        
        if self.data <> None:
            self.classifier = self.learner(self.newData)
            self.classifier.name = self.name
            printOUT(self.classifier)
            self.send("Classifier", self.classifier)
            self.send("lrClassifier", self.classifier)
#if self.stepwiseLR:
            #self.send("attributes", self.attributes)                        
        # if stepwise is on return attribute list

    def activateLoadedSettings(self):
        self.removeSingularCB.setChecked(self.removeSingular)
        self.univariateCB.setChecked(self.univariate)
        self.use_swlr_CB.setChecked(self.stepwiseLR)
        self.addEdt = str(self.addCrit)
        self.removeEdt = str(self.removeCrit)
        self.attrEdt = self.numAttr
        self.refreshControls()
        
        
    def cdata(self,data):
        print "cdata"
        self.data=data
        print "self data v cdata = " + str(self.data)
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

    def setStepwiseLR(self):
        self.stepwiseLR = self.use_swlr_CB.isOn()
        self.refreshControls()

    def setAddCrit(self, value):
        self.addCrit = value
        
    def setRemoveCrit(self, value):
        self.removeCrit = value

    def setNumAttr(self, value):
        self.numAttr = float(value)

        
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

    dataset = orange.ExampleTable('d:\\data\\titanic.tab')
    od = OrangeData(dataset)
    ow.cdata(od)

    ow.show()
    a.exec_loop()
    ow.saveSettings() 
                