"""
<name>CN2Classifier</name>
<desctiption>Constructs CN2 learner and classifier from example table </description>
<icon>CN2Classifier.png</icon>
<priority> 60</priority>
"""

from OWWidget import *
import OWGUI
import orange
import orngCN2
import qt
import sys

class OWCN2Classifier(OWWidget):
    settingsList=["QualityButton","CoveringButton","m", "MaxRuleLength", "MinCoverage",
         "BeamWidth", "Alpha"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"CN2Classifier")
        #OWWidget.__init__(self,parent,"Rules")
        self.inputs=[("ExampleTable", orange.ExampleTable, self.dataset)]
        self.outputs=[("Learner", orange.Learner),("Classiffier",orange.Classifier)]
        self.QualityButton=0
        self.CoveringButton=0
        self.Alpha=0.05
        self.BeamWidth=50
        self.MinCoverage=0
        self.MaxRuleLength=0
        self.Weight=50
        self.m=2
        self.LearnerName=""
        self.loadSettings()

        self.data=None

        ##GUI
        labelWidth=150
        self.learnerName=OWGUI.lineEdit(self.controlArea, self, "LearnerName", box="Learner/Classifier Name",tooltip="Name to be used by other widgets to identify yor Learner/Classifier")
        self.learnerName.setText("Rule classifier")
        self.ruleQualityGroup=OWGUI.widgetBox(self.controlArea, self)
        self.ruleQualityGroup.setTitle("Rule Quality Estimation")
        self.ruleValidationGroup=OWGUI.widgetBox(self.controlArea, self)
        self.ruleValidationGroup.setTitle("Pre-Prunning(LRS)")

        OWGUI.spin(self.controlArea, self, "BeamWidth", 1, 100, box="Beam Width", tooltip=" Specify the width of the search beam")

        self.coveringAlgGroup=OWGUI.widgetBox(self.controlArea, self)
        self.coveringAlgGroup.setTitle("Covering algorithm settings")

        self.ruleQualityBG=OWGUI.radioButtonsInBox(self.ruleQualityGroup, self, "QualityButton",
                            btnLabels=["Laplace","m-estimate","WRACC"],
                            box="Rule quality", callback=self.qualityButtonPressed,
                            tooltips=["Laplace rule evaluator", "m-estimate rule evaluator",
                            "WRACC rule evaluator"])

        self.mSpin=Spin=OWGUI.spin(self.ruleQualityGroup, self, "m", 0, 100, label="m",
                orientation="horizontal", labelWidth=labelWidth-100, tooltip="m value for m estimate rule evaluator")
        
        a=OWGUI.doubleSpin(self.ruleValidationGroup, self, "Alpha", 0, 1,0.05, label="Alpha",
                orientation="horizontal", labelWidth=labelWidth)
        OWGUI.spin(self.ruleValidationGroup, self, "MinCoverage", 0, 100,label="Minimum Coverage",
                orientation="horizontal", labelWidth=labelWidth)
        OWGUI.spin(self.ruleValidationGroup, self, "MaxRuleLength", 0, 100,label="Max. Rule Length",
                orientation="horizontal", labelWidth=labelWidth)
        
        self.coveringAlgBG=OWGUI.radioButtonsInBox(self.coveringAlgGroup, self, "CoveringButton",
                            btnLabels=["Exclusive covering ","Weighted Covering"],
                            box="Covering algorithm", callback=self.coveringAlgButtonPressed)
        self.weightSpin=OWGUI.doubleSpin(self.coveringAlgGroup, self, "Weight",0, 1,0.05,label= "Weight",
                orientation="horizontal", labelWidth=labelWidth)
 
        #layout=QVBoxLayout(self.controlArea)
        #layout.add(self.ruleQualityGroup)
        #layout.add(self.ruleValidationGroup)
        #layout.add(self.coveringAlgGroup)

        OWGUI.button(self.controlArea, self, "&Apply Settings", callback=self.applySettings)

        self.qualityButtonPressed()
        self.coveringAlgButtonPressed()
        self.controlArea.setMinimumWidth(300)
        self.resize(100,100)
        self.setLearner()

    def setLearner(self):
        self.learner=orngCN2.CN2UnorderedLearner()
        self.learner.name=self.LearnerName
        ruleFinder=orange.RuleBeamFinder()

        if self.QualityButton==0:
            ruleFinder.evaluator=orange.RuleEvaluator_Laplace()
        elif self.QualityButton==1:
            pass
            #ruleFinder.evaluator=orange.ProbabilityEstimatorConstructor_m(m=self.m)
            ruleFinder.evaluator=orngCN2.mEstimate(self.m)
        elif self.QualityButton==2:
            ruleFinder.evaluator=orngCN2.WRACCEvaluator()

        ruleFinder.ruleStoppingValidator=orange.RuleValidator_LRS(alpha=self.Alpha/100,
                    min_coverage=self.MinCoverage, max_rule_complexity=self.MaxRuleLength)
        ruleFinder.ruleFilter=orange.RuleBeamFilter_Width(width=self.BeamWidth)
        self.learner.ruleFinder=ruleFinder

        if self.CoveringButton==0:
            self.learner.coverAndRemove=orange.RuleCovererAndRemover_Default()
        elif self.CoveringButton==1:
            self.learner.coverAndRemove=orngCN2.CovererAndRemover_multWeights(mult=self.Weight/100)

        self.send("Learner",self.learner)
        self.classifier=None
        if self.data:
            try:
                self.classifier=self.learner(self.data)
                self.classifier.name=self.LearnerName
                self.classifier.setattr("data",self.data)
            except orange.KernelException, (errValue):
                self.classifier=None
        print self.classifier
        print self.learner
        self.send("Classifier", self.classifier)

    def dataset(self, data):
        self.data=data
        if self.data:
            self.setLearner()
        else:
            self.send("Learner",None)
            self.send("Classifier",None)


    def qualityButtonPressed(self):
        id=self.QualityButton=self.ruleQualityBG.id(self.ruleQualityBG.selected())
        if id==1:
            self.mSpin.setDisabled(0)
        else:
            self.mSpin.setDisabled(1)

    def coveringAlgButtonPressed(self):
        id=self.CoveringButton=self.coveringAlgBG.id(self.coveringAlgBG.selected())
        if id==1:
            self.weightSpin.setDisabled(0)
        else:
            self.weightSpin.setDisabled(1)

    def applySettings(self):
        self.setLearner()

class DecSpinBox(QSpinBox):
    def __init__(self, *args):
        apply(QSpinBox.__init__,(self,)+args)
        self.setValidator(QDoubleValidator(self))

    def mapValueToText(self,i):
        return "%i.%i%i" % (i/100,i/10,i%10)
    def interpretText(self):
        self.setValue(int(self.text().toFloat()[0]*100))
        
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWCN2Classifier()
    w.dataset(orange.ExampleTable("../../doc/datasets/titanic.tab"))
    app.setMainWidget(w)
    w.show()
    app.exec_loop()
