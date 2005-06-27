"""
<name>CN2</name>
<description>Rule-based learner and classifier based on CN2 algorithm</description>
<icon>CN2.png</icon>
<priority>300</priority>
"""

from OWWidget import *
import OWGUI
import orange
import orngCN2
import qt
import sys

class OWCN2(OWWidget):
    settingsList=["QualityButton","CoveringButton","m", "MaxRuleLength", "MinCoverage",
         "BeamWidth", "Alpha", "Weight"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"CN2")
        #OWWidget.__init__(self,parent,"Rules")
        self.inputs=[("ExampleTable", ExampleTable, self.dataset)]
        self.outputs=[("Learner", orange.Learner),("Classifier",orange.Classifier),("CN2UnorderedClassifier", orngCN2.CN2UnorderedClassifier)]
        self.QualityButton=0
        self.CoveringButton=0
        self.Alpha=0.2
        self.BeamWidth=50
        self.MinCoverage=0
        self.MaxRuleLength=0
        self.Weight=0.9
        self.m=2
        self.LearnerName=""
        self.loadSettings()

        self.data=None

        ##GUI
        labelWidth=150
        self.learnerName=OWGUI.lineEdit(self.controlArea, self, "LearnerName", box="Learner/Classifier Name",tooltip="Name to be used by other widgets to identify yor Learner/Classifier")
        self.learnerName.setText("Rule classifier")
        self.ruleQualityBG=QVButtonGroup(self.controlArea)
        self.ruleQualityBG.setTitle("Rule Quality Estimation")
        self.ruleValidationGroup=OWGUI.widgetBox(self.controlArea, self)
        self.ruleValidationGroup.setTitle("Pre-Prunning(LRS)")

        OWGUI.spin(self.controlArea, self, "BeamWidth", 1, 100, box="Beam Width", tooltip=" Specify the width of the search beam")

        self.coveringAlgBG=QVButtonGroup(self.controlArea)
        self.coveringAlgBG.setTitle("Covering algorithm settings")
        """
        self.ruleQualityBG=OWGUI.radioButtonsInBox(self.ruleQualityGroup, self, "QualityButton",
                            btnLabels=["Laplace","m-estimate","WRACC"],
                            box="Rule quality", callback=self.qualityButtonPressed,
                            tooltips=["Laplace rule evaluator", "m-estimate rule evaluator",
                            "WRACC rule evaluator"])
        self.mSpin=Spin=OWGUI.spin(self.ruleQualityGroup, self, "m", 0, 100, label="m",
                orientation="horizontal", labelWidth=labelWidth-100, tooltip="m value for m estimate rule evaluator")
        """
        
        QRadioButton("Laplace",self.ruleQualityBG)
        g=QHBox(self.ruleQualityBG)
        b=QRadioButton("m-estimate",g)
        self.ruleQualityBG.insert(b)
        self.mSpin=OWGUI.spin(g,self,"m",0,100)
        self.mSpin.setDisabled(1)
        QRadioButton("WRACC",self.ruleQualityBG)
        self.connect(self.ruleQualityBG,SIGNAL("released(int)"),self.qualityButtonPressed)
        
        a=OWGUI.doubleSpin(self.ruleValidationGroup, self, "Alpha", 0, 1,0.05, label="Alpha",
                orientation="horizontal", labelWidth=labelWidth)
        OWGUI.spin(self.ruleValidationGroup, self, "MinCoverage", 0, 100,label="Minimum Coverage",
                orientation="horizontal", labelWidth=labelWidth, tooltip=
                "Minimum number of examples a rule must\ncover (use 0 for dont care)")
        OWGUI.spin(self.ruleValidationGroup, self, "MaxRuleLength", 0, 100,label="Max. Rule Length",
                orientation="horizontal", labelWidth=labelWidth, tooltip=
                "Maximum number of conditions in the left\npart of the rule (use 0 for dont care)")
        
        """
        self.coveringAlgBG=OWGUI.radioButtonsInBox(self.coveringAlgGroup, self, "CoveringButton",
                            btnLabels=["Exclusive covering ","Weighted Covering"],
                            tooltips=["Each example will only be used once\n for the construction of a rule",
                                      "Examples can take part in the construction\n of many rules(CN2-SD Algorthim)"],
                            box="Covering algorithm", callback=self.coveringAlgButtonPressed)
        self.weightSpin=OWGUI.doubleSpin(self.coveringAlgGroup, self, "Weight",0, 0.95,0.05,label= "Weight",
                orientation="horizontal", labelWidth=labelWidth, tooltip=
                "Multiplication constant by which the weight of\nthe example will be reduced")
        """

        QRadioButton("Exclusive covering",self.coveringAlgBG)
        g=QHBox(self.coveringAlgBG)
        b=QRadioButton("Weighted covering",g)
        self.coveringAlgBG.insert(b)
        self.weightSpin=OWGUI.doubleSpin(g,self,"Weight",0,0.95,0.05)
        self.weightSpin.setDisabled(1)
        self.connect(self.coveringAlgBG,SIGNAL("released(int)"),self.coveringAlgButtonPressed)
        #layout=QVBoxLayout(self.controlArea)
        #layout.add(self.ruleQualityGroup)
        #layout.add(self.ruleValidationGroup)
        #layout.add(self.coveringAlgGroup)

        OWGUI.button(self.controlArea, self, "&Apply Settings", callback=self.applySettings)

        self.Alpha=float(self.Alpha)
        self.Weight=float(self.Weight)

        self.ruleQualityBG.setButton(self.QualityButton)
        self.coveringAlgBG.setButton(self.CoveringButton)
        self.qualityButtonPressed()
        self.coveringAlgButtonPressed()
        self.controlArea.setMinimumWidth(300)
        self.resize(100,100)
        self.setLearner()

    def setLearner(self):
        self.learner=orngCN2.CN2UnorderedLearner()
        self.learner.name=self.LearnerName
        self.send("Learner",self.learner)
        ruleFinder=orange.RuleBeamFinder()
        #print self.Alpha
        #print self.Weight
        if self.QualityButton==0:
            ruleFinder.evaluator=orange.RuleEvaluator_Laplace()
        elif self.QualityButton==1:
            ruleFinder.evaluator=orngCN2.mEstimate(self.m)
        elif self.QualityButton==2:
            ruleFinder.evaluator=orngCN2.WRACCEvaluator()

        ruleFinder.ruleStoppingValidator=orange.RuleValidator_LRS(alpha=self.Alpha,
                    min_coverage=self.MinCoverage, max_rule_complexity=self.MaxRuleLength)
        ruleFinder.ruleFilter=orange.RuleBeamFilter_Width(width=self.BeamWidth)
        self.learner.ruleFinder=ruleFinder

        if self.CoveringButton==0:
            self.learner.coverAndRemove=orange.RuleCovererAndRemover_Default()
        elif self.CoveringButton==1:
            self.learner.coverAndRemove=orngCN2.CovererAndRemover_multWeights(mult=self.Weight)

        self.classifier=None
        if self.data:
            try:
                self.classifier=self.learner(self.data)
                self.classifier.name=self.LearnerName
                self.classifier.setattr("data",self.data)
            except orange.KernelException, (errValue):
                self.classifier=None
        #print self.classifier
        #print self.learner
        self.send("Classifier", self.classifier)
        self.send("CN2UnorderedClassifier", self.classifier)

    def dataset(self, data):
        self.data=data
        if self.data:
            self.setLearner()
        else:
            self.send("Learner",None)
            self.send("Classifier",None)
            self.send("CN2UnorderedClassifier",None)


    def qualityButtonPressed(self,id=0):
        id=self.QualityButton=self.ruleQualityBG.id(self.ruleQualityBG.selected())
        if id==1:
            self.mSpin.setDisabled(0)
        else:
            self.mSpin.setDisabled(1)

    def coveringAlgButtonPressed(self,id=0):
        id=self.CoveringButton=self.coveringAlgBG.id(self.coveringAlgBG.selected())
        if id==1:
            self.weightSpin.setDisabled(0)
        else:
            self.weightSpin.setDisabled(1)

    def applySettings(self):
        self.setLearner()
        
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWCN2()
    w.dataset(orange.ExampleTable("../../doc/datasets/titanic.tab"))
    app.setMainWidget(w)
    w.show()
    app.exec_loop()
