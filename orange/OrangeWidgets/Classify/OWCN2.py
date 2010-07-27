"""
<name>CN2</name>
<description>Rule-based (CN2) learner/classifier.</description>
<icon>icons/CN2.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>300</priority>
"""
from OWWidget import *
import OWGUI, orange, orngCN2, sys

class CN2ProgressBar(orange.ProgressCallback):
    def __init__(self, widget, start=0.0, end=0.0):
        self.start = start
        self.end = end
        self.widget = widget
        orange.ProgressCallback.__init__(self)
    def __call__(self,value,a):
        self.widget.progressBarSet(100*(self.start+(self.end-self.start)*value))

class OWCN2(OWWidget):
    settingsList=["name", "QualityButton", "CoveringButton", "m",
                  "MaxRuleLength", "useMaxRuleLength",
                  "MinCoverage", "BeamWidth", "Alpha", "Weight", "stepAlpha"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"CN2", wantMainArea = 0, resizingEnabled = 0)

        self.inputs = [("Example Table", ExampleTable, self.dataset)]
        self.outputs = [("Learner", orange.Learner),("Classifier",orange.Classifier),("Unordered CN2 Classifier", orngCN2.CN2UnorderedClassifier)]
        self.QualityButton = 0
        self.CoveringButton = 0
        self.Alpha = 0.05
        self.stepAlpha = 0.2
        self.BeamWidth = 5
        self.MinCoverage = 0
        self.MaxRuleLength = 0
        self.useMaxRuleLength = False
        self.Weight = 0.9
        self.m = 2
        self.name = "CN2 rules"
        self.loadSettings()

        self.data=None

        ##GUI
        labelWidth = 150
        self.learnerName = OWGUI.lineEdit(self.controlArea, self, "name", box="Learner/classifier name", tooltip="Name to be used by other widgets to identify the learner/classifier")
        #self.learnerName.setText(self.name)
        OWGUI.separator(self.controlArea)

        self.ruleQualityBG = OWGUI.widgetBox(self.controlArea, "Rule quality estimation")
        self.ruleQualityBG.buttons = []

        OWGUI.separator(self.controlArea)
        self.ruleValidationGroup = OWGUI.widgetBox(self.controlArea, "Pre-prunning (LRS)")

        OWGUI.separator(self.controlArea)
        OWGUI.spin(self.controlArea, self, "BeamWidth", 1, 100, box="Beam width", tooltip="The width of the search beam\n(number of rules to be specialized)")

        OWGUI.separator(self.controlArea)
        self.coveringAlgBG = OWGUI.widgetBox(self.controlArea, "Covering algorithm")
        self.coveringAlgBG.buttons = []

        """
        self.ruleQualityBG=OWGUI.radioButtonsInBox(self.ruleQualityGroup, self, "QualityButton",
                            btnLabels=["Laplace","m-estimate","WRACC"],
                            box="Rule quality", callback=self.qualityButtonPressed,
                            tooltips=["Laplace rule evaluator", "m-estimate rule evaluator",
                            "WRACC rule evaluator"])
        self.mSpin=Spin=OWGUI.spin(self.ruleQualityGroup, self, "m", 0, 100, label="m",
                orientation="horizontal", labelWidth=labelWidth-100, tooltip="m value for m estimate rule evaluator")
        """

        b1 = QRadioButton("Laplace", self.ruleQualityBG); self.ruleQualityBG.layout().addWidget(b1)
        g = OWGUI.widgetBox(self.ruleQualityBG, orientation = "horizontal");
        b2 = QRadioButton("m-estimate", g)
        g.layout().addWidget(b2)
        self.mSpin = OWGUI.doubleSpin(g,self,"m",0,100)
        b3 = QRadioButton("EVC", self.ruleQualityBG)
        self.ruleQualityBG.layout().addWidget(b3)
        b4 = QRadioButton("WRACC", self.ruleQualityBG)
        self.ruleQualityBG.layout().addWidget(b4)
        self.ruleQualityBG.buttons = [b1, b2, b3, b4]

        for i, button in enumerate([b1, b2, b3, b4]):
            self.connect(button, SIGNAL("clicked()"), lambda v=i: self.qualityButtonPressed(v))

        OWGUI.doubleSpin(self.ruleValidationGroup, self, "Alpha", 0, 1,0.001, label="Alpha (vs. default rule)",
                orientation="horizontal", labelWidth=labelWidth,
                tooltip="Required significance of the difference between the class distribution on all example and covered examples")
        OWGUI.doubleSpin(self.ruleValidationGroup, self, "stepAlpha", 0, 1,0.001, label="Stopping Alpha (vs. parent rule)",
                orientation="horizontal", labelWidth=labelWidth,
                tooltip="Required significance of each specialization of a rule.")
        OWGUI.spin(self.ruleValidationGroup, self, "MinCoverage", 0, 100,label="Minimum coverage",
                orientation="horizontal", labelWidth=labelWidth, tooltip=
                "Minimum number of examples a rule must\ncover (use 0 for not setting the limit)")
        OWGUI.checkWithSpin(self.ruleValidationGroup, self, "Maximal rule length", 0, 100, "useMaxRuleLength", "MaxRuleLength", labelWidth=labelWidth,
                            tooltip="Maximal number of conditions in the left\npart of the rule (use 0 for don't care)")

        """
        self.coveringAlgBG=OWGUI.radioButtonsInBox(self.coveringAlgGroup, self, "CoveringButton",
                            btnLabels=["Exclusive covering ","Weighted Covering"],
                            tooltips=["Each example will only be used once\n for the construction of a rule",
                                      "Examples can take part in the construction\n of many rules(CN2-SD Algorithm)"],
                            box="Covering algorithm", callback=self.coveringAlgButtonPressed)
        self.weightSpin=OWGUI.doubleSpin(self.coveringAlgGroup, self, "Weight",0, 0.95,0.05,label= "Weight",
                orientation="horizontal", labelWidth=labelWidth, tooltip=
                "Multiplication constant by which the weight of\nthe example will be reduced")
        """

        B1 = QRadioButton("Exclusive covering", self.coveringAlgBG); self.coveringAlgBG.layout().addWidget(B1)
        g = OWGUI.widgetBox(self.coveringAlgBG, orientation = "horizontal")
        B2 = QRadioButton("Weighted covering", g); g.layout().addWidget(B2)
        self.coveringAlgBG.buttons = [B1, B2]
        self.weightSpin=OWGUI.doubleSpin(g,self,"Weight",0,0.95,0.05)

        for i, button in enumerate([B1, B2]):
            self.connect(button, SIGNAL("clicked()"), lambda v=i: self.coveringAlgButtonPressed(v))

        OWGUI.separator(self.controlArea)
        self.btnApply = OWGUI.button(self.controlArea, self, "&Apply", callback=self.applySettings)

        self.Alpha=float(self.Alpha)
        self.stepAlpha=float(self.stepAlpha)
        self.Weight=float(self.Weight)

        #self.ruleQualityBG.buttons[self.QualityButton].setChecked(1)
        self.qualityButtonPressed(self.QualityButton)
        self.coveringAlgButtonPressed(self.CoveringButton)
        self.resize(100,100)
        self.setLearner()

    def sendReport(self):
        self.reportSettings("Learning parameters",
                            [("Rule quality estimation", ["Laplace", "m-estimate with m=%.2f" % self.m, "WRACC"][self.QualityButton]),
                             ("Pruning alpha (vs. default rule)", "%.3f" % self.Alpha),
                             ("Stopping alpha (vs. parent rule)", "%.3f" % self.stepAlpha),
                             ("Minimum coverage", "%.3f" % self.MinCoverage),
                             ("Maximal rule length", self.MaxRuleLength if self.useMaxRuleLength else "unlimited"),
                             ("Beam width", self.BeamWidth),
                             ("Covering", ["Exclusive", "Weighted with a weight of %.2f" % self.Weight][self.CoveringButton])])
        self.reportData(self.data)

    def setLearner(self):
        if hasattr(self, "btnApply"):
            self.btnApply.setFocus()
        #progress bar
        self.progressBarInit()

        #learner / specific handling in case of EVC learning (completely different type of class)
        if self.useMaxRuleLength:
            maxRuleLength = self.MaxRuleLength
        else:
            maxRuleLength = -1
        
        if self.QualityButton == 2:
            self.learner=orngCN2.CN2EVCUnorderedLearner(width=self.BeamWidth, rule_sig=self.Alpha, att_sig=self.stepAlpha,
                                                        min_coverage = self.MinCoverage, max_rule_complexity = maxRuleLength)
            self.learner.name = self.name
#            self.learner.progressCallback=CN2ProgressBar(self)
            self.send("Learner",self.learner)
        else:
            self.learner=orngCN2.CN2UnorderedLearner()
            self.learner.name = self.name
#            self.learner.progressCallback=CN2ProgressBar(self)
            self.send("Learner",self.learner)

            ruleFinder=orange.RuleBeamFinder()
            if self.QualityButton==0:
                ruleFinder.evaluator=orange.RuleEvaluator_Laplace()
            elif self.QualityButton==1:
                ruleFinder.evaluator=orngCN2.mEstimate(self.m)
            elif self.QualityButton==3:
                ruleFinder.evaluator=orngCN2.WRACCEvaluator()


            ruleFinder.ruleStoppingValidator=orange.RuleValidator_LRS(alpha=self.stepAlpha,
                        min_coverage=self.MinCoverage, max_rule_complexity=maxRuleLength)
            ruleFinder.validator=orange.RuleValidator_LRS(alpha=self.Alpha,
                        min_coverage=self.MinCoverage, max_rule_complexity=maxRuleLength)
            ruleFinder.ruleFilter=orange.RuleBeamFilter_Width(width=self.BeamWidth)
            self.learner.ruleFinder=ruleFinder

            if self.CoveringButton==0:
                self.learner.coverAndRemove=orange.RuleCovererAndRemover_Default()
            elif self.CoveringButton==1:
                self.learner.coverAndRemove=orngCN2.CovererAndRemover_multWeights(mult=self.Weight)

        self.classifier=None
        self.error()
        if self.data:
            oldDomain = orange.Domain(self.data.domain)
            learnData = orange.ExampleTable(oldDomain, self.data)
            self.learner.progressCallback=CN2ProgressBar(self)
            self.classifier=self.learner(learnData)
            self.learner.progressCallback=None
            self.classifier.name=self.name
            for r in self.classifier.rules:
                r.examples = orange.ExampleTable(oldDomain, r.examples)
            self.classifier.examples = orange.ExampleTable(oldDomain, self.classifier.examples)
            self.classifier.setattr("data",self.classifier.examples)
            self.error("")
##            except orange.KernelException, (errValue):
##                self.classifier=None
##                self.error(errValue)
##            except Exception:
##                self.classifier=None
##                if not self.data.domain.classVar:
##                    self.error("Classless domain.")
##                elif self.data.domain.classVar.varType == orange.VarTypes.Continuous:
##                    self.error("CN2 can learn only from discrete class.")
##                else:
##                    self.error("Unknown error")
        self.send("Classifier", self.classifier)
        self.send("Unordered CN2 Classifier", self.classifier)
        self.progressBarFinished()

    def dataset(self, data):
        #self.data=data
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete, checkMissing=True) and data or None
        self.setLearner()

    def qualityButtonPressed(self, id=0):
        self.QualityButton = id
        for i in range(len(self.ruleQualityBG.buttons)):
            self.ruleQualityBG.buttons[i].setChecked(id == i)
        self.mSpin.control.setEnabled(id == 1)
        self.coveringAlgBG.setEnabled(not id == 2)

    def coveringAlgButtonPressed(self,id=0):
        self.CoveringButton = id
        for i in range(len(self.coveringAlgBG.buttons)):
            self.coveringAlgBG.buttons[i].setChecked(id == i)
        self.weightSpin.control.setEnabled(id == 1)

    def applySettings(self):
        self.setLearner()

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWCN2()
    #w.dataset(orange.ExampleTable("titanic.tab"))
    w.dataset(orange.ExampleTable("titanic.tab"))
    w.show()
    app.exec_()
    w.saveSettings()
