"""
<name>ABCN2</name>
<description>Argument based rule (ABCN2) learner/classifier.</description>
<icon>CN2.png</icon>
<priority>300</priority>
"""

from OWWidget import *
import OWGUI
import orange
import orngCN2
import qt
import sys
import pickle

import orngABCN2
import extremeValues
from orngArg import *
import computeDist

class CN2ProgressBar(orange.ProgressCallback):
    def __init__(self,widget,start=0.0,end=100.0):
        self.start = start
        self.end = end
        self.widget = widget
        orange.ProgressCallback.__init__(self)
    def __call__(self,value,a):
##        print self.start, self.end, value, 100*(self.start+(self.end-self.start)*value)
        self.widget.progressBarSet(100*(self.start+(self.end-self.start)*value))

class OWABCN2(OWWidget):
    settingList = ["max_rule_length", "min_rule_coverage", "always_compute_evd", "max_evd_rule_length", "filter_rules_zero", "learn_rules", "use_arguments", "classifierID"]

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"ABCN2")
        self.inputs=[("ExampleTable", ExampleTable, self.dataset)]
        self.outputs=[("Learner", orange.Learner),("Classifier",orange.Classifier),("RuleClassifier", orange.RuleClassifier_logit)]

        self.max_rule_length = 5
        self.min_rule_coverage = 0
        self.always_compute_evd = False
        self.max_evd_rule_length = 5
        self.number_of_iterations = 200
        self.filter_rules_zero = False
        self.learn_rules = False
        self.use_arguments = True
        self.classifierID = 1

        self.LearnerName=""
        self.loadSettings()

        self.data=None

        #GUI
        self.learnerName=OWGUI.lineEdit(self.controlArea, self, "LearnerName", box="Learner/Classifier Name",tooltip="Name to be used by other widgets to identify yor Learner/Classifier")
        self.learnerName.setText("ABRule classifier")
        self.learnerGroup=OWGUI.widgetBox(self.controlArea, self)
        self.learnerGroup.setTitle("Learner settings")
        self.evdGroup=OWGUI.widgetBox(self.controlArea, self)
        self.evdGroup.setTitle("EVD estimation settings")
        self.classifierGroup=OWGUI.widgetBox(self.controlArea, self)
        self.classifierGroup.setTitle("Classifier settings")

        OWGUI.spin(self.learnerGroup, self, "max_rule_length", 0, 100, box="Max. rule length", tooltip="Maximum number of conditions in the left\npart of the rule (use 0 for dont care)")
        OWGUI.spin(self.learnerGroup, self, "min_rule_coverage", 0, 100, box="Min. rule coverage", tooltip="Minimum number of examples a rule must\ncover (use 0 for dont care)")
        OWGUI.checkBox(self.learnerGroup,self, "learn_rules", label="Induce rules (or send only learner).")        
        OWGUI.checkBox(self.learnerGroup,self, "use_arguments", label="Use arguments for rule induction?")        

        OWGUI.checkBox(self.evdGroup,self, "always_compute_evd", label="Recompute EVD parameters")
        OWGUI.spin(self.evdGroup, self, "max_evd_rule_length", 0, 100, box="Maximum exp. rule length", tooltip="Number of conditions in rule (must be equal or higher to allowed max. rule length).")
        OWGUI.spin(self.evdGroup, self, "number_of_iterations", 0, 100, box="Number of iterations", tooltip="Number of randomly induced rules in EVD parameters estimation (more means better estimations, but takes longer).")

        OWGUI.radioButtonsInBox(self.classifierGroup, self, "classifierID", ["Logit", "Logit_bestRule", "Minimax"], box="Classifier type")
        OWGUI.checkBox(self.classifierGroup,self, "filter_rules_zero", label="Filter rules with zero influence (beta in logit is zero).", callback = self.changeFilterRules)
       
        self.btnApply = OWGUI.button(self.controlArea, self, "&Apply Settings", callback=self.setLearner)
        
        self.controlArea.setMinimumWidth(300)
        self.resize(100,400)
        self.setLearner()

    def setLearner(self):
        if hasattr(self, "btnApply"):
            self.btnApply.setFocus()

        if not self.data:
            self.error("Data should be available for this learner (uncheck learning rules for sending only learner).")
            return

        #General learner settings
        self.learner=orngABCN2.ABCN2Unordered_EDE_PC(argumentID = "Arguments")
        self.learner.name=self.LearnerName
        self.learner.progressCallback=CN2ProgressBar(self)
        self.learner.ruleStoppingValidator = orange.RuleValidator_LRS(alpha = 1.0, max_rule_complexity = self.max_rule_length-1,
                                                              min_coverage = self.min_rule_coverage)
        self.learner.coverAndRemoveAfterEachIteration = True
        
        if self.classifierID == 0: 
            self.learner.classifier = orange.RuleClassifier_logit
        elif self.classifierID == 1:
            self.learner.classifier = orange.RuleClassifier_logit_bestRule
        else:
            if len(self.data.domain.classVar.values)>2:
                self.error("This classifier works only on two class domains!")
                return
            self.learner.classifier = orngABCN2.ClassifyMinimax
            
        self.learner.learnFromArgs = self.use_arguments

        dataPath = getattr(self.data, "path", "")
        if not dataPath:
            self.error("Could not extract path from data. Perhaps was not loaded with ABFile.")
            return
        
        dirName = os.path.dirname(dataPath)
        baseName = os.path.basename(dataPath).split(".")[0]
        distsName = baseName+"dists.evd"
        dists_argsName = baseName+"dists_args.evd"
        if not self.always_compute_evd:
            try:
                self.learner.setattr("dists", pickle.load(file(dirName+"\\"+distsName)))
                self.learner.setattr("dists_args", pickle.load(file(dirName+"\\"+dists_argsName)))
            except:
                pass
        if self.data and not getattr(self.learner, "dists", ""):
            self.progressBarInit()
            self.learner.setattr("dists", computeDist.computeDists(self.data, learner=self.learner, N=self.number_of_iterations,maxLength=self.max_evd_rule_length, progressBar = CN2ProgressBar(self)))
            self.progressBarFinished()

            self.progressBarInit()
            self.learner.setattr("dists_args", computeDist.computeDists_exampleBased(self.data,learner=self.learner,N=self.number_of_iterations,maxLength=self.max_evd_rule_length, progressBar = CN2ProgressBar(self)))
            self.progressBarFinished()

            pickle.dump(self.learner.dists, file(dirName+"\\"+distsName,"w"))
            pickle.dump(self.learner.dists_args, file(dirName+"\\"+dists_argsName,"w"))
        
        #progress bar
        self.progressBarInit()

        #Sending learner
        self.send("Learner",self.learner)

        #Learning classifier
        if self.data and self.learn_rules:
            try:
                self.classifier=self.learner(self.data)
                self.classifier.name=self.LearnerName
                self.classifier.data=self.data
                self.classifier.setattr("allRules", orange.RuleList())
                for r in self.classifier.rules:
                    self.classifier.allRules.append(r)
                self.classifier.setattr("allRuleBetas", self.classifier.ruleBetas[:])
                if self.filter_rules_zero:
                    self.classifier.rules = orange.RuleList()
                    self.classifier.ruleBetas = []
                    for r_i, r in enumerate(self.classifier.allRules):
                        if self.classifier.allRuleBetas[r_i]:
                            self.classifier.rules.append(r)
                            self.classifier.ruleBetas.append(self.classifier.allRuleBetas[r_i])
                
            except orange.KernelException, (errValue):
                self.classifier=None
                self.error("Unknown error:"+str(errValue))
                print errValue
            except Exception, (errValue):
                self.classifier=None
                if not self.data.domain.classVar:
                    self.error("Classless domain!")
                elif self.data.domain.classVar.varType == orange.VarTypes.Continuous:
                    self.error("CN2 can learn only from discrete class!")
                else:
                    self.error("Unknown error:"+str(errValue))
                print errValue
                
            self.send("Classifier", self.classifier)
            self.send("RuleClassifier", self.classifier)
        self.progressBarFinished()
        self.error()

    def dataset(self, data):
        self.data=data
        if self.data:
            self.setLearner()
        else:
            self.send("Learner",None)
            self.send("Classifier",None)
            self.send("RuleClassifier",None)

    def changeFilterRules(self):
        if not self.classifier:
            return
        if self.filter_rules_zero:
            self.classifier.rules = orange.RuleList()
            self.classifier.ruleBetas = []
            for r_i, r in enumerate(self.classifier.allRules):
                if self.classifier.allRuleBetas[r_i]:
                    self.classifier.rules.append(r)
                    self.classifier.ruleBetas.append(self.classifier.allRuleBetas[r_i])
        else:
            self.classifier.rules = orange.RuleList()
            self.classifier.ruleBetas = []
            for r_i, r in enumerate(self.classifier.allRules):
                self.classifier.rules.append(r)
                self.classifier.ruleBetas.append(self.classifier.allRuleBetas[r_i])
        self.send("Classifier", self.classifier)
        self.send("RuleClassifier", self.classifier)

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWABCN2()
    w.dataset(ExampleTableArg("titanic"))
    app.setMainWidget(w)
    w.show()
    app.exec_loop()
