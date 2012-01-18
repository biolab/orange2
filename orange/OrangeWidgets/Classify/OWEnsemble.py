"""
<name>Ensemble</name>
<description>Ensemble classification</description>
<icons>icons/Ensemble.png</icons>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>330</priority>
<keywords>ensemble, bagging, boosting, adaboost</keywords>

"""

from OWWidget import *

import OWGUI

import orange
import orngEnsemble

import os, sys


class OWEnsemble(OWWidget):
    settingsList = ["method", "t"]
    
    METHODS = [("Boosting", orngEnsemble.BoostedLearner),
               ("Bagging", orngEnsemble.BaggedLearner)]
    
    def __init__(self, parent=None, signalManager=None, name="Ensemble"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea=False)
        
        self.inputs = [("Learner", orange.Learner, self.setLearner), ("Data", ExampleTable, self.setData)]
        self.outputs = [("Learner", orange.Learner), ("Classifier", orange.Classifier)]
        
        self.method = 0
        self.t = 10
        
        self.loadSettings()
        
        box = OWGUI.radioButtonsInBox(self.controlArea, self, "method", [name for name, _ in self.METHODS], box="Ensemble", callback=self.onChange)
        i_box = OWGUI.indentedBox(box, sep=OWGUI.checkButtonOffsetHint(box.buttons[0]))
        
        OWGUI.spin(i_box, self, "t", min=1, max=100, step=1, label="Number of created classifiers:")
        OWGUI.rubber(self.controlArea)
        OWGUI.button(self.controlArea, self, "&Apply", callback=self.commit)
        
        self.data = None
        self.learner = None
        
        self.resize(100, 100)
        
    def setLearner(self, learner=None):
        self.learner = learner
        
    def setData(self, data):
        self.data = data
        
    def handleNewSignals(self):
        self.commit()
        
    def onChange(self):
        pass
    
    def commit(self):
        wrapped = None
        if self.learner:
            wrapped = self.METHODS[self.method][1](self.learner, t=self.t)
            self.send("Learner", wrapped)
            
        if self.data and wrapped:
            classifier = wrapped(self.data)
            self.send("Classifier", classifier)
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWEnsemble()
    w.setLearner(orange.BayesLearner())
    w.setData(orange.ExampleTable("../../doc/datasets/iris"))
    w.show()
    app.exec_()
            
        
        
          
