"""<name>Linear Regression</name>
"""

import os, sys
from OWWidget import *

import Orange
from Orange.regression import linear
from orngWrap import PreprocessedLearner

class OWLinearRegression(OWWidget):
    settingsList = []
    
    def __init__(self, parent=None, signalManager=None, title="Linear Regression"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Training data", Orange.data.Table, self.set_data), ("Preprocessor", PreprocessedLearner, self.set_preprocessor)]
        self.outputs = [("Learner", Orange.core.Learner), ("Predictor", Orange.core.Classifier)]
        
        ##########
        # Settings
        ##########
         
        self.name = "Linear Regression"
#        self.beta0 = True
        
        
        #####
        # GUI
        #####
        
        OWGUI.lineEdit(self.controlArea, self, "name", box="Learner/predictor name", 
                       tooltip="Name of the learner/predictor")
        
#        OWGUI.checkBox(self.controlArea, self, "beta0", "Include intercept.",
#                       box="Settings",
#                       tooltip="Add an intercept to the linear model")
        
        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.apply,
                     tooltip="Send the learner on",
                     autoDefault=True)
        
        self.data = None
        self.preprocessor = None
        self.resize(300, 100)
        self.apply()
        
    def set_data(self, data=None):
        self.data = data
            
    def set_preprocessor(self, pproc=None):
        self.preprocessor = pproc
        
    def handleNewSignals(self):
        self.apply()
            
    def apply(self):
        learner = linear.LinearRegressionLearner(name=self.name)
        predictor = None
        if self.preprocessor:
            learner = self.preprocessor.wrapLearner(learner)
        
        self.error(0)
        if self.data is not None:
            try:
                predictor = learner(self.data)
                predictor.name = self.name
            except Exception, ex:
                self.error(0, "An error during learning: %r" % ex)
            
        self.send("Learner", learner)
        self.send("Predictor", predictor)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWLinearRegression()
    w.set_data(Orange.data.Table("auto-mpg"))
    w.show()
    app.exec_()
#    w.saveSettings()               
                
