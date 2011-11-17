"""
<name>Linear Regression</name>
<description>Linear Regression</name>
<icon>icons/LinearRegression.png</icon>
<priority>10</priority>
<category>Regression</category>
<keywords>linear, model</keywords>

"""

import os, sys
from OWWidget import *

import Orange
from Orange.regression import linear
from orngWrap import PreprocessedLearner

class OWLinearRegression(OWWidget):
    settingsList = ["name", "use_ridge", "ridge_lambda"]
    
    def __init__(self, parent=None, signalManager=None, title="Linear Regression"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Training data", Orange.data.Table, self.set_data), ("Preprocessor", PreprocessedLearner, self.set_preprocessor)]
        self.outputs = [("Learner", Orange.core.Learner), ("Predictor", Orange.core.Classifier)]
        
        ##########
        # Settings
        ##########
         
        self.name = "Linear Regression"
        self.use_ridge = False
        self.ridge_lambda = 1.0
        self.loadSettings()
        
        #####
        # GUI
        #####
        
        OWGUI.lineEdit(self.controlArea, self, "name", box="Learner/predictor name",
                       tooltip="Name of the learner/predictor")

        OWGUI.checkWithSpin(self.controlArea, self, "Ridge lambda", 1, 10,
                            "use_ridge", "ridge_lambda", step=1,
                            tooltip="Ridge lambda for ridge regression")
        
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
        if self.use_ridge:
            learner = linear.LinearRegressionLearner(name=self.name,
                                                ridgeLambda=self.ridge_lambda)
        else:
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
                
