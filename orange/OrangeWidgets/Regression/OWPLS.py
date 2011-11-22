"""
<name>PLS Regression</name>
<description>Partial Least Squares Regression</name>
<icon>icons/PLSRegression.png</icon>
<priority>15</priority>
<category>Regression</category>
<keywords>linear, model, PCA</keywords>

"""

from OWWidget import *
import OWGUI

import Orange
from Orange.regression import pls
from Orange.optimization import PreprocessedLearner

class OWPLS(OWWidget):
    settingsList = []
    
    def __init__(self, parent=None, signalManager=None, title="PLS Regression"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Training data", Orange.data.Table, self.set_data),
                       ("Preprocessor", PreprocessedLearner, self.set_preprocessor)]
        
        self.outputs = [("Learner", Orange.core.Learner), 
                        ("Predictor", Orange.core.Classifier)]
        
        
        ##########
        # Settings
        ##########
         
        self.name = "PLS Regression"
        self.n_comp = 2
        self.deflation_mode = "Regression"
        self.mode = "PLS"
        self.algorithm = "svd"
        
        #####
        # GUI
        #####
        
        box = OWGUI.widgetBox(self.controlArea, "Learner/Predictor Name",  
                              addSpace=True)
        
        OWGUI.lineEdit(box, self, "name",
                       tooltip="Name to use for the learner/predictor.")
        
        box = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
        
        OWGUI.spin(box, self, "n_comp", 2, 15, 1, 
                   label="Number of components:", 
                   tooltip="Number of components to keep.")
        
        OWGUI.comboBox(box, self, "deflation_mode", 
                       label="Deflation mode", 
                       items=["Cannonical", "Regression"],
#                       tooltip="",
                       sendSelectedValue=True)
        
        OWGUI.comboBox(box, self, "mode", 
                       label="Mode", 
                       items=["CCA", "PLS"],
#                       tooltip="", 
                       sendSelectedValue=True)
        
        OWGUI.rubber(self.controlArea)
        
        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.apply,
                     tooltip="Send the learner on",
                     autoDefault=True)
        
        self.data = None
        self.preprocessor = None
        
        self.apply()
    
    def set_data(self, data=None):
        self.data = data
            
    def set_preprocessor(self, pproc=None):
        self.preprocessor = pproc
        
    def handleNewSignals(self):
        self.apply()
        
    def apply(self):
        learner = pls.PLSRegressionLearner(nComp=self.n_comp,
                        deflationMode=self.deflation_mode.lower(),
                        mode=self.mode,
                        name=self.mode
                        )
        predictor = None
        
        if self.preprocessor is not None:
            learner = self.preprocessor.wrapLearner(learner)

        self.error(0)
        try:
            if self.data is not None:
                predictor = learner(self.data)
                predictor.name = self.name
        except Exception, ex:
            self.error(0, "An error during learning: %r" % ex)
            
        self.send("Learner", learner)
        self.send("Predictor", predictor)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPLS()
    w.set_data(Orange.data.Table("housing"))
    w.show()
    app.exec_()
#    w.saveSettings()
