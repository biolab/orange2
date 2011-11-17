"""
<name>SVM Regression</name>
<description>Support Vector Machine Regression.</description>
<icon>icons/BasicSVM.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>100</priority>
<keywords>Support, Vector, Machine, Regression</keywords>

"""

from OWSVM import *

class OWSVMRegression(OWSVM):
    def __init__(self, parent=None, signalManager=None, title="SVM Regression"):
        OWSVM.__init__(self, parent, signalManager, title)
        
        self.inputs=[("Example Table", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs=[("Learner", orange.Learner, Default),("Classifier", orange.Classifier, Default),("Support Vectors", ExampleTable)]
        
        
        self.probability = False
        self.probBox.hide()
        
        