"""
<name>Mean</name>
<description>Mean regression</description>
<icon>icons/Mean.png</icon>
<priority>5</priority>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<keywords>mean, average</keywords>

"""

from OWMajority import *

class OWMean(OWMajority):
    def __init__(self, parent=None, signalManager=None, title="Mean"):
        OWMajority.__init__(self, parent, signalManager, "title")

        self.inputs = [("Examples", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier)]
        
        