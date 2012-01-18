"""
<name>k Nearest Neighbours Regression</name>
<description>K-nearest neighbours learner/predictor.</description>
<icon>icons/kNearestNeighbours.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>20</priority>
<keywords>knn</keywords>
"""

from OWKNN import *

class OWKNNRegression(OWKNN):
    def __init__(self, parent=None, signalManager=None, title="kNN Regression"):
        OWKNN.__init__(self, parent, signalManager, title)
        
        self.inputs = [("Data", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner),("kNN Classifier", orange.kNNClassifier)]
            
    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Continuous, checkMissing=True) and data or None
        self.setLearner()