"""
<name>Random Forest Regression</name>
<description>Random forest regression.</description>
<icon>icons/RandomForest.png</icon>
<contact>Marko Toplak (marko.toplak(@at@)gmail.com)</contact>
<priority>320</priority>
<keywords>bagging, ensemble</keywords>

"""

from OWRandomForest import *

class OWRandomForestRegression(OWRandomForest):
    def __init__(self, parent=None, signalManager=None, title="Random forest regression"):
        OWRandomForest.__init__(self, parent, signalManager, title)
        
        self.inputs = [("Examples", ExampleTable, self.setData),
                       ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        
        self.outputs = [("Learner", orange.Learner),
                        ("Random Forest Classifier", orange.Classifier)]

    def setData(self, data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Continuous, checkMissing=True) and data or None
        
        if self.data:
            learner = self.constructLearner()
            pb = OWGUI.ProgressBar(self, iterations=self.trees)
            learner.callback = pb.advance
            try:
                self.classifier = learner(self.data)
                self.classifier.name = self.name
            except Exception, (errValue):
                self.error(str(errValue))
                self.classifier = None
            pb.finish()
        else:
            self.classifier = None

        self.send("Random Forest Classifier", self.classifier)
        