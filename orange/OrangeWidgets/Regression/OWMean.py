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
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)

        self.inputs = [("Data", ExampleTable, self.setData),
                       ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        
        self.outputs = [("Learner", orange.Learner),
                        ("Predictor", orange.Classifier)]
        
        
        self.name = 'Mean'
        
        self.loadSettings()
        
        self.data = None
        self.preprocessor = None

        OWGUI.lineEdit(self.controlArea, self, 'name', 
                       box='Learner/Predictor Name', \
                       tooltip='Name to be used by other widgets to identify your learner/predictor.')

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply", 
                     callback=self.setLearner,
                     disabled=0,
                     default=True)
        
        OWGUI.rubber(self.controlArea)
        
        self.learner = orange.MajorityLearner()
        self.setLearner()
        self.resize(100,100)
        
    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Continuous, checkMissing=True) and data or None

        if self.data:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.name = self.name
                self.error(1)
            except Exception, (errValue):
                self.classifier = None
                self.error(1, str(errValue))
        else:
            self.classifier = None
        self.send("Predictor", self.classifier)
        