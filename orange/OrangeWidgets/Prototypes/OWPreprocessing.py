"""
<name>Preprocessing</name>
<description>Constructs data preprocessors.</description>
<icon>icons/FeatureConstructor.png</icon>
<priority>11</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import OWGUI, math, re
from orngWrap import PreprocessedLearner

class OWPreprocessing(OWWidget):
    contextHandlers = {"": PerfectDomainContextHandler()}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Preprocessing")

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Preprocessor", PreprocessedLearner), ("Examples", ExampleTable)]

        OWGUI.button(self.controlArea, self, "Apply", callback=self.apply)

        self.loadSettings()
        self.apply()
        self.adjustSize()


    def setData(self, data):
        self.data = data
        self.sendData()

    def sendData(self):        
        if not self.data or not self.preprocessor:
            self.preprocessed = self.data
        else:
            self.preprocessed = self.preprocessor.processData(self.data)
        self.send("Examples", self.preprocessed)
        

    def apply(self):
        # The widget needs to construct a new instance of Preprocessor
        # If it modified and send the same instance every time, it would
        # modify an instance which has been passed to another widget which
        # might have a disabled connection and should not get any modifications
        # (and would even not get notified about the preprocessor having been changed)
        self.preprocessor = PreprocessedLearner()
        self.send("Preprocessor", self.preprocessor)
        