"""
<name>Calibrated Classifier</name>
<description>Given a learner, it builds a classifier calibrated for optimal classification accuracy</description>
<icon>icons/CalibratedClassifier.png</icon>
<priority>1030</priority>
"""

from OWWidget import *
from OWGraph import *
import OWGUI

import orngWrap

class ThresholdGraph(OWGraph):
    def __init__(self, parent = None, title = ""):
        OWGraph.__init__(self, parent)
        self.setYRlabels(None)
        self.enableGridXB(0)
        self.enableGridYL(1)
        self.setAxisMaxMajor(QwtPlot.xBottom, 10)
        self.setAxisMaxMinor(QwtPlot.xBottom, 3)
        self.setAxisMaxMajor(QwtPlot.yLeft, 10)
        self.setAxisMaxMinor(QwtPlot.yLeft, 5)
        self.setAxisScale(QwtPlot.xBottom, -0.0, 100.0, 0)
        self.setAxisScale(QwtPlot.yLeft, -0.0, 1.0, 0)
        self.setYLaxisTitle('classification accuracy')
        self.setShowYLaxisTitle(1)
        self.setXaxisTitle('threshold')
        self.setShowXaxisTitle(1)
        self.setShowMainTitle(1)
        self.setMainTitle(title)

        self.curve = self.addCurve("")
        self.thresholdCurve = self.addCurve("Threashold")

    def setCurve(self, coords):
        self.curve.setData([100*x[0] for x in coords], [x[1] for x in coords])
        self.curve.setPen(QPen(Qt.black, 2))
        self.replot()

    def setThreshold(self, threshold):
        self.thresholdCurve.setData([threshold, threshold], [0, 1])
        self.thresholdCurve.setPen(QPen(Qt.blue, 1))
        self.replot()

    
class OWCalibratedClassifier(OWWidget):
    settingsList = ["name", "optimalThreshold", "threshold"]
    def __init__(self, parent=None, signalManager = None, title = "Calibrated Classifier"):
        OWWidget.__init__(self, parent, signalManager, title)

        self.inputs = [("Examples", ExampleTable, self.setData), ("Base Learner", orange.Learner, self.setBaseLearner)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier)]

        # Settings
        self.name = 'Calibrated Learner'
        self.optimalThreshold = 0
        self.threshold = self.accuracy = 50
        self.loadSettings()

        self.learner = None
        self.baseLearner = None
        self.data = None

        OWGUI.lineEdit(self.controlArea, self, 'name',
                       box='Learner/Classifier Name', 
                       tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        self.wbThreshold = OWGUI.widgetBox(self.controlArea, "Threshold", addSpace=True)
        self.cbOptimal = OWGUI.checkBox(self.wbThreshold, self, "optimalThreshold",
                                        "Use optimal threshold",
                                        callback=self.setThreshold)
        
        self.spThreshold = OWGUI.spin(self.wbThreshold, self, "threshold", 1, 99, step=5,
                                      label = "Threshold",
                                      orientation = "horizontal",
                                      callback = self.setThreshold)
        
        self.lbNotice = OWGUI.widgetLabel(self.wbThreshold, "Notice: If the widget is connected to a widget that takes a Learner, not a Classifier (eg 'Test Learners'), the automatically computed threshold can differ from the above.")
        self.lbNotice.setWordWrap(True)
         
        self.cbOptimal.disables = [self.lbNotice]
        self.cbOptimal.makeConsistent()
        self.spThreshold.setDisabled(self.optimalThreshold)
        
        OWGUI.rubber(self.controlArea)
        
        OWGUI.button(self.controlArea, self, "&Apply Setting",
                     callback = self.btApplyCallback,
                     disabled=0)

        self.btSave = OWGUI.button(self.controlArea, self, "&Save Graph", callback = self.saveToFile, disabled=1)

        self.graph = ThresholdGraph()
        self.mainArea.layout().addWidget(self.graph)

        self.resize(700, 330)

    def setData(self, data):
        self.error([0])
        if data and len(data.domain.classVar.values) == 2:
            self.data = data
        else:
            self.error(0, "ThresholdLearner handles binary classes only!")
            self.data = None
        self.compute_baseClassifier_curve_threshold()
        self.construct_classifier()

    def setBaseLearner(self, baseLearner):
        self.baseLearner = baseLearner
        self.construct_learner()
        self.compute_baseClassifier_curve_threshold()
        self.construct_classifier()

    def btApplyCallback(self):
        self.construct_learner()
        self.construct_classifier()

    def setThreshold(self):
        self.spThreshold.setDisabled(self.optimalThreshold)
        if self.optimalThreshold:
            self.threshold = self.computedThreshold*100
        self.graph.setThreshold(self.threshold)

    def construct_learner(self):
        if self.baseLearner:
            if self.optimalThreshold:
                self.learner = orngWrap.ThresholdLearner(learner=self.baseLearner, storeCurve = 1)
            else:
                self.learner = orngWrap.ThresholdLearner_fixed(learner=self.baseLearner, threshhold=self.threshold/100.0)
            self.learner.name = self.name
        else:
            self.learner = None
        self.send("Learner", self.learner)

    def compute_baseClassifier_curve_threshold(self):
        if not self.learner or not self.data:
            self.baseClassifier = None
            self.computedThreshold = 0.5
            self.curve = []
        else:
            self.baseClassifier = self.baseLearner(self.data)
            self.computedThreshold, CA, self.curve = orange.ThresholdCA(self.baseClassifier, self.data)
            if self.optimalThreshold:
                self.threshold = self.computedThreshold*100
        self.graph.setCurve(self.curve)
        self.graph.setThreshold(self.threshold)
        self.btSave.setDisabled(not self.curve)

    def construct_classifier(self):
        if not self.baseClassifier:
            self.classifier = None
        else:
            self.classifier = orngWrap.ThresholdClassifier(self.baseClassifier, self.threshold)
            self.classifier.name = self.name
        self.send("Classifier", self.classifier)

    def saveToFile(self):
        from OWDlgs import OWChooseImageSizeDlg
        dlg = OWChooseImageSizeDlg(self.graph)
        dlg.exec_()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWCalibratedClassifier()

    data = orange.ExampleTable("../../doc/datasets/breast-cancer")
    learner = orange.BayesLearner()
    owdm.setData(data)
    owdm.setBaseLearner(learner)
    
    owdm.show()
    a.exec_()
    owdm.saveSettings()
