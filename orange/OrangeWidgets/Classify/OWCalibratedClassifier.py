"""
<name>Calibrated Classifier</name>
<description>Given a learner, it builds a classifier calibrated for optimal classification accuracy</description>
<icon>CalibratedClassifier.png</icon>
<priority>1030</priority>
"""

from OWWidget import *
from OWGraph import *
import OWGUI
import qt

import orngWrap

class ThresholdGraph(OWGraph):
    def __init__(self, parent = None, name = None, title = ""):
        OWGraph.__init__(self, parent, name)
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

        self.curveIdx = self.insertCurve('')
        self.thresholdIdx = self.insertCurve('')
        self.setMinimumHeight(300)

    def setCurve(self, coords):
        self.setCurveData(self.curveIdx, [100*x[0] for x in coords], [x[1] for x in coords])
        self.setCurvePen(self.curveIdx, QPen(Qt.black, 2))
        self.replot()

    def setThreshold(self, threshold):
        self.setCurveData(self.thresholdIdx, [threshold, threshold], [0, 1])
        self.setCurvePen(self.thresholdIdx, QPen(Qt.blue, 1))
        self.replot()

    
class OWCalibratedClassifier(OWWidget):
    settingsList = ["name", "optimalThreshold", "threshold"]
    def __init__(self,parent=None, signalManager = None, name = "Calibrated Classifier"):
        OWWidget.__init__(self, parent, signalManager, name, "Construct a calibration wrapper for classifier")
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.setData), ("Base Learner", orange.Learner, self.setBaseLearner)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier)]

        # Settings
        self.name = 'CalibratedClassifier'
        self.optimalThreshold = 0
        self.threshold = self.accuracy = 50
        self.loadSettings()

        self.learner = self.baseLearner = self.data = None

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
        tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        self.wbThreshold = OWGUI.widgetBox(self.controlArea, "Threshold")
        self.cbOptimal = OWGUI.checkBox(self.wbThreshold, self, "optimalThreshold", "Use optimal threshold", callback = self.setThreshold)
        self.spThreshold = OWGUI.spin(self.wbThreshold, self, "threshold", 1, 99, label = "Threshold", step=5, orientation = "horizontal", callback = self.setThreshold)
        self.lbNotice = qt.QLabel(self.wbThreshold)
        self.lbNotice.setText("Notice: If the widget is connected to a widget that takes a Learner, not a Classifier (eg 'Test Learners'), the automatically computed threshold can differ from the above.")
        self.lbNotice.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter | qt.Qt.ExpandTabs | qt.Qt.ShowPrefix | qt.Qt.WordBreak)
        self.cbOptimal.disables = [self.lbNotice]
        self.cbOptimal.makeConsistent()
        self.spThreshold.setDisabled(self.optimalThreshold)
        OWGUI.separator(self.controlArea)
        
        OWGUI.button(self.controlArea, self, "&Apply Setting", callback = self.btApplyCallback, disabled=0)
        OWGUI.separator(self.controlArea)

        self.btSave = OWGUI.button(self.controlArea, self, "&Save Graph", callback = self.saveToFile, disabled=1)
        OWGUI.separator(self.controlArea)

        self.g = QVBoxLayout(self.mainArea, 1)
        self.graph = ThresholdGraph(self.mainArea)
        self.g.addWidget(self.graph)

        self.resize(700, 330)

    def setData(self, data):
        if data and len(data.domain.classVar.values) == 2:
            self.data = data
        else:
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
                self.learner = orngWrap.ThresholdLearner(self.baseLearner, storeCurve = 1)
            else:
                self.learner = orngWrap.ThresholdLearner_fixed(self.baseLearner, self.threshold/100.0)
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
        if self.graph:
            self.graph.saveToFile()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWCalibratedClassifier()

    data = orange.ExampleTable("breast-cancer")
    learner = orange.BayesLearner()
    owdm.setData(data)
    owdm.setBaseLearner(learner)
    
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()
    owdm.saveSettings()


