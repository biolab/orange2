"""
<name>Calibrated Classifier</name>
<description>Given a learner, it builds a classifier calibrated for optimal classification accuracy</description>
<category>Classify</category>
<icon>CalibratedClassifier.png</icon>
<priority>1030</priority>
"""

from OData import *
from OWTools import *
from OWWidget import *
from OWGraph import *
import OWGUI
from OWCalibrationPlotOptions import *
import qwt
import qt

import orngTest, orngWrap
import statc, math

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
#        self.setFixedHeight(300)

        self.curveIdx = self.insertCurve('')
        self.thresholdIdx = self.insertCurve('')
        self.setMinimumHeight(300)

    def setCurve(self, coords, threshold):
        #OWGraph.removeCurves(self)
        #self.curveIdx = self.insertCurve('')
        self.setCurveData(self.curveIdx, [100*x[0] for x in coords], [x[1] for x in coords])
        self.setCurvePen(self.curveIdx, QPen(Qt.black, 2))

        #self.thresholdIdx = self.insertCurve('')
        self.setCurveData(self.thresholdIdx, [threshold, threshold], [0, 1])
        self.setCurvePen(self.thresholdIdx, QPen(Qt.blue, 1))
        self.replot()

    def clear(self):
##        self.setCurveData(self.curveIdx, [], [])
##        self.setCurveData(self.thresholdIdx, [], [])
##        self.replot()
        pass

    
class OWCalibratedClassifier(OWWidget):
    settingsList = ["name", "manual", "threshold"]
    def __init__(self,parent=None, name = "Calibrated Classifier"):
        OWWidget.__init__(self, parent, name, "Construct a calibration wrapper for classifier")
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.setData, 1), ("Base Learner", orange.Learner, self.setBaseLearner, 1)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier)]

        # Settings
        self.name = 'CalibratedClassifier'
        self.manual = 0
        self.threshold = self.accuracy = 50
        self.loadSettings()

        self.learner = self.baseLearner = self.data = None

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
        tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        self.wbThreshold = OWGUI.widgetBox(self.controlArea, "Threshold")
        self.cbManual = OWGUI.checkBox(self.wbThreshold, self, "manual", "Manually set the threshold", callback = self.computeThreshold)
        self.spThreshold = OWGUI.spin(self.wbThreshold, self, "threshold", 1, 99, label = "Threshold", step=5, orientation = "horizontal", callback = self.computeThreshold)
        self.cbManual.disables = [self.spThreshold]
        self.cbManual.makeConsistent()

        self.lbNotice = qt.QLabel(self.wbThreshold)
        self.lbNotice.setText("Notice: If the widget is connected to a widget that takes a Learner, not a Classifier (eg 'Test Learners'), the automatically computed threshold can differ from the above.")
        self.lbNotice.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter | qt.Qt.ExpandTabs | qt.Qt.ShowPrefix | qt.Qt.WordBreak)
        
        OWGUI.separator(self.controlArea)
        
        OWGUI.button(self.controlArea, self, "&Apply Setting", callback = self.setLearner, disabled=0)

        self.g = QVBoxLayout(self.mainArea, 1)
        self.graph = ThresholdGraph(self.mainArea)
        self.g.addWidget(self.graph)

        self.resize(400, 330)

    def setData(self, data):
        if data and len(data.domain.classVar.values) == 2:
            self.data = data
            self.learn()
        else:
            self.data = None
            self.graph.clear()

    def setBaseLearner(self, baseLearner):
        self.baseLearner = baseLearner
        self.setLearner()

    def setLearner(self):
        if self.baseLearner:
            self.curveLearner = orngWrap.ThresholdLearner(self.baseLearner, storeCurve = 1)
            if self.manual:
                self.learner = orngWrap.ThresholdLearner_fixed(self.baseLearner, self.threshold/100.0)
            else:
                self.learner = self.curveLearner
            self.learner.name = self.name
            self.send("Learner", self.learner)

            self.learn()
        else:
            self.graph.clear()

    def learn(self):
        if not self.learner or not self.data:
            self.classifier = None
            self.send("Classifier", None)
            return
        
        self.classifier = self.learner(self.data)
        self.classifier.name = self.name
        self.send("Classifier", self.classifier)
        
        self.curve = getattr(self.classifier, "curve", None)
        if not self.curve:
            curveclassifier = self.curveLearner(self.data)
            self.curve = curveclassifier.curve
            self.computedThreshold = curveclassifier.threshold * 100
        else:
            self.computedThreshold = self.classifier.threshold * 100

        self.computeThreshold()

    def computeThreshold(self):
        self.lbNotice.setDisabled(self.manual)
        if not getattr(self, "curve", None):
            return
        if not self.manual:
            self.threshold = self.computedThreshold
        self.graph.setCurve(self.curve, self.threshold)


    def removeGraphs(self):
        for g in self.graphs:
            g.removeCurves()

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


