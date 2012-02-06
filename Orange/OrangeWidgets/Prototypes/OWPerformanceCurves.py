"""<name>Performance Curves</name>
<description>Model performance at different thresholds</description>
<icon>icons/PerformanceCurves.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from OWDlgs import OWChooseImageSizeDlg
import sip
import orngTest
from OWGraph import *

class PerformanceGraph(OWGraph):
    def __init__(self, master, *arg):
        OWGraph.__init__(self, *arg)
        self.master = master
        self.mousePressed = False
        
    def mousePressEvent(self, e):
        self.mousePressed = True
        canvasPos = self.canvas().mapFrom(self, e.pos())
        self.master.thresholdChanged(self.invTransform(QwtPlot.xBottom, canvasPos.x()))

    def mouseReleaseEvent(self, e):
        self.mousePressed = False
        
    def mouseMoveEvent(self, e):
        if self.mousePressed:
            self.mousePressEvent(e)

# Remove if this widget ever goes multilingual!
_ = lambda x:x

class OWPerformanceCurves(OWWidget):
    settingsList = ["selectedScores", "threshold"]

    def __init__(self, parent=None, signalManager=None, name="Performance Curves"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs=[("Evaluation Results", orngTest.ExperimentResults, self.setTestResults, Default)]
        self.outputs=[]

        self.selectedScores = []
        self.classifiers = []
        self.selectedClassifier = []
        self.targetClass = -1
        self.threshold = 0.5
        self.thresholdCurve = None
        self.statistics = ""

        self.resize(980, 420)
        self.loadSettings()

        self.scores = [_('Classification accuracy'), _('Sensitivity (Recall)'), _('Specificity'),
                      _('Positive predictive value (Precision)'), _('Negative predictive value'),
                      _('F-measure')]
        self.colors = [Qt.black, Qt.green, Qt.darkRed,
                       Qt.blue, Qt.red,
                       QColor(255, 128, 0)]
        self.res = None
        self.allScores = None

        OWGUI.listBox(self.controlArea, self, 'selectedClassifier', 'classifiers', box = "Models", callback=self.classifierChanged, selectionMode = QListWidget.SingleSelection)
        self.comTarget = OWGUI.comboBox(self.controlArea, self, 'targetClass', box="Target Class", callback=self.classifierChanged, valueType=0)
        OWGUI.listBox(self.controlArea, self, 'selectedScores', 'scores', box = _("Performance scores"), callback=self.selectionChanged, selectionMode = QListWidget.MultiSelection)

        sip.delete(self.mainArea.layout())
        self.layout = QHBoxLayout(self.mainArea)
       
        self.dottedGrayPen = QPen(QBrush(Qt.gray), 1, Qt.DotLine)
        self.graph = graph = PerformanceGraph(self, self.mainArea)
        graph.state = NOTHING
        graph.setAxisScale(QwtPlot.xBottom, 0.0, 1.0, 0.0)
        graph.setAxisScale(QwtPlot.yLeft, 0.0, 1.0, 0.0)
        graph.useAntialiasing = True
        graph.insertLegend(QwtLegend(), QwtPlot.BottomLegend)
        graph.gridCurve.enableY(True)
        graph.gridCurve.setMajPen(self.dottedGrayPen)
        graph.gridCurve.attach(graph)
        self.mainArea.layout().addWidget(graph)

        b1 = OWGUI.widgetBox(self.mainArea, "Statistics")
        OWGUI.label(b1, self, "%(statistics)s").setTextFormat(Qt.RichText)
        OWGUI.rubber(b1)
        
        self.controlArea.setFixedWidth(220)
    
    def setTestResults(self, res):
        self.res = res
        if res and res.classifierNames:
            self.classifiers = res.classifierNames
            self.selectedClassifier = [0]
            self.comTarget.clear()
            self.comTarget.addItems(self.res.classValues)
            self.targetClass=min(1, len(self.res.classValues))
            self.classifierChanged()
        else:
            self.graph.clear()
            self.thresholdCurve = None
            self.allScores = None

    def classifierChanged(self):
        self.allScores = []
        self.probs = []
        classNo = self.selectedClassifier[0]
        probsClasses = sorted((tex.probabilities[classNo][self.targetClass], self.targetClass==tex.actualClass) for tex in self.res.results)
        self.all = all = len(probsClasses)
        TP = self.P = P = float(sum(x[1] for x in probsClasses))
        FP = self.N = N = all-P
        TN = FN = 0.
        prevprob = probsClasses[0][0]
        for Nc, (prob, kls) in enumerate(probsClasses):
            if kls:
                TP -= 1
                FN += 1
            else:
                FP -= 1
                TN += 1
            if prevprob != prob:
                self.allScores.append(((TP+TN)/all, TP/(P or 1), TN/(N or 1), TP/(all-Nc), TN/Nc, 2*TP/(P+all-Nc), TP, TN, FP, FN, Nc))
                self.probs.append(prevprob)
            prevprob = prob
        self.allScores.append(((TP+TN)/all, TP/(P or 1), TN/(N or 1), TP/(all-Nc), TN/Nc, 2*TP/(P+all-Nc), TP, TN, FP, FN, Nc))
        self.probs.append(prevprob)
        self.allScores = zip(*self.allScores)
        self.selectionChanged()
        
    def selectionChanged(self):
        self.graph.clear()
        self.thresholdCurve = None
        if not self.allScores:
            return            
        for c in self.selectedScores:
            self.graph.addCurve(self.scores[c], self.colors[c], self.colors[c], 1, xData=self.probs, yData=self.allScores[c], style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, lineWidth=3, enableLegend=1)
        self.thresholdChanged()
        # self.graph.replot is called in thresholdChanged
        
    def thresholdChanged(self, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        if self.thresholdCurve:
            self.thresholdCurve.detach()
        self.thresholdCurve = self.graph.addCurve("threshold", Qt.black, Qt.black, 1, xData=[self.threshold]*2, yData=[0,1], style=QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, lineWidth=1)
        self.graph.replot()
        if not self.allScores:
            self.statistics = ""
            return
        ind = 0
        while self.probs[ind] < self.threshold and ind+1 < len(self.probs):
            ind += 1
        alls = self.allScores
        stat = "<b>Sample size: %i instances</b><br/>  Positive: %i<br/>  Negative: %i<br/><br/>" % (self.all, self.P, self.N)
        stat += "<b>Current threshold: %.2f</b><br/><br/>" % self.threshold
        stat += "<b>Positive predictions: %i</b><br/>  True positive: %i<br/>  False positive: %i<br/><br/>" % (self.all-alls[-1][ind], alls[-5][ind], alls[-3][ind])
        stat += "<b>Negative predictions: %i</b><br/>  True negative: %i<br/>  False negative: %i<br/><br/>" % (alls[-1][ind], alls[-4][ind], alls[-3][ind])
        if self.selectedScores:
            stat += "<b>Performance</b><br/>"
        stat += "<br/>".join("%s: %.2f" % (self.scores[i], alls[i][ind]) for i in self.selectedScores)
        self.statistics = stat
        
    def sendReport(self):
        if self.res:
            self.reportSettings(_("Performance Curves"), 
                                [(_("Model"), self.res.classifierNames[self.selectedClassifier[0]]),
                                 (_("Target class"), self.res.classValues[self.targetClass])])
            self.reportImage(self.graph.saveToFileDirect, QSize(790, 390))
            self.reportSection("Performance")
            self.reportRaw(self.statistics)
