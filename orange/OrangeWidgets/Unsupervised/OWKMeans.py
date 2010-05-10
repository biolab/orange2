"""
<name>k-Means Clustering</name>
<description>k-means clustering.</description>
<icon>icons/KMeans.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>2300</priority>
"""

from OWWidget import *
import OWGUI
import orange
import orngClustering
import math
import random
import statc
from PyQt4.Qwt5 import *
from itertools import izip

##############################################################################
# main class

class OWKMeans(OWWidget):
    settingsList = ["K", "optimized", "optimizationFrom", "optimizationTo", "scoring", "distanceMeasure", "classifySelected", "addIdAs", "classifyName",
                    "initializationType", "restarts", "runAnyChange"]
    
    distanceMeasures = [
        ("Euclidean", orange.ExamplesDistanceConstructor_Euclidean),
        ("Pearson Correlation", orngClustering.ExamplesDistanceConstructor_PearsonR),
        ("Spearman Rank Correlation", orngClustering.ExamplesDistanceConstructor_SpearmanR),
        ("Manhattan", orange.ExamplesDistanceConstructor_Manhattan),
        ("Maximal", orange.ExamplesDistanceConstructor_Maximal),
        ("Hamming", orange.ExamplesDistanceConstructor_Hamming),
        ]

    initializations = [
        ("Random", orngClustering.kmeans_init_random),
        ("Diversity", orngClustering.kmeans_init_diversity),
        ("Agglomerative clustering", orngClustering.KMeans_init_hierarchicalClustering(n=100)),
        ]
    
    scoringMethods = [
        ("Silhouette (heuristic)", orngClustering.score_fastsilhouette),
        ("Silhouette", orngClustering.score_silhouette),
        ("Between cluster distance", orngClustering.score_betweenClusterDistance),
        ("Distance to centroids", orngClustering.score_distance_to_centroids)
        ] 

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, 'k-Means Clustering')

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable), ("Centroids", ExampleTable)]

        #set default settings
        self.K = 2
        self.optimized = True
        self.optimizationFrom = 2
        self.optimizationTo = 5
        self.scoring = 0
        self.distanceMeasure = 0
        self.initializationType = 0
        self.restarts = 1
        self.classifySelected = 1
        self.addIdAs = 0
        self.runAnyChange = 1
        self.classifyName = "Cluster"
        
        self.settingsChanged = False
        
        self.loadSettings()

        self.data = None # holds input data
        self.km = None   # holds clustering object

        # GUI definition
        # settings
        
        box = OWGUI.widgetBox(self.controlArea, "Clusters (k)")
        bg = OWGUI.radioButtonsInBox(box, self, "optimized", [], callback=self.setOptimization)
        fixedBox = OWGUI.widgetBox(box, orientation="horizontal")
        button = OWGUI.appendRadioButton(bg, self, "optimized", "Fixed", 
                                         insertInto=fixedBox, tooltip="Fixed number of clusters")
        self.fixedSpinBox = OWGUI.spin(OWGUI.widgetBox(fixedBox), self, "K", min=2, max=30, tooltip="Fixed number of clusters",
                                       callback=self.update, callbackOnReturn=True)
        OWGUI.rubber(fixedBox)
        
        optimizedBox = OWGUI.widgetBox(box)
        button = OWGUI.appendRadioButton(bg, self, "optimized", "Optimized", insertInto=optimizedBox)
        option = QStyleOptionButton()
        option.initFrom(button)
        box = OWGUI.indentedBox(optimizedBox, qApp.style().subElementRect(QStyle.SE_CheckBoxIndicator,
                                                                           option, button).width() - 3)
        self.optimizationBox = box
        OWGUI.spin(box, self, "optimizationFrom", label="From", min=2, max=99,
                   tooltip="Minimum number of clusters to try", callback=self.updateOptimizationFrom, callbackOnReturn=True)
        b = OWGUI.spin(box, self, "optimizationTo", label="To", min=3, max=100,
                   tooltip="Maximum number of clusters to try", callback=self.updateOptimizationTo, callbackOnReturn=True)
#        b.control.setLineEdit(OWGUI.LineEditWFocusOut(b))
        OWGUI.comboBox(box, self, "scoring", label="Scoring", orientation="horizontal",
                       items=[m[0] for m in self.scoringMethods], callback=self.update)
        
        
        
        box = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
#        OWGUI.spin(box, self, "K", label="Number of clusters"+"  ", min=1, max=30, step=1,
#                   callback = self.initializeClustering)
        OWGUI.comboBox(box, self, "distanceMeasure", label="Distance measures",
                       items=[name for name, _ in self.distanceMeasures],
                       tooltip=None, indent=20,
                       callback = self.update)
        cb = OWGUI.comboBox(box, self, "initializationType", label="Initialization",
                       items=[name for name, _ in self.initializations],
                       tooltip=None, indent=20,
                       callback = self.update)
        OWGUI.spin(cb.box, self, "restarts", label="Restarts", orientation="horizontal",
                   min=1, max=100, callback=self.update, callbackOnReturn=True)

        box = OWGUI.widgetBox(self.controlArea, "Cluster IDs")
        cb = OWGUI.checkBox(box, self, "classifySelected", "Append cluster indices")
        box = OWGUI.indentedBox(box)
        form = QWidget()
        le = OWGUI.lineEdit(form, self, "classifyName", None, #"Name" + "  ",
                            orientation="horizontal", #controlWidth=100, 
                            valueType=str,
#                            callback=self.sendData,
#                            callbackOnReturn=True
                            )
        
        cc = OWGUI.comboBox(form, self, "addIdAs", label = " ", #"Place" + "  ",
                            orientation="horizontal", items = ["Class attribute", "Attribute", "Meta attribute"],
                            )
        
        layout = QFormLayout()
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignJustify)
        layout.addRow("Name  ", le)
        layout.addRow("Place  ", cc)
#        le.setFixedWidth(cc.sizeHint().width())
        form.setLayout(layout)
        box.layout().addWidget(form)
        cb.disables.append(box)
#        cb.disables.append(cc.box)
        cb.makeConsistent()
#        OWGUI.separator(box)
        
        box = OWGUI.widgetBox(self.controlArea, "Run", addSpace=True)
        cb = OWGUI.checkBox(box, self, "runAnyChange", "Run after any change")
        self.runButton = b = OWGUI.button(box, self, "Run Clustering", callback = self.run)
        OWGUI.setStopper(self, b, cb, "settingsChanged", callback=self.run)

        OWGUI.rubber(self.controlArea)
        # display of clustering results
        
        
        self.optimizationReportBox = OWGUI.widgetBox(self.mainArea)
        tableBox = OWGUI.widgetBox(self.optimizationReportBox, "Optimization Report")
        self.table = OWGUI.table(tableBox, selectionMode=QTableWidget.SingleSelection)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["k", "Best", "Score"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setItemDelegateForColumn(2, OWGUI.TableBarItem(self, self.table))
        self.table.setItemDelegateForColumn(1, OWGUI.IndicatorItemDelegate(self))
        self.table.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.table.hide()
        
        self.connect(self.table, SIGNAL("itemSelectionChanged()"), self.tableItemSelected)
        
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.mainArea.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        
        
        OWGUI.rubber(self.topWidgetPart)
        
        self.updateOptimizationGui()

#        self.resize(100,100)

    def adjustSize(self):
        self.ensurePolished()
        s = self.sizeHint()
        self.resize(s)
        
    def sizeHint(self):
        s = self.leftWidgetPart.sizeHint()
        if self.optimized and not self.mainArea.isHidden():
            s.setWidth(s.width() + self.mainArea.sizeHint().width() + self.childrenRect().x() * 4)
        return s
            
        
    def updateOptimizationGui(self):
        state = [True, False, False] if self.optimized else [False, True, True]
        for s, func in zip(state, [self.fixedSpinBox.setDisabled,
                                self.optimizationBox.setDisabled,
                                self.mainArea.setHidden]):
            func(s)
            
        qApp.processEvents()
        self.adjustSize()
            
    def updateOptimizationFrom(self):
#        self.optimizationFrom = min([self.optimizationFrom, self.optimizationTo - 1])
        self.optimizationTo = max([self.optimizationFrom + 1, self.optimizationTo])
        self.update()
        
    def updateOptimizationTo(self):
        self.optimizationFrom = min([self.optimizationFrom, self.optimizationTo - 1])
#        self.optimizationTo = max([self.optimizationFrom + 1, self.optimizationTo])
        self.update()
        
    def setOptimization(self):
        self.updateOptimizationGui()
        self.update()
            
    def runOptimization(self):
        if self.optimizationTo > len(set(self.data)):
            self.error("Not enough unique data instances (%d) for given number of clusters (%d)." % \
                       (len(set(self.data)), self.optimizationTo))
            return
        
        random.seed(0)
        try:
            self.progressBarInit()
            Ks = range(self.optimizationFrom, self.optimizationTo + 1)
            self.optimizationRun =[(k, orngClustering.KMeans(
                    self.data,
                    centroids = k,
                    minscorechange=0,
                    nstart = self.restarts,
                    initialization = self.initializations[self.initializationType][1],
                    distance = self.distanceMeasures[self.distanceMeasure][1],
                    scoring = self.scoringMethods[self.scoring][1],
                    inner_callback = lambda val: self.progressBarSet(min(self.progressEstimate(val)/len(Ks) + k * 100.0 / len(Ks), 100.0))
                    )) for k in Ks]
            self.progressBarFinished()
            self.bestRun = (min if getattr(self.scoringMethods[self.scoring][1], "minimize", False) else max)(self.optimizationRun, key=lambda (k, run): run.score)
            self.showResults()
            self.sendData()
        except Exception, ex:
            self.error(0, "An error occured while running optimization. Reason: " + str(ex))
            raise
        
    def cluster(self):
        if self.K > len(set(self.data)):
            self.error("Not enough unique data instances (%d) for given number of clusters (%d)." % \
                       (len(set(self.data)), self.K))
            return
        random.seed(0)
        
        self.km = orngClustering.KMeans(
            self.data,
            centroids = self.K,
            minscorechange=0,
            nstart = self.restarts,
            initialization = self.initializations[self.initializationType][1],
            distance = self.distanceMeasures[self.distanceMeasure][1],
            scoring = self.scoringMethods[self.scoring][1],
            initialize_only = True,
            inner_callback = self.clusterCallback,
            )
        self.progressBarInit()
        self.km.run()
        self.sendData()
        self.progressBarFinished()

    def clusterCallback(self, km):
        norm = math.log(len(self.data), 10)
        if km.iteration < norm:
            self.progressBarSet(80.0 * km.iteration / norm)
        else:
            self.progressBarSet(80.0 + 0.15 * (1.0 - math.exp(norm - km.iteration)))
            
    def progressEstimate(self, km):
        norm = math.log(len(self.data), 10)
        if km.iteration < norm:
            return min(80.0 * km.iteration / norm, 90.0)
        else:
            return min(80.0 + 0.15 * (1.0 - math.exp(norm - km.iteration)), 90.0)
        
    def showResults(self):
        self.table.setRowCount(len(self.optimizationRun))
        bestScore = self.bestRun[1].score
        worstScore = (max if getattr(self.scoringMethods[self.scoring][1], "minimize", False) else min)([km.score for k, km in self.optimizationRun])
        for i, (k, run) in enumerate(self.optimizationRun):
            item = OWGUI.tableItem(self.table, i, 0, k)
            item.setData(Qt.TextAlignmentRole, QVariant(Qt.AlignCenter))
            
            item = OWGUI.tableItem(self.table, i, 1, None)#" " if (k, run) == self.bestRun else "")
            item.setData(OWGUI.IndicatorItemDelegate.IndicatorRole, QVariant((k, run) == self.bestRun))
#            item.setData(Qt.DecorationRole, QVariant(QIcon(os.path.join(os.path.dirname(OWGUI.__file__), "icons", "circle.png"))))
            item.setData(Qt.TextAlignmentRole, QVariant(Qt.AlignCenter))
            
            fmt = lambda score, max_decimals=10: "%%.%if" % min(int(abs(math.log(max(score, 1e-10)))) + 2, max_decimals) if score > 0 and score < 1 else "%.1f"
            item = OWGUI.tableItem(self.table, i, 2, fmt(run.score) % run.score)
            item.setData(OWGUI.TableBarItem.BarRole, QVariant((bestScore - run.score) / ((bestScore - worstScore) or 1) * 0.95))
            if (k, run) == self.bestRun:
                self.table.selectRow(i)
            
        for i in range(2):
            self.table.resizeColumnToContents(i)
        self.table.show()
        qApp.processEvents()
        self.adjustSize()

    def run(self):
        self.error(0)
        if not self.data:
            return
        if self.optimized:
            self.runOptimization()
        else:
            self.cluster()

    def update(self):
        if self.runAnyChange:
            self.run()
        else:
            self.settingsChanged = True
            
    def tableItemSelected(self):
        selectedItems = self.table.selectedItems()
        rows = set([item.row() for item in selectedItems])
        if len(rows) == 1:
            row = rows.pop()
            self.sendData(self.optimizationRun[row][1])

    def sendData(self, km=None):
        if km is None:
            km = self.bestRun[1] if self.optimized else self.km 
        if not self.data or not km:
            self.send("Examples", None)
            self.send("Centroids", None)
            return
        clustVar = orange.EnumVariable(self.classifyName, values = ["C%d" % (x+1) for x in range(km.k)])

        origDomain = self.data.domain
        if self.addIdAs == 0:
            domain=orange.Domain(origDomain.attributes,clustVar)
            if origDomain.classVar:
                domain.addmeta(orange.newmetaid(), origDomain.classVar)
            aid = -1
        elif self.addIdAs == 1:
            domain=orange.Domain(origDomain.attributes+[clustVar], origDomain.classVar)
            aid = len(origDomain.attributes)
        else:
            domain=orange.Domain(origDomain.attributes, origDomain.classVar)
            aid=orange.newmetaid()
            domain.addmeta(aid, clustVar)

        domain.addmetas(origDomain.getmetas())

        # construct a new data set, with a class as assigned by k-means clustering
        new = orange.ExampleTable(domain, self.data)
        for ex, midx in izip(new, km.clusters):
            ex[aid] = midx

        self.send("Examples", new)
        self.send("Centroids", orange.ExampleTable(km.centroids))
        
    def setData(self, data):
        """Handle data from the input signal."""
        self.runButton.setEnabled(bool(data))
        if not data:
            self.data = None
            self.table.setRowCount(0)
        else:
            self.data = data
            self.run()

    def sendReport(self):
        settings = [("Distance measure", self.distanceMeasures[self.distanceMeasure][0]),
                    ("Initialization", self.initializations[self.initializationType][0]),
                    ("Restarts", self.restarts)]
        if self.optimized:
            self.reportSettings("Settings", settings)
            self.reportSettings("Optimization", [("Minimum num. of clusters", self.optimizationFrom),
                                                 ("Maximum num. of clusters", self.optimizationTo),
                                                 ("Scoring method", self.scoringMethods[self.scoring][0])])
        else:
            self.reportSettings("Settings", settings + [("Number of clusters (K)", self.K)])
        self.reportData(self.data)
        if self.optimized:
            self.reportSection("Cluster size optimization report")
            import OWReport 
            self.reportRaw(OWReport.reportTable(self.table))


##############################################################################

class colorItem(QTableWidgetItem):
    def __init__(self, table, i, j, text, flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable, color=Qt.lightGray):
        self.color = color
        QTableWidgetItem.__init__(self, unicode(text))
        self.setFlags(flags)
        table.setItem(i, j, self)

    def paint(self, painter, colorgroup, rect, selected):
        g = QPalette(colorgroup)
        g.setColor(QPalette.Base, self.color)
        QTableWidgetItem.paint(self, painter, g, rect, selected)


##################################################################################################
# Test this widget

if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = OWKMeans()
    d = orange.ExampleTable("../../doc/datasets/iris.tab")
    ow.setData(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
