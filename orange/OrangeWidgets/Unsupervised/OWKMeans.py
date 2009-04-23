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
import statc
from PyQt4.Qwt5 import *
from itertools import izip

##############################################################################
# main class

class OWKMeans(OWWidget):
    settingsList = ["K", "distanceMeasure", "classifySelected", "addIdAs", "classifyName",
                    "initializationType", "runAnyChange"]
    
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

    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, 'k-Means Clustering')

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable), ("Centroids", ExampleTable)]

        #set default settings
        self.K = 3
        self.distanceMeasure = 0
        self.initializationType = 0
        self.classifySelected = 1
        self.addIdAs = 0
        self.runAnyChange = 1
        self.classifyName = "Cluster"
        self.loadSettings()

        self.data = None # holds input data
        self.km = None   # holds clustering object

        # GUI definition
        # settings
        box = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
        OWGUI.spin(box, self, "K", label="Number of clusters"+"  ", min=1, max=30, step=1,
                   callback = self.initializeClustering)
        OWGUI.comboBox(box, self, "distanceMeasure", label="Distance measures",
                       items=[name for name, _ in self.distanceMeasures],
                       tooltip=None,
                       callback = self.initializeClustering)
        OWGUI.comboBox(box, self, "initializationType", label="Initialization",
                       items=[name for name, _ in self.initializations],
                       tooltip=None,
                       callback = self.initializeClustering)

        box = OWGUI.widgetBox(self.controlArea, "Run", addSpace=True)
        OWGUI.checkBox(box, self, "runAnyChange", "Run after any change")
        OWGUI.button(box, self, "Run Clustering", callback = self.cluster)
        OWGUI.rubber(box)

        box = OWGUI.widgetBox(self.controlArea, "Cluster IDs")
        cb = OWGUI.checkBox(box, self, "classifySelected", "Append cluster indices")
        le = OWGUI.lineEdit(box, self, "classifyName", "Name" + "  ",
                            orientation="horizontal", controlWidth=60,
                            valueType=str, callback=self.sendData)
        OWGUI.separator(box, height = 4)
        cc = OWGUI.comboBox(box, self, "addIdAs", label = "Place" + "  ",
                            orientation="horizontal", items = ["Class attribute", "Attribute", "Meta attribute"])
        cb.disables.append(le)
        cb.disables.append(cc)
        cb.makeConsistent()
        OWGUI.separator(box)

        # display of clustering results
        self.table = OWGUI.table(self.mainArea, selectionMode = QTableWidget.NoSelection)
        self.table.hide()

        self.resize(100,100)

    def cluster(self):
        if self.K > len(self.data):
            self.error("Not enough data instances (%d) for given number of clusters (%d)." % \
                       (len(self.data), self.K))
            return
        self.progressBarInit()
        self.km.run()
        # self.showResults()
        self.sendData()
        self.progressBarFinished()

    def clusterCallback(self, km):
        norm = math.log(len(self.data), 10)
        if km.iteration < norm:
            self.progressBarSet(80.0 * km.iteration / norm)
        else:
            self.progressBarSet(80.0 + 0.15 * (1.0 - math.exp(norm - km.iteration)))
        
    def showResults(self):
        self.table.clear() # clears the table
        return
        if not self.mc:
            return
        
        actualK = self.K
        self.table.setColumnCount(4)
        self.table.setRowCount(actualK+1)

        self.header = self.table.horizontalHeader()
        header = ["ID", "Items", "Fitness", "BIC"]
        for (i, h) in enumerate(header):
            self.table.setHorizontalHeaderItem(i, QTableWidgetItem(h))

        dist = [0] * actualK
        for m in self.mc.mapping:
            dist[m-1] += 1

        bic, cbic = self.compute_bic()
        for k in range(actualK):
            self.table.setItem(k, 0, QTableWidgetItem(str(k+1)))
            self.table.setItem(k, 1, QTableWidgetItem(str(dist[k])))
            self.table.setItem(k, 2, QTableWidgetItem("%5.3f" % self.mc.cdisp[k]))
            self.table.setItem(k, 3, QTableWidgetItem(bic is None and u"\u221E" or ("%6.2f" % cbic[k])))

        colorItem(self.table, actualK, 0, "Total")
        colorItem(self.table, actualK, 1, str(len(self.data)))
        colorItem(self.table, actualK, 2, "%5.3f" % self.mc.disp)
        colorItem(self.table, actualK, 3, bic is None and u"\u221E" or ("%6.2f" % bic))

        for i in range(4):
            self.table.resizeColumnToContents(i)
        self.table.show()

    def sendData(self):
        if not self.data or not self.km:
            self.send("Examples", None)
            self.send("Centroids", None)
            return

        clustVar = orange.EnumVariable(self.classifyName, values = ["C%d" % (x+1) for x in range(self.K)])

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
        for ex, midx in izip(new, self.km.clusters):
            ex[aid] = midx

        self.send("Examples", new)
        self.send("Centroids", orange.ExampleTable(self.km.centroids))
        
    def setData(self, data):
        """Handle data from the input signal."""
        if not data:
            self.data = None
        else:
            self.data = data
            self.initializeClustering()

    def initializeClustering(self):
        self.km = orngClustering.KMeans(
            self.data,
            centroids = self.K,
            initialization = self.initializations[self.initializationType][1],
            distance = self.distanceMeasures[self.distanceMeasure][1],
            initialize_only = True,
            inner_callback = self.clusterCallback,
            )
        if self.runAnyChange:
            self.cluster()

    def sendReport(self):
        self.reportSettings("Settings", [("Distance measure", self.distanceMeasures[self.distanceMeasure]),
                                         ("Number of clusters", self.K)])
        self.reportData(self.data)
        self.reportSection("Cluster data")
        res = "<table><tr>"+"".join('<td align="right"><b>&nbsp;&nbsp;%s&nbsp;&nbsp;</b></td>' % n for n in ("ID", "Items", "Fitness", "BIC")) + "</tr>\n"
        for i in range(self.K):
            res += "<tr>"+"".join('<td align="right">&nbsp;&nbsp;%s&nbsp;&nbsp;</td>' % str(self.table.item(i, j).text()) for j in range(4)) + "</tr>\n"
        res += "<tr>"+"".join('<td align="right"><b>&nbsp;&nbsp;%s&nbsp;&nbsp;</b></td>' % str(self.table.item(self.K, j).text()) for j in range(4)) + "</tr>\n"
        res += "</table>"
        self.reportRaw(res)


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
