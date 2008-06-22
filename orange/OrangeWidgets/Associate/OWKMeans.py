"""
<name>K-Means Clustering</name>
<description>K-means clustering.</description>
<icon>icons/KMeans.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>2000</priority>
"""

import orngOrangeFoldersQt4
import orange, orngCluster
import OWGUI
import math, statc
from OWWidget import *

##############################################################################
# main class

class OWKMeans(OWWidget):
    settingsList = ["K", "DistanceMeasure"]

    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, 'k-Means Clustering')

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable)]

        #set default settings
        self.K = 3
        self.DistanceMeasure = 0
        self.loadSettings()

        self.data = None

        # GUI definition
        # settings
        box = OWGUI.widgetBox(self.controlArea, "Settings")
        OWGUI.spin(box, self, "K", label="Number of clusters"+"  ", min=1, max=30, step=1, callback=self.settingsChanged)
        OWGUI.separator(box)
        OWGUI.comboBox(box, self, "DistanceMeasure", label="Distance measure", items=["Euclidean", "Manhattan"], tooltip=None, callback=self.settingsChanged)
        OWGUI.separator(box)
        self.applyBtn = OWGUI.button(box, self, "&Apply", callback = self.cluster)
        self.applyBtn.setDisabled(TRUE)

        # display of clustering results
        self.table = OWGUI.table(self.mainArea, selectionMode = QTableWidget.NoSelection)
        self.table.hide()

        self.resize(100,100)


    def settingsChanged(self):
        if self.data:
            self.applyBtn.setDisabled(FALSE)

    def showResults(self):
        self.table.clear() # clears the table
        self.table.setColumnCount(4)
        self.table.setRowCount(self.K+1)

        # set the header (attribute names)
        self.header = self.table.horizontalHeader()
        header = ["ID", "Items", "Fitness", "BIC"]
        for (i, h) in enumerate(header):
            self.table.setHorizontalHeaderItem(i, QTableWidgetItem(h))

        dist = [0] * self.K
        for m in self.mc.mapping:
            dist[m-1] += 1

        bic, cbic = compute_bic(self.cdata, self.mc.medoids)
        for k in range(self.K):
            self.table.setItem(k, 0, QTableWidgetItem(str(k+1)))
            self.table.setItem(k, 1, QTableWidgetItem(str(dist[k])))
            self.table.setItem(k, 2, QTableWidgetItem("%5.3f" % self.mc.cdisp[k]))
            self.table.setItem(k, 3, QTableWidgetItem("%6.2f" % cbic[k]))

        colorItem(self.table, self.K, 0, "Total")
        colorItem(self.table, self.K, 1, str(len(self.data)))
        colorItem(self.table, self.K, 2, "%5.3f" % self.mc.disp)
        colorItem(self.table, self.K, 3, "%6.2f" % bic)

        # adjust the width of the table
        for i in range(4):
            self.table.resizeColumnToContents(i)
        self.table.show()

    def cluster(self):
        self.K = int(self.K)
        if not self.data:
            return

        examples = []
        for d in self.data:
            examples.append([float(x) for x in d])
        # call the clustering method
        self.mc = orngCluster.MClustering(examples, self.K, self.DistanceMeasure+1)

        # construct a new data set, with a class as assigned by k-means clustering
        cl = orange.EnumVariable("cluster")
        cl.values = [str(x+1) for x in range(self.K)]
        domain = orange.Domain(self.data.domain.attributes, cl)
        metas = self.data.domain.getmetas()
        for id in metas:
            domain.addmeta(id, metas[id])
        if self.data.domain.classVar:
            domain.addmeta(orange.newmetaid(), self.data.domain.classVar)
        self.cdata = orange.ExampleTable(domain, self.data)
        for (i,d) in enumerate(self.cdata):
            d.setclass(self.mc.mapping[i]-1)
        self.mc.medoids = [x-1 for x in self.mc.medoids]

        self.showResults()
        self.applyBtn.setDisabled(TRUE)
        self.send("Examples", self.cdata)

    def setData(self, data):
        if not data:
            pass
        else:
            self.data = orange.Preprocessor_dropMissing(data)
            self.cluster()

##################################################################################################
# Clustering (following should be replaced by a call to the rutines in the orngCluster, once
# they are ready)

# computes BIC for a classified data set, given the medoids
# medoids is a matrix (list of lists of attribute values
def compute_bic(data, medoids):
    cv = data.domain.classVar
    M = len(data.domain.attributes)
    K = len(data.domain.classVar.values)
    R = float(len(data))
    Ri = [0] * K
    for x in data:
        Ri[int(x.getclass())] += 1
    numFreePar = (M+1.) * K * math.log(R, 2.) / 2.
    # sigma**2
    s2 = 0.
    cidx = [i for i, attr in enumerate(data.domain.attributes) if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]
    for x in data:
        medoid = data[medoids[int(x.getclass())]]
        s2 += sum( [(float(x[i]) - float(medoid[i]))**2 for i in cidx] )
    s2 /= (R - K)
    # log-lokehood of clusters: l(Dn)
    # log-likehood of clustering: l(D)
    ld = 0
    bicc = []
    for k in range(K):
        ldn = -1. * Ri[k] * ((math.log(2. * math.pi, 2) / -2.) - (M * math.log(s2, 2) / 2.) + (K / 2.) + math.log(Ri[k], 2) - math.log(R, 2))
        ld += ldn
        bicc.append(ldn - numFreePar)
    return ld - numFreePar, bicc

##############################################################################

class colorItem(QTableWidgetItem):
    def __init__(self, table, i, j, text, flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable, color=Qt.lightGray):
        self.color = color
        QTableWidgetItem.__init__(self, str(text))
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
    #d = orange.ExampleTable('glass')
    d = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab")
    ow.setData(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
