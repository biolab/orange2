"""
<name>K-Means Clustering</name>
<description>K-means clustering.</description>
<icon>icons/KMeans.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>2300</priority>
"""

import orange, orngCluster
import OWGUI
import math, statc
from OWWidget import *
from itertools import izip

##############################################################################
# main class

class OWKMeans(OWWidget):
    settingsList = ["K", "DistanceMeasure", "classifySelected", "addIdAs", "classifyName"]

    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, 'k-Means Clustering')

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable), ("Medoids", ExampleTable)]

        #set default settings
        self.K = 3
        self.DistanceMeasure = 0
        self.classifySelected = 1
        self.addIdAs = 0
        self.classifyName = "clusterId"
        self.loadSettings()

        self.data = None

        # GUI definition
        # settings
        box = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
        OWGUI.spin(box, self, "K", label="Number of clusters"+"  ", min=1, max=30, step=1)
        OWGUI.comboBox(box, self, "DistanceMeasure", label="Distance measure", items=["Euclidean", "Manhattan"], tooltip=None)
        OWGUI.button(box, self, "Run Clustering", callback = self.cluster)
        OWGUI.rubber(box)

        box = OWGUI.widgetBox(self.controlArea, "Cluster IDs")
        cb = OWGUI.checkBox(box, self, "classifySelected", "Append cluster indices")
        self.classificationBox = ib = OWGUI.indentedBox(box)
        le = OWGUI.lineEdit(ib, self, "classifyName", "Name" + "  ", orientation=0, controlWidth=75, valueType = str)
        OWGUI.separator(ib, height = 4)
        aa = OWGUI.comboBox(ib, self, "addIdAs", label = "Place" + "  ", orientation = 0, items = ["Class attribute", "Attribute", "Meta attribute"])
        cb.disables.append(ib)
        cb.makeConsistent()
        OWGUI.separator(box)
        OWGUI.button(box, self, "Apply Changes", callback = self.sendData)


        # display of clustering results
        self.table = OWGUI.table(self.mainArea, selectionMode = QTableWidget.NoSelection)
        self.table.hide()

        self.resize(100,100)


    def cluster(self):
        self.error()
        if self.data:
            examples = [[float(x) for x in d] for d in self.data]
            self.mc = orngCluster.MClustering(examples, int(self.K), self.DistanceMeasure+1)
            # This fix is needed since orngCluster.MClustering does not report errors,
            # and only returns erroneous results instead
            if max(self.mc.mapping) > self.K:
                self.error("Check whether your data contains enough distinct examples\nfor the desired number of clusters")  
                self.mc = None
            else:
                self.mc.medoids = [x-1 for x in self.mc.medoids]
        else:
            self.mc = None

        self.showResults()
        self.sendData()
        

    def showResults(self):
        self.table.clear() # clears the table
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
        if not self.data or not self.mc:
            self.send("Examples", None)
            self.send("Medoids", None)
            return

        clustVar = orange.EnumVariable(self.classifyName, values = [str(x) for x in range(1, 1+self.K)])

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
        table1=orange.ExampleTable(domain)
        table1.extend(orange.ExampleTable(self.data))
        for ex, midx in izip(table1, self.mc.mapping):
            ex[aid] = clustVar(str(midx))

        self.send("Examples", table1)
        self.send("Medoids", table1.getitems(self.mc.medoids))
        

    def setData(self, data):
        if not data:
            self.data = None
        else:
            self.data = orange.Preprocessor_dropMissing(data)
            self.cluster()

    ##################################################################################################
    # Clustering (following should be replaced by a call to the rutines in the orngCluster, once
    # they are ready)
    
    # computes BIC for a classified data set, given the medoids
    # medoids is a matrix (list of lists of attribute values
    def compute_bic(self):
        data = self.data
        medoids = [0] + [data[x] for x in self.mc.medoids] # indices in mapping are 1-based
        mapping = self.mc.mapping
        K = self.K

        M = len(data.domain.attributes)
        R = float(len(data))
        Ri = [mapping.count(x) for x in range(1+K)]
        numFreePar = (M+1.) * K * math.log(R, 2.) / 2.
        # sigma**2
        s2 = 0.
        cidx = [i for i, attr in enumerate(data.domain.attributes) if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]
        for x, midx in izip(data, mapping):
            medoid = medoids[midx] # medoids has a dummy element at the beginning, so we don't need -1 
            s2 += sum( [(float(x[i]) - float(medoid[i]))**2 for i in cidx] )
        s2 /= (R - K)
        if s2 < 1e-20:
            return None, [None]*K
        # log-lokehood of clusters: l(Dn)
        # log-likehood of clustering: l(D)
        ld = 0
        bicc = []
        for k in range(1, 1+K):
            ldn = -1. * Ri[k] * ((math.log(2. * math.pi, 2) / -2.) - (M * math.log(s2, 2) / 2.) + (K / 2.) + math.log(Ri[k], 2) - math.log(R, 2))
            ld += ldn
            bicc.append(ldn - numFreePar)
        return ld - numFreePar, bicc

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
    #d = orange.ExampleTable('glass')
    d = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab")
    ow.setData(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
