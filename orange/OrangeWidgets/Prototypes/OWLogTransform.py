"""<name>Log Transform</name>
<description>Logarithmic transformation of attributes</description>
<icon>icons/LogTransform.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
from math import log

class OWLogTransform(OWWidget):
    # We cannot put attribute selection to context settings; see comment at func settingsFromWidget
    contextHandlers = {"": PerfectDomainContextHandler("", ["transformed"])}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Compare Examples")
        self.inputs = [("Examples", ExampleTable, self.setData, Default)]
        self.outputs = [("Examples", ExampleTable, Default)]
        
        self.transformed = []
        self.controlArea.setFixedWidth(0)
        
        self.pushing = False

        self.table = OWGUI.table(self.mainArea, rows = 0, columns = 0, selectionMode = QTableWidget.SingleSelection)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Attribute", "Skewness", "Kurtosis"])
        self.table.verticalHeader().hide()
        self.connect(self.table, SIGNAL("itemChanged(QTableWidgetItem *)"), self.checksChanged)

        self.setFixedSize(280, 500)

    def setData(self, data):
        self.closeContext()
        self.data = data
        if data:
            self.table.show()
            self.contAttrs = [attr for attr in data.domain if attr.varType == orange.Variable.Continuous]
            self.basStat = orange.DomainBasicAttrStat(self.data)
            self.table.setRowCount(len(self.contAttrs))
            row = 0
            self.pushing = True
            for stat in self.basStat:
                if stat:
                    var = stat.variable
                    it = QTableWidgetItem(" "+var.name)
                    it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    it.setCheckState(Qt.Unchecked)
                    self.table.setItem(row, 0, it)
                    row += 1 
        
            self.table.resizeColumnsToContents()
            self.table.resizeRowsToContents()
        else:
            self.table.hide()
        self.pushing = False
        self.checksChanged()
        self.adjustSize()
        self.openContext("", data)

    def checksChanged(self, *x):
        import sys
        if not self.pushing:
            transs = {}
            for i, attr in enumerate(self.contAttrs):
                if self.table.item(i, 0).checkState()==Qt.Checked:
                    newAttr = orange.FloatVariable("log"+attr.name)
                    newAttr.getValueFrom = lambda ex, foo=0, at=attr, newAt=newAttr: orange.Value(newAt, log(ex[at]) if not ex[at].isSpecial() and ex[at]>0 else "?")
                    transs[attr.name] = newAttr
            oldDomain = self.data.domain
            newDomain = orange.Domain([transs.get(attr.name, attr) for attr in oldDomain.attributes],
                                      transs.get(oldDomain.classVar.name, oldDomain.classVar))
            self.send("Examples", orange.ExampleTable(newDomain, self.data))