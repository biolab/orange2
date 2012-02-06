"""<name>Missing Values</name>
<description>Helps remove features with too many missing values</description>
<icon>icons/MissingValues.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
from math import log

class PropBarItem(QItemDelegate):
    _colors = [QColor(255, 255, 153), QColor(255, 102,51)]
    
    def paint(self, painter, option, index):
        col = index.column()
        if col == 1:
            try:
                text = index.data(Qt.DisplayRole).toString()
                prop = float(str(text).split()[1][1:-2])/100
                row = index.row()
                painter.save()
                self.drawBackground(painter, option, index)
                painter.fillRect(option.rect.adjusted(1, 1, -option.rect.width()*(1-prop), -1), self._colors[self.parent().check[row]])
                self.drawDisplay(painter, option, option.rect, text)
                painter.restore()
                return
            except:
                pass
        return QItemDelegate.paint(self, painter, option, index)
 
        
class OWMissingValues(OWWidget):
    contextHandlers = {"": PerfectDomainContextHandler("", ["check"])}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Compare Examples", noReport=True)
        self.inputs = [("Data", ExampleTable, self.setData, Default)]
        self.outputs = [("Data", ExampleTable, Default), ("Selected Data", ExampleTable)]
        
        self.check = []
        self.resize(520, 560)
        self.loadSettings()
        self.controlArea.setFixedSize(0, 0)
        self.pushing = False
        self.table = OWGUI.table(self.mainArea, rows = 0, columns = 0, selectionMode = QTableWidget.SingleSelection)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Feature", "Instances with missing value", "Data increase if removed"])
        self.table.verticalHeader().hide()
        self.table.setItemDelegate(PropBarItem(self))
        self.connect(self.table, SIGNAL("itemChanged(QTableWidgetItem *)"), self.checksChanged)
        self.connect(self.table, SIGNAL("itemSelectionChanged()"), self.sendSelected)

        b = OWGUI.widgetBox(self.mainArea, orientation=0)
        b1 = OWGUI.widgetBox(b)
        OWGUI.button(b1, self, "Select All", self.selectAll, width=150, debuggingEnabled=0)
        self.reportButton = OWGUI.button(b1, self, "&Report", self.reportAndFinish, width=150, debuggingEnabled=0)
        self.reportButton.setAutoDefault(0)
        b1 = OWGUI.widgetBox(b, "Info", orientation=0)
        self.lbInput = OWGUI.label(b1, self, "")
        OWGUI.separator(b1, 16)
        self.lbOutput = OWGUI.label(b1, self, "")

    def setData(self, data):
        self.closeContext()
        self.data = data
        if data:
            self.check = [1]*len(data.domain)
            self.openContext("", data)
            
            self.pushing = True
            self.table.setRowCount(len(data.domain))

            filt_all_unk = orange.Filter_isDefined(domain=data.domain, negate=1)
            self.notMissing = len(data.filterref(filt_all_unk))
            self.lbInput.setText("Input: %i instances\nWith missing data: %i (%.2f%%)" % (len(data), self.notMissing, self.notMissing*100./(len(data) or 1)))

            filt_all_unk.check = [0]*len(data.domain)
            for row, var in enumerate(data.domain):
                it = QTableWidgetItem(" "+var.name)
                it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                it.setCheckState(Qt.Checked if self.check[row] else Qt.Unchecked)
                self.table.setItem(row, 0, it)
                
                filt_all_unk.check[row] = 1
                miss = len(data.filterref(filt_all_unk))
                filt_all_unk.check[row] = 0
                it = QTableWidgetItem(("%i (%.2f%%)" % (miss, miss*100./len(data))) if miss else "-")
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setTextAlignment(Qt.AlignRight)
                self.table.setItem(row, 1, it)

                it = QTableWidgetItem()
                it.setTextAlignment(Qt.AlignRight)
                self.table.setItem(row, 2, it)
            self.table.resizeColumnsToContents()
            self.table.resizeRowsToContents()
        else:
            self.table.setRowCount(0)
            self.lbInput.setText("")
            self.lbOutput.setText("")
        self.pushing = False
        self.table.clearSelection()
        self.checksChanged(None)

    def selectAll(self):
        for row in range(len(self.data.domain)):
            self.table.item(row, 0).setCheckState(Qt.Checked)
        self.checksChanged(None)
        
    def checksChanged(self, item=None):
        if self.pushing or item and item.column():
            return
        self.missingReport = []
        if self.data:
            data = self.data
            filt_unk = orange.Filter_isDefined(domain = data.domain)
            self.check = [self.table.item(i, 0).checkState()==Qt.Checked for i in range(len(data.domain))]
            filt_unk.check = self.check
            dataout = data.filterref(filt_unk)
            totOut = len(dataout)
            percOut = len(dataout)*100./(len(self.data) or 1)
            self.lbOutput.setText("Output: %i (%.2f%%) instances\nRemoved: %i (%.2f%%)" % 
                                  (totOut, percOut, len(self.data)-totOut, 100-percOut))
            variables = []
            for i, checked in enumerate(self.check):
                it = self.table.item(i, 2)
                if checked:
                    filt_unk.check[i] = 0
                    datatest = data.filterref(filt_unk)
                    filt_unk.check[i] = 1
                    miss = len(datatest) - len(dataout)
                    txt = "%i" % miss if miss else "-"
                    it.setText(txt)
                    it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    var = self.data.domain[i]
                    self.missingReport.append((var.name, txt))
                    variables.append(var)
                else:
                    it.setText("(removed)")
                    it.setFlags(Qt.NoItemFlags)
                # this is ridiculous, but triggering an update by calling update doesn't work
                it = self.table.item(i, 1)
                tt = it.text()
                it.setText(str(tt).strip() + " "*(tt and tt[-1]!=" "))
            self.dataout = orange.ExampleTable(orange.Domain(variables), dataout)
        else:
            self.dataout = None
        self.send("Data", self.dataout)
            
    def sendSelected(self):
        toSend = None
        if self.data:
            selection = self.table.selectedItems()
            if len(selection)==1 and selection[0].column():
                row = selection[0].row()
                data = self.data
                filt_unk = orange.Filter_isDefined(domain = data.domain)
                if selection[0].column()==2:
                    filt_unk.check = [i!=row and self.table.item(i, 0).checkState()==Qt.Checked for i in range(len(data.domain))]
                    data = data.filterref(filt_unk)
                filt_unk.negate=1
                filt_unk.check = [i==row for i in range(len(data.domain))]
                toSend = data.filterref(filt_unk)
        self.send("Selected Data", toSend)

    def sendReport(self):
        import OWReport
        self.reportData(self.data, "Original data")
        self.reportData(self.dataout, "Output data")
        self.reportSection("Missing values by features")
        self.reportRaw(OWReport.reportTable(self.table))