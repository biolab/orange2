"""<name>Pivot</name>
<description>Pivot</description>
<icon>icons/Pivot.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *

disctype, conttype = orange.Variable.Discrete, orange.Variable.Continuous
 
class OWPivot(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["rowAttribute", "columnAttribute", "attribute"])}
    # settingsList is computed from aggregates in __init__
    aggregates = [("Count", "count", [disctype, conttype]),
                  ("Sum", "sum" ,[conttype]), 
                  ("Average", "average", [conttype]),
                  ("Deviation", "deviation", [conttype]),
                  ("Minimum", "minimum", [conttype]),
                  ("Maximum", "maximum", [conttype]),
                  ("Most common", "mostCommon", [disctype]),
                  ("Distribution", "distribution", [disctype]),
                  ("Relative frequencies", "frequencies", [disctype])]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Pivot")
        self.inputs = [("Data", ExampleTable, self.setData, Default)]
        self.outputs = [("Selected Group", ExampleTable, Default)]
        self.settingsList = [a[1] for a in self.aggregates]
        self.icons = self.createAttributeIconDict() 

        self.rowAttribute = self.columnAttribute = self.attribute = ""
        for agg, fld, flag in self.aggregates:
            setattr(self, fld, fld=="count")
        self.resize(640, 480)
        self.loadSettings()

        self.rowCombo = OWGUI.comboBox(self.controlArea, self, "rowAttribute", box="Row attribute", callback=self.rowColAttributeChanged,
                                       sendSelectedValue=1, valueType=str, addSpace=True)
        self.colCombo = OWGUI.comboBox(self.controlArea, self, "columnAttribute", box="Column attribute", callback=self.rowColAttributeChanged,
                                       sendSelectedValue=1, valueType=str, addSpace=True)
        b = OWGUI.widgetBox(self.controlArea, box="Content")
        self.attrCombo = OWGUI.comboBox(b, self, "attribute", callback=self.attributeChanged, sendSelectedValue=1, valueType=str, addSpace=True)
        self.checkBoxes = []
        for agg, fld, flag in self.aggregates:
            cb = OWGUI.checkBox(b, self, fld, agg, callback=self.updateMatrix)
            self.checkBoxes.append((cb, flag))
        OWGUI.rubber(self.controlArea)
            
        self.table = OWGUI.table(self.mainArea, rows=0, columns=0, selectionMode=QTableWidget.SingleSelection)
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.setGridStyle(Qt.DotLine)
        self.connect(self.table, SIGNAL("itemSelectionChanged()"), self.selectionChanged)

    def setData(self, data):
        self.closeContext()
        self.rowCombo.clear()
        self.colCombo.clear()
        self.attrCombo.clear()
        self.data = data
        if data:
            self.discAttrs = [attr for attr in data.domain if attr.varType == orange.Variable.Discrete]
            for attr in self.discAttrs:
                self.rowCombo.addItem(self.icons[attr.varType], attr.name)
                self.colCombo.addItem(self.icons[attr.varType], attr.name)
            for attr in data.domain:
                self.attrCombo.addItem(self.icons[attr.varType], attr.name)
            if self.discAttrs:
                self.rowAttribute = self.discAttrs[0].name
                self.columnAttribute = self.discAttrs[min(1, len(self.discAttrs))].name
            if self.attrCombo.count():
                self.attribute = self.data.domain[0].name
        self.openContext("", self.data)
        self.rowColAttributeChanged()
        
    def rowColAttributeChanged(self):
        if self.rowCombo.count():
            self.realRowAttr = self.data.domain[self.rowAttribute]
            self.realColAttr = self.data.domain[self.columnAttribute]
            ncols = len(self.realColAttr.values)
            nrows = len(self.realRowAttr.values)
            self.subsets = [[orange.ExampleTable(self.data.domain) for i in range(ncols+1)] for j in range(nrows+1)]
            for ex in self.data:
                rowVal, colVal =ex[self.realRowAttr], ex[self.realColAttr]
                if not (rowVal.isSpecial() or colVal.isSpecial()):
                    self.subsets[int(rowVal)][int(colVal)].append(ex)
                    self.subsets[int(rowVal)][ncols].append(ex)
                    self.subsets[nrows][int(colVal)].append(ex)
                    self.subsets[nrows][ncols].append(ex)
            self.horizontalValues = list(self.realColAttr.values)
            self.verticalValues = list(self.realRowAttr.values)
            self.table.show()
        else:
            self.subsets = None
            self.table.hide()
        self.attributeChanged()

            
    def attributeChanged(self):
        self.stats = None
        if self.attrCombo.count():
            self.realAttribute = self.data.domain[self.attribute]
            for cb, flag in self.checkBoxes:
                cb.setDisabled(self.realAttribute.varType not in flag)
            if self.subsets:
                if self.realAttribute.varType == orange.Variable.Continuous:
                    self.stats = [[orange.BasicAttrStat(self.realAttribute, examples) for examples in row] for row in self.subsets]
                else:
                    self.stats = [[orange.Distribution(self.realAttribute, examples) for examples in row] for row in self.subsets]
        self.updateMatrix()
        
    def rowData(self, shw, row):
        attr = self.data.domain[self.attribute]
        if shw == "Count":
            return [str(stat.n) if type(stat)==orange.BasicAttrStat else str(stat.cases) for stat in row]
        elif shw == "Sum":
            return [str(attr(stat.sum)) for stat in row]
        elif shw == "Average":
            return [(str(attr(stat.avg)) if stat.n else "") for stat in row]
        elif shw == "Deviation":
            return [(str(attr(stat.dev)) if stat.sum > 1 else "") for stat in row]
        elif shw == "Minimum":
            return [(str(attr(stat.min)) if stat.n else "") for stat in row]
        elif shw == "Maximum":
            return [(str(attr(stat.max)) if stat.n else "") for stat in row]
        elif shw == "Most common":
            values = []
            for stat in row:
                if stat.cases:
                    m = stat.modus()
                    values.append(("%s (%.2f%%)" % (str(m), stat[int(m)]*100./stat.cases)).decode("utf-8"))
                else:
                    values.append("")
            return values
        elif shw == "Distribution":
            values = []
            for stat in row:
                if stat.cases:
                    values.append(", ".join("%s:%i" % (str(val), stat[i]) for i, val in enumerate(attr.values)).decode("utf-8"))
                else:
                    values.append("")
            return values
        elif shw == "Relative frequencies":
            values = []
            for stat in row:
                if stat.cases:
                    values.append(", ".join("%s:%.2f%%" % (str(val), stat[i]*100./(stat.cases or 1)) for i, val in enumerate(attr.values)).decode("utf-8"))
                else:
                    values.append("")
            return values
        else:
            return [""]*len(row)

    def updateMatrix(self):
        if self.attrCombo.count() and self.stats:
            self.table.clearSelection()
            attr = self.data.domain[self.attribute]
            vtype = attr.varType
            cont = self.cont = [name for name, field, flag in self.aggregates if getattr(self, field) and vtype in flag]
            if len(cont)!=1:
                cont.append("")

            verticalValues = self.verticalValues + ["(Total)"]
            realRows = len(verticalValues)*len(cont) + (len(cont)==1)
            self.table.setRowCount(realRows)
            self.table.setColumnCount(len(self.horizontalValues)+3)
            
            totalBrush = QBrush(QColor(240, 240, 240))
            aggBrush = QBrush(QColor(216, 216, 216))
            whiteBrush = QBrush(QColor(255, 255, 255))
            for coli, val in enumerate(["", ""]+self.horizontalValues+["(Total)"]):
                w = QTableWidgetItem(val)
                w.setBackground(aggBrush if coli>1 else whiteBrush)
                w.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.table.setItem(0, coli, w)

            rrowi = 1
            for rowi, row in enumerate(self.stats):
                lastRow = rowi == len(self.stats)-1
                for shwi, shw in enumerate(cont):
                    if lastRow and not shw:
                        break
                    w = QTableWidgetItem("" if shwi else verticalValues[rowi])
                    w.setBackground(whiteBrush if not shw else aggBrush)
                    w.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.table.setItem(rrowi, 0, w)
                    
                    w = QTableWidgetItem(shw)
                    if shw:
                        w.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        w.setBackground(aggBrush)
                    self.table.setItem(rrowi, 1, w)
                    values = self.rowData(shw, row)
                    for coli, val in enumerate(values):
                        w = QTableWidgetItem(val)
                        w.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        if val and (lastRow or coli == len(values)-1):
                            w.setBackground(totalBrush)
                            w.font().setBold(True)
                        self.table.setItem(rrowi, coli+2, w)
                    rrowi += 1
            self.table.resizeColumnsToContents()
            self.table.resizeRowsToContents()
        
    def selectionChanged(self):
        selected = [(x.row(), x.column()) for x in self.table.selectedIndexes()]
        if not selected:
            data = None
        else:
            row, column = selected[0]
            if row and not self.cont[(row-1) % len(self.cont)]:
                data = None
            else:
                dataRow = self.subsets[(row-1)/len(self.cont) if row else -1]
                data = dataRow[column-2 if column>=2 else -1] 
        self.send("Selected Group", data)

    def sendReport(self):
        if self.attrCombo.count() and self.stats:
            self.startReport()
            self.reportSettings("Rotation",
                                [("Rows", self.rowAttribute),
                                 ("Columns", self.columnAttribute),
                                 ("Contents", self.attribute)])
            self.reportSection("Matrix")
            res = '<table style="border: 0; cell-padding: 3; cell-border: 0">\n'
    
            attr = self.data.domain[self.attribute]
            verticalValues = self.verticalValues + ["(Total)"]
            res += "<tr><td/><td/>"+"".join('<td style="background-color: #d8d8d8; font-weight: bold">%s</td>' % x for x in self.horizontalValues+["(Total)"])+"</tr>\n"
            
            rrowi = 1
            for rowi, row in enumerate(self.stats):
                lastRow = rowi == len(self.stats)-1
                rowcol = 'style="'+('background-color: #f0f0f0' if lastRow else '')+'; text-align: right"'
                for shwi, shw in enumerate(self.cont):
                    if lastRow and not shw:
                        break
                    res += "<tr>"
                    if not shw:
                        res += "<tr><td/></tr>\n"
                        continue
                    if not shwi:
                        res += '<td rowspan="%i" valign="top" style="background-color: #d8d8d8; font-weight: bold">%s</td>' % (len(self.cont)-1 or 1, verticalValues[rowi])
                    res += '<td style="background-color: #d8d8d8">%s</td>' % shw
                    values = self.rowData(shw, row)
                    res += ''.join("<td %s>%s</td>" % (rowcol, val) for val in values[:-1])
                    res += '<td style="background-color: #f0f0f0; font-weight: bold; text-align: right">%s</td>' % values[-1]
                    res += '</tr>'
            res += '</table>'
            self.reportRaw(res)
