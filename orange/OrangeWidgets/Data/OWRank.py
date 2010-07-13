"""
<name>Rank</name>
<description>Ranks and filters attributes by their relevance.</description>
<icon>icons/Rank.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>1102</priority>
"""
from OWWidget import *

import OWGUI

class OWRank(OWWidget):
    settingsList =  ["nDecimals", "reliefK", "reliefN", "nIntervals", "sortBy", "nSelected", "selectMethod", "autoApply", "showDistributions", "distColorRgb"]
    measures          = ["ReliefF", "Information Gain", "Gain Ratio", "Gini Gain", "Log Odds Ratio"]
    measuresShort     = ["ReliefF", "Inf. gain", "Gain ratio", "Gini", "log OR"]
    measuresAttrs     = ["computeReliefF", "computeInfoGain", "computeGainRatio", "computeGini", "computeLogOdds"]
    estimators        = [orange.MeasureAttribute_relief, orange.MeasureAttribute_info, orange.MeasureAttribute_gainRatio, orange.MeasureAttribute_gini, orange.MeasureAttribute_logOddsRatio]
    handlesContinuous = [True, False, False, False, False]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Rank")

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Reduced Example Table", ExampleTable, Default + Single), ("ExampleTable Attributes", ExampleTable, NonDefault)]

        self.settingsList += self.measuresAttrs
        self.logORIdx = self.measuresShort.index("log OR")

        self.nDecimals = 3
        self.reliefK = 10
        self.reliefN = 20
        self.nIntervals = 4
        self.sortBy = 0
        self.selectMethod = 2
        self.nSelected = 5
        self.autoApply = True
        self.showDistributions = 1
        self.distColorRgb = (220,220,220, 255)
        self.distColor = QColor(*self.distColorRgb)
        self.minmax = {}
        
        self.data = None

        for meas in self.measuresAttrs:
            setattr(self, meas, True)

        self.loadSettings()

        labelWidth = 80

        box = OWGUI.widgetBox(self.controlArea, "Scoring", addSpace=True)
        for meas, valueName in zip(self.measures, self.measuresAttrs):
            if valueName == "computeReliefF":
                hbox = OWGUI.widgetBox(box, orientation = "horizontal")
                OWGUI.checkBox(hbox, self, valueName, meas, callback=self.measuresChanged)
                hbox.layout().addSpacing(5)
                smallWidget = OWGUI.SmallWidgetLabel(hbox, pixmap = 1, box = "ReliefF Parameters", tooltip = "Show ReliefF parameters")
                OWGUI.spin(smallWidget.widget, self, "reliefK", 1, 20, label="Neighbours", labelWidth=labelWidth, orientation=0, callback=self.reliefChanged, callbackOnReturn = True)
                OWGUI.spin(smallWidget.widget, self, "reliefN", 20, 100, label="Examples", labelWidth=labelWidth, orientation=0, callback=self.reliefChanged, callbackOnReturn = True)
                OWGUI.button(smallWidget.widget, self, "Load defaults", callback = self.loadReliefDefaults)
                OWGUI.rubber(hbox)
            else:
                OWGUI.checkBox(box, self, valueName, meas, callback=self.measuresChanged)
        OWGUI.separator(box)

        OWGUI.comboBox(box, self, "sortBy", label = "Sort by"+"  ", items = ["No Sorting", "Attribute Name", "Number of Values"] + self.measures, orientation=0, valueType = int, callback=self.sortingChanged)


        box = OWGUI.widgetBox(self.controlArea, "Discretization", addSpace=True)
        OWGUI.spin(box, self, "nIntervals", 2, 20, label="Intervals: ", orientation=0, callback=self.discretizationChanged, callbackOnReturn = True)

        box = OWGUI.widgetBox(self.controlArea, "Precision", addSpace=True)
        OWGUI.spin(box, self, "nDecimals", 1, 6, label="No. of decimals: ", orientation=0, callback=self.decimalsChanged)

        box = OWGUI.widgetBox(self.controlArea, "Score bars", orientation="horizontal", addSpace=True)
        self.cbShowDistributions = OWGUI.checkBox(box, self, "showDistributions", 'Enable', callback = self.cbShowDistributions)
#        colBox = OWGUI.indentedBox(box, orientation = "horizontal")
        OWGUI.rubber(box)
        box = OWGUI.widgetBox(box, orientation="horizontal")
        wl = OWGUI.widgetLabel(box, "Color: ")
        OWGUI.separator(box)
        self.colButton = OWGUI.toolButton(box, self, self.changeColor, width=20, height=20, debuggingEnabled = 0)
        self.cbShowDistributions.disables.extend([wl, self.colButton])
        self.cbShowDistributions.makeConsistent()
#        OWGUI.rubber(box)

        
        selMethBox = OWGUI.widgetBox(self.controlArea, "Select attributes", addSpace=True)
        self.clearButton = OWGUI.button(selMethBox, self, "Clear", callback=self.clearSelection)
        self.clearButton.setDisabled(True)
        
        buttonGrid = QGridLayout()
        selMethRadio = OWGUI.radioButtonsInBox(selMethBox, self, "selectMethod", [], callback=self.selectMethodChanged)
        b1 = OWGUI.appendRadioButton(selMethRadio, self, "selectMethod", "All", insertInto=selMethRadio, callback=self.selectMethodChanged, addToLayout=False)
        b2 = OWGUI.appendRadioButton(selMethRadio, self, "selectMethod", "Manual", insertInto=selMethRadio, callback=self.selectMethodChanged, addToLayout=False)
        b3 = OWGUI.appendRadioButton(selMethRadio, self, "selectMethod", "Best ranked", insertInto=selMethRadio, callback=self.selectMethodChanged, addToLayout=False)
#        brBox = OWGUI.widgetBox(selMethBox, orientation="horizontal", margin=0)
#        OWGUI.appendRadioButton(selMethRadio, self, "selectMethod", "Best ranked", insertInto=brBox, callback=self.selectMethodChanged)
        spin = OWGUI.spin(OWGUI.widgetBox(selMethRadio, addToLayout=False), self, "nSelected", 1, 100, orientation=0, callback=self.nSelectedChanged)
        buttonGrid.addWidget(b1, 0, 0)
        buttonGrid.addWidget(b2, 1, 0)
        buttonGrid.addWidget(b3, 2, 0)
        buttonGrid.addWidget(spin, 2, 1)
        selMethRadio.layout().addLayout(buttonGrid)
        OWGUI.separator(selMethBox)

        applyButton = OWGUI.button(selMethBox, self, "Commit", callback = self.apply)
        autoApplyCB = OWGUI.checkBox(selMethBox, self, "autoApply", "Commit automatically")
        OWGUI.setStopper(self, applyButton, autoApplyCB, "dataChanged", self.apply)

        OWGUI.rubber(self.controlArea)
        
        self.table = QTableWidget()
        self.mainArea.layout().addWidget(self.table)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.table.verticalHeader().setResizeMode(QHeaderView.ResizeToContents)
        self.table.setItemDelegate(RankItemDelegate(self, self.table))

        self.topheader = self.table.horizontalHeader()
        self.topheader.setSortIndicatorShown(1)
        self.topheader.setHighlightSections(0)

        self.setMeasures()
        self.resetInternals()

        self.connect(self.table.horizontalHeader(), SIGNAL("sectionClicked(int)"), self.headerClick)
        self.connect(self.table, SIGNAL("clicked (const QModelIndex&)"), self.selectItem)
        self.connect(self.table, SIGNAL("itemSelectionChanged()"), self.onSelectionChanged)
        
        self.resize(690,500)
        self.updateColor()

    def cbShowDistributions(self):
        self.table.reset()

    def changeColor(self):
        color = QColorDialog.getColor(self.distColor, self)
        if color.isValid():
            self.distColorRgb = color.getRgb()
            self.updateColor()

    def updateColor(self):
        self.distColor = QColor(*self.distColorRgb)
        w = self.colButton.width()-8
        h = self.colButton.height()-8
        pixmap = QPixmap(w, h)
        painter = QPainter()
        painter.begin(pixmap)
        painter.fillRect(0,0,w,h, QBrush(self.distColor))
        painter.end()
        self.colButton.setIcon(QIcon(pixmap))
        self.table.viewport().update()


    def resetInternals(self):
        self.data = None
        self.discretizedData = None
        self.attributeOrder = []
        self.selected = []
        self.measured = {}
        self.usefulAttributes = []
        self.dataChanged = False
        self.lastSentAttrs = None

        self.table.setRowCount(0)

    def onSelectionChanged(self):
        if not getattr(self, "_reselecting", False):
            selected = sorted(set(item.row() for item in self.table.selectedItems()))
            self.clearButton.setEnabled(bool(selected))
            selected = [self.attributeOrder[row] for row in selected]
            if set(selected) != set(self.selected):
                self.selected = selected
                self.selectMethod = 1
            self.applyIf()

    def clearSelection(self):
        self.selected = [] 
        self.reselect()

    def selectMethodChanged(self):
        if self.selectMethod == 0:
            self.selected = self.attributeOrder[:]
            self.reselect()
        elif self.selectMethod == 2:
            self.selected = self.attributeOrder[:self.nSelected]
            self.reselect()
        self.applyIf()

    def nSelectedChanged(self):
        self.selectMethod = 2
        self.selectMethodChanged()

    def sendSelected(self):
        attrs = self.data and [attr for i, attr in enumerate(self.attributeOrder) if self.table.isRowSelected(i)]
        if not attrs:
            self.send("ExampleTable Attributes", None)
            return

        nDomain = orange.Domain(attrs, self.data.domain.classVar)
        for meta in [a.name for a in self.data.domain.getmetas().values()]:
            nDomain.addmeta(orange.newmetaid(), self.data.domain[meta])

        self.send("ExampleTable Attributes", orange.ExampleTable(nDomain, self.data))


    def setData(self,data):
        self.resetInternals()

        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None
        if self.data:
            self.usefulAttributes = filter(lambda x:x.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous], self.data.domain.attributes)
            self.table.setRowCount(len(self.data.domain.attributes))
            self.reprint()

        self.setLogORTitle()
        self.resendAttributes()
        self.applyIf()


    def discretizationChanged(self):
        self.discretizedData = None

        removed = False
        for meas, cont in zip(self.measuresAttrs, self.handlesContinuous):
            if not cont and self.measured.has_key(meas):
                del self.measured[meas]
                removed = True

        if self.data and self.data.domain.hasContinuousAttributes(False):
            sortedByThis = self.sortBy>=3 and not self.handlesContinuous[self.sortBy-3]
            if removed or sortedByThis:
                self.reprint()
                self.resendAttributes()
                if sortedByThis and self.selectMethod == 2:
                    self.applyIf()


    def reliefChanged(self):
        removed = False
        if self.measured.has_key("computeReliefF"):
            del self.measured["computeReliefF"]
            removed = True

        if self.data:
            sortedByReliefF = self.sortBy-3 == self.measuresAttrs.index("computeReliefF")
            if removed or sortedByReliefF:
                self.reprint()
                self.resendAttributes()
                if sortedByReliefF and self.selectMethod == 2:
                    self.applyIf()

    def loadReliefDefaults(self):
        self.reliefK = 5
        self.reliefN = 20
        self.reliefChanged()


    def selectItem(self, index):
        pass
#        row = index.row()
#        attr = self.attributeOrder[row]
#        if attr in self.selected:
#            self.selected.remove(attr)
#        else:
#            self.selected.append(attr)
#        self.selectMethod = 1
#        self.applyIf()

    def headerClick(self, index):
        if index < 0: return

        if index < 2:
            self.sortBy = 1 + index
        else:
            self.sortBy = 3 + self.measuresShort.index(str(self.table.horizontalHeader().model().headerData(index, Qt.Horizontal).toString()))
        self.sortingChanged()

    def sortingChanged(self):
        self.reprint()
        self.resendAttributes()
        if self.selectMethod == 2:
            self.applyIf()


    def setLogORTitle(self):
        selectedMeasures = list(self.selectedMeasures)
        if self.logORIdx in selectedMeasures:
            loi = selectedMeasures.index(self.logORIdx)
            if  self.data and self.data.domain.classVar \
                and self.data.domain.classVar.varType == orange.VarTypes.Discrete \
                and len(self.data.domain.classVar.values) == 2:
                    title = "log OR (for '%s')" % (self.data.domain.classVar.values[1][:10])
            else:
                title = "log OR"
                
            self.table.setHorizontalHeaderItem(2+loi, QTableWidgetItem(title))
            self.table.resizeColumnToContents(2+loi)
        

    def setMeasures(self):
        self.selectedMeasures = [i for i, ma in enumerate(self.measuresAttrs) if getattr(self, ma)]
        self.table.setColumnCount(2 + len(self.selectedMeasures))
        for col, meas_idx in enumerate(self.selectedMeasures):
            #self.topheader.setLabel(col+2, self.measuresShort[meas_idx])
            self.table.setColumnWidth(col+2, 80)
        self.table.setHorizontalHeaderLabels(["Attribute", "#"] + [self.measuresShort[idx] for idx in self.selectedMeasures])
        self.setLogORTitle()


    def measuresChanged(self):
        self.setMeasures()
        if self.data:
            self.reprint(True)
            self.resendAttributes()


    def sortByColumn(self, col):
        if col < 2:
            self.sortBy = 1 + col
        else:
            self.sortBy = 3 + self.selectedMeasures[col-2]
        self.sortingChanged()


    def decimalsChanged(self):
        self.reprint(True)


    def getMeasure(self, meas_idx):
        measAttr = self.measuresAttrs[meas_idx]
        mdict = self.measured.get(measAttr, False)
        if mdict:
            return mdict

        estimator = self.estimators[meas_idx]()
        if measAttr == "computeReliefF":
            estimator.k, estimator.m = self.reliefK, self.reliefN

        handlesContinuous = self.handlesContinuous[meas_idx]
        mdict = {}
        for attr in self.data.domain.attributes:
            if handlesContinuous or attr.varType == orange.VarTypes.Discrete:
                act_attr, act_data = attr, self.data
            else:
                if not self.discretizedData:
                    discretizer = orange.EquiNDiscretization(numberOfIntervals=self.nIntervals)
                    contAttrs = filter(lambda attr: attr.varType == orange.VarTypes.Continuous, self.data.domain.attributes)
                    at = []
                    attrDict = {}
                    for attri in contAttrs:
                        try:
                            nattr = discretizer(attri, self.data)
                            at.append(nattr)
                            attrDict[attri] = nattr
                        except:
                            pass
                    self.discretizedData = self.data.select(orange.Domain(at, self.data.domain.classVar))
                    self.discretizedData.setattr("attrDict", attrDict)

                act_attr, act_data = self.discretizedData.attrDict.get(attr, None), self.discretizedData

            try:
                if act_attr:
                    mdict[attr] = act_attr and estimator(act_attr, act_data)
                    if measAttr == "computeLogOdds":
                        if mdict[attr] == -999999:
                            act_attr = u"-\u221E"
                        elif mdict[attr] == 999999:
                            act_attr = u"\u221E"
                        mdict[attr] = ("%%.%df" % self.nDecimals + " (%s)") % (mdict[attr], act_attr.values[1])
                else:
                    mdict[attr] = None
            except:
                mdict[attr] = None

        self.measured[measAttr] = mdict
        return mdict


    def reprint(self, noSort = False):
        if not self.data:
            return

        prec = " %%.%df" % self.nDecimals

        if not noSort:
            self.resort()

        for row, attr in enumerate(self.attributeOrder):
            OWGUI.tableItem(self.table, row, 0, attr.name)
            OWGUI.tableItem(self.table, row, 1, attr.varType==orange.VarTypes.Continuous and "C" or str(len(attr.values)))

        self.minmax = {}

        for col, meas_idx in enumerate(self.selectedMeasures):
            mdict = self.getMeasure(meas_idx)
            values = filter(lambda val: val != None, mdict.values())
            if values != []:
                self.minmax[col] = (min(values), max(values))
            else:
                self.minmax[col] = (0,1)
            for row, attr in enumerate(self.attributeOrder):
                if mdict[attr] is None:
                    mattr = "NA"
                elif isinstance(mdict[attr], (int, float)):
                    mattr = prec % mdict[attr]
                else:
                    mattr = mdict[attr]
                OWGUI.tableItem(self.table, row, col+2, mattr)

        self.reselect()

        if self.sortBy < 3:
            self.topheader.setSortIndicator(self.sortBy-1, Qt.DescendingOrder)
        elif self.sortBy-3 in self.selectedMeasures:
            self.topheader.setSortIndicator(2 + self.selectedMeasures.index(self.sortBy-3), Qt.DescendingOrder)
        else:
            self.topheader.setSortIndicator(-1, Qt.DescendingOrder)

        #self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.setColumnWidth(0, 100)
        self.table.setColumnWidth(1, 20)
        for col in range(len(self.selectedMeasures)):
            self.table.setColumnWidth(col+2, 80)


    def sendReport(self):
        self.reportData(self.data)
        self.reportRaw(OWReport.reportTable(self.table))


    def resendAttributes(self):
        if not self.data:
            self.send("ExampleTable Attributes", None)
            return

        attrDomain = orange.Domain(  [orange.StringVariable("attributes"), orange.EnumVariable("D/C", values = "DC"), orange.FloatVariable("#")]
                                   + [orange.FloatVariable(self.measuresShort[meas_idx]) for meas_idx in self.selectedMeasures],
                                     None)
        attrData = orange.ExampleTable(attrDomain)
        measDicts = [self.measured[self.measuresAttrs[meas_idx]] for meas_idx in self.selectedMeasures]
        for attr in self.attributeOrder:
            cont = attr.varType == orange.VarTypes.Continuous
            attrData.append([attr.name, cont, cont and "?" or len(attr.values)] + [meas[attr] or "?" for meas in measDicts])

        self.send("ExampleTable Attributes", attrData)


    def reselect(self):
        self._reselecting = True
        try:
            self.table.clearSelection()
    
            if not self.data:
                return
    
            for attr in self.selected:
                self.table.selectRow(self.attributeOrder.index(attr))
    
            if self.selectMethod == 1 or self.selectMethod == 2 and self.selected == self.attributeOrder[:self.nSelected]:
                pass
            elif self.selected == self.attributeOrder:
                self.selectMethod = 0
            else:
                self.selectMethod = 1
        finally:
            self._reselecting = False
            self.onSelectionChanged()

    def resort(self):
        self.attributeOrder = self.usefulAttributes

        if self.sortBy:
            if self.sortBy == 1:
                st = [(attr, attr.name) for attr in self.attributeOrder]
                st.sort(lambda x,y: cmp(x[1], y[1]))
            elif self.sortBy == 2:
                st = [(attr, attr.varType == orange.VarTypes.Continuous and 1e30 or len(attr.values)) for attr in self.attributeOrder]
                st.sort(lambda x,y: cmp(x[1], y[1]))
                self.topheader.setSortIndicator(1, Qt.DescendingOrder)
            else:
                st = [(m, a == None and -1e20 or a) for m, a in self.getMeasure(self.sortBy-3).items()]
                st.sort(lambda x,y: -cmp(x[1], y[1]) or cmp(x[0], y[0]))

            self.attributeOrder = [attr for attr, meas in st]

        if self.selectMethod == 2:
            self.selected = self.attributeOrder[:self.nSelected]


    def applyIf(self):
        if self.autoApply:
            self.apply()
        else:
            self.dataChanged = True

    def apply(self):
        if not self.data or not self.selected:
            self.send("Reduced Example Table", None)
            self.lastSentAttrs = []
        else:
            if self.lastSentAttrs != self.selected:
                nDomain = orange.Domain(self.selected, self.data.domain.classVar)
                for meta in [a.name for a in self.data.domain.getmetas().values()]:
                    nDomain.addmeta(orange.newmetaid(), self.data.domain[meta])

                self.send("Reduced Example Table", orange.ExampleTable(nDomain, self.data))
                self.lastSentAttrs = self.selected[:]

        self.dataChanged = False



class RankItemDelegate(QItemDelegate):
    def __init__(self, widget = None, table = None):
        QItemDelegate.__init__(self, widget)
        self.table = table
        self.widget = widget

    def paint(self, painter, option, index):
        if not self.widget.showDistributions:
            QItemDelegate.paint(self, painter, option, index)
            return

        col = index.column()
        row = index.row()

        if col < 2 or not self.widget.minmax.has_key(col-2):        # we don't paint first two columns
            QItemDelegate.paint(self, painter, option, index)
            return

        min, max = self.widget.minmax[col-2]

        painter.save()
        self.drawBackground(painter, option, index)
        value, ok = index.data(Qt.DisplayRole).toDouble()

        if ok:        # in case we get "?" it is not ok
            smallerWidth = option.rect.width() * (max - value) / (max-min or 1)
            painter.fillRect(option.rect.adjusted(0,0,-smallerWidth,0), self.widget.distColor)

        self.drawDisplay(painter, option, option.rect, index.data(Qt.DisplayRole).toString())
        painter.restore()


if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRank()
    #ow.setData(orange.ExampleTable("../../doc/datasets/wine.tab"))
    ow.setData(orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab"))
    ow.show()
    a.exec_()
    ow.saveSettings()

