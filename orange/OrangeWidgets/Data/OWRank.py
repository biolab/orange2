"""
<name>Rank</name>
<description>Ranks and filters attributes by their relevance.</description>
<icon>icons/Rank.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>1102</priority>
"""
import orngOrangeFoldersQt4
from OWWidget import *
import OWGUI

class OWRank(OWWidget):
    settingsList =  ["nDecimals", "reliefK", "reliefN", "nIntervals", "sortBy", "nSelected", "selectMethod", "autoApply"]
    measures          = ["ReliefF", "Information gain", "Gain ratio", "Gini gain"]
    measuresShort     = ["ReliefF", "Inf. gain", "Gain ratio", "Gini"]
    measuresAttrs     = ["computeReliefF", "computeInfoGain", "computeGainRatio", "computeGini"]
    estimators        = [orange.MeasureAttribute_relief, orange.MeasureAttribute_info, orange.MeasureAttribute_gainRatio, orange.MeasureAttribute_gini]
    handlesContinuous = [True, False, False, False]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Rank")

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Reduced Example Table", ExampleTable, Default + Single), ("ExampleTable Attributes", ExampleTable, NonDefault)]

        self.settingsList += self.measuresAttrs

        self.nDecimals = 3
        self.reliefK = 10
        self.reliefN = 20
        self.nIntervals = 4
        self.sortBy = 0
        self.selectMethod = 3
        self.nSelected = 5
        self.autoApply = True
        self.data = None

        for meas in self.measuresAttrs:
            setattr(self, meas, True)

        self.loadSettings()

        labelWidth = 80

        box = OWGUI.widgetBox(self.controlArea, "Measures", addSpace=True)
        for meas, valueName in zip(self.measures, self.measuresAttrs):
            OWGUI.checkBox(box, self, valueName, meas, callback=self.measuresChanged)
            if valueName == "computeReliefF":
                ibox = OWGUI.indentedBox(box)
                OWGUI.spin(ibox, self, "reliefK", 1, 20, label="Neighbours", labelWidth=labelWidth, orientation=0, callback=self.reliefChanged, callbackOnReturn = True)
                OWGUI.spin(ibox, self, "reliefN", 20, 100, label="Examples", labelWidth=labelWidth, orientation=0, callback=self.reliefChanged, callbackOnReturn = True)
        OWGUI.separator(box)

        OWGUI.comboBox(box, self, "sortBy", label = "Sort by"+"  ", items = ["No sorting", "Attribute name", "Number of values"] + self.measures, orientation=0, valueType = int, callback=self.sortingChanged)


        box = OWGUI.widgetBox(self.controlArea, "Discretization", addSpace=True)
        OWGUI.spin(box, self, "nIntervals", 2, 20, label="Intervals", labelWidth=labelWidth, orientation=0, callback=self.discretizationChanged, callbackOnReturn = True)

        box = OWGUI.widgetBox(self.controlArea, "Precision", addSpace=True)
        OWGUI.spin(box, self, "nDecimals", 1, 6, label="No. of decimals", labelWidth=labelWidth, orientation=0, callback=self.reprint)

        OWGUI.rubber(self.controlArea)

        selMethBox = OWGUI.radioButtonsInBox(self.controlArea, self, "selectMethod", ["None", "All", "Manual", "Best ranked"], box="Select attributes", callback=self.selectMethodChanged)
        OWGUI.spin(OWGUI.indentedBox(selMethBox), self, "nSelected", 1, 100, label="No. selected"+"  ", orientation=0, callback=self.nSelectedChanged)

        OWGUI.separator(selMethBox)

        applyButton = OWGUI.button(selMethBox, self, "Commit", callback = self.apply)
        autoApplyCB = OWGUI.checkBox(selMethBox, self, "autoApply", "Commit automatically")
        OWGUI.setStopper(self, applyButton, autoApplyCB, "dataChanged", self.apply)

        self.table = QTableView()
        self.mainArea.layout().addWidget(self.table)
        self.table.verticalHeader().setVisible(1)
        self.table.horizontalHeader().setVisible(1)
                        
        self.table.setModel(RankTableModel(self, self))
        self.table.setItemDelegate(RankTableItemDelgate(self))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.table.verticalHeader().setResizeMode(QHeaderView.ResizeToContents)
        self.topheader = self.table.horizontalHeader()
        
        self.activateLoadedSettings()
        self.resetInternals()

        self.connect(self.table.horizontalHeader(), SIGNAL("sectionClicked(int)"), self.headerClick)
        self.connect(self.table, SIGNAL("clicked (const QModelIndex&)"), self.selectItem)
        self.resize(690,500)


    def activateLoadedSettings(self):
        self.setMeasures()

    def resetInternals(self):
        self.data = None
        self.discretizedData = None
        self.attributeOrder = []
        self.selected = []
        self.measured = {}
        self.usefulAttributes = []
        self.dataChanged = False
        self.lastSentAttrs = None


    def selectMethodChanged(self):
        if self.selectMethod == 0:
            self.selected = []
        elif self.selectMethod == 1:
            self.selected = self.attributeOrder[:]
        elif self.selectMethod == 3:
            self.selected = self.attributeOrder[:self.nSelected]
        self.table.model().reset()
        self.applyIf()


    def nSelectedChanged(self):
        self.selectMethod = 3
        self.selectMethodChanged()


    def setData(self,data):
        self.resetInternals()

        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None
        if self.data:
            self.usefulAttributes = filter(lambda x:x.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous], self.data.domain.attributes)
        
        self.reprint()
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
                if sortedByThis and self.selectMethod == 3:
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
                if sortedByReliefF and self.selectMethod == 3:
                    self.applyIf()


    def setMeasures(self):
        self.selectedMeasures = [i for i, ma in enumerate(self.measuresAttrs) if getattr(self, ma)]
        self.reprint()


    def measuresChanged(self):
        self.setMeasures()
        if self.data:
            self.reprint(False)
            self.resendAttributes()


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
                else:
                    mdict[attr] = None
            except:
                mdict[attr] = None

        self.measured[measAttr] = mdict
        return mdict


    def reprint(self, sort = True):
        if not self.data:
            self.table.model().reset()
            return

        if sort:
            self.resort()

        #for col, meas_idx in enumerate(self.selectedMeasures):
        #    mdict = self.getMeasure(meas_idx)

        self.table.model().reset()
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        

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


    def selectItem(self, index):
        row = index.row()
        attr = self.data.domain[str(self.table.model().data(self.table.model().createIndex(row, 0)).toString())]
        if attr in self.selected:
            self.selected.remove(attr)
        else:
            self.selected.append(attr)
        self.table.model().reset()
        self.applyIf()
        

    def headerClick(self, index):
        if index < 0: return
        
        if index < 2:
            self.sortBy = 1 + index
        else:
            self.sortBy = 3 + self.measuresShort.index(str(self.table.horizontalHeader().model().headerData(index, Qt.Horizontal).toString()))
        self.sortingChanged()
        

    def sortingChanged(self):
        self.table.horizontalHeader().setSortIndicatorShown(self.sortBy > 0)
        self.table.horizontalHeader().setSortIndicator(self.sortBy-1, Qt.AscendingOrder)
        
        self.reprint()
        self.resendAttributes()
        if self.selectMethod == 3:
            self.applyIf()


    def resort(self):
        self.attributeOrder = self.usefulAttributes

        if self.sortBy:
            if self.sortBy == 1:
                st = [(attr, attr.name) for attr in self.attributeOrder]
                st.sort(lambda x,y: cmp(x[1], y[1]))
                
            elif self.sortBy == 2:
                st = [(attr, attr.varType == orange.VarTypes.Continuous and 1e30 or len(attr.values)) for attr in self.attributeOrder]
                st.sort(lambda x,y: cmp(x[1], y[1]))
                #self.topheader.setSortIndicatorShown(False)
            else:
                st = [(m, a == None and -1e20 or a) for m, a in self.getMeasure(self.sortBy-3).items()]
                st.sort(lambda x,y: -cmp(x[1], y[1]) or cmp(x[0], y[0]))

            self.attributeOrder = [attr for attr, meas in st]
#        else:
#            self.topheader.setSortIndicatorShown(False)

        if self.selectMethod == 3:
            self.selected = self.attributeOrder[:self.nSelected]
        
        #self.table.update()
        self.applyIf()
                                    


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
                self.send("Reduced Example Table", orange.ExampleTable(orange.Domain(self.selected, self.data.domain.classVar), self.data))
                self.lastSentAttrs = self.selected[:]

        self.dataChanged = False

class RankTableItemDelgate(QItemDelegate):
    def __init__(self, parent = None):
        QItemDelegate.__init__(self, parent)
        self.widget = parent
  
    def paint(self, painter, option, index):
        col = index.column()
        row = index.row()
        
        attr = self.widget.attributeOrder[row]

        myOption = QStyleOptionViewItem(option)
        s = [n.name for n in self.widget.selected]
        if attr in self.widget.selected:
            myOption.state |= QStyle.State_Selected 
        else:
            myOption.state &= ~QStyle.State_Selected
        myOption.state |= QStyle.State_Active        # show as active although other controls have focus
        QItemDelegate.paint(self, painter, myOption, index)
        
    

class RankTableModel(QAbstractTableModel):
    def __init__(self, widget, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self.widget = widget

    def rowCount(self, parent):
        if self.widget.data:
            return len(self.widget.data.domain.attributes)
        else:
            return 0

    def columnCount(self, parent):
        if self.widget.data:
            return 2 + len(self.widget.selectedMeasures)
        else:
            return 0

    def headerData(self, section, orientation, role = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Vertical:
            return QVariant(str(section+1))
        
        if section == 0:   return QVariant("Attribute")
        elif section == 1: return QVariant("#")
        else:              return QVariant(self.widget.measuresShort[self.widget.selectedMeasures[section-2]])

    def data(self, index, role = Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return QVariant()

        col = index.column()
        row = index.row()
        if col == 0: 
            return QVariant(self.widget.attributeOrder[row].name) 
        elif col == 1:
            attr =  self.widget.attributeOrder[row]
            return QVariant(attr.varType == orange.VarTypes.Continuous and "C" or str(len(attr.values)))
        else: 
            attr =  self.widget.attributeOrder[row]
            prec = " %%.%df" % self.widget.nDecimals
            mdict = self.widget.getMeasure(col-2)
            return QVariant(mdict[attr] != None and prec % mdict[attr] or "NA")

 

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRank()
    #ow.setData(orange.ExampleTable("../../doc/datasets/wine.tab"))
    ow.setData(orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab"))
    ow.show()
    a.exec_()
    ow.saveSettings()

