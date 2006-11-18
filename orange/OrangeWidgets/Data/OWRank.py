"""
<name>Rank</name>
<description>Ranks and filters attributes by their relevance.</description>
<icon>icons/Rank.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact> 
<priority>1102</priority>
"""

from OWWidget import *
from qttable import *
import OWGUI

class OWRank(OWWidget):
    settingsList =  ["nDecimals", "reliefK", "reliefN", "nIntervals", "sortBy", "nSelected", "selectMethod", "autoApply"]
    measures          = ["ReliefF", "Information Gain", "Gain Ratio", "Gini Index"]
    measuresShort     = ["ReliefF", "Inf. gain", "Gain ratio", "Gini"]
    measuresAttrs     = ["computeReliefF", "computeInfoGain", "computeGainRatio", "computeGini"]
    estimators        = [orange.MeasureAttribute_relief, orange.MeasureAttribute_info, orange.MeasureAttribute_gainRatio, orange.MeasureAttribute_gini]
    handlesContinuous = [True, False, False, False]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Rank")
        
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata)]
        self.outputs = [("Reduced Example Table", ExampleTableWithClass, 8), ("ExampleTable Attributes", ExampleTable, 16)]

        self.settingsList += self.measuresAttrs
        
        self.nDecimals = 3
        self.reliefK = 10
        self.reliefN = 20
        self.nIntervals = 4
        self.sortBy = 0
        self.selectMethod = 3
        self.nSelected = 5
        self.autoApply = True

        for meas in self.measuresAttrs:
            setattr(self, meas, True)
            
        self.loadSettings()

        labelWidth = 80        

        box = OWGUI.widgetBox(self.controlArea, "Measures", addSpace=True)
        for meas, valueName in zip(self.measures, self.measuresAttrs):
            OWGUI.checkBox(box, self, valueName, meas, callback=self.measuresChanged)
            if valueName == "computeReliefF":
                ibox = OWGUI.indentedBox(box)
                OWGUI.spin(ibox, self, "reliefK", 1, 20, label="Neighbours", labelWidth=labelWidth, orientation=0, callback=self.reliefChanged)
                OWGUI.spin(ibox, self, "reliefN", 20, 100, label="Examples", labelWidth=labelWidth, orientation=0, callback=self.reliefChanged)
        OWGUI.separator(box)

        OWGUI.comboBox(box, self, "sortBy", label = "Sort by"+"  ", items = ["No Sorting", "Attribute Name", "Number of Values"] + self.measures, orientation=0, valueType = int, callback=self.sortingChanged)


        box = OWGUI.widgetBox(self.controlArea, "Discretization", addSpace=True)
        OWGUI.spin(box, self, "nIntervals", 2, 20, label="Intervals", labelWidth=labelWidth, orientation=0, callback=self.discretizationChanged)
        
        box = OWGUI.widgetBox(self.controlArea, "Precision", addSpace=True)
        OWGUI.spin(box, self, "nDecimals", 1, 6, label="No. of decimals", labelWidth=labelWidth, orientation=0, callback=self.decimalsChanged)

        OWGUI.rubber(self.controlArea)        

        selMethBox = OWGUI.radioButtonsInBox(self.controlArea, self, "selectMethod", ["None", "All", "Manual", "Best ranked"], box="Select Attributes", callback=self.selectMethodChanged)
        OWGUI.spin(OWGUI.indentedBox(selMethBox), self, "nSelected", 1, 100, label="No. selected"+"  ", orientation=0, callback=self.nSelectedChanged)
        
        OWGUI.separator(selMethBox)

        applyButton = OWGUI.button(selMethBox, self, "Commit", callback = self.apply)
        autoApplyCB = OWGUI.checkBox(selMethBox, self, "autoApply", "Commit automatically")
        OWGUI.setStopper(self, applyButton, autoApplyCB, "dataChanged", self.apply)
        
        self.layout=QVBoxLayout(self.mainArea)
        box = OWGUI.widgetBox(self.mainArea, orientation=0)
        self.table = QTable(0, 6, self.mainArea)
        self.table.setSelectionMode(QTable.Multi)
        self.layout.add(self.table)
        self.topheader = self.table.horizontalHeader()
        self.topheader.setLabel(0, "Attribute")
        self.topheader.setLabel(1, "#")
        self.table.setColumnWidth(1, 40)

        self.activateLoadedSettings()
        self.resetInternals()

        self.connect(self.topheader, SIGNAL("clicked(int)"), self.sortByColumn)
        self.connect(self.table, SIGNAL("clicked(int,int,int,const QPoint &)"), self.selectRow)
        self.connect(self.table.verticalHeader(), SIGNAL("clicked(int)"), self.selectRow)
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
        self.adjustCol0 = True
        self.lastSentAttrs = None
        
        self.table.setNumRows(0)
        self.table.adjustSize()

        
    def selectMethodChanged(self):
        if self.selectMethod == 0:
            self.selected = []
            self.reselect()
        elif self.selectMethod == 1:
            self.selected = self.attributeOrder[:]
            self.reselect()
        elif self.selectMethod == 3:
            self.selected = self.attributeOrder[:self.nSelected]
            self.reselect()
        self.applyIf()


    def nSelectedChanged(self):
        self.selectMethod = 3
        self.selectMethodChanged()


    def sendSelected(self):
        attrs = self.data and [attr for i, attr in enumerate(self.attributeOrder) if self.table.isRowSelected(i)]
        if not attrs:
            self.send("ExampleTable Attributes", None)
            return

        self.send("ExampleTable Attributes", orange.ExampleTable(orange.Domain(attrs, self.data.domain.classVar), self.data))

        
    def cdata(self,data):
        self.resetInternals()
        
        self.data = data
        if self.data:
            self.adjustCol0 = True
            self.usefulAttributes = filter(lambda x:x.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous], self.data.domain.attributes)
            self.table.setNumRows(len(self.data.domain.attributes))
            self.reprint()
            self.table.adjustSize()
            
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


    def sortingChanged(self):
        self.reprint()
        self.resendAttributes()
        if self.selectMethod == 3:
            self.applyIf()


    def setMeasures(self):
        self.selectedMeasures = [i for i, ma in enumerate(self.measuresAttrs) if getattr(self, ma)]
        self.table.setNumCols(2 + len(self.selectedMeasures))
        for col, meas_idx in enumerate(self.selectedMeasures):
            self.topheader.setLabel(col+2, self.measuresShort[meas_idx])
            self.table.setColumnWidth(col+2, 80)

        
    def measuresChanged(self):
        self.setMeasures()
        if self.data:
            self.reprint(True)
            self.resendAttributes()
#            self.repaint()
        self.table.adjustSize()


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
                    at = [discretizer(attri, self.data) for attri in contAttrs]
                    self.discretizedData = self.data.select(orange.Domain(at, self.data.domain.classVar))
                    self.discretizedData.setattr("attrDict", dict(zip(contAttrs, at)))

                act_attr, act_data = self.discretizedData.attrDict[attr], self.discretizedData

            try:
                mdict[attr] = estimator(act_attr, act_data)
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
            self.table.setText(row, 0, attr.name)
            self.table.setText(row, 1, attr.varType==orange.VarTypes.Continuous and "C" or str(len(attr.values)))

        if self.adjustCol0:
            self.table.adjustColumn(0)
            if self.table.columnWidth(0) < 80:
                self.table.setColumnWidth(0, 80)
            self.adjustCol0 = False

        for col, meas_idx in enumerate(self.selectedMeasures):
            mdict = self.getMeasure(meas_idx)
            for row, attr in enumerate(self.attributeOrder):
                self.table.setText(row, col+2, mdict[attr] != None and prec % mdict[attr] or "NA")
                
        self.reselect()

        if self.sortBy < 3:
            self.topheader.setSortIndicator(self.sortBy-1, False)
        elif self.sortBy-3 in self.selectedMeasures:
            self.topheader.setSortIndicator(2 + self.selectedMeasures.index(self.sortBy-3))
        else:
            self.topheader.setSortIndicator(-1)


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
                              

    def selectRow(self, i, *foo):
        if i < 0:
            return
        
        attr = self.attributeOrder[i]
        if attr in self.selected:
            self.selected.remove(attr)
        else:
            self.selected.append(attr)
        self.reselect()
        self.applyIf()


    def reselect(self):
        self.table.clearSelection()

        if not self.data:
            return

        for attr in self.selected:
            sel = QTableSelection()
            i = self.attributeOrder.index(attr)
            sel.init(i, 0)
            sel.expandTo(i, self.table.numCols()-1)
            self.table.addSelection(sel)

        if self.selectMethod == 2 or self.selectMethod == 3 and self.selected == self.attributeOrder[:self.nSelected]:
            pass
        elif self.selected == self.attributeOrder:
            self.selectMethod = 1
        else:
            self.selectMethod = 2


    def resort(self):
        self.attributeOrder = self.usefulAttributes

        if self.sortBy:
            if self.sortBy == 1:
                st = [(attr, attr.name) for attr in self.attributeOrder]
                st.sort(lambda x,y: cmp(x[1], y[1]))
            elif self.sortBy == 2:
                st = [(attr, attr.varType == orange.VarTypes.Continuous and 1e30 or len(attr.values)) for attr in self.attributeOrder]
                st.sort(lambda x,y: cmp(x[1], y[1]))
                self.topheader.setSortIndicator(1, False)
            else:
                st = [(m, a == None and -1e20 or a) for m, a in self.getMeasure(self.sortBy-3).items()]
                st.sort(lambda x,y: -cmp(x[1], y[1]) or cmp(x[0], y[0]))

            self.attributeOrder = [attr for attr, meas in st]

        if self.selectMethod == 3:
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
                self.send("Reduced Example Table", orange.ExampleTable(orange.Domain(self.selected, self.data.domain.classVar), self.data))
                self.lastSentAttrs = self.selected[:]
            
        self.dataChanged = False
    

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRank()
    a.setMainWidget(ow)
    ow.cdata(orange.ExampleTable("../../doc/datasets/iris.tab"))
    ow.show()
    a.exec_loop()
    
    ow.saveSettings()

