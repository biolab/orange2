"""
<name>Confusion Matrix</name>
<description>Shows a confusion matrix.</description>
<contact>Janez Demsar</contact>
<icon>ConfusionMatrx.png</icon>
<priority>1001</priority>
"""

from OWWidget import *
from qt import *
from qttable import *
import OWGUI
import orngStat, orngTest
import statc, math

class ConfusionTable(QTable):
    def paintEmptyArea(self, p, cx, cy, cw, ch):
        pass#p.fillRect(cx, cy, cw, ch, QBrush(QColor(255, 0, 0)))

class ConfusionTableItem(QTableItem):
    def __init__(self, isBold, *args):
        QTableItem.__init__(self, *args)
        self.isBold = isBold
        
    def alignment(self):
        return QWidget.AlignRight

    def paint(self, painter, cg, cr, selected):
        painter.font().setBold(self.isBold)
        QTableItem.paint(self, painter, cg, cr, selected)
    
class OWConfusionMatrix(OWWidget):
    settings = ["shownQuantity", "autoApply"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Confusion Matrix", 1)

        # inputs
        self.inputs=[("Evaluation Results", orngTest.ExperimentResults, self.test_results, Default)]
        self.outputs=[("Selected Examples", ExampleTableWithClass, 8)]

        self.selectedLearner = [0]
        self.learnerNames = []
        self.selectionDirty = 0
        self.autoApply = True
        self.shownQuantity = 0

        self.learnerList = OWGUI.listBox(self.controlArea, self, "selectedLearner", "learnerNames", box = "Learners", callback = self.learnerChanged)
        self.learnerList.setMinimumHeight(300)
        OWGUI.separator(self.controlArea)


        OWGUI.comboBox(self.controlArea, self, "shownQuantity", items = ["Number of examples", "Observed and expected examples", "Proportions of predicted", "Proportions of true"], box = "Show", callback=self.reprint)

        box = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
#        OWGUI.checkBox(box, self, "colorCells", "Color 

        box = OWGUI.widgetBox(self.controlArea, "Selection", addSpace=True)
        OWGUI.button(box, self, "Correct", callback=self.selectCorrect)
        OWGUI.button(box, self, "Misclassified", callback=self.selectWrong)
        OWGUI.button(box, self, "None", callback=self.selectNone)

        box = OWGUI.widgetBox(self.controlArea, "Commit")
        applyButton = OWGUI.button(box, self, "Commit", callback = self.sendData)
        autoApplyCB = OWGUI.checkBox(box, self, "autoApply", "Commit automatically")
        OWGUI.setStopper(self, applyButton, autoApplyCB, "dataChanged", self.sendData)

        self.layout=QGridLayout(self.mainArea, 4, 3)
        self.layout.setAutoAdd(False)
        labpred = OWGUI.widgetLabel(self.mainArea, "Prediction")
        labpred.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        self.layout.addWidget(labpred, 0, 1, QWidget.AlignLeft)
        self.layout.addWidget(OWGUI.separator(self.mainArea),1, 0)
        
        labpred = OWGUI.widgetLabel(self.mainArea, "True Class  ")
    #    labpred.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        self.layout.addWidget(labpred, 2, 0, QWidget.AlignTop)
        self.layout.addMultiCellWidget(OWGUI.rubber(self.mainArea), 3, 3, 0, 2)
      #  self.layout.addWidget(OWGUI.rubber(self.mainArea), 0, 2)

        self.table = QTable(0, 0, self.mainArea)
        self.table.setSelectionMode(QTable.Multi)
        self.layout.addWidget(self.table, 2, 1)
        self.topHeader = self.table.horizontalHeader()
        self.leftHeader = self.table.verticalHeader()
        self.table.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        
        self.connect(self.table, SIGNAL("selectionChanged()"), self.sendIf)
        
        self.resize(750,450)

    def resizeEvent(self, *args):
        if hasattr(self, "table"):
            self.table.adjustSize()
        OWWidget.resizeEvent(self, *args)
        
    def test_results(self, res):
        self.res = res
        if not res:
            self.table.setNumRows(0)
            self.table.setNumCols(0)
            return
        
        self.matrix = orngStat.confusionMatrices(res)

        dim = len(res.classValues)
        
        self.table.setNumRows(dim+1)
        self.table.setNumCols(dim+1)
        for i, v in enumerate(res.classValues):
            self.topHeader.setLabel(i, v)
            self.leftHeader.setLabel(i, v)
        self.topHeader.setLabel(dim, "")
        self.leftHeader.setLabel(dim, "")

        for ri in range(dim+1):
            for ci in range(dim+1):
                self.table.setItem(ri, ci, ConfusionTableItem(ri==dim or ci==dim, self.table, QTableItem.Never, ""))

        self.learnerNames = res.classifierNames[:]

        # This also triggers a callback (learnerChanged)
        self.selectedLearner = [self.selectedLearner[0] < res.numberOfLearners and self.selectedLearner[0]]

        self.table.clearSelection()
        # if the above doesn't call sendIf, you should call it here

    def learnerChanged(self):
        self.reprint()
        self.sendIf()        
            

    def reprint(self):
        cm = self.matrix[self.selectedLearner[0]]

        dim = len(cm)
        rowSums = [sum(r) for r in cm]
        colSums = [sum([r[i] for r in cm]) for i in range(dim)]
        total = sum(rowSums)
        rowPriors = [r/total for r in rowSums]
        colPriors = [r/total for r in colSums]

        for ri, r in enumerate(cm):
            for ci, c in enumerate(r):
                item = self.table.item(ri, ci)
                if self.shownQuantity == 0:
                    self.table.setText(ri, ci, " %5.3f " % c)
                elif self.shownQuantity == 1:
                    item.setText(" %5.3f / %5.3f " % (c, total*rowPriors[ri]*colPriors[ci]))
                elif self.shownQuantity == 2:
                    if colSums[ci] > 1e-5:
                        item.setText(" %2.1f %%  " % (100 * c / colSums[ci]))
                    else:
                        item.setText(" N/A ")
                elif self.shownQuantity == 3:
                    if rowSums[ri] > 1e-5:
                        item.setText(" %2.1f %%  " % (100 * c / rowSums[ri]))
                    else:
                        item.setText(" N/A ")
                self.table.updateCell(ri, ci)

        for ci in range(len(cm)):
            self.table.setText(dim, ci, " %5.3f " % colSums[ci])
            self.table.setText(ci, dim, "   %5.3f " % rowSums[ci])
            self.table.adjustColumn(ci)
        self.table.setText(dim, dim, "   %5.3f " % total)
        self.table.adjustColumn(dim)

        self.table.adjustSize()

            

    def selectCorrect(self):
        if not self.res:
            return
        
        self.table.clearSelection()
        for i in range(len(self.matrix[0])):
            ts = QTableSelection()
            ts.init(i, i)
            ts.expandTo(i, i)
            self.table.addSelection(ts)
        self.sendIf()

    def selectWrong(self):
        if not self.res:
            return
        
        self.table.clearSelection()
        dim = len(self.matrix[0])
        for i in range(dim):
            if i:
                ts = QTableSelection()
                ts.init(i, 0)
                ts.expandTo(i, i-1)
                self.table.addSelection(ts)
            if i+1 <= dim-1:
                ts = QTableSelection()
                ts.init(i, i+1)
                ts.expandTo(i, dim-1)
                self.table.addSelection(ts)
        self.sendIf()


    def selectNone(self):
        self.table.clearSelection()
        # clearSelection for some reason calls the callback, while add doesn't


    def sendIf(self):
        if self.autoApply:
            self.sendData()
        else:
            self.selectionDirty = True


    def sendData(self):
        self.selectionDirty = False
        
        res = self.res
        if not res or not self.table.numSelections():
            self.send("Selected Examples", None)
            return
        
        from sets import Set
        selected = Set()
        for seli in range(self.table.numSelections()):
            sel = self.table.selection(seli)
            for ri in range(sel.topRow(), sel.bottomRow()+1):
                for ci in range(sel.leftCol(), sel.rightCol()+1):
                    selected.add((ri, ci))

        learnerI = self.selectedLearner[0]
        data = res.examples.getitemsref([i for i, rese in enumerate(res.results) if (rese.actualClass, rese.classes[learnerI]) in selected])

        self.send("Selected Examples", data)

        
if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWConfusionMatrix()
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()
