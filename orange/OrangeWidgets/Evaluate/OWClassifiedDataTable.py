"""
<name>Classifications</name>
<description>Shows a set of data instances and predictions of one or more classifiers.</description>
<icon>icons/Classifications.png</icon>
<priority>300</priority>
"""

from qttable import *
from OWWidget import *
import OWGUI

##############################################################################

class colorItem(QTableItem):
    def __init__(self, table, editType, text):
        QTableItem.__init__(self, table, editType, str(text))

    def paint(self, painter, colorgroup, rect, selected):
        g = QColorGroup(colorgroup)
        g.setColor(QColorGroup.Base, Qt.lightGray)
        QTableItem.paint(self, painter, g, rect, selected)

##############################################################################

class OWClassifiedDataTable(OWWidget):
    settingsList = ["ShowProb", "ShowClass", "ShowTrueClass", "ShowAttributeMethod", "sendDataType", "commitOnChange"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Classifications")

        self.callbackDeposit = []
        self.inputs = [("Examples", ExampleTable, self.dataset),("Classifiers", orange.Classifier, self.classifier, 0)]
        self.outputs = [("Selected Examples", ExampleTable)]
        self.classifiers = {}

        # saveble settings
        self.ShowProb = 1; self.ShowClass = 1; self.ShowTrueClass = 0
        self.ShowAttributeMethod = 0
        self.sendDataType = 0; self.commitOnChange = 0
        self.loadSettings()

        self.freezeAttChange = 0 # 1 to block table update followed by changes in attribute list box
        self.data=None
        
        # GUI - Options
        self.options = QVButtonGroup("Options", self.controlArea)
        self.options.setDisabled(1)
        OWGUI.checkBox(self.options, self, 'ShowProb', "Show predicted probabilities", callback=self.updateTableOutcomes)

        self.lbClasses = QListBox(self.options)
        self.lbClasses.setSelectionMode(QListBox.Multi)
        self.connect(self.lbClasses, SIGNAL("selectionChanged()"), self.updateTableOutcomes)
        
        OWGUI.checkBox(self.options, self, 'ShowClass', "Show predicted class", callback=[self.updateTableOutcomes, self.checkenable])
        self.trueClassCheckBox = OWGUI.checkBox(self.options, self, 'ShowTrueClass', "Show true class", callback=self.updateTrueClass, disabled=1)

        OWGUI.separator(self.controlArea)
        self.att = QVButtonGroup("Data Attributes", self.controlArea)
        OWGUI.radioButtonsInBox(self.att, self, 'ShowAttributeMethod', ['Show all', 'Hide all'], callback=self.updateAttributes)
        self.att.setDisabled(1)

        OWGUI.separator(self.controlArea)
        self.outBox = QVButtonGroup("Output", self.controlArea)
        OWGUI.radioButtonsInBox(self.outBox, self, 'sendDataType', ['None', 'Data with class conflict', 'Data with class agreement'], box='Data Selection',
                                tooltips=['No data will be sent to the output channel', 'Send data for which the predicted (and true class, if shown) are different.', 'Send data for which the predicted (and true class, if shown) match.'],
                                callback=self.checksenddata)
        OWGUI.checkBox(self.outBox, self, 'commitOnChange', 'Commit data on any change')
        self.commitBtn = OWGUI.button(self.outBox, self, "Commit", callback=self.senddata)

        self.outBox.setDisabled(1)

        # GUI - Table        
        self.layout = QVBoxLayout(self.mainArea)
        self.table = QTable(self.mainArea)
        self.table.setSelectionMode(QTable.NoSelection)
        self.header = self.table.horizontalHeader()
        self.vheader = self.table.verticalHeader()
        # manage sorting (not correct, does not handle real values)
        self.connect(self.header, SIGNAL("pressed(int)"), self.sort)
        self.sortby = -1

        self.layout.add(self.table)

    # updates the columns associated with the classifiers
    def updateTableOutcomes(self):
        if self.freezeAttChange: # program-based changes should not alter the table immediately
            return
        if not self.data or not self.classifiers:
            return
        
        attsel = [self.lbClasses.isSelected(i) for i in range(len(self.data.domain.attributes))]
        showatt = attsel.count(1)
        # sindx is the column where these start
        sindx = 1 + len(self.data.domain.attributes) + 1 * (self.data.domain.classVar<>None)
        col = sindx
        if self.ShowClass or self.ShowProb:
            for (cid, c) in enumerate(self.classifiers.values()):
                for (i, d) in enumerate(self.data):
                    (cl, p) = c(d, orange.GetBoth)
                    self.classifications[i].append(cl)
                    s = ''
                    if self.ShowProb and showatt:
                        s += reduce(lambda x,y: x+' : '+y, map(lambda x: "%8.6f"%x[1], filter(lambda x,s=attsel: s[x[0]], enumerate(p))))
                        if self.ShowClass:
                            s += ' -> '
                    if self.ShowClass:
                        s += str(cl)
                    self.table.setText(self.rindx[i], col, s)
                col += 1
        else:
            for i in range(len(self.data)):
                for c in range(len(self.classifiers)):
                    self.table.setText(self.rindx[i], col+c, '')
    
        for i in range(sindx, col):
            self.table.adjustColumn(i)

    def updateTrueClass(self):
        if self.classifiers:
            col = 1+len(self.data.domain.attributes)
            if self.ShowTrueClass and self.data.domain.classVar:
                self.table.showColumn(col)
                self.table.adjustColumn(col)
            else:
                self.table.hideColumn(col)

    def updateAttributes(self):
        if self.ShowAttributeMethod == 0:
            for i in range(len(self.data.domain.attributes)):
                self.table.showColumn(i+1)
                self.table.adjustColumn(i+1)
        if self.ShowAttributeMethod == 1:
            for i in range(len(self.data.domain.attributes)):
                self.table.hideColumn(i+1)

    # defines the table and paints its contents    
    def setTable(self):
        if self.data==None:
            return

        self.table.setNumCols(0)
        self.table.setNumCols(1 + len(self.data.domain.attributes) + (self.ShowTrueClass) + len(self.classifiers))
        self.table.setNumRows(len(self.data))

        # set the header (attribute names)
        self.header.setLabel(0, '#')
        for col in range(len(self.data.domain.attributes)):
            self.header.setLabel(col+1, self.data.domain.attributes[col].name)
        col = len(self.data.domain.attributes)+1
        self.header.setLabel(col, self.data.domain.classVar.name)
        col += 1
        for (i,c) in enumerate(self.classifiers.values()):
            self.header.setLabel(col+i, c.name)

        # set the contents of the table (values of attributes), data first
        for i in range(len(self.data)):
            self.table.setText(i, 0, str(i+1))
            for j in range(len(self.data.domain.attributes)):
                self.table.setText(i, j+1, str(self.data[i][j].native()))
        col = 1+len(self.data.domain.attributes)

        self.classifications = [[]] * len(self.data)
        if self.data.domain.classVar:
            # column for the true class
            for i in range(len(self.data)):
                c = self.data[i].getclass()
                item = colorItem(self.table, QTableItem.WhenCurrent, c.native())
                self.table.setItem(i, col, item)
                self.classifications[i] = [c]
            col += 1
    
        for i in range(col):
            self.table.adjustColumn(i)

        # include classifications
        self.updateTableOutcomes()
        self.updateAttributes()
        self.updateTrueClass()
        self.table.hideColumn(0) # hide column with indices, we will use vertical header to show this info
                
    def sort(self, col):
        "sorts the table by column col"
        self.sortby = - self.sortby
        self.table.sortColumn(col, self.sortby>=0, TRUE)

        # the table may be sorted, figure out data indices
        for i in range(len(self.data)):
            self.rindx[int(str(self.table.item(i,0).text()))-1] = i
        for (i, indx) in enumerate(self.rindx):
            self.vheader.setLabel(i, self.table.item(i,0).text())
        
    ##############################################################################
    # Input signals

    def dataset(self,data):
        self.data = data
        if not data:
            self.table.hide()
        else:
            if not self.classifiers:
                self.ShowTrueClass = 1
            
            lb = self.lbClasses
            lb.clear()
            for v in self.data.domain.classVar.values:
                lb.insertItem(str(v))
            self.freezeAttChange = 1
            for i in range(len(self.data.domain.classVar.values)):
                lb.setSelected(i, 1)
            self.freezeAttChange = 0
            lb.show()

            self.rindx = range(len(self.data))
            self.setTable()
            self.table.show()
            self.checkenable()

    def classifier(self, c, id):
        if not c:
            if self.classifiers.has_key(id):
                del self.classifiers[id]
        else:
            self.classifiers[id] = c
        if self.data:
            self.setTable()
        self.checkenable()

    # based on the data and classifiers enables/disables the control boxes
    def checkenable(self):
        # following should be more complicated and depends on what data are we showing
        cond = self.data<>None and (len(self.classifiers)>1 or len(self.classifiers)>0 and self.ShowTrueClass)
        self.outBox.setEnabled(cond)
        if cond and self.commitOnChange:
            self.senddata()

        self.trueClassCheckBox.setEnabled(self.data<>None and self.data.domain.classVar<>None)
##        self.options.setEnabled(len(self.classifiers)>0)
        self.att.setEnabled(self.data<>None)            
        self.options.setEnabled(self.data<>None)            
        

    ##############################################################################
    # Ouput signals

    def checksenddata(self):
        if self.commitOnChange and self.outBox.isEnabled():
            self.senddata()

    # assumes that the data and display conditions (enough classes are displayed) have been checked
    def senddata(self):
        def cmpclasses(clist):
            ref = clist[0]
            for c in clist[1:]:
                if c<>ref: return 0
            return 1
        
        if not self.sendDataType:
            self.send("Selected Examples", None)
            return

        # list of columns to check
        selclass = [[],[0]][self.ShowTrueClass>0]
        for (i, classifier) in enumerate(self.classifiers):
            selclass.append(i+1)
        
##        s = [reduce(lambda x,y: [x, None][x==None or x<>y],  map(lambda x: cls[x], selclass)) <> None for cls in self.classifications]
        s = [cmpclasses(map(lambda x: cls[x], selclass)) for cls in self.classifications]
        if self.sendDataType == 1:
            s = [not x for x in s]
        data_selection = self.data.select(s)
        self.send("Selected Examples", data_selection)

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWClassifiedDataTable()
    a.setMainWidget(ow)
    ow.show()

    if 0:
        data = orange.ExampleTable('sailing')
        ow.dataset(data)
    elif 0:
        data = orange.ExampleTable('outcome')
        test = orange.ExampleTable('cheat', uses=data.domain)
        bayes = orange.BayesLearner(data)
        bayes.name = 'Naive Bayes'
        import orngTree
        tree = orngTree.TreeLearner(data)
        tree.name = 'Tree'
        ow.classifier(bayes, 1)
        ow.classifier(tree, 2)
        ow.dataset(test)
    else:
        #data = orange.ExampleTable(r'../../doc/datasets/titanic')
        #data = orange.ExampleTable('voting')
        data = orange.ExampleTable('sailing.txt')
        bayes = orange.BayesLearner(data)
        bayes.name = 'Naive Bayes'
        ow.classifier(bayes, 1)
##        import orngTree
##        tree = orngTree.TreeLearner(data)
##        tree.name = 'Tree'
##        ow.classifier(tree, 2)
        maj = orange.MajorityLearner(data)
        maj.name = 'Default'
        ow.classifier(maj, 2)
        ow.dataset(data)

    a.exec_loop()
    ow.saveSettings()
