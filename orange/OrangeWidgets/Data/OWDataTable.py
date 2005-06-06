"""
<name>Data Table</name>
<description>DataTable shows the data set in a spreadsheet.</description>
<icon>icons/DataTable.png</icon>
<priority>100</priority>
"""

# OWDataTable.py
#
# wishes:
# ignore attributes, filter examples by attribute values, do
# all sorts of preprocessing (including discretization) on the table,
# output a new table and export it in variety of formats.

from qttable import *
from OWWidget import *
import OWGUI

##############################################################################

class OWDataTable(OWWidget):
    settingsList = []

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Table")

        self.inputs = [("Examples", ExampleTable, self.dataset, 0)]
        self.outputs = []
        
        self.data = {}
        self.showMetas = 1

        # info
        self.infoBox = QVGroupBox("Info", self.controlArea)
        self.infoEx = QLabel('No data loaded.', self.infoBox)
        self.infoMiss = QLabel('', self.infoBox)
        QLabel('', self.infoBox)
        self.infoAttr = QLabel('', self.infoBox)
        self.infoMeta = QLabel('', self.infoBox)
        QLabel('', self.infoBox)
        self.infoClass = QLabel('', self.infoBox)
        self.infoBox.setMinimumWidth(200)

##        # GUI
##        layout=QVBoxLayout(self.mainArea)
##        table=QTable(self.mainArea)
##        table.setSelectionMode(QTable.NoSelection)
##        layout.add(table)
##        table.hide()
##        self.grid.setColStretch(0,0)

        # GUI tabs
        layout=QVBoxLayout(self.mainArea)
        self.tabs = QTabWidget(self.mainArea, 'tabWidget')
        self.id2table = {}  # key: widget id, value: table
        self.table2id = {}  # key: table, value: widget id
        layout.addWidget(self.tabs)
        self.connect(self.tabs,SIGNAL("currentChanged(QWidget*)"),self.updateInfo)
        

    def dataset(self, data, id=None):
        if data:
            if self.data.has_key(id):
                # remove existing table
                self.data.pop(id)
                self.id2table[id].hide()
                self.tabs.removePage(self.id2table[id])
                self.table2id.pop(self.id2table.pop(id))
            self.data[id] = data
            self.progressBarInit()
            table=QTable(None)
            table.setSelectionMode(QTable.NoSelection)
            self.id2table[id] = table
            self.table2id[table] = id
##            tabName = data.name
##            if not tabName: tabName = str(id)
            tabName = data.name + " " + str(id)
            self.tabs.insertTab(table, tabName)
            self.set_table(table, data)
            self.tabs.showPage(table)
            self.progressBarFinished()
            self.set_info(data)
        elif self.data.has_key(id):
            self.data.pop(id)
            self.id2table[id].hide()
            self.tabs.removePage(self.id2table[id])
            self.table2id.pop(self.id2table.pop(id))
            self.set_info(self.data.get(self.table2id.get(self.tabs.currentPage(),None),None))


    def updateInfo(self, qTableInstance):
        #self.set_info(self.data.get(self.table2id.get(self.tabs.currentPage(),None),None))
        self.set_info(self.data.get(self.table2id.get(qTableInstance,None),None))

    def set_info(self, data):
        """Updates data info.
        """
        def sp(l):
            n = len(l)
            if n <> 1: return n, 's'
            else: return n, ''
        
        if not data:
            self.infoEx.setText('No data loaded.')
            self.infoMiss.setText('')
            self.infoAttr.setText('')
            self.infoMeta.setText('')
            self.infoClass.setText('')
        else:
            self.infoEx.setText("%i example%s," % sp(data))
            missData = orange.Preprocessor_takeMissing(data)
            self.infoMiss.setText('%i (%.1f%s) with missing values.' % (len(missData), 100.*len(missData)/len(data), "%"))
            self.infoAttr.setText("%i attribute%s," % sp(data.domain.attributes))
            self.infoMeta.setText("%i meta%s." % sp(data.domain.getmetas()))
            if data.domain.classVar:
                if data.domain.classVar.varType == orange.VarTypes.Discrete:
                    self.infoClass.setText('Discrete class with %d value%s.' % sp(data.domain.classVar.values))
                elif data.domain.classVar.varType == orange.VarTypes.Continuous:
                    self.infoClass.setText('Continuous class.')
                else:
                    self.infoClass.setText("Class neither descrete nor continuous.")
            else:
                self.infoClass.setText('Classless domain.')


    def set_table(self, table, data):
        if data==None:
            return
        #print data.domain, data.domain.getmetas()
        cols = len(data.domain.attributes)
        if data.domain.classVar:
            cols += 1
        if self.showMetas:
            m = data.domain.getmetas() # getmetas returns a dictionary
            ml = [(k, m[k]) for k in m]
            ml.sort(lambda x,y: cmp(y[0], x[0]))
            metas = [x[1] for x in ml]
            cols += len(metas)
        table.setNumCols(cols)
        table.setNumRows(len(data))

        # set the header (attribute names)
        self.header=table.horizontalHeader()
        for i in range(len(data.domain.attributes)):
            self.header.setLabel(i, data.domain.attributes[i].name)
        col = len(data.domain.attributes)
        if data.domain.classVar:
            self.header.setLabel(col, data.domain.classVar.name)
            col += 1
        if self.showMetas:
            for (j,m) in enumerate(metas):
                self.header.setLabel(j+col, m.name)

        # set the contents of the table (values of attributes)
        instances = len(data)
        for i in range(instances):
            self.progressBarSet(i*50/instances)
            for j in range(len(data.domain.attributes)):
                table.setText(i, j, str(data[i][j]))
        col = len(data.domain.attributes)
        if data.domain.classVar:
            self.progressBarSet(50+i*20/instances)
            for i in range(instances):
                OWGUI.tableItem(table, i, col, str(data[i].getclass()), editType=QTableItem.Never, background=QColor(160,160,160))
            col += 1
        mlen = len(metas)
        for (j,m) in enumerate(metas):
            self.progressBarSet(70+j*30/mlen)
            for i in range(instances):
                #print m.name, m.varType, data[i][m].valueType, data[i][m].varType
                OWGUI.tableItem(table, i, j+col, str(data[i][m]), editType=QTableItem.Never, background=QColor(220,220,220))

        # adjust the width of the table
        for i in range(cols):
            table.adjustColumn(i)

        # manage sorting (not correct, does not handle real values)
        self.connect(self.header,SIGNAL("clicked(int)"),self.sort)
        self.sortby = 0
        #table.setColumnMovingEnabled(1)
        table.show()
##        self.layout.activate() # this is needed to scale the widget correctly

    def sort(self, col):
        "sorts the table by column col"
        if col == self.sortby-1:
            self.sortby = - self.sortby
        else:
            self.sortby = col+1
        self.tabs.currentPage().sortColumn(col, self.sortby>=0, TRUE)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDataTable()
    a.setMainWidget(ow)

    d1 = orange.ExampleTable(r'..\..\doc\datasets\auto-mpg')
    d2 = orange.ExampleTable(r'..\..\doc\datasets\voting.tab')
    d3 = orange.ExampleTable(r'..\..\doc\datasets\sponge.tab')
#    d = orange.ExampleTable('wtclassed')
    ow.show()
    ow.dataset(d1,0)
    ow.dataset(d2,1)
##    ow.dataset(None,0)
    ow.dataset(d3,2)
##    ow.dataset(None,1)
##    ow.dataset(None,2)
    a.exec_loop()
