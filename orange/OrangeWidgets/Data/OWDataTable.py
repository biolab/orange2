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
import math

##############################################################################

class OWDataTable(OWWidget):
    settingsList = []

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Table")

        self.inputs = [("Examples", ExampleTable, self.dataset, 0)]
        self.outputs = []
        
        self.data = {}
        self.showMetas = True

        # info box
        #self.controlArea.layout().setResizeMode(QLayout.Minimum)
        infoBox = QVGroupBox("Info", self.controlArea)
        self.infoEx = QLabel('No data loaded.', infoBox)
        self.infoMiss = QLabel('', infoBox)
        QLabel('', infoBox)
        self.infoAttr = QLabel('', infoBox)
        self.infoMeta = QLabel('', infoBox)
        QLabel('', infoBox)
        self.infoClass = QLabel('', infoBox)
        infoBox.setMinimumWidth(200)
        #infoBox.setMaximumHeight(infoBox.sizeHint().height())

        # settings box        
        boxSettings = QVGroupBox("Settings", self.controlArea)
##        self.cbShowMeta = OWGUI.checkBox(boxSettings, self, 'graph.showAttrValues', 'Show meta attributes', callback = not(self.showMetas))
        self.cbShowMeta = QCheckBox('Show meta attributes', boxSettings)
        self.connect(self.cbShowMeta, SIGNAL("clicked()"), self.cbShowMetaClicked)
        self.btnResetSort = QPushButton("Reset Sorting", boxSettings)
        self.connect(self.btnResetSort, SIGNAL("clicked()"), self.btnResetSortClicked)
        boxSettings.setMaximumHeight(boxSettings.sizeHint().height())
        
        # GUI with tabs
        layout=QVBoxLayout(self.mainArea)
        self.tabs = QTabWidget(self.mainArea, 'tabWidget')
        self.id2table = {}  # key: widget id, value: table
        self.table2id = {}  # key: table, value: widget id
        layout.addWidget(self.tabs)
        self.connect(self.tabs,SIGNAL("currentChanged(QWidget*)"),self.updateInfo)
        

    def dataset(self, data, id=None):
        """Generates a new table and adds it to a new tab when new data arrives;
        or hides the table and removes a tab when data==None;
        or replaces the table when new data arrives together with already existing id.
        """
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
            tabName = data.name
            if not tabName: tabName = str(id)
            #tabName = data.name + " " + str(id)
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
        """Updates the info box when a tab is clicked.
        """
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
        """Writes data into table, adjusts the column width.
        """
        qApp.setOverrideCursor(QWidget.waitCursor)
        if data==None:
            return
        vars = data.domain.variables
        if self.showMetas:
            m = data.domain.getmetas() # getmetas returns a dictionary
            ml = [(k, m[k]) for k in m]
            ml.sort(lambda x,y: cmp(y[0], x[0]))
            metas = [x[1] for x in ml]
            metaKeys = [x[0] for x in ml]
        else:
            metas = []
            metaKeys = []
        varsMetas = vars + metas
        numVars = len(data.domain.variables)
        numMetas = len(metas)
        numVarsMetas = numVars + numMetas
        numEx = len(data)
        numSpaces = int(math.log(numEx, 10))+1

        table.setNumCols(numVarsMetas)
        table.setNumRows(numEx)

        # set the header (attribute names)
        self.header=table.horizontalHeader()
        for i,var in enumerate(varsMetas):
            self.header.setLabel(i, var.name)

        # set the contents of the table (values of attributes)
        # iterate variables
        for j,(key,attr) in enumerate(zip(range(numVars) + metaKeys, varsMetas)):
            self.progressBarSet(j*100.0/numVarsMetas)
            if attr == data.domain.classVar:
                bgColor = QColor(160,160,160)
            elif attr in metas:
                bgColor = QColor(220,220,200)
            else:
                bgColor = Qt.white
            # generate list of tuples (attribute value, instance index) and sort by attrVal
            valIdx = [(ex[key].native(),idx) for idx,ex in enumerate(data)]
            valIdx.sort()
            # generate a dictionary where key: instance index, value: rank
            idx2rankDict = dict(zip([valIdx[1] for valIdx in valIdx], range(numEx)))
            for i in range(numEx):
                # set sorting key to str(rank) with leading spaces, i.e. 001, 002, ...
                OWGUI.tableItem(table, i, j, str(data[i][key]), editType=QTableItem.Never, background=bgColor, sortingKey=self.sortingKey(idx2rankDict[i], numSpaces))
            # adjust the width of the table
            table.adjustColumn(j)

        # manage sorting (not correct, does not handle real values)
        self.connect(self.header,SIGNAL("clicked(int)"),self.sort)
        self.sortby = 0
        #table.setColumnMovingEnabled(1)
        qApp.restoreOverrideCursor()
        table.show()


    def sort(self, col):
        """Sorts the table by column col.
        """
        qApp.setOverrideCursor(QWidget.waitCursor)
        if col == self.sortby-1:
            self.sortby = - self.sortby
        else:
            self.sortby = col+1
        table = self.tabs.currentPage()
        table.sortColumn(col, self.sortby>=0, TRUE)
        table.horizontalHeader().setSortIndicator(col, self.sortby>=0)
##        table.setSortIndicator(col, self.sortby>=0)
        qApp.restoreOverrideCursor()


    def sortingKey(self, val, len):
        """Returns a string with leading spaces followed by str(val), whose length is at least len.
        """
        s = "%0" + str(len) + "s"
        return s % str(val)


##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDataTable()
    a.setMainWidget(ow)

##    d = orange.ExampleTable('wtclassed')
    d1 = orange.ExampleTable(r'..\..\doc\datasets\auto-mpg')
    d2 = orange.ExampleTable(r'..\..\doc\datasets\voting.tab')
    d3 = orange.ExampleTable(r'..\..\doc\datasets\sponge.tab')
    d4 = orange.ExampleTable(r'..\..\doc\datasets\wpbc.csv')
    ow.show()
    ow.dataset(d1,"auto-mpg")
    ow.dataset(d2,"voting")
##    ow.dataset(None,"auto-mpg")
##    ow.dataset(d3,"sponge")
##    ow.dataset(None,"voting")
##    ow.dataset(None,"sponge")
##    ow.dataset(d4,"wpbc")
    a.exec_loop()
