"""
<name>Data Table</name>
<description>DataTable shows the data set in a spreadsheet.</description>
<category>Data</category>
<icon>icons/DataTable.png</icon>
<priority>500</priority>
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

    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, "Data Table", "DataTable shows the data set in a spreadsheet.\n")

        self.inputs = [("Examples", ExampleTable, self.dataset, 1)]
        self.outputs = []
        
        self.data = None
        self.showMetas = 1

        # GUI
        self.layout=QVBoxLayout(self.mainArea)
        self.table=QTable(self.mainArea)
        self.table.setSelectionMode(QTable.NoSelection)
        self.layout.add(self.table)
        self.table.hide()

    def dataset(self,data):
        self.data = data
        if not data:
            self.table.hide()
        else:
            self.set_table()

    def set_table(self):
        if self.data==None:
            return
        cols = len(self.data.domain.attributes)
        if hasattr(self.data.domain, 'classVar'):
            cols += 1
        if self.showMetas:
            metas = self.data.domain.getmetas().values() # getmetas returns a dictionary
            cols += len(metas)
        self.table.setNumCols(cols)
        self.table.setNumRows(len(self.data))

        # set the header (attribute names)
        self.header=self.table.horizontalHeader()
        for i in range(len(self.data.domain.attributes)):
            self.header.setLabel(i, self.data.domain.attributes[i].name)
        col = len(self.data.domain.attributes)
        if self.data.domain.classVar:
            self.header.setLabel(col, self.data.domain.classVar.name)
            col += 1
        if self.showMetas:
            for (j,m) in enumerate(metas):
                self.header.setLabel(j+col, m.name)

        # set the contents of the table (values of attributes)
        for i in range(len(self.data)):
            for j in range(len(self.data.domain.attributes)):
                self.table.setText(i, j, str(self.data[i][j].native()))
        col = len(self.data.domain.attributes)
        if self.data.domain.classVar:
            for i in range(len(self.data)):
                OWGUI.tableItem(self.table, i, col, self.data[i].getclass().native(), color=Qt.lightGray)
            col += 1
##        for (j,m) in enumerate(metas):
##            for i in range(len(self.data)):
##                OWGUI.tableItem(self.table, i, j+col, self.data[m].native(), color=Qt.yellow)

        # adjust the width of the table
        for i in range(cols):
            self.table.adjustColumn(i)

        # manage sorting (not correct, does not handle real values)
        self.connect(self.header,SIGNAL("clicked(int)"),self.sort)
        self.sortby = 0
        #self.table.setColumnMovingEnabled(1)
        self.table.show()
        self.layout.activate() # this is needed to scale the widget correctly

    def sort(self, col):
        "sorts the table by column col"
        if col == self.sortby-1:
            self.sortby = - self.sortby
        else:
            self.sortby = col+1
        self.table.sortColumn(col, self.sortby>=0, TRUE)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDataTable()
    a.setMainWidget(ow)

#    d = orange.ExampleTable('adult_sample')
    d = orange.ExampleTable('wtclassed')
    ow.show()
    ow.dataset(d)
    a.exec_loop()
