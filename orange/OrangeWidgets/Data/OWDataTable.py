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

##############################################################################

class colorItem(QTableItem):
    def __init__(self, table, editType, text):
        QTableItem.__init__(self, table, editType, str(text))

    def paint(self, painter, colorgroup, rect, selected):
        g = QColorGroup(colorgroup)
        g.setColor(QColorGroup.Base, Qt.lightGray)
        QTableItem.paint(self, painter, g, rect, selected)

##############################################################################

class OWDataTable(OWWidget):
    settingsList = []

    def __init__(self, parent=None):
        OWWidget.__init__(self,
        parent,
        "DataTable",
        """DataTable shows the data set in a spreadsheet.
""",
        FALSE,
        FALSE)

        self.inputs = [("Examples", ExampleTable, self.dataset, 1)]
        self.outputs = []
        
        self.data=None
        
        # GUI
        self.layout=QVBoxLayout(self.mainArea)
        self.table=QTable(self.mainArea)
        self.table.setSelectionMode(QTable.NoSelection)
        self.layout.add(self.table)

    def dataset(self,data):
        self.data = data
        self.set_table()
    
    def set_table(self):
        if self.data==None:
            return
        if hasattr(self.data.domain, 'classVar') and self.data.domain.classVar:
            self.table.setNumCols(len(self.data.domain.attributes)+1)
        else:   
            self.table.setNumCols(len(self.data.domain.attributes))
        self.table.setNumRows(len(self.data))

        # set the header (attribute names)
        self.header=self.table.horizontalHeader()
        for i in range(len(self.data.domain.attributes)):
            self.header.setLabel(i, self.data.domain.attributes[i].name)
        if self.data.domain.classVar:
            self.header.setLabel(len(self.data.domain.attributes), self.data.domain.classVar.name)

        # set the contents of the table (values of attributes)
        for i in range(len(self.data)):
            for j in range(len(self.data.domain.attributes)):
                self.table.setText(i, j, str(self.data[i][j].native()))
        if self.data.domain.classVar:
            j = len(self.data.domain.attributes)
            for i in range(len(self.data)):
                item = colorItem(self.table, QTableItem.WhenCurrent, self.data[i].getclass().native())
                self.table.setItem(i, j, item)
#                self.table.setText(i, j, self.data[i].getclass().native())

        # adjust the width of the table
        for i in range(len(self.data.domain.attributes)):
            self.table.adjustColumn(i)

        # manage sorting (not correct, does not handle real values)
        self.connect(self.header,SIGNAL("clicked(int)"),self.sort)
        self.sortby = 0
        #self.table.setColumnMovingEnabled(1)
                
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

    data = orange.ExampleTable('adult_sample')
    ow.dataset(data)
    ow.show()
    a.exec_loop()
