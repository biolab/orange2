"""
<name>Data table</name>
<description>DataTable shows the data set in a spreadsheet.</description>
<category>Data</category>
<icon>icons/DataTable.png</icon>
<priority>500</priority>
"""

#
# OWDataTable.py
#
# wishes:
# ignore attributes, filter examples by attribute values, do
# all sorts of preprocessing (including discretization) on the table,
# output a new table and export it in variety of formats.

from qttable import *
from OData import *
from OWWidget import *

##############################################################################

class OWDataTable(OWWidget):
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "DataTable",
        """DataTable shows the data set in a spreadsheet.
""",
        FALSE,
        FALSE)
        
        self.dataset=None
        self.addInput("cdata")
        self.addInput("data")
        
        # GUI
        # set the table widget
        self.layout=QVBoxLayout(self.mainArea)
        #add your components here
        self.table=QTable(self.mainArea)
        self.table.setSelectionMode(QTable.NoSelection)
        self.layout.add(self.table)
        
        #self.table.resize(self.mainArea.size())
        #self.resize(100,100)

    def cdata(self,dataset):
        self.data(dataset)

    def data(self,dataset):
        self.dataset=dataset.table
        self.set_table()
    
    def set_table(self):
        if self.dataset==None:
            return
        if self.dataset.domain.classVar:
            self.table.setNumCols(len(self.dataset.domain.attributes)+1)
        else:   
            self.table.setNumCols(len(self.dataset.domain.attributes))
        self.table.setNumRows(len(self.dataset))

        # set the header (attaribute names)
        self.header=self.table.horizontalHeader()
        for i in range(len(self.dataset.domain.attributes)):
            self.header.setLabel(i, self.dataset.domain.attributes[i].name)
        if self.dataset.domain.classVar:
            self.header.setLabel(len(self.dataset.domain.attributes), self.dataset.domain.classVar.name)

        # set the contents of the table (values of attributes)
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset.domain.attributes)):
                self.table.setText(i, j, str(self.dataset[i][j].native()))
        if self.dataset.domain.classVar:
            j = len(self.dataset.domain.attributes)
            for i in range(len(self.dataset)):
                self.table.setText(i, j, self.dataset[i].getclass().native())

        # adjust the width of the table
        for i in range(len(self.dataset.domain.attributes)):
            self.table.adjustColumn(i)

        # manage sorting (not correct, does not handle real values)
        self.connect(self.header,SIGNAL("clicked(int)"),self.sort)
        self.sortby = len(self.dataset.domain.attributes)+2
        #self.resize(100,100)
                
    def sort(self, col):
        "sorts the table by column col"
        if col == self.sortby:
            self.sortby = - self.sortby
        else:
            self.sortby = col
        self.table.sortColumn(abs(self.sortby),self.sortby>=0,TRUE)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWDataTable()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('adult_sample')
    od = OrangeData(dataset)
    ow.data(od)

    ow.show()
    a.exec_loop()
