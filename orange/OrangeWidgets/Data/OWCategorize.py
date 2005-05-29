"""
<name>Categorize</name>
<description>Categorize treats all coninuously-valued attribute in a table
and categorizes them using the same method. The data in the table needs
to be classified. The three methods for categorization
this widget can use are: entropy-based discretization (finds most approapriate
cut-off by MDL-based technique proposed by Fayyad & Iranni), equal-frequency intervals
(intervals contain about the same number of instances), and equal-width intervals.</description>
<icon>icons/Categorize.png</icon>
<priority>2100</priority>
"""
#
# OWDataTable.py
#
# wishes:
# ignore attributes, filter examples by attribute values, do
# all sorts of preprocessing (including discretization) on the table,
# output a new table and export it in variety of formats.

from qttable import *
from OWWidget import *

##############################################################################

class OWCategorize(OWWidget):
    settingsList = ["Categorization", "NumberOfIntervals", "ShowIntervals"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Categorize")

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.data)]
        self.outputs = [("Classified Examples", ExampleTableWithClass)]
        self.dataset=None

        # Settings
        self.Categorization = 0  # default to entropy
        self.NumberOfIntervals = 5
        self.ShowIntervals = 0
        self.loadSettings()
        
        # GUI: CATEGORIZATION DETAILS
        self.catBox = QVGroupBox(self.controlArea)
        self.catBox.setTitle('Categorization Method')
        QToolTip.add(self.catBox,"Method that will be used for categorization")

        self.catMethods=["Entropy-based discretization", "Equal-Frequency Intervals", "Equal-Width Intervals"]
        self.cat = QComboBox(self.catBox)
        for cm in self.catMethods:
            self.cat.insertItem(cm)
        self.cat.setCurrentItem(self.Categorization)

        self.nIntLab = QLabel("Number of intervals: %i" % self.NumberOfIntervals, self.controlArea)
        self.nInt = QSlider(2, 20, 1, self.NumberOfIntervals, QSlider.Horizontal, self.controlArea)
        self.nInt.setTickmarks(QSlider.Below)

        self.nIntLab.setDisabled(self.Categorization==0)
        self.nInt.setDisabled(self.Categorization==0)

        QWidget(self.controlArea).setFixedSize(16, 16)
        sp = QSpacerItem(20,20)
        #self.controlArea.addItem(sp)

        self.applyBtn = QPushButton("&Apply", self.controlArea)

        QWidget(self.controlArea).setFixedSize(16, 16)

        self.showInt = QCheckBox("Show &Intervals", self.controlArea)
        self.showInt.setChecked(self.ShowIntervals)

        # GUI: DISPLAY THE RESULTS        
        #self.resBox = QVGroupBox(self.mainArea)
        #self.resBox.setTitle('Categorization Results')

        # set the table widget
        self.layout=QVBoxLayout(self.mainArea)
        self.g = QVGroupBox(self.mainArea)
        self.g.setTitle('Categorization Results')
        self.res=QTable(self.g)
        self.res.setSelectionMode(QTable.NoSelection)
        
        self.resHeader=self.res.horizontalHeader()
        self.resHeader.setLabel(0, 'Attribute')
        self.resHeader.setLabel(1, 'Values')
        self.layout.add(self.g)
        self.lab = QLabel(self.g)
        self.lab.setText('Atributes Removed:')
        self.remList=QListBox(self.g)
        

        # event handling
        self.connect(self.cat,SIGNAL("activated(int)"), self.setCatMethod)
        self.connect(self.nInt,SIGNAL("valueChanged(int)"), self.setNInt)
        #self.connect(self.nInt,SIGNAL("textChanged(const QString &)"), self.setNInt)
        #self.connect(self.freqInt,SIGNAL("stateChanged(int)"), self.setIntMethod)
        #self.connect(self.widthInt,SIGNAL("stateChanged(int)"), self.setIntMethod)
        self.connect(self.applyBtn,SIGNAL("clicked()"),self.categorize)
        self.connect(self.showInt,SIGNAL("toggled(bool)"), self.showIntervals)

        self.resize(500,500)
        
    def data(self,dataset):
        self.dataset=dataset
        self.categorize()
        self.setTable()

    def setTable(self):
        self.res.setNumCols(2)
        self.res.setNumRows(len(self.discretizedAtts) - len(self.removedAtt))

        self.resHeader=self.res.horizontalHeader()
        self.resHeader.setLabel(0, 'Attribute')
        self.resHeader.setLabel(1, 'Values')

        i=0
        for att in self.discretizedAtts:
            if 'D_' + att not in self.removedAtt:
                self.res.setText(i, 0, att)
                if self.ShowIntervals:
                    values = reduce(lambda x,y: x+', '+y,  self.catData.domain['D_'+att].values)
                else:
                    discretizer = self.catData.domain['D_'+att].getValueFrom.transformer
                    if type(discretizer)==orange.EquiDistDiscretizer:
                        values = reduce(lambda x,y: x+', '+y, ["%.2f" % (discretizer.firstVal + t*discretizer.step) for t in range(1, discretizer.numberOfIntervals)])
                    elif type(discretizer)==orange.IntervalDiscretizer:
                        values = reduce(lambda x,y: x+', '+y, ["%.2f" % x for x in discretizer.points])
                    else:
                        values = "<unknown discretization type>"
                        
                self.res.setText(i, 1, values+" ")
                i += 1
        self.res.adjustColumn(0)
        self.res.adjustColumn(1)

        self.remList.clear()
        if len(self.removedAtt)==0:
            self.remList.insertItem('(none)')
        else:
            for i in self.removedAtt:
                self.remList.insertItem(i[2:])

    # data categorization (what this widget does)
    
    def categorize(self):
        if self.dataset == None: return

        # remember which attributes will be discretized
        self.discretizedAtts = []
        for a in self.dataset.domain.attributes:
            if a.varType == orange.VarTypes.Continuous:
                self.discretizedAtts.append(a.name)

        # set appropriate discretizer object
        catMethod = self.Categorization
        if catMethod == 0:
            discretizer = orange.EntropyDiscretization()
        else:
            nInt = self.NumberOfIntervals
            if catMethod==1:
                discretizer = orange.EquiNDiscretization(numberOfIntervals=nInt)
            else:
                discretizer = orange.EquiDistDiscretization(numberOfIntervals=nInt)

        self.catData = orange.Preprocessor_discretize(self.dataset, method=discretizer)

        # remove attributes that were discretized to a constant
        attrlist = []
        self.removedAtt = []
        nrem=0
        for i in self.catData.domain.attributes:
            if (len(i.values)>1):
                attrlist.append(i)
            else:
                self.removedAtt.append(i.name)

        attrlist.append(self.catData.domain.classVar)
        self.newData = self.catData.select(attrlist)

        self.send("Classified Examples", self.newData)
        self.setTable()

    # management of signals (parameter setting)    

    def setCatMethod(self, value):
        self.Categorization = value
        self.nIntLab.setDisabled(value==0)
        self.nInt.setDisabled(value==0)

    def setNInt(self, value):
        if str(value) == '': value = '5'
        v = int(str(value))
        if (v<2) or (v>20): v = 5
        self.nIntLab.setText ("Number Of Intervals: %i" % self.NumberOfIntervals)
        self.NumberOfIntervals = v

    def showIntervals(self, value):
        self.ShowIntervals = value
        self.setTable()

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWCategorize()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable(r'..\datasets\adult_sample')
    ow.cdata(dataset)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
