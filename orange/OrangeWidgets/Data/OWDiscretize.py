"""
<name>Discretize</name>
<description>Discretization of continuous attributes.</description>
<icon>icons/Discretize.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact> 
<priority>2100</priority>
"""

from qttable import *
from OWWidget import *
import OWGUI, warnings

##############################################################################

warnings.filterwarnings("ignore", ".*'numberOfIntervals' is not a builtin attribute of 'EntropyDiscretization'", orange.AttributeWarning)

class OWDiscretize(OWWidget):
    settingsList = ["Discretization", "NumberOfIntervals", "ShowIntervals", "DiscretizeClass"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Discretize")

        self.inputs = [("Examples", ExampleTable, self.data)]
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass)]
        self.dataset=None

        # Settings#            self.discClassBtn.setDisabled(0)

        self.Discretization = 0  # default to entropy
        self.NumberOfIntervals = 5
        self.ShowIntervals = 0
        self.DiscretizeClass = 0 # applies only for regression data sets
        self.loadSettings()

        self.methods = [("Entropy-based discretization", orange.EntropyDiscretization()),
                        ("Equal-Frequency Intervals", orange.EquiNDiscretization()),
                        ("Equal-Width Intervals", orange.EquiDistDiscretization())]

        # GUI: DISCRETIZATION DETAILS
        box = OWGUI.widgetBox(self.controlArea, "Discretization")

        items = [x[0] for x in self.methods]
        self.methodBtns = OWGUI.radioButtonsInBox(box, self, "Discretization", items, callback=self.setDiscMethod, box="Method")

        self.nInt = OWGUI.qwtHSlider(box, self, "NumberOfIntervals", label="Intervals: ", 
                                     minValue=2, maxValue=20, step=1, precision=0, maxWidth=120).box
        self.nInt.setDisabled(self.Discretization==0)

#        self.discClassBtn = OWGUI.checkBox(box, self, "DiscretizeClass", "Discretize Class", disabled=1)

        OWGUI.button(box, self, "&Apply", callback = self.discretize)

        #sp = QSpacerItem(20,20)
        #self.controlArea.addItem(sp)

        #OWGUI.checkBox(self.controlArea, self, "ShowIntervals", "Show &Intervals", callback=self.setTable)
        OWGUI.radioButtonsInBox(self.controlArea, self, "ShowIntervals", ["Cut-Off Points", "Intervals"], box="Show",
                                callback=self.setTable)

        # set the table widget
        self.layout=QVBoxLayout(self.mainArea)
        self.g = QVGroupBox(self.mainArea)
        self.g.setTitle('Discretization Results')
        self.res=QTable(self.g)
        self.res.setSelectionMode(QTable.NoSelection)

        self.resHeader=self.res.horizontalHeader()
        self.resHeader.setLabel(0, 'Attribute')
        self.resHeader.setLabel(1, 'Values')
        self.layout.add(self.g)
        self.lab = QLabel(self.g)
        self.lab.setText('Removed atributes:')
        self.remList=QListBox(self.g)

        self.resize(600,300)

    def data(self, dataset):
        self.dataset=dataset
        if not self.dataset:
            self.res.setNumRows(0)
            self.remList.clear()
            return
        
        if self.dataset.domain.classVar and self.dataset.domain.classVar.varType == orange.VarTypes.Continuous or not self.dataset.domain.classVar:
            self.methodBtns.buttons[0].setDisabled(1)
            if self.Discretization == 0:
                self.setDiscMethod(1)
        else:
            self.methodBtns.buttons[0].setDisabled(0)
            if not self.dataset.domain.classVar and self.Discretization==0:
                self.Discretization=1

        self.discretize()

    def setTable(self):
        if not self.dataset:
            return
        self.res.setNumCols(2)
        self.res.setNumRows(len(self.discretizedAtts) - len(self.removed))

        self.resHeader=self.res.horizontalHeader()
        self.resHeader.setLabel(0, 'Attribute')
        self.resHeader.setLabel(1, ["Cut-Off Values", "Intervals"][self.ShowIntervals])

        i=0
        removed = [x.name for x in self.removed]
        for att in self.discretizedAtts:
            if 'D_' + att.name not in removed:
                self.res.setText(i, 0, att.name)
                if self.ShowIntervals:
                    values = reduce(lambda x,y: x+', '+y, self.discData.domain['D_'+att.name].values)
                else:
                    discretizer = self.discData.domain['D_'+att.name].getValueFrom.transformer
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
        if len(self.removed)==0:
            self.remList.insertItem('(none)')
        else:
            for i in self.removed:
                self.remList.insertItem(i.name[2:])

    # data discretization (what this widget does)
    
    def discretize(self):
        if self.dataset == None: return

        # remember which attributes will be discretized
        self.discretizedAtts = filter(lambda x: x.varType == orange.VarTypes.Continuous, self.dataset.domain.attributes)

        discretizer = self.methods[self.Discretization][1]
        discretizer.numberOfIntervals=int(self.NumberOfIntervals)
        self.discData = orange.Preprocessor_discretize(self.dataset, method=discretizer)

        # remove attributes that were discretized to a constant
        self.kept = filter(lambda x: len(x.values)>1, self.discData.domain.attributes)
        self.removed = filter(lambda x: len(x.values)<=1, self.discData.domain.attributes)
        if self.discData.domain.classVar:
            self.kept.append(self.discData.domain.classVar)
        self.newData = self.discData.select(self.kept)

        if self.dataset.name:
            self.newData.name = 'Discretized ' + self.dataset.name
        else:
            self.newData.name = 'Discretized'
        self.send("Examples", self.newData)
        if self.dataset.domain.classVar:
            self.send("Classified Examples", self.newData)
        self.setTable()

    def setDiscMethod(self, method = None):
        if method <> None:
            self.Discretization = method
        self.nInt.setDisabled(self.Discretization==0)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWDiscretize()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable(r'../../doc/datasets/adult_sample')
    dataset = orange.ExampleTable(r'../../doc/datasets/auto-mpg')
#    dataset = orange.ExampleTable(r'../../doc/datasets/bridges')
    ow.data(dataset)
    ow.show()
#    ow.data(None)
    a.exec_loop()
    ow.saveSettings()
