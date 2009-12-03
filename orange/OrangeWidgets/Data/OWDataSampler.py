r"""
<name>Data Sampler</name>
<description>Selects a subset of instances from the data set.</description>
<icon>icons/DataSampler.png</icon>
<contact>Aleksander Sadikov (aleksander.sadikov(@at@)fri.uni-lj.si)</contact>
<priority>1125</priority>
"""
from OWWidget import *
import OWGUI
import random

class OWDataSampler(OWWidget):
    settingsList=["Stratified", "Repeat", "UseSpecificSeed", "RandomSeed",
    "GroupSeed", "outFold", "Folds", "SelectType", "useCases", "nCases", "selPercentage", "LOO",
    "CVFolds", "CVFoldsInternal", "nGroups", "pGroups", "GroupText"]
    
    contextHandlers = {"":DomainContextHandler("", ["nCases","selPercentage"])}
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'SampleData', wantMainArea = 0)

        self.inputs = [("Data", ExampleTable, self.setData)]
        self.outputs = [("Sample", ExampleTable), ("Remaining Examples", ExampleTable)]

        # initialization of variables
        self.data = None                        # dataset (incoming stream)
        self.indices = None                     # indices that control sampling
        self.ind = None                         # indices that control sampling

        self.Stratified = 1                     # use stratified sampling if possible?
        self.Repeat = 0                         # can elements repeat in a sample?
        self.UseSpecificSeed = 0                # use a specific random seed?
        self.RandomSeed = 1                     # specific seed used
        self.GroupSeed = 1                      # current seed for multiple group selection
        self.outFold = 1                        # folder/group to output
        self.Folds = 1                          # total number of folds/groups

        self.SelectType = 0                     # sampling type (LOO, CV, ...)
        self.useCases = 0                       # use a specific number of cases?
        self.nCases = 25                        # number of cases to use
        self.selPercentage = 30                 # sample size in %
        self.LOO = 1                            # use LOO?
        self.CVFolds = 10                       # number of CV folds
        self.CVFoldsInternal = 10               # number of CV folds (for internal use)
        self.nGroups = 3                        # number of groups
        self.pGroups = [0.1,0.25,0.5]           # sizes of groups
        self.GroupText = '0.1,0.25,0.5'         # assigned to Groups Control (for internal use)

        self.loadSettings()
        # GUI
        
        # Info Box
        box1 = OWGUI.widgetBox(self.controlArea, "Information")
        self.infoa = OWGUI.widgetLabel(box1, 'No data on input.')
        self.infob = OWGUI.widgetLabel(box1, ' ')
        self.infoc = OWGUI.widgetLabel(box1, ' ')
        
        # Options Box
        box2 = OWGUI.widgetBox(self.controlArea, 'Options')
        OWGUI.checkBox(box2, self, 'Stratified', 'Stratified (if possible)', callback=self.settingsChanged)
        OWGUI.checkWithSpin(box2, self, 'Set random seed:', 0, 32767, 'UseSpecificSeed', 'RandomSeed', checkCallback=self.settingsChanged, spinCallback=self.settingsChanged)
        OWGUI.separator(self.controlArea)

        # Sampling Type Box
        self.s = [None, None, None, None]
        self.sBox = OWGUI.widgetBox(self.controlArea, "Sampling type")
        self.sBox.buttons = []

        # Random Sampling
        self.s[0] = OWGUI.appendRadioButton(self.sBox, self, "SelectType", 'Random sampling')
        # repeat checkbox
        self.h1Box = OWGUI.indentedBox(self.sBox, orientation = "horizontal")
        OWGUI.checkBox(self.h1Box, self, 'Repeat', 'With replacement', callback=self.settingsChanged)

        # specified number of elements checkbox
        self.h2Box = OWGUI.indentedBox(self.sBox, orientation = "horizontal")
        OWGUI.checkWithSpin(self.h2Box, self, 'Sample size (instances):', 1, 1000000000, 'useCases', 'nCases', checkCallback=[self.uCases, self.settingsChanged], spinCallback=self.settingsChanged)
        OWGUI.rubber(self.h2Box)
        
        # percentage slider
        self.h3Box = OWGUI.indentedBox(self.sBox, orientation = "horizontal")
        OWGUI.widgetLabel(self.h3Box, "Sample size:")
        self.slidebox = OWGUI.indentedBox(self.sBox, orientation = "horizontal")
        OWGUI.hSlider(self.slidebox, self, 'selPercentage', minValue=1, maxValue=100, step=1, ticks=10, labelFormat="   %d%%", callback=self.settingsChanged)

        # Cross Validation
        self.s[1] = OWGUI.appendRadioButton(self.sBox, self, "SelectType", 'Cross validation')
        
        box = OWGUI.indentedBox(self.sBox, orientation = "horizontal")
        OWGUI.spin(box, self, 'CVFolds', 2, 100, step=1, label='Number of folds:  ', callback=[self.changeCombo, self.settingsChanged])
        OWGUI.rubber(box)

        # Leave-One-Out
        self.s[2] = OWGUI.appendRadioButton(self.sBox, self, "SelectType", 'Leave-one-out')

        # Multiple Groups
        self.s[3] = OWGUI.appendRadioButton(self.sBox, self, "SelectType", 'Multiple subsets')
        gbox = OWGUI.indentedBox(self.sBox, orientation = "horizontal")
        OWGUI.lineEdit(gbox, self, 'GroupText', label='Subset sizes (e.g. "0.1, 0.2, 0.5"):', callback=self.multipleChanged)

        # Output Group Box
        OWGUI.separator(self.controlArea)
        self.foldcombo = OWGUI.comboBox(self.controlArea, self, "outFold", 'Output Data for Fold / Group', 'Fold / group:', orientation = "horizontal", items = range(1,101), callback = self.foldChanged, sendSelectedValue = 1, valueType = int)
        self.foldcombo.setEnabled(False)

        # Select Data Button
        OWGUI.separator(self.controlArea)
        self.sampleButton = OWGUI.button(self.controlArea, self, 'Sample &Data', callback = self.process)
        self.s[self.SelectType].setChecked(True)    # set initial radio button on (default sample type)
        OWGUI.separator(self.controlArea)

        # CONNECTIONS
        # set connections for RadioButton (SelectType)
        self.dummy1 = [None]*len(self.s)
        for i in range(len(self.s)):
            self.dummy1[i] = lambda x, v=i: self.sChanged(x, v)
            self.connect(self.s[i], SIGNAL("toggled(bool)"), self.dummy1[i])

        # final touch
        self.resize(200, 275)

    # CONNECTION TRIGGER AND GUI ROUTINES
    # enables RadioButton switching
    def sChanged(self, value, id):
        self.SelectType = id
        self.process()

    def multipleChanged(self):
        try:
            self.pGroups = [float(x) for x in self.GroupText.split(',')]
            self.nGroups = len(self.pGroups)
            self.error(1)
        except:
            self.error("Invalid specification for sizes of subsets.", 1)
        self.changeCombo()
        self.settingsChanged()

    # reflect user's actions that change combobox contents
    def changeCombo(self):
        # refill combobox
        self.Folds = 1
        if self.SelectType == 1:
            self.Folds = self.CVFolds
        elif self.SelectType == 2:
            if self.data:
                self.Folds = len(self.data)
            else:
                self.Folds = 1
        elif self.SelectType == 3:
            self.Folds = self.nGroups
        self.foldcombo.clear()
        for x in range(self.Folds):
            self.foldcombo.addItem(str(x+1))

    # triggered on change in output fold combobox
    def foldChanged(self):
        #self.outFold = int(ix+1)
        if self.data:
            self.sdata()

    # switches between cases and percentage (random sampling)
    def uCases(self):
        if self.useCases == 1:
            self.h3Box.setEnabled(False)
            self.slidebox.setEnabled(False)
        else:
            self.h3Box.setEnabled(True)
            self.slidebox.setEnabled(True)

    # I/O STREAM ROUTINES
    # handles changes of input stream
    def setData(self, dataset):
        self.closeContext()
        if dataset:
            self.infoa.setText('%d instances in input data set.' % len(dataset))
            self.data = dataset
            self.openContext("", dataset)
            self.process()
        else:
            self.infoa.setText('No data on input.')
            self.infob.setText('')
            self.infoc.setText('')
            self.send("Sample", None)
            self.send("Remaining Examples", None)
            self.data = None

    # feeds the output stream
    def sdata(self):
        # select data
        if self.SelectType == 0:
            if self.useCases == 1 and self.Repeat == 1:
                sample = orange.ExampleTable(self.data.domain)
                for x in range(self.nCases):
                    sample.append(self.data[random.randint(0,len(self.data)-1)])
                remainder = None
                self.infob.setText('Random sampling with repetitions, %d instances.' % self.nCases)
            else:
                sample = self.data.select(self.ind, 0)
                remainder = self.data.select(self.ind, 1)
            self.infoc.setText('Output: %d instances.' % len(sample))
        elif self.SelectType == 3:
            self.ind = self.indices(self.data, p0 = self.pGroups[self.outFold-1])
            sample = self.data.select(self.ind, 0)
            remainder = self.data.select(self.ind, 1)
            self.infoc.setText('Output: subset %(outFold)d of %(folds)d, %(instances)d instance(s).' % {"outFold": self.outFold, "folds": self.Folds, "instances": len(sample)})
        else:
            sample = self.data.select(self.ind, self.outFold-1)
            remainder = self.data.select(self.ind, self.outFold-1, negate=1)
            self.infoc.setText('Output: fold %(outFold)d of %(folds)d, %(instances)d instance(s).' % {"outFold": self.outFold, "folds": self.Folds, "instances": len(sample)})
        # set name (by PJ)
        if sample:
            sample.name = self.data.name
        if remainder:
            remainder.name = self.data.name
        # send data
        self.nSample = len(sample)
        self.nRemainder = len(remainder) if remainder is not None else 0
        self.send("Sample", sample)
        self.send("Remaining Examples", remainder)
        
        self.sampleButton.setEnabled(False)

    # MAIN SWITCH
    # processes data after the user requests it
    def process(self):
        # reset errors, fold selected
        self.error(0)
        self.outFold = 1

        # check for data
        if self.data == None:
            return
        else:
            self.infob.setText('')
            self.infoc.setText('')

        # Random Selection
        if self.SelectType == 0:
            # apply selected options
            if self.useCases == 1 and self.Repeat != 1:
                if self.nCases > len(self.data):
                    self.error(0, "Sample size (w/o repetitions) larger than dataset.")
                    return
                self.indices = orange.MakeRandomIndices2(p0=int(self.nCases))
                self.infob.setText('Random sampling, using exactly %d instances.' % self.nCases)
            else:
                if self.selPercentage == 100:
                    self.indices = orange.MakeRandomIndices2(p0=int(len(self.data)))
                else:
                    self.indices = orange.MakeRandomIndices2(p0=float(self.selPercentage/100.0))
                self.infob.setText('Random sampling, %d%% of input instances.' % self.selPercentage)
            if self.Stratified == 1: self.indices.stratified = self.indices.StratifiedIfPossible
            else:                    self.indices.stratified = self.indices.NotStratified
            if self.UseSpecificSeed == 1: self.indices.randseed = self.RandomSeed
            else:                         self.indices.randomGenerator = orange.RandomGenerator(random.randint(0,65536))

            # call output stream handler to send data
            self.ind = self.indices(self.data)

        # Cross Validation / LOO
        elif self.SelectType == 1 or self.SelectType == 2:
            # apply selected options
            if self.SelectType == 2:
                self.CVFoldsInternal = len(self.data)
                self.infob.setText('Leave-one-out.')
            else:
                self.CVFoldsInternal = self.CVFolds
                self.infob.setText('%d-fold cross validation.' % self.CVFolds)
            self.indices = orange.MakeRandomIndicesCV(folds = self.CVFoldsInternal)
            if self.Stratified == 1:
                self.indices.stratified = self.indices.StratifiedIfPossible
            else:
                self.indices.stratified = self.indices.NotStratified
            if self.UseSpecificSeed == 1:
                #self.indices.randomGenerator = orange.RandomGenerator(random.randint(0,65536))
                self.indices.randseed = self.RandomSeed
            else:
                #self.indices.randomGenerator = orange.RandomGenerator(random.randint(0,65536))
                self.indices.randseed = random.randint(0,65536)

            # call output stream handler to send data
            self.ind = self.indices(self.data)

        # MultiGroup
        elif self.SelectType == 3:
            self.infob.setText('Multiple subsets.')
            #parse group specification string

            #prepare indices generator
            self.indices = orange.MakeRandomIndices2()
            if self.Stratified == 1: self.indices.stratified = self.indices.StratifiedIfPossible
            else:                    self.indices.stratified = self.indices.NotStratified
            if self.UseSpecificSeed == 1: self.indices.randseed = self.RandomSeed
            else:                         self.indices.randomGenerator = orange.RandomGenerator(random.randint(0,65536))

        # enable fold selection and fill combobox if applicable
        if self.SelectType == 0:
            self.foldcombo.setEnabled(False)
        else:
            self.foldcombo.setEnabled(True)
            self.changeCombo()

        # call data output routine
        self.sdata()
        
    def settingsChanged(self):
        self.sampleButton.setEnabled(True)

    def sendReport(self):
        if self.SelectType == 0:
            if self.useCases:
                stype = "Random sample of %i instances" % self.nCases
            else:
                stype = "Random sample with %i%% instances" % self.selPercentage
        elif self.SelectType == 1:
            stype = "%i-fold cross validation" % self.CVFolds
        elif self.SelectType == 2:
            stype = "Leave one out"
        elif self.SelectType == 3:
            stype = "Multiple subsets"
        self.reportSettings("Settings", [("Sampling type", stype), ("Stratification", OWGUI.YesNo[self.Stratified]), ("Random seed", str(self.RandomSeed) if self.UseSpecificSeed else "auto")])
                             
        self.reportSettings("Data", [("Input", "%i examples" % len(self.data)), ("Sample", "%i examples" % self.nSample), ("Rest", "%i examples" % self.nRemainder)])

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWDataSampler()
    data = orange.ExampleTable('../../doc/datasets/iris.tab')
    ow.setData(data)
    ow.show()
    appl.exec_()
    ow.saveSettings()