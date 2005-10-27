"""
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
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'SampleData')
        
        self.inputs = [("Data", ExampleTable, self.cdata)]
        self.outputs = [("Sampled Data", ExampleTable), ("Classified Sampled Data", ExampleTableWithClass), ("Remaining Data", ExampleTable), ("Classified Remaining Data", ExampleTableWithClass)]

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
 
        # GUI
        # Info Box
        box1 = QVGroupBox("Information", self.controlArea)
        self.infoa = QLabel('No data on input.', box1)
        self.infob = QLabel('', box1)
        self.infoc = QLabel('', box1)
        OWGUI.separator(self.controlArea)

        # Options Box
        box2 = QVGroupBox('Options', self.controlArea)
        OWGUI.checkBox(box2, self, 'Stratified', 'Stratified if possible')
        OWGUI.checkWithSpin(box2, self, 'Use a specific random seed:', 0, 32767, 'UseSpecificSeed', 'RandomSeed')
        OWGUI.separator(self.controlArea)

        # Sampling Type Box
        self.s = [None, None, None, None]
        self.sBox = QVButtonGroup("Sampling Type", self.controlArea)        

        # Random Sampling
        self.s[0] = QRadioButton('Random Sampling', self.sBox)
        # repeat checkbox
        self.h1Box = QHBox(self.sBox)
        QWidget(self.h1Box).setFixedSize(19, 8)
        OWGUI.checkBox(self.h1Box, self, 'Repeat', 'Elements in sample can repeat')
        # specified number of elements checkbox
        self.h2Box = QHBox(self.sBox)
        QWidget(self.h2Box).setFixedSize(19, 8)
        OWGUI.checkWithSpin(self.h2Box, self, 'Select a specified number of elements:', 1, 1000000000, 'useCases', 'nCases', checkCallback=self.uCases)
        # percentage slider
        self.h3Box = QHBox(self.sBox)
        QWidget(self.h3Box).setFixedSize(19, 8)
        QLabel("Sample size:", self.h3Box)
        self.slidebox = QHBox(self.sBox)
        QWidget(self.slidebox).setFixedSize(19, 8)
        OWGUI.hSlider(self.slidebox, self, 'selPercentage', minValue=1, maxValue=100, step=1, ticks=10, labelFormat="   %d%%")        
        
        # Cross Validation
        self.s[1] = QRadioButton('Cross Validation', self.sBox)
        box = QHBox(self.sBox)
        QWidget(box).setFixedSize(19, 8)
        OWGUI.spin(box, self, 'CVFolds', 2, 100, step=1, label='Number of folds:  ', callback=self.changeCombo)

        # Leave-One-Out
        self.s[2] = QRadioButton('Leave-One-Out', self.sBox)

        # Multiple Groups
        self.s[3] = QRadioButton('Multiple Groups', self.sBox)        
        gbox = QHBox(self.sBox)
        QWidget(gbox).setFixedSize(19, 8)
        OWGUI.lineEdit(gbox, self, 'GroupText', label='Sizes of groups:', callback=self.changeCombo)

        # Select Data Button
        QWidget(self.sBox).setFixedSize(0, 8)
        OWGUI.button(self.sBox, self, 'Select &Data', callback = self.process)
        self.s[self.SelectType].setChecked(True)    # set initial radio button on (default sample type)

        # Output Group Box
        OWGUI.separator(self.controlArea)
        self.foldBox = QHGroupBox('Fold / Group', self.controlArea)
        QLabel('Select fold / group to output:', self.foldBox)
        self.foldcombo = QComboBox(self.foldBox)
        # fill the combo box (later make it sensitive to number of folds)
        self.foldcombo.clear()
        for x in range(100):
            self.foldcombo.insertItem(str(x+1))
        self.foldBox.setEnabled(False)
        
        # CONNECTIONS        
        # set connections for RadioButton (SelectType)
        self.dummy1 = [None]*len(self.s)
        for i in range(len(self.s)):
            self.dummy1[i] = lambda x, v=i: self.sChanged(x, v)
            self.connect(self.s[i], SIGNAL("toggled(bool)"), self.dummy1[i])

        # set connection for ComboBox (fold to output)
        self.connect(self.foldcombo, SIGNAL('activated(int)'), self.foldChanged)
        
        # final touch
        self.resize(200, 275)

    # CONNECTION TRIGGER AND GUI ROUTINES
    # enables RadioButton switching
    def sChanged(self, value, id):
        self.SelectType = id
        self.process()

    # reflect user's actions that change combobox contents
    def changeCombo(self):
        # refill combobox
        self.Folds = 1
        if self.SelectType == 1: self.Folds = self.CVFolds
        if self.SelectType == 2:
            if self.data: self.Folds = len(self.data)
            else:         self.Folds = 1
        if self.SelectType == 3: self.Folds = self.nGroups
        self.foldcombo.clear()
        for x in range(self.Folds):
            self.foldcombo.insertItem(str(x+1))
     
    # triggered on change in output fold combobox
    def foldChanged(self, ix):
        self.outFold = int(ix+1)
        if self.data: self.sdata()

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
    def cdata(self, dataset):
        if dataset:
            self.infoa.setText('%d instances in input data set.' % len(dataset))
            self.data = dataset
            self.process()
        else:
            self.infoa.setText('No data on input.')
            self.infob.setText('')
            self.infoc.setText('')
            self.send("Sampled Data", None)
            self.send("Remaining Data", None)
            self.send("Classified Sampled Data", None)
            self.send("Classified Remaining Data", None)
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
                self.infob.setText('Method: Random Sampling with Repetitions, using exactly %d instances.' % self.nCases)
            else:                
                sample = self.data.select(self.ind, 0)
                remainder = self.data.select(self.ind, 1)
            self.infoc.setText('Outputing %d instances.' % len(sample))
        elif self.SelectType == 3:
            self.ind = self.indices(self.data, p0 = self.pGroups[self.outFold-1])
            sample = self.data.select(self.ind, 0)
            remainder = self.data.select(self.ind, 1)
            self.infoc.setText('Outputing group %d of %d, %d instance(s).' % (self.outFold, self.Folds, len(sample)))
        else:
            sample = self.data.select(self.ind, self.outFold-1)
            remainder = self.data.select(self.ind, self.outFold-1, negate=1)
            self.infoc.setText('Outputing fold %d of %d, %d instance(s).' % (self.outFold, self.Folds, len(sample)))
        # set name (by PJ)
        if sample:
            sample.name = self.data.name
        if remainder:
            remainder.name = self.data.name
        # send data
        self.send("Sampled Data", sample)
        self.send("Remaining Data", remainder)
        # send classified data (if class exists)
        if self.data.domain.classVar:
            self.send("Classified Sampled Data", sample)
            self.send("Classified Remaining Data", remainder)

    # MAIN SWITCH
    # processes data after the user requests it
    def process(self):
        # reset errors, fold selected
        self.error()
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
                #print int(self.nCases)
                if self.nCases > len(self.data):
                    self.error("Sample size (w/o repetitions) larger than dataset.")
                    return                    
                self.indices = orange.MakeRandomIndices2(p0=int(self.nCases))
                self.infob.setText('Method: Random Sampling, using exactly %d instances.' % self.nCases)
            else:
                #print float(self.selPercentage/100.0)
                self.indices = orange.MakeRandomIndices2(p0=float(self.selPercentage/100.0))
                self.infob.setText('Method: Random Sampling, using %d percent of instances.' % self.selPercentage)
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
                self.infob.setText('Method: Leave-One-Out.')
            else:
                self.CVFoldsInternal = self.CVFolds
                self.infob.setText('Method: %d-fold Cross Validation.' % self.CVFolds)
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
            self.infob.setText('Method: Multiple Groups.')
            #parse group specification string
            try:
                self.pGroups = []
                for x in self.GroupText.split(','):
                    self.pGroups.append(float(x))
                self.nGroups = len(self.pGroups)
            except:
                self.error("Invalid group specification string entered.")
                return

            #prepare indices generator
            self.indices = orange.MakeRandomIndices2()
            if self.Stratified == 1: self.indices.stratified = self.indices.StratifiedIfPossible  
            else:                    self.indices.stratified = self.indices.NotStratified
            if self.UseSpecificSeed == 1: self.indices.randseed = self.RandomSeed
            else:                         self.indices.randomGenerator = orange.RandomGenerator(random.randint(0,65536))    

        # enable fold selection and fill combobox if applicable
        if self.SelectType == 0:
            self.foldBox.setEnabled(False)
        else:
            self.foldBox.setEnabled(True)
            self.changeCombo()

        # call data output routine        
        self.sdata()

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWDataSamplerA()
    appl.setMainWidget(ow)
    ow.show()
    dataset = orange.ExampleTable('iris.tab')
    ow.data(dataset)
    appl.exec_loop()
