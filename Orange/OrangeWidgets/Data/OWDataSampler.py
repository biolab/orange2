import random

import Orange
from Orange.data import sample

import OWGUI
from OWWidget import *

NAME = "Data Sampler"
DESCRIPTION = "Samples data from a data set."
ICON = "icons/DataSampler.svg"
PRIORITY = 1125
CATEGORY = "Data"
MAINTAINER = "Aleksander Sadikov"
MAINTAINER_EMAIL = "aleksander.sadikov(@at@)fri.uni-lj.si"
INPUTS = [("Data", Orange.data.Table, "setData", Default)]
OUTPUTS = [("Data Sample", Orange.data.Table, ),
           ("Remaining Data", Orange.data.Table, )]


class OWDataSampler(OWWidget):
    settingsList = [
        "Stratified", "Repeat", "UseSpecificSeed", "RandomSeed",
        "GroupSeed", "outFold", "Folds", "SelectType", "useCases", "nCases",
        "selPercentage", "CVFolds", "nGroups",
        "pGroups", "GroupText", "autocommit"]

    contextHandlers = {
        "": DomainContextHandler("", ["nCases", "selPercentage"])
    }

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'SampleData',
                          wantMainArea=0)

        self.inputs = [("Data", ExampleTable, self.setData)]
        self.outputs = [("Data Sample", ExampleTable),
                        ("Remaining Data", ExampleTable)]

        # initialization of variables
        self.data = None                        # dataset (incoming stream)
        self.indices = None                     # indices that control sampling

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
        self.CVFolds = 10                       # number of CV folds
        self.nGroups = 3                        # number of groups
        self.pGroups = [0.1, 0.25, 0.5]         # sizes of groups
        self.GroupText = '0.1,0.25,0.5'         # assigned to Groups Control (for internal use)
        self.autocommit = False

        # Invalidated settings flag.
        self.outputInvalidateFlag = False

        self.loadSettings()

        # GUI

        # Info Box
        box1 = OWGUI.widgetBox(self.controlArea, "Information", addSpace=True)
        # Input data set info
        self.infoa = OWGUI.widgetLabel(box1, 'No data on input.')
        # Sampling type/parameters info
        self.infob = OWGUI.widgetLabel(box1, ' ')
        # Output data set info
        self.infoc = OWGUI.widgetLabel(box1, ' ')

        # Options Box
        box2 = OWGUI.widgetBox(self.controlArea, 'Options', addSpace=True)
        OWGUI.checkBox(box2, self, 'Stratified', 'Stratified (if possible)',
                       callback=self.settingsChanged)

        OWGUI.checkWithSpin(
            box2, self, 'Set random seed:', 0, 32767,
            'UseSpecificSeed',
            'RandomSeed',
            checkCallback=self.settingsChanged,
            spinCallback=self.settingsChanged
        )

        # Sampling Type Box
        self.s = [None, None, None, None]
        self.sBox = OWGUI.widgetBox(self.controlArea, "Sampling type",
                                    addSpace=True)
        self.sBox.buttons = []

        # Random Sampling
        self.s[0] = OWGUI.appendRadioButton(self.sBox, self, "SelectType",
                                            'Random sampling')

        # indent
        indent = OWGUI.checkButtonOffsetHint(self.s[0])
        # repeat checkbox
        self.h1Box = OWGUI.indentedBox(self.sBox, sep=indent,
                                       orientation="horizontal")
        OWGUI.checkBox(self.h1Box, self, 'Repeat', 'With replacement',
                       callback=self.settingsChanged)

        # specified number of elements checkbox
        self.h2Box = OWGUI.indentedBox(self.sBox, sep=indent,
                                       orientation="horizontal")
        check, _ = OWGUI.checkWithSpin(
            self.h2Box, self, 'Sample size (instances):', 1, 1000000000,
            'useCases', 'nCases',
            checkCallback=self.settingsChanged,
            spinCallback=self.settingsChanged
        )

        # percentage slider
        self.h3Box = OWGUI.indentedBox(self.sBox, sep=indent)
        OWGUI.widgetLabel(self.h3Box, "Sample size:")

        self.slidebox = OWGUI.widgetBox(self.h3Box, orientation="horizontal")
        OWGUI.hSlider(self.slidebox, self, 'selPercentage',
                      minValue=1, maxValue=100, step=1, ticks=10,
                      labelFormat="   %d%%",
                      callback=self.settingsChanged)

        # Sample size (instances) check disables the Percentage slider.
        # TODO: Should be an exclusive option (radio buttons)
        check.disables.extend([(-1, self.h3Box)])
        check.makeConsistent()

        # Cross Validation sampling options
        self.s[1] = OWGUI.appendRadioButton(self.sBox, self, "SelectType",
                                            "Cross validation")

        box = OWGUI.indentedBox(self.sBox, sep=indent,
                                orientation="horizontal")
        OWGUI.spin(box, self, 'CVFolds', 2, 100, step=1,
                   label='Number of folds:  ',
                   callback=self.settingsChanged)

        # Leave-One-Out
        self.s[2] = OWGUI.appendRadioButton(self.sBox, self, "SelectType",
                                            "Leave-one-out")

        # Multiple Groups
        self.s[3] = OWGUI.appendRadioButton(self.sBox, self, "SelectType",
                                            'Multiple subsets')
        gbox = OWGUI.indentedBox(self.sBox, sep=indent,
                                 orientation="horizontal")
        OWGUI.lineEdit(gbox, self, 'GroupText',
                       label='Subset sizes (e.g. "0.1, 0.2, 0.5"):',
                       callback=self.multipleChanged)

        # Output Group Box
        box = OWGUI.widgetBox(self.controlArea, 'Output Data for Fold / Group',
                              addSpace=True)
        self.foldcombo = OWGUI.comboBox(
            box, self, "outFold", items=range(1, 101),
            label='Fold / group:', orientation="horizontal",
            sendSelectedValue=1, valueType=int,
            callback=self.invalidate
        )
        self.foldcombo.setEnabled(self.SelectType != 0)

        # Sample Data box
        OWGUI.rubber(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Sample Data")
        cb = OWGUI.checkBox(box, self, "autocommit", "Sample on any change")
        self.sampleButton = OWGUI.button(box, self, 'Sample &Data',
                                         callback=self.sdata, default=True)
        OWGUI.setStopper(self, self.sampleButton, cb, "outputInvalidateFlag",
                         callback=self.sdata)

        # set initial radio button on (default sample type)
        self.s[self.SelectType].setChecked(True)

        # Connect radio buttons (SelectType)
        for i, button in enumerate(self.s):
            button.toggled[bool].connect(
                lambda state, i=i: self.samplingTypeChanged(state, i)
            )

        self.process()

        self.resize(200, 275)

    # CONNECTION TRIGGER AND GUI ROUTINES
    # enables RadioButton switching
    def samplingTypeChanged(self, value, i):
        """Sampling type changed."""
        self.SelectType = i
        self.settingsChanged()

    def multipleChanged(self):
        """Multiple subsets (Groups) changed."""
        self.error(1)
        try:
            self.pGroups = [float(x) for x in self.GroupText.split(',')]
            self.nGroups = len(self.pGroups)
        except:
            self.error(1, "Invalid specification for sizes of subsets.")
        else:
            self.settingsChanged()

    def updateFoldCombo(self):
        """Update the 'Folds' combo box contents."""
        fold = self.outFold
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
            self.foldcombo.addItem(str(x + 1))
        self.outFold = min(fold, self.Folds)

    def setData(self, dataset):
        """Set the input data set."""
        self.closeContext()
        if dataset is not None:
            self.infoa.setText('%d instances in input data set.' %
                               len(dataset))
            self.data = dataset
            self.openContext("", dataset)
            self.process()
            self.sdata()
        else:
            self.infoa.setText('No data on input.')
            self.infob.setText('')
            self.infoc.setText('')
            self.send("Data Sample", None)
            self.send("Remaining Data", None)
            self.data = None

    # feeds the output stream
    def sdata(self):
        if not self.data:
            return

        # select data
        if self.SelectType == 0:
            if self.useCases == 1 and self.Repeat == 1:
                indices = self.indices(self.data)
                sample = [self.data[i] for i in indices]
                sample = Orange.data.Table(self.data.domain, sample)
                remainder = None
            else:
                indices = self.indices(self.data)
                sample = self.data.select(indices, 0)
                remainder = self.data.select(indices, 1)
            self.infoc.setText('Output: %d instances.' % len(sample))
        elif self.SelectType == 3:
            indices = self.indices(self.data, p0=self.pGroups[self.outFold - 1])
            sample = self.data.select(indices, 0)
            remainder = self.data.select(indices, 1)
            self.infoc.setText(
                'Output: subset %(fold)d of %(folds)d, %(len)d instance(s).' %
                {"fold": self.outFold, "folds": self.Folds, "len": len(sample)}
            )
        else:
            # CV/LOO
            indices = self.indices(self.data)
            sample = self.data.select(indices, self.outFold - 1)
            remainder = self.data.select(indices, self.outFold - 1, negate=1)
            self.infoc.setText(
                'Output: fold %(fold)d of %(folds)d, %(len)d instance(s).' %
                {"fold": self.outFold, "folds": self.Folds, "len": len(sample)}
            )

        if sample is not None:
            sample.name = self.data.name
        if remainder is not None:
            remainder.name = self.data.name

        # send data
        self.nSample = len(sample)
        self.nRemainder = len(remainder) if remainder is not None else 0
        self.send("Data Sample", sample)
        self.send("Remaining Data", remainder)

        self.outputInvalidateFlag = False

    def process(self):
        self.error(0)
        self.warning(0)

        self.infob.setText('')

        if self.SelectType == 0:
            # Random Selection
            if self.useCases == 1:
                ncases = self.nCases
                if self.Repeat == 0:
                    ncases = self.nCases
                    if self.data is not None and ncases > len(self.data):
                        self.warning(0, "Sample size (w/o repetitions) larger than dataset.")
                        ncases = len(self.data)
                    p0 = ncases + 1e-7 if ncases == 1 else ncases
                    self.indices = sample.SubsetIndices2(p0=p0)
                    self.infob.setText('Random sampling, using exactly %d instances.' % ncases)
                else:
                    p0 = ncases + 1e-7 if ncases == 1 else ncases
                    self.indices = sample.SubsetIndicesMultiple(p0=p0)
                    self.infob.setText('Random sampling with repetitions, %d instances.' % ncases)
            else:
                if self.selPercentage == 100:
                    p0 = len(self.data) if self.data is not None else 1.0
                else:
                    p0 = float(self.selPercentage) / 100.0
                self.indices = sample.SubsetIndices2(p0=p0)
                self.infob.setText('Random sampling, %d%% of input instances.' % self.selPercentage)
            if self.Stratified == 1:
                self.indices.stratified = self.indices.StratifiedIfPossible
            else:
                self.indices.stratified = self.indices.NotStratified
            if self.UseSpecificSeed == 1:
                self.indices.randseed = self.RandomSeed
            else:
                self.indices.randomGenerator = Orange.misc.Random(random.randint(0,65536))

        # Cross Validation / LOO
        elif self.SelectType == 1 or self.SelectType == 2:
            # apply selected options
            if self.SelectType == 2:
                folds = len(self.data) if self.data is not None else 1
                self.infob.setText('Leave-one-out.')
            else:
                folds = self.CVFolds
                self.infob.setText('%d-fold cross validation.' % self.CVFolds)
            self.indices = sample.SubsetIndicesCV(folds=folds)
            if self.Stratified == 1:
                self.indices.stratified = self.indices.StratifiedIfPossible
            else:
                self.indices.stratified = self.indices.NotStratified
            if self.UseSpecificSeed == 1:
                self.indices.randseed = self.RandomSeed
            else:
                self.indices.randseed = random.randint(0, 65536)

        # MultiGroup
        elif self.SelectType == 3:
            self.infob.setText('Multiple subsets.')
            #prepare indices generator
            self.indices = sample.SubsetIndices2()
            if self.Stratified == 1:
                self.indices.stratified = self.indices.StratifiedIfPossible
            else:
                self.indices.stratified = self.indices.NotStratified
            if self.UseSpecificSeed == 1:
                self.indices.randseed = self.RandomSeed
            else:
                self.indices.randomGenerator = Orange.misc.Random(random.randint(0,65536))

    def settingsChanged(self):
        # enable fold selection and fill combobox if applicable
        if self.SelectType == 0:
            self.foldcombo.setEnabled(False)
        else:
            self.foldcombo.setEnabled(True)
            self.updateFoldCombo()

        self.process()
        self.invalidate()

    def invalidate(self):
        """Invalidate current output."""
        self.infoc.setText('...')
        if self.autocommit:
            self.sdata()
        else:
            self.outputInvalidateFlag = True

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
        self.reportSettings("Settings", [("Sampling type", stype), 
                                         ("Stratification", OWGUI.YesNo[self.Stratified]),
                                         ("Random seed", str(self.RandomSeed) if self.UseSpecificSeed else "auto")])
        if self.data is not None:
            self.reportSettings("Data", [("Input", "%i examples" % len(self.data)), 
                                         ("Sample", "%i examples" % self.nSample), 
                                         ("Rest", "%i examples" % self.nRemainder)])
        else:
            self.reportSettings("Data", [("Input", "None")])


if __name__ == "__main__":
    appl = QApplication(sys.argv)
    ow = OWDataSampler()
    data = Orange.data.Table('iris.tab')
    ow.setData(data)
    ow.show()
    appl.exec_()
    ow.saveSettings()
