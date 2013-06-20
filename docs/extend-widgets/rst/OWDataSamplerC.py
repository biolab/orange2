"""
<name>Data Sampler (C)</name>
<description>Randomly selects a subset of instances from the data set</description>
<icon>icons/DataSamplerC.svg</icon>
<priority>30</priority>
"""
import Orange

from OWWidget import *
import OWGUI

class OWDataSamplerC(OWWidget):
    settingsList = ['proportion', 'commitOnChange']
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'SampleDataC')
        
        self.inputs = [("Data", Orange.data.Table, self.data)]
        self.outputs = [("Sampled Data", Orange.data.Table),
                        ("Other Data", Orange.data.Table)]

        self.proportion = 50
        self.commitOnChange = 0
        self.loadSettings()

        # GUI
        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.infoa = OWGUI.widgetLabel(box, 'No data on input yet, waiting to get something.')
        self.infob = OWGUI.widgetLabel(box, '')

        OWGUI.separator(self.controlArea)
        self.optionsBox = OWGUI.widgetBox(self.controlArea, "Options")
        OWGUI.spin(self.optionsBox, self, 'proportion', min=10, max=90, step=10,
                   label='Sample Size [%]:', callback=[self.selection, self.checkCommit])
        OWGUI.checkBox(self.optionsBox, self, 'commitOnChange', 'Commit data on selection change')
        OWGUI.button(self.optionsBox, self, "Commit", callback=self.commit)
        self.optionsBox.setDisabled(1)

        self.resize(100,50)

    def data(self, dataset):
        if dataset:
            self.dataset = dataset
            self.infoa.setText('%d instances in input data set' % len(dataset))
            self.optionsBox.setDisabled(0)
            self.selection()
            self.commit()
        else:
            self.send("Sampled Data", None)
            self.optionsBox.setDisabled(1)
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')

    def selection(self):
        indices = Orange.data.sample.SubsetIndices2(p0=self.proportion / 100.)
        ind = indices(self.dataset)
        self.sample = self.dataset.select(ind, 0)
        self.otherdata = self.dataset.select(ind, 1)
        self.infob.setText('%d sampled instances' % len(self.sample))

    def commit(self):
        self.send("Sampled Data", self.sample)
        self.send("Other Data", self.otherdata)

    def checkCommit(self):
        if self.commitOnChange:
            self.commit()


if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWDataSamplerC()
    ow.show()
    dataset = Orange.data.Table('iris.tab')
    ow.data(dataset)
    appl.exec_()
