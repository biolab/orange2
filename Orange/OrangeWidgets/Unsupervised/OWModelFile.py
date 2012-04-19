"""
<name>Model File</name>
<description>Load a Model Map</description>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact>
<icon>icons/DistanceFile.png</icon>
<priority>6510</priority>
"""

from OWWidget import *

import OWGUI
import Orange
import orngMisc
import exceptions
import os.path
import cPickle as pickle
import bz2

class OWModelFile(OWWidget):
    settingsList = ["files", "file_index"]

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, name='Model File', wantMainArea=0, resizingEnabled=1)

        self.outputs = [("Distances", Orange.misc.SymMatrix),
                        ("Model Meta-data", Orange.data.Table),
                        ("Original Data", Orange.data.Table)]

        #self.dataFileBox.setTitle("Model File")
        self.files = []
        self.file_index = 0

        self.matrix = None
        self.model_data = None
        self.original_data = None

        self.loadSettings()

        self.fileBox = OWGUI.widgetBox(self.controlArea, "Model File", addSpace=True)
        hbox = OWGUI.widgetBox(self.fileBox, orientation=0)
        self.filecombo = OWGUI.comboBox(hbox, self, "file_index", callback=self.loadFile)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback=self.browseFile)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


#        Moved to SymMatrixTransform widget
#
#        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "normalizeMethod", [], "Normalize method", callback = self.setNormalizeMode)
#        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "None", callback = self.setNormalizeMode)
#        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "To interval [0,1]", callback = self.setNormalizeMode)
#        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "Sigmoid function: 1 / (1 + e^x)", callback = self.setNormalizeMode)
#        
#        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "invertMethod", [], "Invert method", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "None", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "-X", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "1 - X", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "Max - X", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "1 / X", callback = self.setInvertMode)

        OWGUI.rubber(self.controlArea)

        self.adjustSize()

        for i in range(len(self.files) - 1, -1, -1):
            if not (os.path.exists(self.files[i]) and os.path.isfile(self.files[i])):
                del self.files[i]

        if self.files:
            self.loadFile()

    def browseFile(self):
        if self.files:
            lastPath = os.path.split(self.files[0])[0]
        else:
            lastPath = "."
        fn = unicode(QFileDialog.getOpenFileName(self, "Open Model Map File",
                                             lastPath, "Model Map (*.bz2)"))
        fn = os.path.abspath(fn)
        if fn in self.files: # if already in list, remove it
            self.files.remove(fn)
        self.files.insert(0, fn)
        self.file_index = 0
        self.loadFile()

    def loadFile(self):
        if self.file_index:
            fn = self.files[self.file_index]
            self.files.remove(fn)
            self.files.insert(0, fn)
            self.file_index = 0
        else:
            fn = self.files[0]

        self.filecombo.clear()
        for file in self.files:
            self.filecombo.addItem(os.path.split(file)[1])
        #self.filecombo.updateGeometry()

        self.matrix = None
        self.model_data = None
        self.original_data = None
        pb = OWGUI.ProgressBar(self, 100)

        self.error()
        try:
            matrix, self.model_data, self.original_data = pickle.load(bz2.BZ2File('%s' % fn, "r"))

            self.matrix = Orange.misc.SymMatrix(len(matrix))
            milestones = orngMisc.progressBarMilestones(self.matrix.dim, 100)
            for i in range(self.matrix.dim):
                for j in range(i + 1):
                    self.matrix[j, i] = matrix[i, j]

                if i in milestones:
                    pb.advance()
            pb.finish()

        except Exception, ex:
            self.error("Error while reading the file: '%s'" % str(ex))
            return
        self.relabel()

    def relabel(self):
        self.error()
        if self.matrix is not None:
            if self.model_data is not None and self.matrix.dim == len(self.model_data):
                self.matrix.setattr("items", self.model_data)
            else:
                self.error("The number of model doesn't match the matrix dimension. Invalid Model Map file.")

            if self.original_data is not None:
                self.matrix.setattr("original_data", self.original_data)

            self.send("Distances", self.matrix)

        if self.model_data is not None:
            self.send("Model Meta-data", self.model_data)

        if self.original_data is not None:
            self.send("Original Data", self.original_data)

if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWModelFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
