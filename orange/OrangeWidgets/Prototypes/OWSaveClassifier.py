"""<name>Save Classifier</name>
<description>Save orange classifiers to a file</description>
<icon>icons/SaveClassifier.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>3000</priority>
"""

from OWWidget import *
import OWGUI
import orange
import sys, os


class OWSaveClassifier(OWWidget):
    settingsList = ["lastSaveFile", "filenameHistory"]
    def __init__(self, parent=None, signalManager=None, name="Save Classifier"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea=False)
        
        self.inputs = [("Classifier", orange.Classifier, self.setClassifier)]
        
        self.lastSaveFile = os.path.expanduser("~/orange_classifier.pck")
        self.filenameHistory = []
        self.selectedFileIndex = 0
        
        self.loadSettings()
        
        #####
        # GUI
        #####
        box = OWGUI.widgetBox(self.controlArea, "File",
                              orientation="horizontal",
                              addSpace=True)
        
        self.filesCombo = OWGUI.comboBox(box, self, "selectedFileIndex",
                                         items=[os.path.basename(f) for f in self.filenameHistory],
                                         tooltip="Select a recently saved file",
                                         callback=self.onRecentSelection)
        
        self.browseButton = OWGUI.button(box, self, "...",
                                         tooltip="Browse local file system",
                                         callback=self.browse)
        
        self.browseButton.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.browseButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
         
        box = OWGUI.widgetBox(self.controlArea, "Save")
        self.saveButton = OWGUI.button(box, self, "Save current classifier",
                                       callback=self.saveCurrentClassifier)
        
        self.saveButton.setEnabled(False)
        
        OWGUI.rubber(self.controlArea)
        
        self.resize(200, 100)
        
        self.classifier = None
        
        
    def onRecentSelection(self):
        filename = self.filenameHistory[self.selectedFileIndex]
        self.filenameHistory.pop(self.selectedFileIndex)
        self.filenameHistory.insert(0, filename)
        self.filesCombo.removeItem(self.selectedFileIndex)
        self.filesCombo.insertItem(0, os.path.basename(filename))
        self.selectedFileIndex = 0
    
    
    def browse(self):
        filename = QFileDialog.getSaveFileName(self, "Save Classifier As ...",
                    self.lastSaveFile, "Pickle files (*.pickle *.pck);; All files (*.*)")
        filename = str(filename)
        if filename:
            if filename in self.filenameHistory:
                self.selectedFileIndex = self.filenameHistory.index(filename)
                self.onRecentSelection()
                return
            
            self.lastSaveFile = filename
            self.filenameHistory.insert(0, filename)
            self.filesCombo.insertItem(0, os.path.basename(filename))
            self.filesCombo.setCurrentIndex(0)
            self.saveButton.setEnabled(self.classifier is not None and bool(self.filenameHistory))
    
            
    def saveCurrentClassifier(self):
        if self.classifier is not None:
            filename = self.filenameHistory[self.selectedFileIndex]
            import cPickle
            self.error(0)
            try:
                cPickle.dump(self.classifier, open(filename, "wb"))
            except Exception, ex:
                self.error(0, "Could not save classifier! %s" % str(ex))
            
            
    def setClassifier(self, classifier=None):
        self.classifier = classifier
        self.saveButton.setEnabled(classifier is not None and bool(self.filenameHistory))
        
    
if __name__ == "__main__":
    app = QApplication([])
    w = OWSaveClassifier()
    import orngTree
    data = orange.ExampleTable("../../doc/datasets/iris.tab")
    w.setClassifier(orngTree.TreeLearner(data))
    w.show()
    app.exec_()
    w.saveSettings()
    