"""
<name>Bag of Words</name>
<description>Computes bag of words from text files and optionally also normalizes them.</description>
<icon>icons/BagOfWords.png</icon>
<contact></contact> 
<priority>1300</priority>
"""

from OWWidget import *
import OWGUI, orngText

class OWBagofWords(OWWidget):
    settingsList=["TFIDF", "norm"]

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"BagofWords")
        self.inputs = [("Example Table", ExampleTable, self.dataset)]
        self.outputs = [("Example Table", ExampleTable)]

        self.TFIDF = 0
        self.norm = 0
        self.textAttribute = "-"
        self.nDocuments = "-"
        self.data = None
        
        self.loadSettings()

        OWGUI.radioButtonsInBox(self.controlArea, self, "TFIDF", ["None", "log(1/f)"], "TFIDF", addSpace = True)
        OWGUI.radioButtonsInBox(self.controlArea, self, "norm", ["None", "L1 (Sum of elements)", "L2 (Euclidean)"], "Normalization", addSpace = True)

        box = OWGUI.widgetBox(self.controlArea, "Info", addSpace = True)
        OWGUI.label(box, self, "Number of documents: %(nDocuments)s")
        OWGUI.label(box, self, "Text attribute: %(textAttribute)s")

        OWGUI.button(self.controlArea, self, "Apply", self.apply)

        self.adjustSize()        
        

    def dataset(self, data):
        if data:
            for i in range(1, len(data.domain.attributes)+1):
                if data.domain.attributes[-i]:
                    self.textAttribute = data.domain.attributes[-i].name
                    self.nDocuments = len(data)
                    self.data = data
                    self.error()
                    break
            else:
                self.error("The data has no string attributes")
                self.textAttribute = "-"
                self.nDocuments = "-"
                self.data = None

        self.apply()

    def apply(self):
        self.send("Example Table", None)        
        if self.data:
            newdata = orngText.bagOfWords(self.data)
            if self.norm:
                newdata = orngText.Preprocessor_norm()(newdata, self.norm)
            if self.TFIDF:
                newdata = orngText.PreprocessorConstructor_tfidf()(newdata)(newdata)
            self.send("Example Table", newdata)
        else:
            self.send("Example Table", None)


if __name__ == "__main__":
    t = orngText.loadFromXML(r'c:\test\orange\msnbc.xml')
    a = QApplication(sys.argv)
    ow = OWBagofWords()
    ow.data = t
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
