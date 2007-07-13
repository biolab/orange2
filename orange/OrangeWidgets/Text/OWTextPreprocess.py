"""
<name>Preprocess</name>
<description>Lower case, tokenizer and lematizer for text.</description>
<icon>icons/TextPreprocess.png</icon>
<contact></contact> 
<priority>1200</priority>
"""

from OWWidget import *
import orngText
import OWGUI

class OWTextPreprocess(OWWidget):
    settingsList=["lowerCase", "stopWords", "lematizer"]

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"Preprocess")
        self.langDict = {0: 'en', 1: 'hr', 2: 'fr'}
        self.selectedLanguage = 0
        #OWWidget.__init__(self,parent,"Rules")
        self.inputs = [("Example Table", ExampleTable, self.dataset)]
        self.outputs = [("Example Table", ExampleTable)]

        self.lowerCase = self.stopWords = self.lematizer = True
        self.textAttribute = "-"
        self.nDocuments = "-"
        self.data = None
        
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Options", addSpace = True)
        OWGUI.checkBox(box, self, "lowerCase", "Convert to lower case")
        OWGUI.checkBox(box, self, "stopWords", "Remove stop words")
        OWGUI.checkBox(box, self, "lematizer", "Lematize")
        OWGUI.radioButtonsInBox(self.controlArea, self, "selectedLanguage", box = "Language", btnLabels = ["en", "hr", "fr"], addSpace = True)

        box = OWGUI.widgetBox(self.controlArea, "Info", addSpace = True)
        OWGUI.label(box, self, "Number of documents: %(nDocuments)s")
        OWGUI.label(box, self, "Text attribute: %(textAttribute)s")

        OWGUI.button(self.controlArea, self, "Apply", self.apply)

        self.adjustSize()        
        

    def dataset(self, data):
        if data:
            for i in range(1, len(data.domain.attributes)+1):
                if data.domain.attributes[-i]:
                    self.textAttributePos = len(data.domain.attributes) - i
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
        if self.data:
            newData = orange.ExampleTable(orange.Domain(self.data.domain), self.data)
            preprocess = orngText.Preprocess(language = self.langDict[self.selectedLanguage])
            if self.lowerCase:
                import string
                newData = preprocess.doOnExampleTable(newData, self.textAttributePos, string.lower)
            if self.stopWords:
                newData = preprocess.removeStopwordsFromExampleTable(newData, self.textAttributePos)
            if self.lematizer:
                newData = preprocess.lemmatizeExampleTable(newData, self.textAttributePos)
        else:
            newData = None
        self.send("Example Table", newData)


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWTextPreprocess()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
