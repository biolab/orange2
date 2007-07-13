"""
<name>Word n-Grams</name>
<description>Computes the word ngram representation.</description>
<icon>icons/WordNgram.png</icon>
<contact>Sasa Petrovic</contact> 
<priority>1410</priority>
"""

from OWWidget import *
import OWGUI, orngText

class OWWordNgram(OWWidget):
    settingsList = ["recentFiles"]

    def __init__(self, parent=None, signalManager=None, name = "WordNgram"):
        OWWidget.__init__(self,parent,signalManager,name)
        self.callbackDeposit = []
        self.inputs = [("Example Table", ExampleTable, self.dataset)]
        self.outputs = [("Example Table", ExampleTable)]

        self.recentFiles=[]
        self.fileIndex = 0
        self.loadSettings()
        self.stopwords = None
        self.size = 0
        self.measure = 0
        self.threshold = 0
        self.data = None
        self.measureDict = {0: 'FREQ', 1: 'MI', 2: 'DICE', 3: 'CHI', 4: 'LL'}

        #GUI        
        optionBox = QHGroupBox('', self.controlArea)
        OWGUI.radioButtonsInBox(optionBox, self, "size", box = "No. of words", btnLabels = ["2", "3", "4", "Named entities"], addSpace = True, callback = self.radioChanged)
        self.ambox = OWGUI.radioButtonsInBox(optionBox, self, "measure", box = "Association measure", btnLabels = ["Frequency", "Mutual information", "Dice coefficient", "Chi square", "Log likelihood"], addSpace = True)
        self.ambox.setEnabled(self.size - 3)
        box = QVGroupBox('', optionBox)
        OWGUI.lineEdit(box, self, "threshold", orientation="horizontal", valueType=float, box="Threshold")

        stopbox = OWGUI.widgetBox(box, "Stopwords File")
        stophbox = OWGUI.widgetBox(stopbox, orientation = 1)
        self.filecombo = OWGUI.comboBox(stophbox, self, "fileIndex", callback = self.loadFile)
        OWGUI.button(stophbox, self, '...', callback = self.browseFile)
        OWGUI.button(self.controlArea, self, "Apply", self.apply)
        self.lblFeatureNo = QLabel("\nNo. of features: ", self.controlArea)
        self.adjustSize()

        if self.recentFiles:
            self.loadFile()

    def radioChanged(self):
        if self.size == 3:
            self.ambox.setEnabled(False)
        else:
            self.ambox.setEnabled(True)


    def browseFile(self):
        if self.recentFiles:
            lastPath = os.path.split(self.recentFiles[0])[0]
        else:
            lastPath = "."

        fn = str(QFileDialog.getOpenFileName(lastPath, "Text files (*.*)", None, "Open Text Files"))
        if not fn:
            return
        
        fn = os.path.abspath(fn)
        if fn in self.recentFiles: # if already in list, remove it
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0, fn)
        self.fileIndex = 0
        self.loadFile()


    def loadFile(self):
        if self.fileIndex:
            fn = self.recentFiles[self.fileIndex]
            self.recentFiles.remove(fn)
            self.recentFiles.insert(0, fn)
            self.fileIndex = 0
        else:
            fn = self.recentFiles[0]

        self.filecombo.clear()
        for file in self.recentFiles:
            self.filecombo.insertItem(os.path.split(file)[1])
        self.filecombo.updateGeometry()

        self.error()
        try:
            self.stopwords = orngText.loadWordSet(fn)
        except:
            self.error("Cannot read the file")
        

    def dataset(self, data):
        if data:
            self.data = orange.ExampleTable(orange.Domain(data.domain), data)
            self.tmpData = orange.ExampleTable(data)
            self.tmpDom = orange.Domain(data.domain)            
            
            self.data.domain = orange.Domain(data.domain)
            #self.apply()

    def apply(self):
        if self.data:
            self.data = orange.ExampleTable(orange.Domain(self.tmpDom), self.tmpData)
            if self.size == 3:
                newdata = orngText.extractNamedEntities(self.data, stopwords = self.stopwords)
            else:
                newdata = orngText.extractWordNGram(self.data, n = self.size + 2, stopwords = self.stopwords, threshold = self.threshold, measure = self.measureDict[self.measure])
            self.lblFeatureNo.setText("\nNo. of features: \n%d" % len(newdata.domain.getmetas()))
            self.send("Example Table", newdata)
        else:
            self.send("Example Table", None)


            

if __name__ == "__main__":
    t = orngText.loadFromXML(r'c:\test\msnbc.xml')
    a = QApplication(sys.argv)
    ow = OWWordNgram()
    ow.data = t
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()        
