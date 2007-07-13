"""
<name>Letter n-Grams</name>
<description>Computes the letter ngram representation.</description>
<icon>icons/LetterNgram.png</icon>
<contact>Sasa Petrovic</contact> 
<priority>1405</priority>
"""

from OWWidget import *
import OWGUI, orngText

class OWLetterNgram(OWWidget):

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"LetterNgram")
        self.inputs = [("Example Table", ExampleTable, self.dataset)]
        self.outputs = [("Example Table", ExampleTable)]

        self.size = 0
        self.data = None
        OWGUI.radioButtonsInBox(self.controlArea, self, "size", box = "Ngram size", btnLabels = ["2", "3", "4"], addSpace = True)
        OWGUI.button(self.controlArea, self, "Apply", self.apply)
        self.lblFeatureNo = QLabel("\nNo. of features: ", self.controlArea)
        self.adjustSize()
        

    def dataset(self, data):
        if data:
            self.data = orange.ExampleTable(data)
            self.tmpData = orange.ExampleTable(data)
            self.tmpDom = orange.Domain(data.domain)            
            #self.data.domain = orange.Domain(data.domain)
            self.apply()

    def apply(self):
        if self.data:
            self.data = orange.ExampleTable(orange.Domain(self.tmpDom), self.tmpData)
            newdata = orngText.extractLetterNGram(self.data, self.size + 2)
            self.lblFeatureNo.setText("\nNo. of features: \n%d" % len(newdata.domain.getmetas()))
            self.send("Example Table", newdata)
        else:
            self.send("Example Table", None)


if __name__ == "__main__":
    t = orngText.loadFromXML(r'c:\test\msnbc.xml')
    a = QApplication(sys.argv)
    ow = OWLetterNgram()
    ow.data = t
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
