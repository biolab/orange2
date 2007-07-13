"""
<name>Feature Selection</name>
<description>Removes features from the data according to given criteria.</description>
<icon>icons/TextFeatureSelection.png</icon>
<contact>Sasa Petrovic</contact> 
<priority>1500</priority>
"""

from OWWidget import *
from copy import deepcopy
import OWGUI, orngText, warnings

class OWTextFeatureSelection(OWWidget):    

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"FeatureSelection")
        self.inputs = [("Example Table", ExampleTable, self.dataset)]
        self.outputs = [("Example Table", ExampleTable)]

        warnings.filterwarnings("ignore", "", orange.AttributeWarning)
        
        self.data = None
        self.chosenMeasure = [0]
        self.measures = ['TF', 'RAND', 'TDF', 'WF', 'NF']
        self.chosenOp = [0]
        self.operators = ['MIN', 'MAX']
        self.tmpData = None
        self.perc = 1
        self.threshold = 90
        self.selections = []

        #GUI
        #ca=QFrame(self.controlArea)
        #gl=QGridLayout(ca)
        selectionbox = OWGUI.QHGroupBox('Feature selection', self.controlArea)

        OWGUI.listBox(selectionbox, self, 'chosenMeasure', 'measures', box = 'Select measure', callback = None)
        OWGUI.listBox(selectionbox, self, 'chosenOp', 'operators', box = 'Select operator', callback = None)

        boxAttrStat = QVGroupBox("Statistics for features", self.controlArea)
        self.lblFeatNo = QLabel("No. of features: ", boxAttrStat)
        self.lblMin = QLabel("Min: ", boxAttrStat)
        self.lblAvg = QLabel("Avg: ", boxAttrStat)
        self.lblMax = QLabel("Max: ", boxAttrStat)

        boxDocStat = QVGroupBox("Statistics for documents", self.controlArea)
        self.lblDocNo = QLabel("No. of documents: ", boxDocStat)
        self.lblDocAvg = QLabel("Avg: ", boxDocStat)
        self.lblDocMax = QLabel("Max: ", boxDocStat)
        self.lblDocMin = QLabel("Min: ", boxDocStat)

        optionBox = OWGUI.QVGroupBox('', selectionbox)        

        OWGUI.button(optionBox, self, "Apply", self.apply)
        OWGUI.button(optionBox, self, "Reset", self.reset)
        OWGUI.checkBox(optionBox, self, "perc", "percentage")
        #OWGUI.spin(optionBox, self, "threshold", 0, 10000, label="Threshold:", callback = None)
        OWGUI.lineEdit(optionBox, self, "threshold", orientation="horizontal", valueType=float, box="Threshold")
        
        legendBox = QVGroupBox("Legend", self.controlArea)
        OWGUI.label(legendBox, self, "TF = Term frequency -> the number of times a feature appears in the corpus")
        OWGUI.label(legendBox, self, "RAND = Random -> randomly selects the features to keep")
        OWGUI.label(legendBox, self, "TDF = Term document frequency -> the number of documents in which a feature appears")
        OWGUI.label(legendBox, self, "WF = Word frequency -> the total number of words in a document")
        OWGUI.label(legendBox, self, "NF = Number of features -> the number of different features in a document")

        self.controlArea.adjustSize()


    def apply(self):
        if self.measures[self.chosenMeasure[0]] == 'WF' or self.measures[self.chosenMeasure[0]] == 'NF':
            self.data = orngText.DSS(self.data, self.measures[self.chosenMeasure[0]], self.operators[self.chosenOp[0]], self.threshold)
        else:
            self.data = orngText.FSS(self.data, self.measures[self.chosenMeasure[0]], self.operators[self.chosenOp[0]], self.threshold, self.perc)
        self.selections.append(self.measures[self.chosenMeasure[0]] + ' ' + self.operators[self.chosenOp[0]] + ' ' + str(self.threshold) + ' percentage=' + str(self.perc))
        self.data.selection = deepcopy(self.selections)
        self.send("Example Table", self.data)
        self.computeStatistics()

    def dataset(self, data):
        if data:
            self.selections = []
            self.data = orange.ExampleTable(data)
            self.tmpData = orange.ExampleTable(data)
            self.tmpDom = orange.Domain(data.domain)
            self.computeStatistics()
        else:
            self.data = None
            
    def reset(self):
        self.selections = []
        self.data = orange.ExampleTable(orange.Domain(self.tmpDom), self.tmpData)
        self.send("Example Table", self.data)
        self.computeStatistics()

    def computeStatistics(self):
        docNo = len(self.data)
        print len(self.data.domain.getmetas())
        if not docNo:
            docNo = 1
        self.lblDocNo.setText("No. of documents: %d" % docNo)
        #compute document statistics        
        #max = min = len(self.data[0].getmetas())
        max = 0
        min = ()
        sum = 0
        for doc in self.data:
            featNo = len(doc.getmetas())
            sum += featNo
            if featNo > max:
                max = featNo
            if featNo < min:
                min = featNo
        avg = sum / docNo
        if min == ():
            min = 0
        self.lblDocAvg.setText("Avg: %.3f" % avg)
        self.lblDocMax.setText("Max: %d" % max)
        self.lblDocMin.setText("Min: %d" % min)

        #compute feature statistics
        words = {}
        sum = 0
        if not self.data.domain.getmetas() or not self.data:
          self.lblFeatNo.setText("No. of features: 0")
          self.lblMin.setText("Min: %d  Min word = 0")
          self.lblMax.setText("Max: %d  Max word = 0")
          self.lblAvg.setText("Avg: 0.0")
          return
        max = 0
        min = ()
        maxword = minword = ''
        
        for ex in self.data:
           for v in ex.getmetas().values():
              varname = v.variable.name
              if words.has_key(varname):
                 words[varname] += v.value
              else:
                 words[varname] = v.value
        for word,freq in words.items():
            if freq > max:
                max = freq
                maxword = word
            if freq < min:
                min = freq
                minword = word
            sum += freq
        avg = sum / len(words)
        if min == ():
            min = 0
        self.lblFeatNo.setText("No. of features: %d, %d" % (len(words), len(self.data.domain.getmetas())))
        self.lblMin.setText("Min: %d  Min word = %s" % (min, minword))
        self.lblMax.setText("Max: %d  Max word = %s" % (max,maxword))
        self.lblAvg.setText("Avg: %.3f" % avg)


if __name__ == "__main__":
    a = QApplication(sys.argv)
    t = orngText.loadFromXML(r'c:\test\orange\msnbc.xml')
    t2 = orngText.extractLetterNGram(t)
    #print t2.domain.getmetas().values()
    ow = OWTextFeatureSelection()
    a.setMainWidget(ow)
    ow.show()
    ow.dataset(t2)
    a.exec_loop()
