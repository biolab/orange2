#
# OWParallelCoordinatesOptions.py
#
# options dialog for distributions graph
#

from OWOptions import *
from OWTools import *

class OWParallelCoordinatesOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "Parallel Coordinate Options", "OrangeWidgetsIcon.png", parent, name)

        self.gSetCanvasColor = QColor(Qt.white) 

        # jittering
        self.spreadButtons = QVButtonGroup("Jittering type", self.top)
        QToolTip.add(self.spreadButtons, "Selected the type of jittering for discrete variables")
        self.spreadButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.spreadButtons)
        self.spreadUniform = QRadioButton('uniform', self.spreadButtons)
        self.spreadTriangle = QRadioButton('triangle', self.spreadButtons)
        self.spreadBeta = QRadioButton('beta', self.spreadButtons)

        self.showDistributions = QCheckBox("Show distributions", self.top)
        
        # attribute ordering
        self.attrButtons = QVButtonGroup("Attribute ordering", self.top)
        QToolTip.add(self.attrButtons, "Select the measure for attribute ordering")
        self.attrButtons.setExclusive(TRUE)
        self.attrNone = QRadioButton('None', self.attrButtons)
        self.attrRelieF = QRadioButton('RelieF', self.attrButtons)
        self.attrGainRatio = QRadioButton('GainRatio', self.attrButtons)
        self.attrGini = QRadioButton('Gini', self.attrButtons)
        self.attrCorrelation = QRadioButton('Correlation', self.attrButtons)

        self.gSetCanvasColorB = QPushButton("Canvas Color", self.top)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(self.gSetCanvasColor)
        if newColor.isValid():
            self.gSetCanvasColor = newColor
            self.emit(PYSIGNAL("canvasColorChange(QColor &)"),(QColor(newColor),))

if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWParallelCoordinatesOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

    
