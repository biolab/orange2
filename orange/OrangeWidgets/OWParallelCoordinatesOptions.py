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

        #####
        # jittering
        self.spreadButtons = QVButtonGroup("Jittering type", self.top)
        QToolTip.add(self.spreadButtons, "Selected the type of jittering for discrete variables")
        self.spreadButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.spreadButtons)
        self.spreadUniform = QRadioButton('uniform', self.spreadButtons)
        self.spreadTriangle = QRadioButton('triangle', self.spreadButtons)
        self.spreadBeta = QRadioButton('beta', self.spreadButtons)

        #####
        self.showDistributions = QCheckBox("Show distributions", self.top)
        self.showAttrValues = QCheckBox("Show attribute values", self.top)
        self.hidePureExamples = QCheckBox("Hide pure examples", self.top)

        #####        
        # continuous attribute ordering
        self.attrContButtons = QVButtonGroup("Continuous attribute ordering", self.top)
        QToolTip.add(self.attrContButtons, "Select the measure for continuous attribute ordering")
        self.attrContButtons.setExclusive(TRUE)
        
        self.attrContNone = QRadioButton('None', self.attrContButtons)
        self.attrContRelieF = QRadioButton('RelieF', self.attrContButtons)
        self.attrCorrelation = QRadioButton('Correlation', self.attrContButtons)

        #####
        # discrete attribute ordering
        self.attrDiscButtons = QVButtonGroup("Discrete attribute ordering", self.top)
        QToolTip.add(self.attrDiscButtons, "Select the measure for discrete attribute ordering")
        self.attrDiscButtons.setExclusive(TRUE)

        self.attrDiscNone = QRadioButton('None', self.attrDiscButtons)
        self.attrDiscRelieF = QRadioButton('RelieF', self.attrDiscButtons)
        self.attrDiscGainRatio = QRadioButton('GainRatio', self.attrDiscButtons)
        self.attrDiscGini = QRadioButton('Gini', self.attrDiscButtons)
        self.attrDiscFD   = QRadioButton('Functional decomposition', self.attrDiscButtons)

        #####
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

    
