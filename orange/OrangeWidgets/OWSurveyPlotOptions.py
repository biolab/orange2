#
# OWParallelCoordinatesOptions.py
#
# options dialog for distributions graph
#

from OWOptions import *
from OWTools import *

class OWSurveyPlotOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "Parallel Coordinate Options", "OrangeWidgetsIcon.png", parent, name)

        self.gSetCanvasColor = QColor(Qt.white)

        #####
        # attribute value scaling
        self.attrValueScalingButtons = QVButtonGroup("Attribute value scaling", self.top)
        self.globalValueScaling = QCheckBox("Global Value Scaling", self.attrValueScalingButtons)

        #####
        # visual settings
        self.visualSettingsButtons = QVButtonGroup("Visual settings", self.top)
        self.exampleTracking = QCheckBox("Enable example tracking", self.visualSettingsButtons)
        

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
        self.attrDiscFD   = QRadioButton('Oblivious decision graphs', self.attrDiscButtons)

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
    w=OWSurveyPlotOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()