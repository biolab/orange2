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

        ######
        # jittering options
        self.jitteringOptionsBG = QVButtonGroup("Jittering options", self.top)
        QToolTip.add(self.jitteringOptionsBG, "Percents of a discrete value to be jittered")
        self.hbox = QHBox(self.jitteringOptionsBG, "Jittering size")
        self.jitterLabel = QLabel('Jittering size (% of size)  ', self.hbox)
        self.jitterSize = QComboBox(self.hbox)

        # attribute axis options
        self.linesDistanceOptionsBG = QVButtonGroup("Attribute axis distance", self.top)
        QToolTip.add(self.linesDistanceOptionsBG, "What is the minimum distance between two adjecent attributes")
        self.hbox2 = QHBox(self.linesDistanceOptionsBG, "Minimum distance")
        self.linesLabel = QLabel('Minimum distance (pixels)  ', self.hbox2)
        self.linesDistance = QComboBox(self.hbox2)        

        #####
        # visual settings
        self.visualSettings = QVButtonGroup("Visual settings", self.top)
        self.showDistributions = QCheckBox("Show distributions", self.visualSettings)
        self.showAttrValues = QCheckBox("Show attribute values", self.visualSettings)
        self.hidePureExamples = QCheckBox("Hide pure examples", self.visualSettings)
        self.showCorrelations = QCheckBox("Show correlations between attributes", self.visualSettings)      # show correlations
        self.useSplines = QCheckBox("Show lines using splines", self.visualSettings)      # show correlations
        self.lineTracking = QCheckBox("Enable line tracking", self.visualSettings)      # show nearest line in bold

        #####
        # attribute value scaling
        self.attrValueScalingButtons = QVButtonGroup("Attribute value scaling", self.top)
        self.globalValueScaling = QCheckBox("Global Value Scaling", self.attrValueScalingButtons)

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
    w=OWParallelCoordinatesOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

    
