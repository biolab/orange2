#
# OWPolyvizOptions.py
#
# options dialog for Polyviz
#

from OWOptions import *
from OWTools import *

class OWPolyvizOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "Polyviz Options", "OrangeWidgetsIcon.png", parent, name)

        self.gSetCanvasColor = QColor(Qt.white) 

        # point width
        widthBox = QHGroupBox("Point Width", self.top)
        QToolTip.add(widthBox, "The width of points")
        self.widthSlider = QSlider(2, 19, 1, 3, QSlider.Horizontal, widthBox)
        self.widthSlider.setTickmarks(QSlider.Below)
        self.widthLCD = QLCDNumber(2, widthBox)

        # scale point position
        self.positionBox = QHGroupBox("Point scaling", self.top)
        self.scaleBox= QHBox(self.positionBox, "scale")
        self.scaleLabel = QLabel("Scale point position by: ", self.scaleBox)
        self.scaleCombo = QComboBox(self.scaleBox)

        # line length
        lengthBox = QHGroupBox("Line Length", self.top)
        QToolTip.add(widthBox, "The length of the line")
        self.lengthSlider = QSlider(1, 5, 1, 2, QSlider.Horizontal, lengthBox)
        self.lengthSlider.setTickmarks(QSlider.Below)
        self.lengthLCD = QLCDNumber(1, lengthBox)

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
        # jittering size
        self.jitteringOptionsBG = QVButtonGroup("Jittering options for discrete attribute", self.top)
        QToolTip.add(self.jitteringOptionsBG, "Percents of a discrete value to be jittered")
        self.hbox = QHBox(self.jitteringOptionsBG, "jittering size")
        self.jitterLabel = QLabel('Jittering size (%)', self.hbox)
        self.jitterSize = QComboBox(self.hbox)

        ######
        # general graph settings
        self.graphSettingsBG = QVButtonGroup("General graph settings", self.top)
        self.useEnhancedTooltips = QCheckBox("Use enhanced tooltips", self.graphSettingsBG)
        self.globalValueScaling  = QCheckBox("Use global value scaling", self.graphSettingsBG)
        self.showFilledSymbols   = QCheckBox('Show filled symbols', self.graphSettingsBG)
        self.showLegend = QCheckBox('Show legend', self.graphSettingsBG)

        # continuous attribute selection
        self.attrContButtons = QVButtonGroup("Continuous attribute selection", self.top)
        QToolTip.add(self.attrContButtons, "Select the measure for measuring importance of continuous attributes")
        self.attrContButtons.setExclusive(TRUE)
        
        self.attrContNone = QRadioButton('None', self.attrContButtons)
        self.attrContRelieF = QRadioButton('RelieF', self.attrContButtons)

        #####
        # discrete attribute selection
        self.attrDiscButtons = QVButtonGroup("Discrete attribute selection", self.top)
        QToolTip.add(self.attrDiscButtons, "Select the measure for measuring importance of discrete attributes")
        self.attrDiscButtons.setExclusive(TRUE)

        self.attrDiscNone = QRadioButton('None', self.attrDiscButtons)
        self.attrDiscRelieF = QRadioButton('RelieF', self.attrDiscButtons)
        self.attrDiscGainRatio = QRadioButton('GainRatio', self.attrDiscButtons)
        self.attrDiscGini = QRadioButton('Gini', self.attrDiscButtons)
        self.attrDiscFD   = QRadioButton('Oblivious decision graphs', self.attrDiscButtons)

        #####
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.top)
        self.connect(self.widthSlider, SIGNAL("valueChanged(int)"), self.widthLCD, SLOT("display(int)"))
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(self.gSetCanvasColor)
        if newColor.isValid():
            self.gSetCanvasColor = newColor
            self.emit(PYSIGNAL("canvasColorChange(QColor &)"),(QColor(newColor),))

if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWPolyvizOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()