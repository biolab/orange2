#
# OWScatterPlotOptions.py
#
# options dialog for ScatterPlot
#

from OWOptions import *
from OWTools import *

class OWScatterPlotOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "ScatterPlot Options", "OrangeWidgetsIcon.png", parent, name)

        self.gSetGridColor = QColor(Qt.black)
        self.gSetCanvasColor = QColor(Qt.white) 

        # point width
        widthBox = QHGroupBox("Point Width", self.top)
        QToolTip.add(widthBox, "The width of points")
        self.widthSlider = QSlider(2, 20, 1, 3, QSlider.Horizontal, widthBox)
        self.widthSlider.setTickmarks(QSlider.Below)
        self.widthLCD = QLCDNumber(2, widthBox)

        #####
        # jittering
        self.jitteringButtons = QVButtonGroup("Jittering type", self.top)
        QToolTip.add(self.jitteringButtons, "Selected the type of jittering for discrete variables")
        self.jitteringButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.jitteringButtons)
        self.spreadUniform = QRadioButton('uniform', self.jitteringButtons)
        self.spreadTriangle = QRadioButton('triangle', self.jitteringButtons)
        self.spreadBeta = QRadioButton('beta', self.jitteringButtons)

        ######
        # jittering options
        self.jitteringOptionsBG = QVButtonGroup("Jittering options", self.top)
        QToolTip.add(self.jitteringOptionsBG, "Percents of a discrete value to be jittered")
        self.hbox = QHBox(self.jitteringOptionsBG, "jittering size")
        self.jitterLabel = QLabel('Jittering size (% of size)', self.hbox)
        self.jitterSize = QComboBox(self.hbox)

        self.jitterContinuous = QCheckBox('jitter continuous attributes', self.jitteringOptionsBG)        


        #####
        self.graphSettings = QVButtonGroup("General graph settings", self.top)
        QToolTip.add(self.graphSettings, "Enable/disable main title, axis title and grid")
        self.gSetXaxisCB = QCheckBox('X axis title ', self.graphSettings)
        self.gSetYaxisCB = QCheckBox('Y axis title ', self.graphSettings)
        self.gSetVgridCB = QCheckBox('vertical gridlines', self.graphSettings)
        self.gSetHgridCB = QCheckBox('horizontal gridlines', self.graphSettings)
        self.gSetLegendCB = QCheckBox('show legend', self.graphSettings)
        self.gShowFilledSymbolsCB = QCheckBox('show filled symbols', self.graphSettings)
        self.gSetGridColorB = QPushButton("Grid Color", self.top)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.top)

        self.connect(self.widthSlider, SIGNAL("valueChanged(int)"), self.widthLCD, SLOT("display(int)"))
        self.connect(self.gSetGridColorB, SIGNAL("clicked()"), self.setGraphGridColor)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

    def setGraphGridColor(self):
        newColor = QColorDialog.getColor(self.gSetGridColor)
        if newColor.isValid():
            self.gSetGridColor = newColor
            self.emit(PYSIGNAL("gridColorChange(QColor &)"),(QColor(newColor),))

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(self.gSetCanvasColor)
        if newColor.isValid():
            self.gSetCanvasColor = newColor
            self.emit(PYSIGNAL("canvasColorChange(QColor &)"),(QColor(newColor),))

if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWScatterPlotOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()