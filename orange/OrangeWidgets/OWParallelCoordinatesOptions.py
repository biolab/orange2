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

        self.gSetGridColor = QColor(Qt.black)
        self.gSetCanvasColor = QColor(Qt.white) 

        widthBox = QHGroupBox("Point Width", self.top)
        QToolTip.add(widthBox, "The width of points")
        self.widthSlider = QSlider(2, 9, 1, 3, QSlider.Horizontal, widthBox)
        self.widthSlider.setTickmarks(QSlider.Below)
        self.widthLCD = QLCDNumber(1, widthBox)

        self.spreadButtons = QVButtonGroup("Random spread type", self.top)
        QToolTip.add(self.spreadButtons, "Selected the type of random spread for discrete variables")
        self.spreadButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.spreadButtons)
        self.spreadUniform = QRadioButton('uniform', self.spreadButtons)
        self.spreadTriangle = QRadioButton('triangle', self.spreadButtons)
        self.spreadBeta = QRadioButton('beta', self.spreadButtons)

        self.graphSettings = QVButtonGroup("General graph settings", self.top)
        QToolTip.add(self.graphSettings, "Enable/disable main title, axis title and grid")
        self.gSetMainTitle = QHBox(self.graphSettings, "main title group")
        self.gSetMainTitleCB = QCheckBox('show main title', self.gSetMainTitle)
        self.gSetMainTitleLE = QLineEdit('main title', self.gSetMainTitle)
        self.gSetXaxisCB = QCheckBox('X axis title ', self.graphSettings)
        self.gSetYaxisCB = QCheckBox('Y axis title ', self.graphSettings)
        self.gSetVgridCB = QCheckBox('vertical gridlines', self.graphSettings)
        self.gSetHgridCB = QCheckBox('horizontal gridlines', self.graphSettings)
        self.gSetLegendCB = QCheckBox('show legend', self.graphSettings)
        self.gSetGridColorB = QPushButton("Grid Color", self.graphSettings)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.graphSettings)

        self.connect(self.gSetMainTitleCB, SIGNAL("toggled(bool)"), self.gSetMainTitleLE, SLOT("setEnabled(bool)"))
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
    w=OW2DInteractionsOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

    
