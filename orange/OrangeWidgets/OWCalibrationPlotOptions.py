#
# OWCalibrationPlotOptions.py
#
# options dialog for the CalibrationPlot widget
#

from OWOptions import *
from OWTools import *

class OWCalibrationPlotOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "Calibration Plot Options", "OrangeWidgetsIcon.png", parent, name)

        lineWidthBox = QHGroupBox("Calibration Curve Width", self.top)
        QToolTip.add(lineWidthBox, "The width of Calibration curves")
        self.lineWidthSlider = QSlider(1, 9, 1, 3, QSlider.Horizontal, lineWidthBox)
        self.lineWidthSlider.setTickmarks(QSlider.Below)
        self.lineWidthLCD = QLCDNumber(1, lineWidthBox)

        diagonalBox = QVGroupBox("Show", self.top)
        diagFrame = QHBox(diagonalBox)
        self.showDiagonalQCB = QCheckBox("Show Diagonal Line", diagonalBox)
        self.showDiagonalQCB.setChecked(1)
        rugFrame = QHBox(diagonalBox)
        self.showRugsQCB = QCheckBox("Show Rugs", rugFrame)
        self.showRugsQCB.setChecked(1)

        self.connect(self.lineWidthSlider, SIGNAL("valueChanged(int)"), self.lineWidthLCD, SLOT("display(int)"))

if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWCalibrationPlotOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()
