#
# OWLiftCurveOptions.py
#
# options dialog for the ROC widget
#

from OWOptions import *
from OWTools import *

class OWLiftCurveOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "Lift Curve Options", "OrangeWidgetsIcon.png", parent, name)

        pointWidthBox = QHGroupBox("Point Width", self.top)
        QToolTip.add(pointWidthBox, "The width of points")
        self.pointWidthSlider = QSlider(4, 15, 1, 7, QSlider.Horizontal, pointWidthBox)
        self.pointWidthSlider.setTickmarks(QSlider.Below)
        self.pointWidthLCD = QLCDNumber(1, pointWidthBox)

        lineWidthBox = QHGroupBox("Lift Curve Curve Width", self.top)
        QToolTip.add(lineWidthBox, "The width of Lift curves")
        self.lineWidthSlider = QSlider(1, 9, 1, 3, QSlider.Horizontal, lineWidthBox)
        self.lineWidthSlider.setTickmarks(QSlider.Below)
        self.lineWidthLCD = QLCDNumber(1, lineWidthBox)

        diagonalBox = QHGroupBox("Show diagonal Lift Curve line", self.top)
        self.showDiagonalQCB = QCheckBox("Show", diagonalBox)
        self.showDiagonalQCB.setChecked(1)

        hullBox = QHGroupBox("Lift Curve convex hull", self.top)
        QToolTip.add(hullBox, "The width of Lift Curve hull")
        self.hullWidthSlider = QSlider(2, 9, 1, 3, QSlider.Horizontal, hullBox)
        self.hullWidthSlider.setTickmarks(QSlider.Below)
        self.hullWidthLCD = QLCDNumber(1, hullBox)
        self.hullColorQPB = QPushButton("Lift Curve Convex Hull Color", hullBox)
        self.hullColor = QColor(Qt.yellow)

        self.connect(self.pointWidthSlider, SIGNAL("valueChanged(int)"), self.pointWidthLCD, SLOT("display(int)"))
        self.connect(self.lineWidthSlider, SIGNAL("valueChanged(int)"), self.lineWidthLCD, SLOT("display(int)"))
        self.connect(self.hullWidthSlider, SIGNAL("valueChanged(int)"), self.hullWidthLCD, SLOT("display(int)"))

        self.connect(self.hullColorQPB, SIGNAL("clicked()"), self.setHullColor)

    def setHullColor(self):
        newColor = QColorDialog.getColor(self.hullColor)
        if newColor.isValid():
            self.hullColor = newColor
            self.emit(PYSIGNAL("hullColorChange(QColor &)"),(QColor(newColor),))

if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWLiftCurveOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

    
