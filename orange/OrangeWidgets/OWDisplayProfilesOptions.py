#
# OWROCOptions.py
#
# options dialog for the Display Profiles widget
#

from OWOptions import *
from OWTools import *

class OWDisplayProfilesOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "Display Motifs Options", "OrangeWidgetsIcon.png", parent, name)

        pointWidthBox = QHGroupBox("Point Width", self.top)
        QToolTip.add(pointWidthBox, "The width of points")
        self.pointWidthSlider = QSlider(0, 15, 1, 7, QSlider.Horizontal, pointWidthBox)
        self.pointWidthSlider.setTickmarks(QSlider.Below)
        self.pointWidthLCD = QLCDNumber(1, pointWidthBox)

        lineWidthBox = QHGroupBox("Curve Width", self.top)
        QToolTip.add(lineWidthBox, "The width of single curves")
        self.lineWidthSlider = QSlider(1, 9, 1, 3, QSlider.Horizontal, lineWidthBox)
        self.lineWidthSlider.setTickmarks(QSlider.Below)
        self.lineWidthLCD = QLCDNumber(1, lineWidthBox)

        averageLineWidthBox = QHGroupBox("Average Curve Width", self.top)
        QToolTip.add(averageLineWidthBox, "The width of average curves")
        self.averageLineWidthSlider = QSlider(1, 9, 1, 6, QSlider.Horizontal, averageLineWidthBox)
        self.averageLineWidthSlider.setTickmarks(QSlider.Below)
        self.averageLineWidthLCD = QLCDNumber(1, averageLineWidthBox)

        self.connect(self.pointWidthSlider, SIGNAL("valueChanged(int)"), self.pointWidthLCD, SLOT("display(int)"))
        self.connect(self.lineWidthSlider, SIGNAL("valueChanged(int)"), self.lineWidthLCD, SLOT("display(int)"))
        self.connect(self.averageLineWidthSlider, SIGNAL("valueChanged(int)"), self.averageLineWidthLCD, SLOT("display(int)"))

if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWDisplayProfilesOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

    
