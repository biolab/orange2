#
# OWRankOptions.py
#
# options dialog for distributions graph
#

from OWOptions import *
from OWTools import *

class OWRankOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self,"Rank Options","OrangeWidgetsIcon.png",parent,name)  

        #precision
        precisionBox=QHGroupBox("Precision",self.top)
        QToolTip.add(precisionBox,"Number of decimal places in output")
        self.precisionSlider=QSlider(0,9,1,3,QSlider.Horizontal,precisionBox)
        self.precisionSlider.setTickmarks(QSlider.Below)
        self.precisionLCD=QLCDNumber(1,precisionBox)
        self.precisionLCD.display(3)
        self.connect(self.precisionSlider,SIGNAL("valueChanged(int)"),self.precisionLCD,SLOT("display(int)"))

        #relief options
        reliefBox=QVGroupBox("Relief options",self.top)
        reliefKBox=QHGroupBox("k",reliefBox)
        self.kSlider=QSlider(1,50,5,11,QSlider.Horizontal,reliefKBox)
        self.kSlider.setTickmarks(QSlider.Below)
        self.kLCD=QLCDNumber(2,reliefKBox)
        self.kLCD.display(11)
        self.connect(self.kSlider,SIGNAL("valueChanged(int)"),self.kLCD,SLOT("display(int)"))
        QToolTip.add(reliefKBox,"k")
        reliefNBox=QHGroupBox("n",reliefBox)
        QToolTip.add(reliefNBox,"n")
        self.nSlider=QSlider(1,50,5,20,QSlider.Horizontal,reliefNBox)
        self.nSlider.setTickmarks(QSlider.Below)
        self.nLCD=QLCDNumber(2,reliefNBox)
        self.nLCD.display(20)
        self.connect(self.nSlider,SIGNAL("valueChanged(int)"),self.nLCD,SLOT("display(int)"))

        #discretization options
        discretizationBox=QHGroupBox("Discretization Method",self.top)
        QToolTip.add(discretizationBox,"Method of discretization")
        self.discretizationStrings=["equal-frequency intervals","entropy-based discretization","equal-width intervals"]
        self.discretization=QComboBox(discretizationBox)
        self.discretization.insertItem(self.discretizationStrings[0])
        self.discretization.insertItem(self.discretizationStrings[1])
        self.discretization.insertItem(self.discretizationStrings[2])

        #display selection
        displayBox=QVGroupBox("Measures used",self.top)
        QToolTip.add(displayBox,"Select which measure to display")
        self.displayReliefF=QCheckBox("ReliefF",displayBox)
        self.displayReliefF.setChecked(TRUE)
        self.displayInfoGain=QCheckBox("InfoGain",displayBox)
        self.displayInfoGain.setChecked(TRUE)
        self.displayGainRatio=QCheckBox("GainRation",displayBox)
        self.displayGainRatio.setChecked(TRUE)
        self.displayGini=QCheckBox("Gini",displayBox)
        self.displayGini.setChecked(TRUE)
               
if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWRankOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

