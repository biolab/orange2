#
# voDisOpt.py
#
# options dialog for distributions graph
#

import sys
from qt import *
from OWOptions import *

class OWDistributionsOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self,"Distributions Options","OrangeWidgetsIcon.py",parent,name)
        self.dist=QVGroupBox(self.top)
        self.dist.setTitle("Distribution Graph")
        self.nb=QHGroupBox(self.dist)
        self.nb.setTitle("Number of Bars")
        QToolTip.add(self.nb,"Number of bars for graphs\nof continuous variables")
        self.numberOfBars=QSlider(5,60,5,5,QSlider.Horizontal,self.nb)
        self.numberOfBars.setTickmarks(QSlider.Below)
        self.numberOfBars.setTracking(0) # no change until the user stop dragging the slider
        self.nbLCD=QLCDNumber(2,self.nb)
        self.nbLCD.display(5)
        self.connect(self.numberOfBars,SIGNAL("valueChanged(int)"),self.nbLCD,SLOT("display(int)"))
        self.bx=QHGroupBox(self.dist)
        self.bx.setTitle("Bar Size")
        QToolTip.add(self.bx,"The size of bars\nin percentage\nof available space (for discrete variables)")
        self.barSize=QSlider(30,100,10,50,QSlider.Horizontal,self.bx)
        self.barSize.setTickmarks(QSlider.Below)
        self.barSize.setLineStep(10)
#        self.barSize.setTracking(0) # no change until the user stop dragging the slider
        self.bxLCD=QLCDNumber(3,self.bx)
        self.bxLCD.display(50)
        self.connect(self.barSize,SIGNAL("valueChanged(int)"),self.bxLCD,SLOT("display(int)"))
        self.pg=QVGroupBox(self.top)
        self.pg.setTitle("Probability graph")
        self.showprob=QCheckBox("Show Probabilities",self.pg)
        self.showprob.setChecked(1)
        self.showcoin=QCheckBox("Show Confidence Intervals",self.pg)
        self.smooth=QCheckBox("Smooth probability lines",self.pg)
        self.lw=QHGroupBox(self.pg)
        self.lw.setTitle("Line width")
        QToolTip.add(self.lw,"The width of lines in pixels")
        self.lineWidth=QSlider(1,9,1,1,QSlider.Horizontal,self.lw)
        self.lineWidth.setTickmarks(QSlider.Below)
#        self.lineWidth.setTracking(0) # no change signaled until the user stop dragging the slider
        self.lwLCD=QLCDNumber(1,self.lw)
        self.lwLCD.display(1)
        self.smooth.setEnabled(0)
        self.connect(self.lineWidth,SIGNAL("valueChanged(int)"),self.lwLCD,SLOT("display(int)"))
        self.connect(self.showprob,SIGNAL("stateChanged(int)"),self.prob2CI)
              
    def prob2CI(self,state):
        if state==0:
            self.showcoin.setChecked(0)
        self.showcoin.setEnabled(state)
   
if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWDistributionsOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()