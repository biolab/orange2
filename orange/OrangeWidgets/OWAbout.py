#
# OWAbout.py
# About Orange & Orange Widgets dialog
#

import sys
from qt import *

class OWAbout(QTabDialog):
    def __init__(self, parent=None, name=None):
        QTabDialog.__init__(self,parent,name)
        self.setCaption("About Orange Widgets")
        self.icon=QPixmap(sys.prefix + "/lib/site-packages/orange/orangeWidgets/icons/OrangeWidgetsIcon.png")
        self.setIcon(self.icon)
        self.setIconText("Orange Widgets")

        #some text about visual orange
        orangewidgets="""Orange Widgets is a free, widget based,
public domain visual data mining software.
It is developed in the python programming language
on top of the Orange data mining software
using the powerful Qt library from Trolltech.

Authors: Marko Kavcic and Blaz Zupan
Version 0.80
January 5, 2003"""
        #some text about orange, copied from its homepage
        orange="""Orange is a free, component-based, 
public domain data mining software. 
Currently it includes a range of preprocessing, 
modeling and data exploration techniques. 
It is based on C++ components, 
that are accessed either directly 
or through Python scripts.

Principal authors: Janez Demsar and Blaz Zupan"""

        tab1=QVBox(self)
        tab2=QVBox(self)
        pixmap1=QPixmap(sys.prefix + "/lib/site-packages/orange/orangeWidgets/icons/OrangeWidgetsLogo.png")
        pixmap2=QPixmap(sys.prefix + "/lib/site-packages/orange/orangeWidgets/icons/OrangeLogo.png")
        l11=QLabel(orangewidgets,tab1)
        l12=QLabel(tab1)
        l12.setPixmap(pixmap1)
        l21=QLabel(orange,tab2)
        l22=QLabel(tab2)
        l22.setPixmap(pixmap2)
        l11.setAlignment(Qt.AlignCenter)
        l12.setAlignment(Qt.AlignCenter)
        l21.setAlignment(Qt.AlignCenter)
        l22.setAlignment(Qt.AlignCenter)
        self.addTab(tab1,"Orange &Widgets")
        self.addTab(tab2,"&Orange")
        self.resize(300,340)
    
    def show(self):
        QTabDialog.show(self)

if __name__== "__main__":
    #as a test, simply show the dialog
    a=QApplication(sys.argv)
    w=OWAbout()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()