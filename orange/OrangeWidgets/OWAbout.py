#
# OWAbout.py
# About Orange & Orange Widgets dialog
#

import sys
from qt import *
import os.path

class OWAbout(QTabDialog):
    def __init__(self, defaultIcon, logo, parent=None, name=None):
        QTabDialog.__init__(self,parent,name)
        self.setCaption("Qt About Orange Widgets")
        self.setIconText("Orange Widgets")

        #some text about visual orange
        orangewidgets="""Orange Widgets is a free platform for embedding
data mining functions in easy-to-use graphical user
interface components. Widgets are developed in Python
programming language on top of Orange data mining and
using the Qt GUI library from Trolltech. Orange Widgets
are best used within Orange Canvas.

   Orange Widgets concept by B. Zupan, J. Demsar, M. Kavcic    
Version 0.90, January - April, 2003"""
        #some text about orange, copied from its homepage
        orange="""Orange is a free, component-based, public domain data
mining software. It includes a range of preprocessing,
modeling and data exploration techniques. It is based
on C++ components, that are accessed either directly
or through Python scripts.

Principal authors of Orange J. Demsar and B. Zupan"""

        tab1=QVBox(self)
        tab2=QVBox(self)
        
        l11=QLabel(orangewidgets,tab1)
        l12=QLabel(tab1)
        l21=QLabel(orange,tab2)
        l22=QLabel(tab2)
        if os.path.exists(defaultIcon): l12.setPixmap(QPixmap(defaultIcon))
        if os.path.exists(logo): l22.setPixmap(QPixmap(logo))
        l11.setAlignment(Qt.AlignCenter)
        l12.setAlignment(Qt.AlignCenter)
        l21.setAlignment(Qt.AlignCenter)
        l22.setAlignment(Qt.AlignCenter)
        self.addTab(tab1,"Orange &Widgets")
        self.addTab(tab2,"&Orange")
        self.resize(300,340)
    
    def show(self):
        QTabDialog.show(self)


class OWAboutX(OWAbout):
    def __init__(
    self,
    title="&X",
    description="X is an Orange Widget that does nothing.",
    widgetIcon = "OrangeWidgetsIcon.png", defaultIcon = "OrangeWidgetsLogo.png", logoIcon = "OrangeLogo.png"):
        """
        Constructor
        title - The title of the widget
        description - The description of the widget, appears in the about box
        logo - The logo of the widget, OrangeWidgetsLogo.gif if omitted, 
            pass an empty string if no logo is wanted
        icon - The icon of the widget, OrangeWidgetsIcon.gif if omitted, 
            pass an empty string if no icon is wanted
        """
        OWAbout.__init__(self, defaultIcon, logoIcon)
        tabx = QVBox(self)
        self.insertTab(tabx,title,0)
        self.showPage(tabx)
        self.setCaption(title.replace("&",""))
        
        if os.path.exists(widgetIcon): self.setIcon(QPixmap(widgetIcon))
        l1=QLabel(description,tabx)
        l2=QLabel(tabx)
        l1.setAlignment(Qt.AlignCenter)
        l2.setAlignment(Qt.AlignCenter)

        if os.path.exists(logoIcon): l2.setPixmap(QPixmap(logoIcon))
        

if __name__== "__main__":
    #as a test, simply show the dialog
    a=QApplication(sys.argv)
    w=OWAbout("icons/OrangeWidgetsLogo.png", "icons/OrangeLogo.png")
    a.setMainWidget(w)
    w.show()
    a.exec_loop()