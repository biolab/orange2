#
# OWAboutX.py
# A general about dialog for any orange widget, inherited from OWAbout
#

import sys
from qt import *
from OWAbout import *

class OWAboutX(OWAbout):
    def __init__(
    self,
    title="&X",
    description="X is an Orange Widget that does nothing.",
    icon="pics\OrangeWidgetsIcon.png",
    logo="pics\OrangeWidgetsLogo.png"
    ):
        """
        Constructor
        title - The title of the widget
        description - The description of the widget, appears in the about box
        logo - The logo of the widget, OrangeWidgetsLogo.gif if omitted, 
            pass an empty string if no logo is wanted
        icon - The icon of the widget, OrangeWidgetsIcon.gif if omitted, 
            pass an empty string if no icon is wanted
        """
        OWAbout.__init__(self)
        tabx=QVBox(self)
        self.insertTab(tabx,title,0)
        self.showPage(tabx)
        self.setCaption(title.replace("&",""))
        pixmapx=QPixmap(logo)
        self.setIcon(QPixmap(icon))
        l1=QLabel(description,tabx)
        l2=QLabel(tabx)
        l2.setPixmap(pixmapx)
        l1.setAlignment(Qt.AlignCenter)
        l2.setAlignment(Qt.AlignCenter)
        
if __name__== "__main__":
    #as a test, simply show the dialog with all the options
    a=QApplication(sys.argv)
    w=OWAboutX("&Y","Y is ...","OrangeWidgetsIcon.png","OrangeLogo.png")
    a.setMainWidget(w)
    w.show()
    a.exec_loop()