#
# voDisOpt.py
#
# options dialog for distributions graph
#

import sys
from qt import *

class OWOptions(QVBox):
    def __init__(self,title="Options",icon="OrangeWidgetsIcon.png",parent=None,name=None):
        QVBox.__init__(self,parent,name)
        self.top=QVBox(self)
        self.ok=QPushButton(self,"ok")
        self.ok.setText("OK")

        # if we want the widget to show the title then the title must start with "Qt"
        if title[:2].upper != "QT":
            title = "Qt " + title
        self.setCaption(title)
        self.setIcon(QPixmap(icon))
        self.connect(self.ok,SIGNAL("clicked()"),self,SLOT("close()"))
               
if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()