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
        self.setCaption(title)
        self.setIcon(QPixmap(icon))
        self.connect(self.ok,SIGNAL("clicked()"),self,SLOT("close()"))
               
if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()