#
# OWWorkspace.py
#

import sys
from OWTools import *

class OWWorkspace(QWidget):
	def __init__(self,parent=None):
		QWidget.__init__(self,parent)
		self.setBackgroundColor(Qt.black)

if __name__ == "__main__":	
	a=QApplication(sys.argv)
	oww=OWWorkSpace()
	a.setMainWidget(oww)
	QObject.connect(oww, PYSIGNAL('exit'),a,SLOT('quit()'))
	oww.show()
	a.exec_loop()
