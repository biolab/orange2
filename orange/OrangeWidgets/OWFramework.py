#
# OWFramework.py
# The Orange Widgets Framework, an aplication for visually creating Orange Widget Applications
#

import sys
from OWTools import *
from OWAbout import *
from OWWorkspace import *

class OWFramework(QMainWindow):
	def test(self):
		self.test=3
		
	def __init__(self,parent=None,name=None):
		"constructor"
		QMainWindow.__init__(self,parent,name)
		self.resize(640,480)
		self.setCaption("Orange Widgets Framework")
		#give it an icon and a title
		self.setMenu()
		self.ad=OWAbout()
		self.workspace=OWWorkspace(self)
		self.setCentralWidget(self.workspace)
		self.statusBar=QStatusBar(self)
		self.statusBar.message("Welcome To Orange Widgets")
		
		self.connect(self.helpAbout,SIGNAL('activated()'),self.ad.show)
		
	def setMenu(self):
        
		self.icon=QPixmap("OrangeIcon.gif")
		self.setIcon(self.icon)
		self.setIconText("Orange Widgets Framework")

		self.file=QPopupMenu()
		self.fileOpen=QAction(self,"Open")
		self.fileOpen.setText("&Open")
		self.fileOpen.setToolTip("&Open Application")
		self.fileOpen.setStatusTip('Open a saved application')
		self.fileOpen.addTo(self.file)
		self.fileSave=QAction(self,"Save")
		self.fileSave.setText("&Save")
		self.fileSave.setToolTip("&Save Application")
		self.fileSave.setStatusTip('Save the application')
		self.fileSave.addTo(self.file)
		self.file.insertSeparator()
		self.fileExit=QAction(self,"Exit")
		self.fileExit.setText("&Exit")
		self.fileExit.setToolTip("&Exit program")
		self.fileExit.setStatusTip('Quit working with OrangeWidgets')
		self.fileExit.addTo(self.file)
		self.menuBar().insertItem("&Application",self.file)
		
		self.widgets=QPopupMenu()
		self.wAdd=QAction(self,"Add")
		self.wAdd.setText("&Add")
		self.wAdd.addTo(self.widgets)
		self.wRemove=QAction(self,"Remove")
		self.wRemove.setText("&Remove")
		self.wRemove.addTo(self.widgets)
		self.menuBar().insertItem("&Widget",self.widgets);
		
		self.help=QPopupMenu()
		self.helpAbout=QAction(self,"About")
		self.helpAbout.setText('&About')
		self.helpAbout.setToolTip('Show About box')
		self.helpAbout.setStatusTip('Display info about Visual Orange')
		self.helpAbout.addTo(self.help)
		self.menuBar().insertSeparator() #NOTE: this does nothing on Windows
		self.menuBar().insertItem("&Help",self.help)

a=QApplication(sys.argv)
owf=OWFramework()
a.setMainWidget(owf)
QObject.connect(owf, PYSIGNAL('exit'),a,SLOT('quit()'))
QObject.connect(owf.fileExit,SIGNAL('activated()'),a,SLOT('quit()'))
owf.show()
a.exec_loop()
