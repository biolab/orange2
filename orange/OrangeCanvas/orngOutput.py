# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:

#
from qt import *
import sys
import os
import string
import time
import traceback
import os.path


TRUE  = 1
FALSE = 0 


class OutputWindow(QToolBar ):
	def __init__(self, canvas, *args):
		apply(QToolBar.__init__,(self,)+args)
		self.textOutput = QTextView(self)
		self.textOutput.setFont(QFont('Courier New',10, QFont.Normal))
		self.setStretchableWidget(self.textOutput)
		self.canvas = canvas

		self.setVerticalStretchable(1)
		self.setHorizontalStretchable(1)

		sys.excepthook = self.exceptionHandler
		sys.stdout = self
		#self.textOutput.setText("")

	def write(self, str):
		if not self.canvas.showOutputWindow:
			self.canvas.menuItemShowOutputWindow()
		self.textOutput.append(str)

		self.textOutput.ensureVisible(0, self.textOutput.contentsHeight())


	def exceptionHandler(self, type, value, tracebackInfo):
		if not self.canvas.showOutputWindow:
			self.canvas.menuItemShowOutputWindow()
			
		t = time.localtime()
		self.textOutput.append("--------- ERROR at %d:%d:%d ------------------------------" % (t[3],t[4],t[5]))
		self.textOutput.append("Exception type: " + str(type))
		self.textOutput.append("Exception value: " + str(value))
		self.textOutput.append("Traceback:")

		list = traceback.extract_tb(tracebackInfo, 5)
		print len(list)

		space = "     "
		totalSpace = ""
		for (file, line, funct, code) in list:
			if code == None: continue
			(dir, filename) = os.path.split(file)
			self.textOutput.append("." + totalSpace + "File: " + filename + "  in line : %4d" %(line))
			self.textOutput.append("." + totalSpace + "Function name: %s" % (funct))
			self.textOutput.append("." + totalSpace + "Code: " + code)
			totalSpace += space
		self.textOutput.ensureVisible(0, self.textOutput.contentsHeight())
		

