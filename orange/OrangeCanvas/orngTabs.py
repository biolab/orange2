# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	tab for showing widgets and widget button class
#
from qt import *
import sys
import os
import string
import orngDoc
from xml.dom.minidom import Document, parse

TRUE  = 1
FALSE = 0 

class WidgetButton(QToolButton):
	def __init__(self, *args):
		apply(QToolButton.__init__,(self,)+ args)

	def setValue(self, name, fileName, inList, outList, icon, description, priority, canvasDlg, useLargeIcons):
		self.name = name
		self.fileName = fileName
		self.iconName = icon
		self.priority = priority
		self.description = description
		self.inList = inList
		self.outList = outList

		if len(inList) == 0:
			formatedInList = "<b>Inputs:</b><br>None"
		else:
			formatedInList = "<b>Inputs:</b><ul>"
			for (signalname, type, handler, single) in inList:
				formatedInList = formatedInList + "<li>" + canvasDlg.getChannelName(signalname) + " (" + type + ")</li>"
			formatedInList += "</ul>"

		if len(outList) == 0:
			formatedOutList = "<b>Outputs:</b><br>None"
		else:
			formatedOutList = "<b>Outputs:</b><ul>"
			for (signalname, type) in outList:
				formatedOutList = formatedOutList + "<li>" + canvasDlg.getChannelName(signalname) + " (" + type + ")</li>"
			formatedOutList += "</ul>"
		
		#tooltipText = name + "\nClass name: " + fileName + "\nin: " + formatedInList + "\nout: " + formatedOutList + "\ndescription: " + description
		tooltipText = "<b>%s</b><br>Class name: %s<br><hr><b>Description:</b><br>%s<hr>%s<hr>%s" % (name, fileName, description, formatedInList, formatedOutList)
		QToolTip.add( self, tooltipText)

		self.canvasDlg = canvasDlg

		self.setTextLabel(name, FALSE)		
		self.setPixmap(QPixmap(self.iconName))
		if useLargeIcons == 1:
			self.setUsesTextLabel (TRUE)
			self.setUsesBigPixmap(TRUE)
			self.setMaximumSize(80, 80)
			self.setMinimumSize(80, 80)
		else:
			self.setMaximumSize(48, 48)
			self.setMinimumSize(48, 48)

	def clicked(self):
		win = self.canvasDlg.workspace.activeWindow()
		if (win != None and isinstance(win, orngDoc.SchemaDoc)):
			win.addWidget(self)

class WidgetTab(QWidget):
	def __init__(self, *args):
		apply(QWidget.__init__,(self,)+ args)
		self.HItemBox = QHBoxLayout(self)
		#self.setMaximumHeight(60)
		self.widgets = []

	def addWidget(self, widget):
		self.HItemBox.addWidget(widget)
		self.widgets.append(widget)

	def finishedAdding(self):
		self.HItemBox.addStretch(10)

class WidgetTabs(QTabWidget):
	def __init__(self, *args):
		apply(QTabWidget.__init__,(self,) + args)
		self.tabs = []
		self.canvasDlg = None
		self.allWidgets = []
		self.useLargeIcons = FALSE
		self.tabDict = {}

	def insertWidgetTab(self, name):
		tab = WidgetTab(self, name)
		self.tabs.append(tab)
		self.insertTab(tab, name)
		self.tabDict[name] = tab
		return tab
		
	def setCanvasDlg(self, canvasDlg):
		self.canvasDlg = canvasDlg

	# read the xml registry and show all installed widgets
	def readInstalledWidgets(self, registryFileName, widgetDir, picsDir, defaultPic, useLargeIcons):
		self.widgetDir = widgetDir
		self.picsDir = picsDir
		self.defaultPic = defaultPic
		self.useLargeIcons = useLargeIcons
		doc = parse(registryFileName)
		orangeCanvas = doc.firstChild
		categories = orangeCanvas.getElementsByTagName("widget-categories")[0]
		if (categories == None):
			return

		categoryList = categories.getElementsByTagName("category")
		for category in categoryList:
			self.addWidgetCategory(category)

		# remove empty categories
		for i in range(len(self.tabs)-1, -1, -1):
			if self.tabs[i].widgets == []:
				self.removePage(self.tabs[i])
				self.tabs.remove(self.tabs[i])

	# add all widgets inside the category to the tab
	def addWidgetCategory(self, category):
		strCategory = str(category.getAttribute("name"))
		if self.tabDict.has_key(strCategory): tab = self.tabDict[strCategory]
		else:	tab = self.insertWidgetTab(strCategory)

		priorityList = []
		inListList = []
		outListList = []
		nameList = []
		iconNameList = []
		descriptionList = []
		fileNameList = []
		
		widgetList = category.getElementsByTagName("widget")
		for widget in widgetList:
			name = widget.getAttribute("name")
			fileName = widget.getAttribute("file")
			inList = eval(widget.getAttribute("in"))
			outList = eval(widget.getAttribute("out"))
			priority = int(widget.getAttribute("priority"))

			icon = widget.getAttribute("icon")
			iconName = icon
			if (icon != ""):
				if os.path.isfile(self.widgetDir + str(category.getAttribute("name")) + icon):
					iconName = self.widgetDir + str(category.getAttribute("name")) + icon
				elif os.path.isfile(self.picsDir + icon):
					iconName = self.picsDir + icon			
				elif os.path.isfile(self.widgetDir + icon):
					iconName = self.widgetDir + icon
				else:
					iconName = self.defaultPic
			
			# it's a complicated way to get to the widget description
			descNode= widget.getElementsByTagName("description")[0]
			description = ""
			for node in widget.childNodes:
				if node.nodeType == node.TEXT_NODE:
					description = description + node.nodeValue
				else:
					for n2 in node.childNodes:
						if n2.nodeType == node.TEXT_NODE:
							description = description + n2.nodeValue

			description = string.strip(description)
			i = 0
			while i < len(priorityList) and priority > priorityList[i]:
				i = i + 1
			priorityList.insert(i, priority)
			nameList.insert(i, name)
			fileNameList.insert(i, fileName)
			inListList.insert(i, inList)
			outListList.insert(i, outList)
			iconNameList.insert(i, iconName)
			descriptionList.insert(i, description)

		exIndex = 0
		for i in range(len(priorityList)):			
			button = WidgetButton(tab)
			button.setValue(nameList[i], fileNameList[i], inListList[i], outListList[i], iconNameList[i], descriptionList[i], priorityList[i], self.canvasDlg, self.useLargeIcons)
			self.connect( button, SIGNAL( 'clicked()' ), button.clicked)
			if exIndex != priorityList[i] / 1000:
				for k in range(priorityList[i]/1000 - exIndex):
					frame = QFrame(tab)
					frame.setMinimumWidth(20)
					frame.setMaximumWidth(20)
					tab.addWidget(frame)
				exIndex = priorityList[i] / 1000
			tab.addWidget(button)
			self.allWidgets.append(button)

		tab.finishedAdding()