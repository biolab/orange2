# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	main file, that creates the MDI environment

from qt import *
import sys
import os
import string
import orngTabs
import orngDoc
import orngDlgs
import orngOutput
import orngResources
import xmlParse
import cPickle
from orngSignalManager import *

TRUE  = 1
FALSE = 0

class OrangeCanvasDlg(QMainWindow):
	def __init__(self,*args):
		apply(QMainWindow.__init__,(self,) + args)
		self.ctrlPressed = 0	# we have to save keystate, so that orngView can access information about keystate
		self.resize(900,800)
		self.setCaption("Qt Orange Canvas")
		self.windows = []	# list of id for windows in Window menu

		# if widget path not registered -> register
		self.widgetDir = "../OrangeWidgets/"
		self.picsDir = "../OrangeWidgets/icons/"
		self.defaultPic = self.picsDir + "Unknown.png"
		self.canvasDir = "./"
		self.registryFileName = self.canvasDir + "widgetregistry.xml"
		
		if sys.path.count(self.widgetDir) == 0:
			sys.path.append(self.widgetDir)

		self.workspace = QWorkspace(self)
		self.workspace.setBackgroundColor(QColor(255,255,255))
		self.setCentralWidget(self.workspace)
		self.statusBar = QStatusBar(self)
		self.connect(self.workspace, SIGNAL("windowActivated(QWidget*)"), self.focusDocument)

		
		self.settings = {}
		self.loadSettings()
		self.rebuildSignals()

		self.toolbar = QToolBar(self, 'toolbar')
		self.widgetsToolBar = QToolBar( self, 'Widgets')

		self.initMenu()
		
		self.toolNew  = QToolButton(QPixmap(orngResources.file_new), "New schema" , QString.null, self.menuItemNewSchema, self.toolbar, 'new schema')
		#self.toolNew.setUsesTextLabel (TRUE)
		self.toolOpen = QToolButton(QPixmap(orngResources.file_open), "Open schema" , QString.null, self.menuItemOpen , self.toolbar, 'open schema') 
		self.toolSave = QToolButton(QPixmap(orngResources.file_save), "Save schema" ,QString.null, self.menuItemSave, self.toolbar, 'save schema')
		self.toolbar.addSeparator()
		toolPrint = QToolButton(QPixmap(orngResources.file_print), "Print" ,QString.null, self.menuItemPrinter, self.toolbar, 'print')
		self.addToolBar(self.toolbar, "Toolbar", QMainWindow.Top, TRUE)
	
		self.widgetsToolBar.setHorizontalStretchable(TRUE)		
		self.createWidgetsToolbar(not os.path.exists(self.registryFileName))

		self.recentDocs = []
		self.readRecentFiles()
		self.show()

		# center window in the desktop
		deskH = app.desktop().height()
		deskW = app.desktop().width()
		h = deskH/2 - self.height()/2
		w = deskW/2 - self.width()/2
		h = max(h,0)	# if the screen is to small position the window in the upper left corner
		w = max(w,0)
		self.move(w,h)

		self.output = orngOutput.OutputWindow(self, self.workspace)
		self.output.show()
		self.output.catchException(self.settings["catchException"])
		self.output.catchOutput(self.settings["catchOutput"])
		self.output.setFocusOnException(self.settings["focusOnCatchException"])
		self.output.setFocusOnOutput(self.settings["focusOnCatchOutput"])

		win = self.menuItemNewSchema()
		win.showMaximized()

	
	def createWidgetsToolbar(self, rebuildRegistry):
		self.widgetsToolBar.clear()
		self.tabs = orngTabs.WidgetTabs(self.widgetsToolBar, 'tabs')
		self.addToolBar(self.widgetsToolBar, "Widgets", QMainWindow.Top, TRUE)
		
		self.tabs.setCanvasDlg(self)
		
		# if registry doesn't exist yet, we create it
		if rebuildRegistry == 1:
			parse = xmlParse.WidgetsToXML()
			parse.ParseDirectory(self.widgetDir, self.canvasDir)
			
		# if registry still doesn't exist then something is very wrong...
		if not os.path.exists(self.registryFileName):
			QMessageBox.error( None, "Orange Canvas", "Unable to create widget registry. Exiting...", QMessageBox.Ok + QMessageBox.Default )
			self.quit()
			
		# read widget registry file and create tab with buttons
		self.tabs.readInstalledWidgets(self.registryFileName, self.widgetDir, self.picsDir, self.defaultPic, self.useLargeIcons)
		
	def initMenu(self):
		# ###################
		# menu items
		# ###################
		self.menuRecent = QPopupMenu(self)
		
		self.menuFile = QPopupMenu( self )
		self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_new)), "&New",  self.menuItemNewSchema, Qt.CTRL+Qt.Key_N )
		self.menuFile.insertItem( "New from template",  self.menuItemNewFromTemplate)
		self.menuFile.insertItem( "New from wizard",  self.menuItemNewWizard)
		self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_open)), "&Open", self.menuItemOpen, Qt.CTRL+Qt.Key_O )
		self.menuFile.insertItem( "&Close", self.menuItemClose )
		self.menuFile.insertSeparator()
		self.menuSaveID = self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_save)), "&Save", self.menuItemSave, Qt.CTRL+Qt.Key_S )
		self.menuSaveAsID = self.menuFile.insertItem( "&Save As..", self.menuItemSaveAs)
		self.menuFile.insertItem( "&Save As Application (Tabs)", self.menuItemSaveAsAppTabs)
		self.menuFile.insertItem( "&Save As Application (Buttons)", self.menuItemSaveAsAppButtons)
		self.menuFile.insertSeparator()
		self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_print)), "&Print", self.menuItemPrinter, Qt.CTRL+Qt.Key_P )
		self.menuFile.insertSeparator()
		self.menuFile.insertItem( "Recent Files", self.menuRecent)
		self.menuFile.insertSeparator()
		self.menuFile.insertItem( "E&xit",  qApp, SLOT( "quit()" ), Qt.CTRL+Qt.Key_Q )

		self.menuEdit = QPopupMenu( self )
		self.menuEdit.insertItem( "Cut",  self.menuItemCut, Qt.CTRL+Qt.Key_X )
		self.menuEdit.insertItem( "Copy",  self.menuItemCopy, Qt.CTRL+Qt.Key_C )
		self.menuEdit.insertItem( "Paste",  self.menuItemPaste, Qt.CTRL+Qt.Key_V )
		self.menuFile.insertSeparator()
		self.menuEdit.insertItem( "Select All",  self.menuItemSelectAll, Qt.CTRL+Qt.Key_A )

		self.menuOptions = QPopupMenu( self )
		#self.menuOptions.insertItem( "Grid",  self.menuItemGrid )
		#self.menuOptions.insertSeparator()
		#self.menuOptions.insertItem( "Show Grid",  self.menuItemShowGrid)

		# snap to grid option
		self.menupopupSnapToGridID = self.menuOptions.insertItem( "Snap to Grid",  self.menuItemSnapToGrid )
		if self.settings.has_key("snapToGrid"): self.snapToGrid = self.settings["snapToGrid"]
		else:									self.snapToGrid = TRUE
		self.menuOptions.setItemChecked(self.menupopupSnapToGridID, self.snapToGrid)

		# use large icons with text option
		self.menupopupLargeIconsID = self.menuOptions.insertItem( "Use large icons",  self.menuItemLargeIcons )
		if self.settings.has_key("useLargeIcons"): self.useLargeIcons = self.settings["useLargeIcons"]
		else:									   self.useLargeIcons = FALSE

		self.menuOptions.insertSeparator()

		self.menuOptions.setItemChecked(self.menupopupLargeIconsID, self.useLargeIcons)
		self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Enable All Links",  self.menuItemEnableAll)
		self.menuOptions.insertItem( "Disable All Links",  self.menuItemDisableAll)
		self.menuOptions.insertItem( "Clear All",  self.menuItemClearWidgets)
		self.menuOptions.insertSeparator()
		#self.menuOptions.insertItem( "Channel preferences",  self.menuItemPreferences)
		#self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Rebuild widget registry",  self.menuItemRebuildWidgetRegistry)
		self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Canvas options",  self.menuItemCanvasOptions)
		
		self.menuWindow = QPopupMenu( self )		
		self.menuWindow.insertItem("Cascade", self.workspace.cascade)
		self.menuWindow.insertItem("Tile", self.workspace.tile)
		self.menuWindow.insertSeparator()
		
		self.connect(self.menuWindow, SIGNAL("aboutToShow()"), self.showWindows)

		self.menupopupShowToolbarID = self.menuWindow.insertItem( "Toolbar",  self.menuItemShowToolbar )
		if self.settings.has_key("showToolbar"): self.showToolbar = self.settings["showToolbar"]
		else:									self.showToolbar = TRUE
		if not self.showToolbar: self.toolbar.hide()
		self.menuWindow.setItemChecked(self.menupopupShowToolbarID, self.showToolbar)

		self.menupopupShowWidgetToolbarID = self.menuWindow.insertItem( "Widget toolbar",  self.menuItemShowWidgetToolbar)
		if self.settings.has_key("showWidgetToolbar"): self.showWidgetToolbar = self.settings["showWidgetToolbar"]
		else:									self.showWidgetToolbar = TRUE
		if not self.showWidgetToolbar: self.widgetsToolBar.hide()
		self.menuWindow.setItemChecked(self.menupopupShowWidgetToolbarID, self.showWidgetToolbar)
		
		self.menuWindow.insertSeparator()
		self.menuWindow.insertItem("Show output window", self.menuItemShowOutputWindow)
		self.menuWindow.insertSeparator()

		self.menuWindow.insertItem("Minimize All", self.menuMinimizeAll)
		self.menuWindow.insertItem("Restore All", self.menuRestoreAll)
		self.menuWindow.insertItem("Close All", self.menuCloseAll)
		self.menuWindow.insertSeparator()

		self.menuBar = QMenuBar( self ) 
		self.menuBar.insertItem( "&File", self.menuFile )
		#self.menuBar.insertItem( "&Edit", self.menuEdit )
		self.menuBar.insertItem( "&Options", self.menuOptions )
		self.menuBar.insertItem("&Window", self.menuWindow)

		self.printer = QPrinter() 


	def showWindows(self):
		for id in self.windows:
			self.menuWindow.removeItem(id)
		self.windows = []
		wins = self.workspace.windowList()
		for win in wins:
			#id = self.menuWindow.insertItem(str(win.caption()), SLOT("windowsMenuActivated( int )"), self.activateWindow)
			id = self.menuWindow.insertItem(str(win.caption()), self.activateWindow)
			self.windows.append(id)

	def activateWindow(self, id):
		caption = self.menuWindow.text(id)
		winList = self.workspace.windowList()
		for win in winList:
			if str(win.caption()) == caption:
				win.setFocus()

	def menuItemNewSchema(self):
		win = orngDoc.SchemaDoc(self, self.workspace)
		win.createView()
		#win.show()
		#win.setFocus()
		return win

	def menuItemNewFromTemplate(self):
		return
	
	def menuItemNewWizard(self):
		return

	def menuItemOpen(self):
		name = QFileDialog.getOpenFileName( os.getcwd(), "Orange Widget Scripts (*.ows)", self, "", "Open File")
		if name.isEmpty():
			return
		win = self.menuItemNewSchema()
		win.loadDocument(str(name))
		self.addToRecentMenu(str(name))

	def menuItemClose(self):
		win = self.workspace.activeWindow()
		win.close()
		
	def menuItemSave(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocument()

	def menuItemSaveAs(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocumentAs()

	def menuItemSaveAsAppButtons(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocumentAsApp(asTabs = 0)

	def menuItemSaveAsAppTabs(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.saveDocumentAsApp(asTabs = 1)	

	def menuItemPrinter(self):
		win = self.workspace.activeWindow()
		if not isinstance(win, orngDoc.SchemaDoc):
			return
		if self.printer.setup(self):
			self.statusBar.message('Printing...')
			p = QPainter()
			p.begin(self.printer)
			p.scale(10,10)
			p.setFont(QFont('Times',1, QFont.Normal))

			for line in win.lines:
				line.printShape(p)
				
			for item in win.widgets:
				item.printShape(p)
			p.end()
			self.statusBar.message('')

	def readRecentFiles(self):
		if not self.settings.has_key("RecentFiles"): return
		recentDocs = self.settings["RecentFiles"]
		self.menuRecent.clear()

		for i in range(len(recentDocs)):
			name = recentDocs[i]
			shortName = os.path.basename(name)
			self.menuRecent.insertItem(shortName, eval("self.menuItemRecent"+str(i+1)))

	def openRecentFile(self, index):
		if len(self.settings["RecentFiles"]) >= index:
			win = self.menuItemNewSchema()
			win.loadDocument(self.settings["RecentFiles"][index-1])
			self.addToRecentMenu(self.settings["RecentFiles"][index-1])

	def addToRecentMenu(self, name):
		recentDocs = []
		if self.settings.has_key("RecentFiles"):
			recentDocs = self.settings["RecentFiles"]
		
		if name in recentDocs:
			recentDocs.remove(name)
		recentDocs.insert(0, name)
	
		if len(recentDocs)> 5:
			recentDocs.remove(recentDocs[5])
		self.settings["RecentFiles"] = recentDocs
		self.readRecentFiles()

	def menuItemRecent1(self):
		self.openRecentFile(1)

	def menuItemRecent2(self):
		self.openRecentFile(2)

	def menuItemRecent3(self):
		self.openRecentFile(3)

	def menuItemRecent4(self):
		self.openRecentFile(4)

	def menuItemRecent5(self):
		self.openRecentFile(5)

	def menuItemCut(self):
		return

	def menuItemCopy(self):
		return

	def menuItemPaste(self):
		return

	def menuItemSelectAll(self):
		return

	def menuItemGrid(self):
		return
		
	def menuItemShowGrid(self):
		return
		
	def menuItemSnapToGrid(self):
		self.snapToGrid = not self.snapToGrid
		self.settings["snapToGrid"] = self.snapToGrid
		self.menuOptions.setItemChecked(self.menupopupSnapToGridID, self.snapToGrid)

		# reposition all widgets in all documents so that the widgets are aligned
		if self.snapToGrid == TRUE:
			for win in self.workspace.windowList():
				if not isinstance(win, orngDoc.SchemaDoc): continue
				for widget in win.widgets:
					widget.setCoords(widget.x(), widget.y())
					widget.moveToGrid()
					widget.repaintAllLines()
				win.canvas.update()

	def menuItemLargeIcons(self):
		self.useLargeIcons = not self.useLargeIcons
		self.settings["useLargeIcons"] = self.useLargeIcons
		self.menuOptions.setItemChecked(self.menupopupLargeIconsID, self.useLargeIcons)
		self.tabs.hide()
		self.tabs = orngTabs.WidgetTabs(self.widgetsToolBar, 'tabs')
		self.tabs.setCanvasDlg(self)
		self.tabs.readInstalledWidgets(self.registryFileName, self.widgetDir, self.picsDir, self.defaultPic, self.useLargeIcons)
		
	def menuItemEnableAll(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.enableAllLines()
		return
		
	def menuItemDisableAll(self):
		win = self.workspace.activeWindow()
		if isinstance(win, orngDoc.SchemaDoc):
			win.disableAllLines()
		return

	def menuItemClearWidgets(self):
		win = self.workspace.activeWindow()
		if win != None:
			win.clear()

	def menuItemShowOutputWindow(self):
		#self.output.hide()
		#self.menuWindow.setItemChecked(self.menupopupShowOutputWindowID, 1)
		self.output.show()
		#self.output.setActiveWindow()
		self.output.setFocus()
		#self.output.clearFocus()

	def menuItemShowToolbar(self):
		self.showToolbar = not self.showToolbar
		self.settings["showToolbar"] = self.showToolbar
		self.menuWindow.setItemChecked(self.menupopupShowToolbarID, self.showToolbar)
		if self.showToolbar: self.toolbar.show()
		else: self.toolbar.hide()

	def menuItemShowWidgetToolbar(self):
		self.showWidgetToolbar = not self.showWidgetToolbar
		self.settings["showWidgetToolbar"] = self.showWidgetToolbar
		self.menuWindow.setItemChecked(self.menupopupShowWidgetToolbarID, self.showWidgetToolbar)
		if self.showWidgetToolbar: self.widgetsToolBar.show()
		else: self.widgetsToolBar.hide()

	def menuItemPreferences(self):
		dlg = orngDlgs.PreferencesDlg(self, None, "", TRUE)
		dlg.exec_loop()
		if dlg.result() == QDialog.Accepted:
			self.rebuildSignals()

	def menuItemRebuildWidgetRegistry(self):
		self.createWidgetsToolbar(TRUE)
		
	def menuCloseAll(self):
		wins = self.workspace.windowList()
		for win in wins:
			win.close()
			
	def menuMinimizeAll(self):
		win.showMinimized()
			
	def menuRestoreAll(self):
		win.showNormal()

	def rebuildSignals(self):
		self.channels = {}
		if self.settings.has_key("Channels"):
			channels = self.settings["Channels"]
			for (key, value) in channels.items():
				items = value.split("::")
				self.channels[key] = items

	def getChannelName(self, symbName):
		if self.channels.has_key(symbName):
			return (self.channels[symbName])[0]
		return symbName
	
	def getChannelInfo(self, symbName):
		if self.channels.has_key(symbName):
			return self.channels[symbName]
		else:
			return [symbName, str(1), "green"]

	def focusDocument(self, w):	
		if w != None:
			w.setFocus()


	def menuItemCanvasOptions(self):
		dlg = orngDlgs.CanvasOptionsDlg(self, None, "", TRUE)

		# set current settings
		dlg.catchExceptionCB.setChecked(self.settings["catchException"])
		dlg.focusOnCatchExceptionCB.setChecked(self.settings["focusOnCatchException"])
		dlg.catchOutputCB.setChecked(self.settings["catchOutput"])
		dlg.focusOnCatchOutputCB.setChecked(self.settings["focusOnCatchOutput"])
        
		dlg.exec_loop()
		if dlg.result() == QDialog.Accepted:
			self.settings["catchException"] = dlg.catchExceptionCB.isChecked()
			self.settings["catchOutput"] = dlg.catchOutputCB.isChecked()
			self.settings["focusOnCatchException"] = dlg.focusOnCatchExceptionCB.isChecked()
			self.settings["focusOnCatchOutput"] = dlg.focusOnCatchOutputCB.isChecked()
			self.output.catchException(self.settings["catchException"])
			self.output.catchOutput(self.settings["catchOutput"])
			self.output.setFocusOnException(self.settings["focusOnCatchException"])
			self.output.setFocusOnOutput(self.settings["focusOnCatchOutput"])


	def setStatusBarEvent(self, text):
		if text == "":
			self.statusBar.message("")
			return
		elif text == "\n": return
		self.statusBar.message(QString("Last event: " + text))
		
	#######################
	# SETTINGS
	#######################

	# Loads settings from the widget's .ini file	
	def loadSettings(self):
		filename = self.canvasDir + "orngCanvas.ini"
		if os.path.exists(filename):
			file = open(filename)
			self.settings = cPickle.load(file)
			file.close()
		else:
			self.settings = {}

		if not self.settings.has_key("catchException"): self.settings["catchException"] = 1
		if not self.settings.has_key("catchOutput"): self.settings["catchOutput"] = 1
		if not self.settings.has_key("focusOnCatchException"): self.settings["focusOnCatchException"] = 1
		if not self.settings.has_key("focusOnCatchOutput"): self.settings["focusOnCatchOutput"] = 0
				

	# Saves settings to this widget's .ini file
	def saveSettings(self):
		filename = self.canvasDir + "orngCanvas.ini"
		file=open(filename, "w")
		cPickle.dump(self.settings, file)
		file.close()


	#######################
	# EVENTS
	#######################

	def closeEvent(self,ce):
		for win in self.workspace.windowList():
			win.close()

		self.saveSettings()
		if len(self.workspace.windowList()) == 0:
			ce.accept()
		else:
			ce.ignore()

	def keyPressEvent(self, e):
		if e.state() & Qt.ControlButton != 0:
			self.ctrlPressed = 1

	def keyReleaseEvent(self, e):
		self.ctrlPressed = 0

	def enableSave(self, enable):
		self.toolSave.setEnabled(enable)
		self.menuFile.setItemEnabled(self.menuSaveID, enable)
		self.menuFile.setItemEnabled(self.menuSaveAsID, enable)

app = QApplication(sys.argv) 
dlg = OrangeCanvasDlg()
app.setMainWidget(dlg)
dlg.show()
app.exec_loop() 