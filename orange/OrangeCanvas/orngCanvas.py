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
import orngResources
import xmlParse
import cPickle

TRUE  = 1
FALSE = 0

class MDIWorkspace(QWorkspace):
	def __init__(self,*args):
		apply(QWorkspace.__init__,(self,) + args)
		self.setBackgroundColor(QColor(255,255,255))
		
class OrangeCanvasDlg(QMainWindow):
	def __init__(self,*args):
		apply(QMainWindow.__init__,(self,) + args)
		self.resize(800,700)
		self.setCaption("Orange Canvas")

		# if widget path not registered -> register
		self.widgetDir = sys.prefix + "/lib/site-packages/Orange/OrangeWidgets/"
		self.picsDir = sys.prefix + "/lib/site-packages/Orange/OrangeWidgets/icons/"
		self.defaultPic = self.picsDir + "Unknown.png"
		self.canvasDir = sys.prefix + "/lib/site-packages/Orange/OrangeCanvas/"
		self.registryFileName = self.canvasDir + "widgetregistry.xml"
		

		if sys.path.count(self.widgetDir) == 0:
			sys.path.append(self.widgetDir)

		self.workspace = MDIWorkspace(self)
		self.setCentralWidget(self.workspace)
		self.statusBar = QStatusBar(self)

		self.initMenu()

		self.toolbar = QToolBar(self, 'test')
		toolNew  = QToolButton(QPixmap(orngResources.file_new), "New schema" , QString.null, self.menuItemNewSchema, self.toolbar, 'new schema') 
		toolOpen = QToolButton(QPixmap(orngResources.file_open), "Open schema" , QString.null, self.menuItemOpen , self.toolbar, 'open schema') 
		toolSave = QToolButton(QPixmap(orngResources.file_save), "Save schema" ,QString.null, self.menuItemSave, self.toolbar, 'save schema')
		self.toolbar.addSeparator()
		toolPrint = QToolButton(QPixmap(orngResources.file_print), "Print" ,QString.null, self.menuItemPrinter, self.toolbar, 'print')
		self.addToolBar(self.toolbar, "Toolbar", QMainWindow.Top, TRUE)
		
		widgetsToolBar = QToolBar( self, 'Widgets')
		widgetsToolBar.setHorizontalStretchable(TRUE)
		self.tabs = orngTabs.WidgetTabs(widgetsToolBar, 'tabs')
		self.addToolBar(widgetsToolBar, "Widgets", QMainWindow.Top, TRUE)
		
		self.settings = {}
		self.loadSettings()
		self.rebuildSignals()
		
		self.recentDocs = []
		self.tabs.setCanvasDlg(self)
		
		# if registry doesn't exist yet, we create it
		if not os.path.exists(self.registryFileName):
			parse = xmlParse.WidgetsToXML()
			parse.ParseDirectory(self.widgetDir, self.canvasDir)
			
		# if registry still doesn't exist then something is very wrong...
		if not os.path.exists(self.registryFileName):
			QMessageBox.error( None, "Orange Canvas", "Unable to create widget registry. Exiting...", QMessageBox.Ok + QMessageBox.Default )
			self.quit()
			
		# read widget registry file and create tab with buttons
		self.tabs.readInstalledWidgets(self.registryFileName, self.widgetDir, self.picsDir, self.defaultPic)
		self.readRecentFiles()

		self.menuItemSnapToGrid()
		win = self.menuItemNewSchema()
		
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
		self.menuFile.insertItem(QIconSet(QPixmap(orngResources.file_save)), "&Save", self.menuItemSave, Qt.CTRL+Qt.Key_S )
		self.menuFile.insertItem( "&Save As..", self.menuItemSaveAs)
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
		self.menupopupSnapToGridID = self.menuOptions.insertItem( "Snap to Grid",  self.menuItemSnapToGrid )
		self.snapToGrid = FALSE
		self.menuOptions.setItemChecked(self.menupopupSnapToGridID, self.snapToGrid)
		self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Enable All",  self.menuItemEnableAll)
		self.menuOptions.insertItem( "Disable All",  self.menuItemDisableAll)
		self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Channel preferences",  self.menuItemPreferences)
		self.menuOptions.insertSeparator()
		self.menuOptions.insertItem( "Rebuild widget registry",  self.menuItemRebuildWidgetRegistry)
		

		self.menuWindow = QPopupMenu( self )		
		self.menuWindow.insertItem("Cascade", self.workspace.cascade)
		self.menuWindow.insertItem("Tile", self.workspace.tile)
		self.menuWindow.insertSeparator()
		self.menuWindow.insertItem("Close All", self.menuCloseAll)
		self.menuWindow.insertSeparator()
		self.menuWindow.insertItem("Minimize All", self.menuMinimizeAll)
		self.menuWindow.insertItem("Restore All", self.menuRestoreAll)

		self.menuBar = QMenuBar( self ) 
		self.menuBar.insertItem( "&File", self.menuFile )
		#self.menuBar.insertItem( "&Edit", self.menuEdit )
		self.menuBar.insertItem( "&Options", self.menuOptions )
		self.menuBar.insertItem("&Window", self.menuWindow)

		self.printer = QPrinter() 

	def menuItemNewSchema(self):
		win = orngDoc.SchemaDoc(self, self.workspace)
		win.show()
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
		win.saveDocument()

	def menuItemSaveAs(self):
		win = self.workspace.activeWindow()
		win.saveDocumentAs()

	def menuItemSaveAsAppButtons(self):
		win = self.workspace.activeWindow()
		win.saveDocumentAsApp(asTabs = 0)

	def menuItemSaveAsAppTabs(self):
		win = self.workspace.activeWindow()
		win.saveDocumentAsApp(asTabs = 1)	

	def menuItemPrinter(self):
		win = self.workspace.activeWindow()
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
		recentDocs = []
		try:
			recentDocs = self.settings["RecentFiles"]
		except:
			pass
		
		self.menuRecent.clear()

		for i in range(5):
			try:
				name = recentDocs[i]
				shortName = os.path.basename(name)
				self.menuRecent.insertItem(shortName, eval("self.menuItemRecent"+str(i+1)))
			except:
				pass

	def openRecentFile(self, index):
		if len(self.settings["RecentFiles"]) >= index:
			win = self.menuItemNewSchema()
			win.loadDocument(self.settings["RecentFiles"][index-1])

	def addToRecentMenu(self, name):
		recentDocs = []
		try:
			recentDocs = self.settings["RecentFiles"]
		except:
			pass

		if name not in recentDocs:
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
		self.menuOptions.setItemChecked(self.menupopupSnapToGridID, self.snapToGrid)

		# reposition all widgets in all documents so that the widgets are aligned
		if self.snapToGrid == TRUE:
			for win in self.workspace.windowList():
				for widget in win.widgets:
					widget.setCoords(widget.x(), widget.y())
					widget.moveToGrid()
					widget.repaintAllLines(win.canvasView)
				win.canvas.update()
		
		
	def menuItemEnableAll(self):
		win = self.workspace.activeWindow()
		win.enableAllLines()
		return
		
	def menuItemDisableAll(self):
		win = self.workspace.activeWindow()
		win.disableAllLines()
		return

	def menuItemPreferences(self):
		dlg = orngDlgs.PreferencesDlg(self, None, "", TRUE)
		dlg.exec_loop()
		if dlg.result() == QDialog.Accepted:
			self.rebuildSignals()

	def menuItemRebuildWidgetRegistry(self):
		parse = xmlParse.WidgetsToXML()
		parse.ParseDirectory(self.widgetDir, self.canvasDir)
		QMessageBox.information( None, "Orange Canvas", "Rebuild completed.\nYou have to restart Orange Canvas for changes to take effect.", QMessageBox.Ok + QMessageBox.Default ) 
		
	def menuCloseAll(self):
		wins = self.workspace.windowList()
		for win in wins:
			win.close()
			
	def menuMinimizeAll(self):
		for win in self.workspace.windowList():
			win.showMinimized()
			
	def menuRestoreAll(self):
		for win in self.workspace.windowList():
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

	# Loads settings from the widget's .ini file	
	def loadSettings(self):
		filename = self.canvasDir + "orngCanvas.ini"
		if os.path.exists(filename):
			file = open(filename)
			self.settings = cPickle.load(file)
			file.close()
		else:
			self.settings = {}

	# Saves settings to this widget's .ini file
	def saveSettings(self):
		filename = self.canvasDir + "orngCanvas.ini"
		file=open(filename, "w")
		cPickle.dump(self.settings, file)
		file.close()

	def closeEvent(self,ce):
		for win in self.workspace.windowList():
			win.close()

		self.saveSettings()
		if len(self.workspace.windowList()) == 0:
			ce.accept()
		else:
			ce.ignore()			

app = QApplication(sys.argv) 
dlg = OrangeCanvasDlg()
app.setMainWidget(dlg)
dlg.show()
app.exec_loop() 