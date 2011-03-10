# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    main file, that creates the MDI environment

# This module has to be imported first because it takes care of the system PATH variable
# Namely, it throws out the MikTeX directories which contain an incompatible Qt .dll's
import orngEnviron, orngAddOns

from PyQt4.QtCore import *
from PyQt4.QtGui import *
    
import sys, os, cPickle, orngRegistry, OWGUI
import orngTabs, orngDoc, orngDlgs, orngOutput, orngHelp, OWReport
import  user, orngMisc

RedR = False
product = "Red R" if RedR else "Orange"


class OrangeCanvasDlg(QMainWindow):
    def __init__(self, app, parent=None, flags=0):
        QMainWindow.__init__(self, parent)
        self.debugMode = 1        # print extra output for debuging
        self.setWindowTitle("%s Canvas" % product)
        self.windows = []    # list of id for windows in Window menu
        self.recentDocs = []
        self.iconNameToIcon = {}
        self.toolbarIconSizeList = [16, 32, 40, 48, 60]
        self.schemeIconSizeList = [32, 40, 48]
        self.widgetsToolBar = None
        self.originalPalette = QApplication.palette()

        self.__dict__.update(orngEnviron.directoryNames)
               
        self.defaultPic = os.path.join(self.picsDir, "Unknown.png")
        self.defaultBackground = os.path.join(self.picsDir, "frame.png")
        canvasPicsDir = os.path.join(self.canvasDir, "icons")
        self.file_new = os.path.join(canvasPicsDir, "doc.png")
        self.outputPix = os.path.join(canvasPicsDir, "output.png")
        self.file_open = os.path.join(canvasPicsDir, "open.png")
        self.file_save = os.path.join(canvasPicsDir, "save.png")
        if RedR:
            self.reload_pic = os.path.join(canvasPicsDir, "update1.png")
        self.text_icon = os.path.join(canvasPicsDir, "text.png")
        self.file_print = os.path.join(canvasPicsDir, "print.png")
        self.file_exit = os.path.join(canvasPicsDir, "exit.png")
        canvasIconName = os.path.join(canvasPicsDir, "CanvasIcon.png")
        if os.path.exists(canvasIconName):
            self.setWindowIcon(QIcon(canvasIconName))
            
        self.settings = {}
        if RedR:
            self.settings['svnSettings'] = {}
            self.settings['versionNumber'] = 'Version1.0'
        self.menuSaveSettingsID = -1
        self.menuSaveSettings = 1

        self.loadSettings()
        if RedR:
            import updateRedR
            self.settings['svnSettings'], self.settings['versionNumber'] = updateRedR.start(self.settings['svnSettings'], self.settings['versionNumber'])
           
#        self.widgetSelectedColor = QColor(*self.settings["widgetSelectedColor"])
#        self.widgetActiveColor = QColor(*self.settings["widgetActiveColor"])
#        self.lineColor = QColor(*self.settings["lineColor"])

        if not self.settings.has_key("WidgetTabs") or self.settings["WidgetTabs"] == []:
            f = open(os.path.join(self.canvasDir, "WidgetTabs.txt"), "r")
            defaultTabs = [c for c in [line.split("#")[0].strip() for line in f.readlines()] if c!=""]
            for i in xrange(len(defaultTabs)-1,0,-1):
                if defaultTabs[i] in defaultTabs[0:i]:
                    del defaultTabs[i]
            self.settings["WidgetTabs"] = [(name, Qt.Checked) for name in defaultTabs] + [("Prototypes", Qt.Unchecked)] 
        
        # output window
        self.output = orngOutput.OutputWindow(self)
        self.output.catchException(1)
        self.output.catchOutput(1)

        # create error and warning icons
        informationIconName = os.path.join(canvasPicsDir, "information.png")
        warningIconName = os.path.join(canvasPicsDir, "warning.png")
        errorIconName = os.path.join(canvasPicsDir, "error.png")
        if os.path.exists(errorIconName) and os.path.exists(warningIconName) and os.path.exists(informationIconName):
            self.errorIcon = QPixmap(errorIconName)
            self.warningIcon = QPixmap(warningIconName)
            self.informationIcon = QPixmap(informationIconName)
            self.widgetIcons = {"Info": self.informationIcon, "Warning": self.warningIcon, "Error": self.errorIcon}
        else:
            self.errorIcon = None
            self.warningIcon = None
            self.informationIcon = None
            self.widgetIcons = None
            print "Unable to load all necessary icons. Please reinstall Orange."

        self.setStatusBar(MyStatusBar(self))
                
        self.widgetRegistry = orngRegistry.readCategories()
        self.updateStyle()
        
        # create toolbar
        self.toolbar = self.addToolBar("Toolbar")
        self.toolbar.setOrientation(Qt.Horizontal)
#        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if not self.settings.get("showToolbar", True):
            self.toolbar.hide()
        
        # create a schema
        self.schema = orngDoc.SchemaDoc(self)
        self.setCentralWidget(self.schema)
        self.schema.setFocus()

        self.toolbar.addAction(QIcon(self.file_open), "Open schema", self.menuItemOpen)
#        self.toolbar.addAction(QIcon(self.style().standardIcon(QStyle.SP_FileIcon)), "Open schema", self.menuItemOpen)
        self.toolSave = self.toolbar.addAction(QIcon(self.file_save), "Save schema", self.menuItemSave)
#        self.toolSave = self.toolbar.addAction(QIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton)), "Save schema", self.menuItemSave)
        if RedR:
            self.toolReloadWidgets = self.toolbar.addAction(QIcon(self.reload_pic), "Reload Widgets", self.reloadWidgets)
        self.toolbar.addSeparator()
        self.toolbar.addAction(QIcon(self.file_print), "Print", self.menuItemPrinter)
        self.toolbar.addSeparator()
        w = QWidget()
        w.setLayout(QHBoxLayout())
        
        items = ["Tool box", "Tree view", "Tree view (no icons)", "Tabs without labels", "Tabs with labels"]
        ind = max(len(items) - 1, self.settings["widgetListType"])
        self.widgetListTypeCB = OWGUI.comboBox(w, self.settings, "widgetListType", label="Style:", orientation="horizontal", items=items, callback=self.createWidgetsToolbar, debuggingEnabled=0)
        self.widgetListTypeCB.setFocusPolicy(Qt.TabFocus)
        self.toolbar.addWidget(w)
        
        self.toolbar.addSeparator()

        w = QWidget()
        w.setLayout(QHBoxLayout())
        items = ["%d x %d" % (v, v) for v in self.toolbarIconSizeList]
        self.settings["toolbarIconSize"] = min(len(items) - 1, self.settings["toolbarIconSize"])
        cb = OWGUI.comboBoxWithCaption(w, self.settings, "toolbarIconSize", "Icon size:", items=items, tooltip="Set the size of the widget icons in the toolbar, tool box, and tree view area", callback=self.createWidgetsToolbar, debuggingEnabled=0)
        cb.setFocusPolicy(Qt.TabFocus)
        
        self.toolbar.addWidget(w)
        
        self.freezeAction = self.toolbar.addAction("Freeze signals")
        self.freezeAction.setCheckable(True)
        self.freezeAction.setIcon(QIcon(self.style().standardIcon(QStyle.SP_MediaPause))) #self.schema_pause))
        
        def toogleSchemaFreeze(freeze):
            self.freezeAction.setIcon(QIcon(self.style().standardIcon(QStyle.SP_MediaPlay if freeze else QStyle.SP_MediaPause))) #self.schema_pause))
            self.schema.setFreeze(freeze)
            
        self.connect(self.freezeAction, SIGNAL("toggled(bool)"), toogleSchemaFreeze) #lambda bool: self.schema.setFreeze(bool))
        
        # Restore geometry before calling createWidgetsToolbar.
        # On Mac OSX with unified title bar the canvas can move up on restarts
        state = self.settings.get("CanvasMainWindowGeometry", None)
        if state is not None:
            state = self.restoreGeometry(QByteArray(state))
            width, height = self.width(), self.height()
        
        if not state:
            width, height = self.settings.get("canvasWidth", 700), self.settings.get("canvasHeight", 600)

        # center window in the desktop
        # on multiheaded desktops it it does not fit
        
        desktop = qApp.desktop()
        space = desktop.availableGeometry(self)
        geometry, frame = self.geometry(), self.frameGeometry()
        
        #Fit the frame size to fit in space
        width = min(space.width() - (frame.width() - geometry.width()), geometry.width())
        height = min(space.height() - (frame.height() - geometry.height()), geometry.height())
        
        self.resize(width, height)
        
        self.addToolBarBreak()
        orngTabs.constructCategoriesPopup(self)
        self.createWidgetsToolbar()
        orngTabs.constructCategoriesPopup(self)
        self.readShortcuts()
        
        def addOnRefreshCallback():
            self.widgetRegistry = orngRegistry.readCategories()
            orngTabs.constructCategoriesPopup(self)
            self.createWidgetsToolbar()
        orngAddOns.addOnRefreshCallback.append(addOnRefreshCallback)

        # create menu
        self.initMenu()
        self.readRecentFiles()

        
        
        #move to center if frame not fully contained in space
        if not space.contains(self.frameGeometry()):
            x = max(0, space.width() / 2 - width / 2)
            y = max(0, space.height() / 2 - height / 2)
            
            self.move(x, y)

        self.helpWindow = orngHelp.HelpWindow(self)
        self.reportWindow = OWReport.ReportWindow()
        self.reportWindow.widgets = self.schema.widgets
        self.reportWindow.saveDir = self.settings["reportsDir"]
        

        # did Orange crash the last time we used it? If yes, you will find a temSchema.tmp file
        if not RedR:
            if os.path.exists(os.path.join(self.canvasSettingsDir, "tempSchema.tmp")):
                mb = QMessageBox('%s Canvas' % product, "Your previous %s Canvas session was not closed successfully.\nYou can choose to reload your unsaved work or start a new session.\n\nIf you choose 'Reload', the links will be disabled to prevent reoccurence of the crash.\nYou can enable them by clicking Options/Enable all links." % product, QMessageBox.Information, QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape, QMessageBox.NoButton)
                mb.setButtonText(QMessageBox.Ok, "Reload")
                mb.setButtonText(QMessageBox.Cancel, "New schema")
                if mb.exec_() == QMessageBox.Ok:
                    self.schema.loadDocument(os.path.join(self.canvasSettingsDir, "tempSchema.tmp"), freeze=1)
        
        if self.schema.widgets == [] and len(sys.argv) > 1 and os.path.exists(sys.argv[-1]) and os.path.splitext(sys.argv[-1])[1].lower() == ".ows":
            self.schema.loadDocument(sys.argv[-1])
        
        # show message box if no numpy
        qApp.processEvents()
        try:
            import numpy
        except:
            if QMessageBox.warning(self, '%s Canvas' % product, 'Several widgets now use numpy module, \nthat is not yet installed on this computer. \nDo you wish to download it?', QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape) == QMessageBox.Ok:
                import webbrowser
                webbrowser.open("http://sourceforge.net/projects/numpy/")

    def createWidgetsToolbar(self):
        if self.widgetsToolBar:
            self.settings["showWidgetToolbar"] = self.widgetsToolBar.isVisible()
            if isinstance(self.widgetsToolBar, QToolBar):
                self.removeToolBar(self.widgetsToolBar)
            elif isinstance(self.widgetsToolBar, orngTabs.WidgetToolBox):
                self.settings["toolboxWidth"] = self.widgetsToolBar.toolbox.width()
                self.removeDockWidget(self.widgetsToolBar)
            elif isinstance(self.widgetsToolBar, orngTabs.WidgetTree):
                self.settings["toolboxWidth"] = self.widgetsToolBar.treeWidget.width()
                self.removeDockWidget(self.widgetsToolBar)
            
        if self.settings["widgetListType"] == 0:
            self.tabs = self.widgetsToolBar = orngTabs.WidgetToolBox(self, self.widgetRegistry)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.widgetsToolBar)
        elif self.settings["widgetListType"] in [1, 2]:
            self.tabs = self.widgetsToolBar = orngTabs.WidgetTree(self, self.widgetRegistry)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.widgetsToolBar)
        else:
            if sys.platform == "darwin":
                self.setUnifiedTitleAndToolBarOnMac(False)   
            self.widgetsToolBar = self.addToolBar("Widgets")
            self.insertToolBarBreak(self.widgetsToolBar)
            self.tabs = orngTabs.WidgetTabs(self, self.widgetRegistry, self.widgetsToolBar)
            self.widgetsToolBar.addWidget(self.tabs)
            
        if sys.platform == "darwin":
            self.setUnifiedTitleAndToolBarOnMac(self.settings["widgetListType"] in [0, 1, 2] and self.settings["style"].lower() == "macintosh (aqua)")

        # find widgets and create tab with buttons
        self.settings["WidgetTabs"] = self.tabs.createWidgetTabs(self.settings["WidgetTabs"], self.widgetRegistry, self.widgetDir, self.picsDir, self.defaultPic)
        if not self.settings.get("showWidgetToolbar", True): 
            self.widgetsToolBar.hide()


    def readShortcuts(self):
        self.widgetShortcuts = {}
        shfn = os.path.join(self.canvasSettingsDir, "shortcuts.txt")
        if os.path.exists(shfn):
            for t in file(shfn).readlines():
                key, info = [x.strip() for x in t.split(":")]
                if len(info) == 0: continue
                if info[0] == "(" and info[-1] == ")":
                    cat, widgetName = eval(info)            # new style of shortcuts are of form F: ("Data", "File")
                else:
                    cat, widgetName = info.split(" - ")   # old style of shortcuts are of form F: Data - File
                if self.widgetRegistry.has_key(cat) and self.widgetRegistry[cat].has_key(widgetName):
                    self.widgetShortcuts[key] = self.widgetRegistry[cat][widgetName]


    def initMenu(self):
        self.menuRecent = QMenu("Recent Schemas", self)

        self.menuFile = QMenu("&File", self)
        self.menuFile.addAction("New Scheme", self.menuItemNewScheme, QKeySequence.New)
        self.menuFile.addAction(QIcon(self.file_open), "&Open...", self.menuItemOpen, QKeySequence.Open)
        self.menuFile.addAction(QIcon(self.file_open), "&Open and Freeze...", self.menuItemOpenFreeze)
        if RedR:
            self.menuFile.addAction("Import Schema", self.importSchema)
        if os.path.exists(os.path.join(self.canvasSettingsDir, "lastSchema.tmp")):
            self.menuFile.addAction("Reload Last Schema", self.menuItemOpenLastSchema, Qt.CTRL + Qt.Key_R)
        #self.menuFile.addAction( "&Clear", self.menuItemClear)
        self.menuFile.addSeparator()
        self.menuReportID = self.menuFile.addAction("&Report", self.menuItemReport, Qt.CTRL + Qt.ALT + Qt.Key_R)
        self.menuFile.addSeparator()
        self.menuSaveID = self.menuFile.addAction(QIcon(self.file_save), "&Save", self.menuItemSave, QKeySequence.Save)
        self.menuSaveAsID = self.menuFile.addAction("Save &as...", self.menuItemSaveAs)
        if not RedR:
            self.menuFile.addAction("&Save as Application (Tabs)...", self.menuItemSaveAsAppTabs)
            self.menuFile.addAction("&Save as Application (Buttons)...", self.menuItemSaveAsAppButtons)
        self.menuFile.addSeparator()
        self.menuFile.addAction(QIcon(self.file_print), "&Print Schema / Save image", self.menuItemPrinter, QKeySequence.Print)
        self.menuFile.addSeparator()
        self.menuFile.addMenu(self.menuRecent)
        self.menuFile.addSeparator()
        self.menuFile.addAction("E&xit", self.close, Qt.CTRL + Qt.Key_Q)

        self.menuOptions = QMenu("&Options", self)
        self.menuOptions.addAction("Enable All Links", self.menuItemEnableAll, Qt.CTRL + Qt.Key_E)
        self.menuOptions.addAction("Disable All Links", self.menuItemDisableAll, Qt.CTRL + Qt.Key_D)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction("Show Output Window", self.menuItemShowOutputWindow)
        self.menuOptions.addAction("Clear Output Window", self.menuItemClearOutputWindow)
        self.menuOptions.addAction("Save Output Text...", self.menuItemSaveOutputWindow)
        if RedR:
            self.menuOptions.addAction("Set to debug mode", self.setDebugMode)
        
        # uncomment this only for debugging
#        self.menuOptions.addSeparator()
#        self.menuOptions.addAction("Dump widget variables", self.dumpVariables)

        self.menuOptions.addSeparator()
#        self.menuOptions.addAction( "Channel preferences",  self.menuItemPreferences)
        #self.menuOptions.addSeparator()
        self.menuOptions.addAction("&Customize Shortcuts", self.menuItemEditWidgetShortcuts)
        self.menuOptions.addAction("&Delete Widget Settings", self.menuItemDeleteWidgetSettings)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction(sys.platform == "darwin" and "&Preferences..." or "Canvas &Options...", self.menuItemCanvasOptions)
        self.menuOptions.addAction("&Add-ons...", self.menuItemAddOns)

        localHelp = 0
        self.menuHelp = QMenu("&Help", self)
        if os.path.exists(os.path.join(self.orangeDir, r"doc/reference/default.htm")): self.menuHelp.addAction("Orange Help", self.menuOpenLocalOrangeHelp)
        if os.path.exists(os.path.join(self.orangeDir, r"doc/catalog/index.html")): self.menuHelp.addAction("Orange Widget Catalog", self.menuOpenLocalWidgetCatalog)
        if os.path.exists(os.path.join(self.orangeDir, r"doc/canvas/default.htm")): self.menuHelp.addAction("Orange Canvas Help", self.menuOpenLocalCanvasHelp)

        self.menuHelp.addAction("Orange Online Widget Catalog", self.menuOpenOnlineOrangeHelp)
        #self.menuHelp.addAction("Orange Canvas Online Help", self.menuOpenOnlineCanvasHelp)

        if os.path.exists(os.path.join(self.orangeDir, r"updateOrange.py")):
            self.menuHelp.addSeparator()
            self.menuHelp.addAction("Check for updates", self.menuCheckForUpdates)
            
        self.menuHelp.addSeparator()
        self.menuHelp.addAction("About Orange", self.menuItemAboutOrange)

        # widget popup menu
        self.widgetPopup = QMenu("Widget", self)
        self.openActiveWidgetAction = self.widgetPopup.addAction("Open", self.schema.canvasView.openActiveWidget)
        self.widgetPopup.addSeparator()
        self.renameActiveWidgetAction = rename = self.widgetPopup.addAction("&Rename", self.schema.canvasView.renameActiveWidget, Qt.Key_F2)
        self.removeActiveWidgetAction = delete = self.widgetPopup.addAction("Remove", self.schema.canvasView.removeActiveWidget, Qt.Key_Delete)
        if sys.platform != "darwin":
            delete.setShortcuts([Qt.Key_Delete, Qt.CTRL + Qt.Key_Backspace, Qt.CTRL + Qt.Key_Delete])
        else:
            delete.setShortcuts([Qt.CTRL + Qt.Key_Backspace, Qt.Key_Delete, Qt.CTRL + Qt.Key_Delete])
        self.widgetPopup.addSeparator()
        self.helpActiveWidgetAction = self.widgetPopup.addAction("Help", self.schema.canvasView.helpOnActiveWidget, Qt.Key_F1)
        self.widgetPopup.setEnabled(0)
        
        if sys.platform == "darwin":
            self.windowPopup = QMenu("Window", self)
            self.windowPopup.addAction("Minimize", self.showMinimized, Qt.CTRL + Qt.Key_M)
            self.windowPopup.addAction("Zoom", self.showMaximized, 0)

        self.menuBar = QMenuBar(self)
        self.menuBar.addMenu(self.menuFile)
        self.menuBar.addMenu(self.menuOptions)
        self.menuBar.addMenu(self.widgetPopup)
        
        if hasattr(self, "windowPopup"):
            self.menuBar.addMenu(self.windowPopup)
            
        self.menuBar.addMenu(self.menuHelp)
        
        self.setMenuBar(self.menuBar)
        
    def setDebugMode(self):   # RedR specific
        if self.output.debugMode:
            self.output.debugMode = 0
        else:
            self.output.debugMode = 1
    def importSchema(self):   # RedR specific
        name = QFileDialog.getOpenFileName(self, "Import File", self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)")
        if name.isEmpty():
            return
        self.schema.clear()
        self.schema.loadDocument(str(name), freeze = 0, importBlank = 1)
        self.addToRecentMenu(str(name))
    
    def openSchema(self, filename):
        if self.schema.isSchemaChanged() and self.schema.widgets:
            ret = QMessageBox.warning(self, "Orange Canvas", "Changes to your present schema are not saved.\nSave them?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if ret == QMessageBox.Save:
                self.schema.saveDocument()
            elif ret == QMessageBox.Cancel:
                return
        self.schema.clear()
        dirname = os.path.dirname(filename)
        os.chdir(dirname)
        self.schema.loadDocument(filename)
        
    def menuItemOpen(self):
        name = QFileDialog.getOpenFileName(self, "Open Orange Schema", self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)")
        if name.isEmpty():
            return
        self.schema.clear()
        dirname = os.path.dirname(str(name))
        os.chdir(dirname)
        self.schema.loadDocument(str(name), freeze=0)
        self.addToRecentMenu(str(name))

    def menuItemOpenFreeze(self):
        name = QFileDialog.getOpenFileName(self, "Open Orange Schema", self.settings["saveSchemaDir"], "Orange Widget Scripts (*.ows)")
        if name.isEmpty():
            return
        self.schema.clear()
        dirname = os.path.dirname(str(name))
        os.chdir(dirname)
        self.schema.loadDocument(str(name), freeze=1)
        self.addToRecentMenu(str(name))

    def menuItemOpenLastSchema(self):
        fullName = os.path.join(self.canvasSettingsDir, "lastSchema.tmp")
        if os.path.exists(fullName):
            self.schema.loadDocument(fullName)

    def menuItemReport(self):
        self.schema.reportAll()
        
    def menuItemSave(self):
        self.schema.saveDocument()
        
    def menuItemSaveAs(self):
        self.schema.saveDocumentAs()

    def menuItemSaveAsAppButtons(self):
        self.schema.saveDocumentAsApp(asTabs=0)

    def menuItemSaveAsAppTabs(self):
        self.schema.saveDocumentAsApp(asTabs=1)

    def menuItemPrinter(self):
        try:
            import OWDlgs
            sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.schema.canvas, defaultName=self.schema.schemaName or "schema", parent=self)
            sizeDlg.exec_()
        except:
            print "Missing file 'OWDlgs.py'. This file should be in OrangeWidgets folder. Unable to print/save image."
        

    def readRecentFiles(self):
        self.menuRecent.clear()
        if not self.settings.has_key("RecentFiles"): return
        recentDocs = self.settings["RecentFiles"]

        # remove missing recent files
        for i in range(len(recentDocs) - 1, -1, -1):
            if not os.path.exists(recentDocs[i]):
                recentDocs.remove(recentDocs[i])

        recentDocs = recentDocs[:9]
        self.settings["RecentFiles"] = recentDocs

        for i in range(len(recentDocs)):
            shortName = "&" + str(i + 1) + " " + os.path.basename(recentDocs[i])
            self.menuRecent.addAction(shortName, lambda ind=i: self.openRecentFile(ind + 1))

    def openRecentFile(self, index):
        if len(self.settings["RecentFiles"]) >= index:
            self.schema.clear()
            name = self.settings["RecentFiles"][index - 1]
            dirname = os.path.dirname(name)
            os.chdir(dirname)
            self.schema.loadDocument(name)
            self.addToRecentMenu(name)

    def addToRecentMenu(self, name):
        recentDocs = []
        if self.settings.has_key("RecentFiles"):
            recentDocs = self.settings["RecentFiles"]

        # convert to a valid file name
        name = os.path.realpath(name)

        if name in recentDocs:
            recentDocs.remove(name)
        recentDocs.insert(0, name)

        if len(recentDocs) > 5:
            recentDocs.remove(recentDocs[5])
        self.settings["RecentFiles"] = recentDocs
        self.readRecentFiles()

    def menuItemSelectAll(self):
        return

    def updateSnapToGrid(self):
        if self.settings["snapToGrid"]:
            for widget in self.schema.widgets:
                widget.setCoords(widget.x(), widget.y())
            self.schema.canvas.update()

    def menuItemEnableAll(self):
        self.schema.enableAllLines()

    def menuItemDisableAll(self):
        self.schema.disableAllLines()

    def menuItemSaveSettings(self):
        self.menuSaveSettings = not self.menuSaveSettings
        self.menuOptions.setItemChecked(self.menuSaveSettingsID, self.menuSaveSettings)

    def menuItemNewScheme(self):
        self.schema.clear()

    def dumpVariables(self):
        self.schema.dumpWidgetVariables()

    def menuItemShowOutputWindow(self):
        self.output.show()
        self.output.raise_()
        self.output.activateWindow()

    def menuItemClearOutputWindow(self):
        self.output.textOutput.clear()
        self.statusBar().showMessage("")

    def menuItemSaveOutputWindow(self):
        qname = QFileDialog.getSaveFileName(self, "Save Output To File", self.canvasSettingsDir + "/Output.html", "HTML Document (*.html)")
        if qname.isEmpty(): return
        name = str(qname)

        text = str(self.output.textOutput.toHtml())
        #text = text.replace("</nobr>", "</nobr><br>")

        file = open(name, "wt")
        file.write(text)
        file.close()


    def menuItemShowToolbar(self):
        self.settings["showToolbar"] = not self.settings.get("showToolbar", True)
        if self.settings["showToolbar"]: self.toolbar.show()
        else: self.toolbar.hide()

    def menuItemShowWidgetToolbar(self):
        self.settings["showWidgetToolbar"] = not self.settings.get("showWidgetToolbar", True)
        if self.settings["showWidgetToolbar"]: self.widgetsToolBar.show()
        else: self.widgetsToolBar.hide()


    def menuItemEditWidgetShortcuts(self):
        dlg = orngDlgs.WidgetShortcutDlg(self, self)
        if dlg.exec_() == QDialog.Accepted:
            self.widgetShortcuts = dict([(y, x) for x, y in dlg.invDict.items()])
            shf = file(os.path.join(self.canvasSettingsDir, "shortcuts.txt"), "wt")
            for k, widgetInfo in self.widgetShortcuts.items():
                shf.write("%s: %s\n" % (k, (widgetInfo.category, widgetInfo.name)))

    def menuItemDeleteWidgetSettings(self):
        if QMessageBox.warning(self, 'Orange Canvas', 'Delete all settings?\nNote that for a complete reset there should be no open schema with any widgets.', QMessageBox.Ok | QMessageBox.Default, QMessageBox.Cancel | QMessageBox.Escape) == QMessageBox.Ok:
            if os.path.exists(self.widgetSettingsDir):
                for f in os.listdir(self.widgetSettingsDir):
                    if os.path.splitext(f)[1].lower() == ".ini":
                        os.remove(os.path.join(self.widgetSettingsDir, f))

    def menuOpenLocalOrangeHelp(self):
        import webbrowser
        webbrowser.open("file:///" + os.path.join(self.orangeDir, "doc/reference/default.htm"))

    def menuOpenLocalWidgetCatalog(self):
        import webbrowser
        webbrowser.open("file:///" + os.path.join(self.orangeDir, "doc/catalog/index.html"))

    def menuOpenLocalCanvasHelp(self):
        import webbrowser
        webbrowser.open(os.path.join(self.orangeDir, "doc/canvas/default.htm"))

    def menuOpenOnlineOrangeHelp(self):
        import webbrowser
        webbrowser.open("http://www.ailab.si/orange/doc/catalog")

    def menuOpenOnlineCanvasHelp(self):
        import webbrowser
        #webbrowser.open("http://www.ailab.si/orange/orangeCanvas") # to be added on the web
        webbrowser.open("http://www.ailab.si/orange")

    def menuCheckForUpdates(self):
        import updateOrange
        self.updateDlg = updateOrange.updateOrangeDlg(None)#, Qt.WA_DeleteOnClose)

    def menuItemAboutOrange(self):
        dlg = orngDlgs.AboutDlg(self)
        dlg.exec_()


## to see the signals you have to call: self.output.catchException(0); self.output.catchOutput(0)
## and run orngCanvas.pyw from command line using "python.exe orngCanvas.pyw"
#    def event(self, e):
#        eventDict = dict([(0, "None"), (130, "AccessibilityDescription"), (119, "AccessibilityHelp"), (86, "AccessibilityPrepare"), (114, "ActionAdded"), (113, "ActionChanged"), (115, "ActionRemoved"), (99, "ActivationChange"), (121, "ApplicationActivated"), (122, "ApplicationDeactivated"), (36, "ApplicationFontChange"), (37, "ApplicationLayoutDirectionChange"), (38, "ApplicationPaletteChange"), (35, "ApplicationWindowIconChange"), (68, "ChildAdded"), (69, "ChildPolished"), (71, "ChildRemoved"), (40, "Clipboard"), (19, "Close"), (82, "ContextMenu"), (52, "DeferredDelete"), (60, "DragEnter"), (62, "DragLeave"), (61, "DragMove"), (63, "Drop"), (98, "EnabledChange"), (10, "Enter"), (150, "EnterEditFocus"), (124, "EnterWhatsThisMode"), (116, "FileOpen"), (8, "FocusIn"), (9, "FocusOut"), (97, "FontChange"), (159, "GraphicsSceneContextMenu"), (164, "GraphicsSceneDragEnter"), (166, "GraphicsSceneDragLeave"), (165, "GraphicsSceneDragMove"), (167, "GraphicsSceneDrop"), (163, "GraphicsSceneHelp"), (160, "GraphicsSceneHoverEnter"), (162, "GraphicsSceneHoverLeave"), (161, "GraphicsSceneHoverMove"), (158, "GraphicsSceneMouseDoubleClick"), (155, "GraphicsSceneMouseMove"), (156, "GraphicsSceneMousePress"), (157, "GraphicsSceneMouseRelease"), (168, "GraphicsSceneWheel"), (18, "Hide"), (27, "HideToParent"), (127, "HoverEnter"), (128, "HoverLeave"), (129, "HoverMove"), (96, "IconDrag"), (101, "IconTextChange"), (83, "InputMethod"), (6, "KeyPress"), (7, "KeyRelease"), (89, "LanguageChange"), (90, "LayoutDirectionChange"), (76, "LayoutRequest"), (11, "Leave"), (151, "LeaveEditFocus"), (125, "LeaveWhatsThisMode"), (88, "LocaleChange"), (153, "MenubarUpdated"), (43, "MetaCall"), (102, "ModifiedChange"), (4, "MouseButtonDblClick"), (2, "MouseButtonPress"), (3, "MouseButtonRelease"), (5, "MouseMove"), (109, "MouseTrackingChange"), (13, "Move"), (12, "Paint"), (39, "PaletteChange"), (131, "ParentAboutToChange"), (21, "ParentChange"), (75, "Polish"), (74, "PolishRequest"), (123, "QueryWhatsThis"), (14, "Resize"), (117, "Shortcut"), (51, "ShortcutOverride"), (17, "Show"), (26, "ShowToParent"), (50, "SockAct"), (112, "StatusTip"), (100, "StyleChange"), (87, "TabletMove"), (92, "TabletPress"), (93, "TabletRelease"), (171, "TabletEnterProximity"), (172, "TabletLeaveProximity"), (1, "Timer"), (120, "ToolBarChange"), (110, "ToolTip"), (78, "UpdateLater"), (77, "UpdateRequest"), (111, "WhatsThis"), (118, "WhatsThisClicked"), (31, "Wheel"), (132, "WinEventAct"), (24, "WindowActivate"), (103, "WindowBlocked"), (25, "WindowDeactivate"), (34, "WindowIconChange"), (105, "WindowStateChange"), (33, "WindowTitleChange"), (104, "WindowUnblocked"), (126, "ZOrderChange"), (169, "KeyboardLayoutChange"), (170, "DynamicPropertyChange")])
#        if eventDict.has_key(e.type()):
#            print str(self.windowTitle()), eventDict[e.type()]
#        return QMainWindow.event(self, e)


    def menuItemCanvasOptions(self):
        dlg = orngDlgs.CanvasOptionsDlg(self, self)

        if dlg.exec_() == QDialog.Accepted:
            if self.settings["snapToGrid"] != dlg.settings["snapToGrid"]:
                self.updateSnapToGrid()

            if self.settings["widgetListType"] != dlg.settings["widgetListType"]:
                self.settings["widgetListType"] = dlg.settings["widgetListType"]
                self.createWidgetsToolbar()
                self.widgetListTypeCB.setCurrentIndex(self.settings["widgetListType"])
            self.settings.update(dlg.settings)
            self.updateStyle()
            
#            self.widgetSelectedColor = dlg.selectedWidgetIcon.color
#            self.widgetActiveColor   = dlg.activeWidgetIcon.color
#            self.lineColor           = dlg.lineIcon.color
            
            # update settings in widgets in current documents
            for widget in self.schema.widgets:
                widget.instance._useContexts = self.settings["useContexts"]
                widget.instance._owInfo = self.settings["owInfo"]
                widget.instance._owWarning = self.settings["owWarning"]
                widget.instance._owError = self.settings["owError"]
                widget.instance._owShowStatus = self.settings["owShow"]
                widget.instance.updateStatusBarState()
                widget.resetWidgetSize()
                widget.updateWidgetState()
                
            # update tooltips for lines in all documents
            for line in self.schema.lines:
                line.showSignalNames = self.settings["showSignalNames"]
                line.updateTooltip()
            
            self.schema.canvasView.repaint()
        
#            import orngEnviron, orngRegistry
#            if dlg.toAdd != []:
#                for (name, dir) in dlg.toAdd: 
#                    orngEnviron.registerAddOn(name, dir)
            
#            if dlg.toRemove != []:
#                for (catName, cat) in dlg.toRemove:
#                    addonsToRemove = set()
#                    for widget in cat.values():
#                        addonDir = widget.directory
#                        while os.path.split(addonDir)[1] in ["prototypes", "widgets"]:
#                            addonDir = os.path.split(addonDir)[0]
#                        addonName = os.path.split(addonDir)[1]
#                        addonsToRemove.add( (addonName, addonDir) )
#                    for addonToRemove in addonsToRemove: 
#                        orngEnviron.registerAddOn(add=False, *addonToRemove)
#            
#            if dlg.toAdd != [] or dlg.toRemove != []:
#                self.widgetRegistry = orngRegistry.readCategories()

            # save tab order settings
            newTabList = [(str(dlg.tabOrderList.item(i).text()), int(dlg.tabOrderList.item(i).checkState())) for i in range(dlg.tabOrderList.count())]
            if newTabList != self.settings["WidgetTabs"]:
                self.settings["WidgetTabs"] = newTabList
                self.createWidgetsToolbar()
                orngTabs.constructCategoriesPopup(self)

    def menuItemAddOns(self):
        dlg = orngDlgs.AddOnManagerDialog(self, self)
        if dlg.exec_() == QDialog.Accepted:
            for (id, addOn) in dlg.addOnsToRemove.items():
                try:
                    addOn.uninstall(refresh=False)
                    if id in dlg.addOnsToAdd.items():
                        orngAddOns.installAddOnFromRepo(dlg.addOnsToAdd[id], globalInstall=False, refresh=False)
                        del dlg.addOnsToAdd[id]
                except Exception, e:
                    print "Problem %s add-on %s: %s" % ("upgrading" if id in dlg.addOnsToAdd else "removing", addOn.name, e)
            for (id, addOn) in dlg.addOnsToAdd.items():
                if id.startswith("registered:"):
                    try:
                        orngAddOns.registerAddOn(addOn.name, addOn.directory, refresh=False, systemWide=False)
                    except Exception, e:
                        print "Problem registering add-on %s: %s" % (addOn.name, e)
                else:
                    try:
                        orngAddOns.installAddOnFromRepo(dlg.addOnsToAdd[id], globalInstall=False, refresh=False)
                    except Exception, e:
                        print "Problem installing add-on %s: %s" % (addOn.name, e)
            if len(dlg.addOnsToAdd)+len(dlg.addOnsToRemove)>0:
                orngAddOns.refreshAddOns(reloadPath=True)
                
                    

    def updateStyle(self):
        QApplication.setStyle(QStyleFactory.create(self.settings["style"]))
#        qApp.setStyleSheet(" QDialogButtonBox { button-layout: 0; }")       # we want buttons to go in the "windows" direction (Yes, No, Cancel)
        if self.settings["useDefaultPalette"]:
            QApplication.setPalette(qApp.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)


    def setStatusBarEvent(self, text):
        if text == "" or text == None:
            self.statusBar().showMessage("")
            return
        elif text == "\n": return
        text = str(text)
        text = text.replace("<nobr>", ""); text = text.replace("</nobr>", "")
        text = text.replace("<b>", ""); text = text.replace("</b>", "")
        text = text.replace("<i>", ""); text = text.replace("</i>", "")
        text = text.replace("<br>", ""); text = text.replace("&nbsp", "")
        self.statusBar().showMessage("Last event: " + str(text), 5000)

    # Loads settings from the widget's .ini file
    def loadSettings(self):
        self.settings = {"widgetListType": 4, "iconSize": "40 x 40", "toolbarIconSize": 2, "toolboxWidth": 200, 'schemeIconSize': 1,
                       "snapToGrid": 1, "writeLogFile": 1, "dontAskBeforeClose": 0, "saveWidgetsPosition": 1,
#                       "widgetSelectedColor": (0, 255, 0), "widgetActiveColor": (0, 0, 255), "lineColor": (0, 255, 0),
                       "reportsDir": self.defaultReportsDir, "saveSchemaDir": self.canvasSettingsDir, "saveApplicationDir": self.canvasSettingsDir,
                       "showSignalNames": 1, "useContexts": 1, "enableCanvasDropShadows": 0,
                       "canvasWidth": 700, "canvasHeight": 600, "useDefaultPalette": 0,
                       "focusOnCatchException": 1, "focusOnCatchOutput": 0, "printOutputInStatusBar": 1, "printExceptionInStatusBar": 1,
                       "outputVerbosity": 0, "synchronizeHelp": 1,
                       "ocShow": 1, "owShow": 0, "ocInfo": 1, "owInfo": 1, "ocWarning": 1, "owWarning": 1, "ocError": 1, "owError": 1,
                       }
        if RedR:
            self.setting.update({"svnSettings": None, "versionNumber": "Version0"})
        try:
            filename = os.path.join(self.canvasSettingsDir, "orngCanvas.ini")
            self.settings.update(cPickle.load(open(filename, "rb")))
        except:
            pass

        if not self.settings.has_key("style"):
            items = [str(n) for n in QStyleFactory.keys()]
            lowerItems = [str(n).lower() for n in QStyleFactory.keys()]
            if sys.platform == "darwin" and qVersion() < "4.6": #On Mac OSX full aqua style isn't supported until Qt 4.6
                currStyle = "cleanlooks"
            else:
                currStyle = str(qApp.style().objectName()).lower()
            self.settings.setdefault("style", items[lowerItems.index(currStyle)])


    # Saves settings to this widget's .ini file
    def saveSettings(self):
        filename = os.path.join(self.canvasSettingsDir, "orngCanvas.ini")
        file = open(filename, "wb")
        if self.settings["widgetListType"] == 1:        # tree view
            self.settings["treeItemsOpenness"] = dict([(key, self.tabs.tabDict[key].isExpanded()) for key in self.tabs.tabDict.keys()])
        cPickle.dump(self.settings, file)
        file.close()

    def closeEvent(self, ce):
        # save the current width of the toolbox, if we are using it
        if isinstance(self.widgetsToolBar, orngTabs.WidgetToolBox):
            self.settings["toolboxWidth"] = self.widgetsToolBar.toolbox.width()
        self.settings["showWidgetToolbar"] = self.widgetsToolBar.isVisible()
        self.settings["showToolbar"] = self.toolbar.isVisible()
        self.settings["reportsDir"] = self.reportWindow.saveDir

        closed = self.schema.close()
        if closed:
            self.canvasIsClosing = 1        # output window (and possibly report window also) will check this variable before it will close the window
            self.output.logFile.close()
            self.output.hide()
            
            ce.accept()
            
            self.helpWindow.close()
            self.reportWindow.close()
        else:
            ce.ignore()
        
        self.reportWindow.removeTemp()
        
        size = self.geometry().size()
        self.settings["canvasWidth"] = size.width()
        self.settings["canvasHeight"] = size.height()
        self.settings["CanvasMainWindowGeometry"] = str(self.saveGeometry())
        
        self.saveSettings()
        

    def setCaption(self, caption=""):
        if caption:
            caption = caption.split(".")[0]
            self.setWindowTitle(caption + " - %s Canvas" % product)
        else:
            self.setWindowTitle("%s Canvas" % product)
    
    def getWidgetIcon(self, widgetInfo):
        if self.iconNameToIcon.has_key(widgetInfo.icon):
            return self.iconNameToIcon[widgetInfo.icon]
        
        iconNames = self.getFullWidgetIconName(widgetInfo)
        iconBackgrounds = self.getFullIconBackgroundName(widgetInfo)
        icon = QIcon()
        if len(iconNames) == 1:
            iconSize = QPixmap(iconNames[0]).width()
            iconBackgrounds = [name for name in iconBackgrounds if QPixmap(name).width() == iconSize]
        for name, back in zip(iconNames, iconBackgrounds):
            image = QPixmap(back).toImage()
            painter = QPainter(image)
            painter.drawPixmap(0, 0, QPixmap(name))
            painter.end()
            icon.addPixmap(QPixmap.fromImage(image))
        if iconNames != [self.defaultPic]:
            self.iconNameToIcon[widgetInfo.icon] = icon
        return icon
            
    
    def getFullWidgetIconName(self, widgetInfo):
        iconName = widgetInfo.icon
        names = []
        name, ext = os.path.splitext(iconName)
        for num in [16, 32, 40, 48, 60]:
            names.append("%s_%d%s" % (name, num, ext))
            
        widgetDir = str(widgetInfo.directory)  #os.path.split(self.getFileName())[0]
        fullPaths = []
        for paths in [(self.widgetDir, widgetInfo.category), (self.widgetDir,), (self.picsDir,), tuple(), (widgetDir,), (widgetDir, "icons")]:
            for name in names + [iconName]:
                fname = os.path.join(*paths + (name,))
                if os.path.exists(fname):
                    fullPaths.append(fname)
            if len(fullPaths) > 1 and fullPaths[-1].endswith(iconName):
                fullPaths.pop()     # if we have the new icons we can remove the default icon
            if fullPaths != []:
                return fullPaths
        return [self.defaultPic]
    
    def getFullIconBackgroundName(self, widgetInfo):
        widgetDir = str(widgetInfo.directory)
        fullPaths = []
        for paths in [(widgetDir, "icons"), (self.widgetDir, widgetInfo.category, "icons"), (self.widgetDir, "icons"), (self.picsDir,), tuple(), (widgetDir,), (widgetDir, "icons")]:
            for name in ["background_%d.png" % num for num in [16, 32, 40, 48, 60]]:
                fname = os.path.join(*paths + (name,))
#                print fname
                if os.path.exists(fname):
                    fullPaths.append(fname)
            if fullPaths != []:
                return fullPaths    
        return [self.defaultBackground]
    
class MyStatusBar(QStatusBar):
    def __init__(self, parent):
        QStatusBar.__init__(self, parent)
        self.parentWidget = parent

    def mouseDoubleClickEvent(self, ev):
        self.parentWidget.menuItemShowOutputWindow()
        
        
class OrangeQApplication(QApplication):
    def __init__(self, *args):
        QApplication.__init__(self, *args)
        
    #QFileOpenEvent are Mac OSX only
    if sys.platform == "darwin":
        def event(self, event):
            if event.type() == QEvent.FileOpen:
                file = str(event.file())
                def send():
                    if hasattr(qApp, "canvasDlg"):
                        qApp.canvasDlg.openSchema(file)
                    else:
                        QTimer.singleShot(100, send)
                send()
            return QApplication.event(self, event)
        
#    def notify(self, receiver, event):
#        eventDict = {0: 'None', 1: 'Timer', 2: 'MouseButtonPress', 3: 'MouseButtonRelease', 4: 'MouseButtonDblClick', 5: 'MouseMove', 6: 'KeyPress', 7: 'KeyRelease', 8: 'FocusIn', 9: 'FocusOut', 10: 'Enter', 11: 'Leave', 12: 'Paint', 13: 'Move', 14: 'Resize', 17: 'Show', 18: 'Hide', 19: 'Close', 21: 'ParentChange', 24: 'WindowActivate', 25: 'WindowDeactivate', 26: 'ShowToParent', 27: 'HideToParent', 31: 'Wheel', 33: 'WindowTitleChange', 34: 'WindowIconChange', 35: 'ApplicationWindowIconChange', 36: 'ApplicationFontChange', 37: 'ApplicationLayoutDirectionChange', 38: 'ApplicationPaletteChange', 39: 'PaletteChange', 40: 'Clipboard', 43: 'MetaCall', 50: 'SockAct', 51: 'ShortcutOverride', 52: 'DeferredDelete', 60: 'DragEnter', 61: 'DragMove', 62: 'DragLeave', 63: 'Drop', 68: 'ChildAdded', 69: 'ChildPolished', 70: 'ChildInserted', 71: 'ChildRemoved', 74: 'PolishRequest', 75: 'Polish', 76: 'LayoutRequest', 77: 'UpdateRequest', 78: 'UpdateLater', 82: 'ContextMenu', 83: 'InputMethod', 86: 'AccessibilityPrepare', 87: 'TabletMove', 88: 'LocaleChange', 89: 'LanguageChange', 90: 'LayoutDirectionChange', 92: 'TabletPress', 93: 'TabletRelease', 94: 'OkRequest', 96: 'IconDrag', 97: 'FontChange', 98: 'EnabledChange', 99: 'ActivationChange', 100: 'StyleChange', 101: 'IconTextChange', 102: 'ModifiedChange', 103: 'WindowBlocked', 104: 'WindowUnblocked', 105: 'WindowStateChange', 109: 'MouseTrackingChange', 110: 'ToolTip', 111: 'WhatsThis', 112: 'StatusTip', 113: 'ActionChanged', 114: 'ActionAdded', 115: 'ActionRemoved', 116: 'FileOpen', 117: 'Shortcut', 118: 'WhatsThisClicked', 119: 'AccessibilityHelp', 120: 'ToolBarChange', 121: 'ApplicationActivate', 122: 'ApplicationDeactivate', 123: 'QueryWhatsThis', 124: 'EnterWhatsThisMode', 125: 'LeaveWhatsThisMode', 126: 'ZOrderChange', 127: 'HoverEnter', 128: 'HoverLeave', 129: 'HoverMove', 130: 'AccessibilityDescription', 131: 'ParentAboutToChange', 132: 'WinEventAct', 150: 'EnterEditFocus', 151: 'LeaveEditFocus', 153: 'MenubarUpdated', 155: 'GraphicsSceneMouseMove', 156: 'GraphicsSceneMousePress', 157: 'GraphicsSceneMouseRelease', 158: 'GraphicsSceneMouseDoubleClick', 159: 'GraphicsSceneContextMenu', 160: 'GraphicsSceneHoverEnter', 161: 'GraphicsSceneHoverMove', 162: 'GraphicsSceneHoverLeave', 163: 'GraphicsSceneHelp', 164: 'GraphicsSceneDragEnter', 165: 'GraphicsSceneDragMove', 166: 'GraphicsSceneDragLeave', 167: 'GraphicsSceneDrop', 168: 'GraphicsSceneWheel', 169: 'KeyboardLayoutChange', 170: 'DynamicPropertyChange', 171: 'TabletEnterProximity', 172: 'TabletLeaveProximity', 173: 'NonClientAreaMouseMove', 174: 'NonClientAreaMouseButtonPress', 175: 'NonClientAreaMouseButtonRelease', 176: 'NonClientAreaMouseButtonDblClick', 177: 'MacSizeChange', 178: 'ContentsRectChange', 181: 'GraphicsSceneResize', 182: 'GraphicsSceneMove', 183: 'CursorChange', 184: 'ToolTipChange', 186: 'GrabMouse', 187: 'UngrabMouse', 188: 'GrabKeyboard', 189: 'UngrabKeyboard'}
#        import time
#        try:
#            if str(receiver.windowTitle()) != "":
#                print time.strftime("%H:%M:%S"), "%15s" % str(receiver.windowTitle()) + ": ",      # print only events for QWidget classes and up
#                if eventDict.has_key(event.type()):
#                    print eventDict[event.type()] 
#                else:
#                    print "unknown event name (" + str(event.type()) + ")"
#        except:
#            pass
#            #print str(receiver.objectName()),
#                
#        return QApplication.notify(self, receiver, event)


def main(argv=None):
    if argv == None:
        argv = sys.argv

    app = OrangeQApplication(sys.argv)
    dlg = OrangeCanvasDlg(app)
    qApp.canvasDlg = dlg
    dlg.show()
    for arg in sys.argv[1:]:
        if arg == "-reload":
            dlg.menuItemOpenLastSchema()
    app.exec_()
    app.closeAllWindows()

if __name__ == "__main__":
    sys.exit(main())
