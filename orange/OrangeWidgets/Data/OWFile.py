"""
<name>File</name>
<description>Reads data from a file.</description>
<icon>icons/File.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>10</priority>
"""

#
# OWFile.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
import OWGUI, string, os.path, user, sys

class OWFile(OWWidget):
    settingsList=["recentFiles", "symbolDC", "symbolDK", "createNewOn"]
    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "File")

        self.inputs = []
        self.outputs = [("Examples", ExampleTable), ("Attribute Definitions", orange.Domain)]

        #set default settings
        self.recentFiles=["(none)"]
        self.symbolDC = ""
        self.symbolDK = ""
        self.createNewOn = orange.Variable.MakeStatus.NoRecognizedValues
        self.domain = None
        #get settings from the ini file, if they exist
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Data File", addSpace = True, orientation=0)
        self.filecombo=QComboBox(box)
        self.filecombo.setMinimumWidth(150)
        button = OWGUI.button(box, self, '...', callback = self.browseFile, disabled=0)
        self.reloadBtn = OWGUI.button(box, self, "Reload", callback = self.reload)
        button.setMaximumWidth(25)

#        OWGUI.rubber(self.controlArea)
        
        box = OWGUI.widgetBox(self.controlArea, "Advanced", addSpace=True)

#        OWGUI.widgetLabel(box, "Undefineds in tab-delimited files (besides default ones)")
#        hbox = OWGUI.indentedBox(box, addSpace=True, orientation=False)
#        le = OWGUI.lineEdit(hbox, self, "symbolDC", "Don't care:  ", orientation="horizontal", tooltip="Default values: empty fields (space), '?' or 'NA'")
#        le.setMaximumWidth(48)
#        OWGUI.separator(hbox, 16, 0)
#        le = OWGUI.lineEdit(hbox, self, "symbolDK", "Don't know:  ", orientation="horizontal", tooltip="Default values: '~' or '*'")
#        le.setMaximumWidth(48)
        OWGUI.comboBox(box, self, "createNewOn", 
                       label = "Create a new attribute when existing attribute(s) ...", 
                       items = ["... Always create a new attribute", 
                                "Miss some values of the new attribute", 
                                "Have no common values with the new (recommended)", 
                                "Have mismatching order of values"
                               ])
        
        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.infoa = OWGUI.widgetLabel(box, 'No data loaded.')
        self.infob = OWGUI.widgetLabel(box, '')
        self.warnings = OWGUI.widgetLabel(box, '')


        self.adjustSize()

    # set the file combo box
    def setFileList(self):
        self.filecombo.clear()
        if not self.recentFiles:
            self.filecombo.insertItem("(none)")
        for file in self.recentFiles:
            if file == "(none)":
                self.filecombo.insertItem("(none)")
            else:
                self.filecombo.insertItem(os.path.split(file)[1])
        self.filecombo.insertItem("Browse documentation data sets...")
        #self.filecombo.adjustSize() #doesn't work properly :(
        self.filecombo.updateGeometry()


    def reload(self):
        if self.recentFiles:
            return self.openFile(self.recentFiles[0], 1)

    def activateLoadedSettings(self):
        # remove missing data set names
        self.recentFiles=filter(os.path.exists, self.recentFiles)
        self.setFileList()

        if len(self.recentFiles) > 0 and os.path.exists(self.recentFiles[0]):
            self.openFile(self.recentFiles[0])

        # connecting GUI to code
        self.connect(self.filecombo, SIGNAL('activated(int)'), self.selectFile)

    # user selected a file from the combo box
    def selectFile(self, n):
        if n < len(self.recentFiles) :
            name = self.recentFiles[n]
            self.recentFiles.remove(name)
            self.recentFiles.insert(0, name)
        elif n:
            self.browseFile(1)

        if len(self.recentFiles) > 0:
            self.setFileList()
            self.openFile(self.recentFiles[0])

    # user pressed the "..." button to manually select a file to load
    def browseFile(self, inDemos=0):
        "Display a FileDialog and select a file"
        if inDemos:
            import os
            try:
                import win32api, win32con
                t = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, "SOFTWARE\\Python\\PythonCore\\%i.%i\\PythonPath\\Orange" % sys.version_info[:2], 0, win32con.KEY_READ)
                t = win32api.RegQueryValueEx(t, "")[0]
                startfile = t[:t.find("orange")] + "orange\\doc\\datasets"
            except:
                startfile = ""

            if not startfile or not os.path.exists(startfile):
                d = OWGUI.__file__
                if d[-8:] == "OWGUI.py":
                    startfile = d[:-22] + "doc/datasets"
                elif d[-9:] == "OWGUI.pyc":
                    startfile = d[:-23] + "doc/datasets"

            if not startfile or not os.path.exists(startfile):
                d = os.getcwd()
                if d[-12:] == "OrangeCanvas":
                    startfile = d[:-12]+"doc/datasets"
                else:
                    if d[-1] not in ["/", "\\"]:
                        d+= "/"
                    startfile = d+"doc/datasets"

            if not os.path.exists(startfile):
                QMessageBox.information( None, "File", "Cannot find the directory with example data sets", QMessageBox.Ok + QMessageBox.Default)
                return
        else:
            if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
                if sys.platform == "darwin":
                    startfile = user.home
                else:
                    startfile="."
            else:
                startfile=self.recentFiles[0]

        filename = str(QFileDialog.getOpenFileName(startfile,
        'Tab-delimited files (*.tab *.txt)\nC4.5 files (*.data)\nAssistant files (*.dat)\nRetis files (*.rda *.rdo)\nBasket files (*.basket)\nAll files(*.*)',
        None,'Open Orange Data File'))

        if filename == "": return
        if filename in self.recentFiles: self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)
        self.setFileList()
        self.openFile(self.recentFiles[0])

    def setInfo(self, info):
        for (i, s) in enumerate(info):
            self.info[i].setText(s)

    # Open a file, create data from it and send it over the data channel
    def openFile(self, fn, throughReload=0, DK=None, DC=None):
        self.error()

        if fn == "(none)":
            self.send("Examples", None)
            self.send("Attribute Definitions", None)
            self.infoa.setText("No data loaded")
            self.infob.setText("")
            self.warnings.setText("")
            return
            
        argdict = {"createNewOn": self.createNewOn}
        if DK:
            argdict["DK"] = DK
        if DC:
            argdict["DC"] = DC

        try:
            data = orange.ExampleTable(fn, **argdict)
        except Exception, (errValue):
            if not data:
                self.error(str(errValue))
                self.dataDomain = None
                self.infoa.setText('No data loaded due to an error')
                self.infob.setText("")
                self.warnings.setText("")
                return
                        
        self.dataDomain = data.domain

        self.infoa.setText('%d example(s), ' % len(data) + '%d attribute(s), ' % len(data.domain.attributes) + '%d meta attribute(s).' % len(data.domain.getmetas()))
        cl = data.domain.classVar
        if cl:
            if cl.varType == orange.VarTypes.Continuous:
                    self.infob.setText('Regression; Numerical class.')
            elif cl.varType == orange.VarTypes.Discrete:
                    self.infob.setText('Classification; Discrete class with %d value(s).' % len(cl.values))
            else:
                self.infob.setText("Class neither descrete nor continuous.")
        else:
            self.infob.setText("Data without a dependent variable.")
        
        warnings = ""
        metas = data.domain.getmetas()
        for status, messageUsed, messageNotUsed in [
                                (orange.Variable.MakeStatus.Incompatible,
                                 "",
                                 "The following attributes already existed but had a different order of values, so new attributes needed to be created"),
                                (orange.Variable.MakeStatus.NoRecognizedValues,
                                 "The following attributes were reused although they share no common values with the existing attribute of the same names",
                                 "The following attributes were not reused since they share no common values with the existing attribute of the same names"),
                                (orange.Variable.MakeStatus.MissingValues, 
                                 "The following attribute(s) were reused although some values needed to be added",
                                 "The following attribute(s) were not reused since they miss some values")
                                ]:
            if self.createNewOn > status:
                message = messageUsed
            else:
                message = messageNotUsed
            if not message:
                continue
            attrs = [attr.name for attr, stat in zip(data.domain, data.attributeLoadStatus) if stat == status] \
                  + [attr.name for id, attr in metas.items() if data.metaAttributeLoadStatus[id] == status]
            if attrs:
                warnings += "<li>%s: %s</li>" % (message, ", ".join(attrs))

        self.warnings.setText(warnings)
        qApp.processEvents()
        self.adjustSize()

        # make new data and send it
        fName = os.path.split(fn)[1]
        if "." in fName:
            data.name = data.name[:data.name.rfind('.')]
        else:
            data.name = fName
            
        self.send("Examples", data)
        self.send("Attribute Definitions", data.domain)




if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWFile()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()
