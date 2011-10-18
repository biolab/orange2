#import orngOrangeFoldersQt4
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os, re, urllib, sys
import md5, cPickle

# This is Orange Update program. It can check on the web if there are any updates available and download them.
# User can select a list of folders that he wants to update and a list of folders that he wants to ignore.
# In case that a file was locally changed and a new version of the same file is available, the program offers the user
# to update to the new file or to keep the old one.
#

defaultIcon = ['16 13 5 1', '. c #040404', '# c #808304', 'a c None', 'b c #f3f704', 'c c #f3f7f3',  'aaaaaaaaa...aaaa',  'aaaaaaaa.aaa.a.a',  'aaaaaaaaaaaaa..a',
    'a...aaaaaaaa...a', '.bcb.......aaaaa', '.cbcbcbcbc.aaaaa', '.bcbcbcbcb.aaaaa', '.cbcb...........', '.bcb.#########.a', '.cb.#########.aa', '.b.#########.aaa', '..#########.aaaa', '...........aaaaa']

CONFLICT_ASK = 0
CONFLICT_OVERWRITE = 1
CONFLICT_KEEP = 2

def splitDirs(path):
    dirs, filename = os.path.split(path)
    listOfDirs = []
    while dirs != "":
        dirs, dir = os.path.split(dirs)
        listOfDirs.insert(0, dir)
    return listOfDirs

class OptionsDlg(QDialog):
    def __init__(self, settings):
        QDialog.__init__(self, None)
        self.setWindowTitle("Update Options")

        self.setLayout(QVBoxLayout())

        self.groupBox = QGroupBox("Updating Options", self)
        self.layout().addWidget(self.groupBox)
        self.check1 = QCheckBox("Update scripts", self.groupBox)
        self.check2 = QCheckBox("Update binary files", self.groupBox)
        self.check3 = QCheckBox("Download new files", self.groupBox)
        self.groupBox.setLayout(QVBoxLayout())
        for c in [self.check1, self.check2, self.check3]:
            self.groupBox.layout().addWidget(c)

        self.groupBox2 = QGroupBox("Solving Conflicts", self)
        self.groupBox2.setLayout(QVBoxLayout())
        self.layout().addWidget(self.groupBox2)
        label = QLabel("When your local file was edited\nand a newer version is available...", self.groupBox2)
        self.groupBox2.layout().addWidget(label)
        self.combo = QComboBox(self.groupBox2)
        for s in ["Ask what to do", "Overwrite your local copy with new file", "Keep your local file"]:
            self.combo.addItem(s)
        self.groupBox2.layout().addWidget(self.combo)

        self.check1.setChecked(settings["scripts"])
        self.check2.setChecked(settings["binary"])
        self.check3.setChecked(settings["new"])
        self.combo.setCurrentIndex(settings["conflicts"])

        widget = QWidget(self)
        self.layout().addWidget(widget)
        widget.setLayout(QHBoxLayout())
        widget.layout().addStretch(1)
        okButton = QPushButton('OK', widget)
        widget.layout().addWidget(okButton)
##        self.topLayout.addWidget(okButton)
        self.connect(okButton, SIGNAL('clicked()'),self,SLOT('accept()'))
        cancelButton = QPushButton('Cancel', widget)
        widget.layout().addWidget(cancelButton)
##        self.topLayout.addWidget(cancelButton)
        self.connect(cancelButton, SIGNAL('clicked()'),self,SLOT('reject()'))

    def accept(self):
        self.settings = {"scripts": self.check1.isChecked(), "binary": self.check2.isChecked(), "new": self.check3.isChecked(), "conflicts": self.combo.currentIndex()}
        QDialog.accept(self)



class FoldersDlg(QDialog):
    def __init__(self, caption):
        QDialog.__init__(self, None)

        self.setLayout(QVBoxLayout())

        self.groupBox = QGroupBox(self)
        self.layout().addWidget(self.groupBox)
        self.groupBox.setTitle(" " + caption.strip() + " ")
        self.groupBox.setLayout(QVBoxLayout())
        self.groupBoxLayout.setMargin(20)

        self.setWindowCaption("Select Folders")
        self.resize(300,100)

        self.folders = []
        self.checkBoxes = []

    def addCategory(self, text, checked = 1, indent = 0):
        widget = QWidget(self.groupBox)
        self.groupBox.layout().addWidget(widget)
        hboxLayout = QHBoxLayout()
        widget.setLayout(hboxLayout)

        if indent:
            sep = QWidget(widget)
            sep.setFixedSize(19, 8)
            hboxLayout.addWidget(sep)
        check = QCheckBox(text, widget)
        hboxLayout.addWidget(check)

        check.setChecked(checked)
        self.checkBoxes.append(check)
        self.folders.append(text)

    def addLabel(self, text):
        label = QLabel(text, self.groupBox)
        self.groupBox.layout().addWidget(label)


    def finishedAdding(self, ok = 1, cancel = 1):
        widget = QWidget(self)
        self.layout().addWidget(widget)
        widgetLayout = QHBoxLayout(widget)
        widget.setLayout(widgetLayout)
        widgetLayout.addStretch(1)

        if ok:
            okButton = QPushButton('OK', widget)
            widgetLayout.addWidget(okButton)
            self.connect(okButton, SIGNAL('clicked()'),self,SLOT('accept()'))
        if cancel:
            cancelButton = QPushButton('Cancel', widget)
            widgetLayout.addWidget(cancelButton)
            self.connect(cancelButton, SIGNAL('clicked()'),self,SLOT('reject()'))

class updateOrangeDlg(QMainWindow):
    def __init__(self,*args):
        QMainWindow.__init__(self, *args)
        self.resize(600,600)
        self.setWindowTitle("Orange Update")

        self.toolbar = self.addToolBar("Toolbar")

        self.text = QTextEdit(self)
        self.text.setReadOnly(1)
        self.text.zoomIn(2)
        self.setCentralWidget(self.text)
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Ready')

        import updateOrange
        self.orangeDir = os.path.split(os.path.abspath(updateOrange.__file__))[0]
        os.chdir(self.orangeDir)        # we have to set the current dir to orange dir since we can call update also from orange canvas

        self.settings = {"scripts":1, "binary":1, "new":1, "conflicts":0}
        if os.path.exists("updateOrange.set"):
            file = open("updateOrange.set", "r")
            self.settings = cPickle.load(file)
            file.close()

        self.re_vLocalLine = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<md5>.*)')
        self.re_vInternetLine = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<location>.*)')
        self.re_widget = re.compile(r'(?P<category>.*)[/,\\].*')
        self.re_documentation = re.compile(r'doc[/,\\].*')

        self.downfile = os.path.join(self.orangeDir, "whatsdown.txt")

        self.updateUrl = "http://orange.biolab.si/download/update/"
        self.binaryUrl = "http://orange.biolab.si/download/binaries/%i%i/" % sys.version_info[:2]
        self.whatsupUrl = "http://orange.biolab.si/download/whatsup.txt"

        self.updateGroups = []
        self.dontUpdateGroups = []
        self.newGroups = []
        self.downstuff = {}

        # read updateGroups and dontUpdateGroups
        self.addText("Welcome to the Orange update.")
        try:
            vf = open(self.downfile)
            self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readLocalVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
        except:
            pass
        self.addText("To download latest versions of files click the 'Update' button.", nobr = 0)

        # create buttons
        iconsDir = os.path.join(self.orangeDir, "OrangeCanvas/icons")
        self.updateIcon = os.path.join(iconsDir, "update.png")
        self.foldersIcon = os.path.join(iconsDir, "folders.png")
        self.optionsIcon = os.path.join(iconsDir, "options.png")
        if not os.path.exists(self.updateIcon): self.updateIcon = defaultIcon
        if not os.path.exists(self.foldersIcon): self.foldersIcon = defaultIcon
        if not os.path.exists(self.optionsIcon): self.optionsIcon = defaultIcon

        def createButton(text, icon, callback, tooltip):
            b = QToolButton(self.toolbar)
            self.toolbar.layout().addWidget(b)
            b.setIcon(icon)
            b.setText(text)
            self.connect(b, SIGNAL("clicked()"), callback)
            b.setToolTip(tooltip)

        self.toolUpdate  = self.toolbar.addAction(QIcon(self.updateIcon), "Update" , self.executeUpdate)
        self.toolbar.addSeparator()
        self.toolFolders = self.toolbar.addAction(QIcon(self.foldersIcon), "Folders" , self.showFolders)
        self.toolOptions = self.toolbar.addAction(QIcon(self.optionsIcon), "Options" , self.showOptions)

        self.setWindowIcon(QIcon(self.updateIcon))
        self.move((qApp.desktop().width()-self.width())/2, (qApp.desktop().height()-self.height())/2)   # center the window
        self.show()


    # ####################################
    # show the list of possible folders
    def showFolders(self):
        self.updateGroups = []
        self.dontUpdateGroups = []
        try:
            vf = open(self.downfile)
            self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readLocalVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
        except:
            self.addText("Failed to locate file 'whatsdown.txt'. There is no information on current versions of Orange files. By clicking 'Update files' you will download the latest versions of files.", nobr = 0)
            return

        groups = [(name, 1) for name in self.updateGroups] + [(name, 0) for name in self.dontUpdateGroups]
        groups.sort()
        groupDict = dict(groups)

        dlg = FoldersDlg("Select Orange folders that you wish to update")
        dlg.setWindowIcon(QIcon(self.foldersIcon))

        dlg.addCategory("Orange Canvas", groupDict.get("Orange Canvas", 1))
        dlg.addCategory("Documentation", groupDict.get("Documentation", 1))
        dlg.addCategory("Orange Root", groupDict.get("Orange Root", 1))
        dlg.addLabel("Orange Widgets:")
        for (group, sel) in groups:
            if group in ["Orange Canvas", "Documentation", "Orange Root"]: continue
            dlg.addCategory(group, sel, indent = 1)

        dlg.finishedAdding(cancel = 1)
        dlg.move((qApp.desktop().width()-dlg.width())/2, (qApp.desktop().height()-400)/2)   # center dlg window

        res = dlg.exec_()
        if res == QDialog.Accepted:
            self.updateGroups = []
            self.dontUpdateGroups = []
            for i in range(len(dlg.checkBoxes)):
                if dlg.checkBoxes[i].isChecked(): self.updateGroups.append(dlg.folders[i])
                else:                             self.dontUpdateGroups.append(dlg.folders[i])
            self.writeVersionFile()
        return

    def showOptions(self):
        dlg = OptionsDlg(self.settings)
        dlg.setWindowIcon(QIcon(self.optionsIcon))
        res = dlg.exec_()
        if res == QDialog.Accepted:
            self.settings = dlg.settings

    def readLocalVersionFile(self, data, updateGroups = 1):
        versions = {}
        updateGroups = []; dontUpdateGroups = []
        for line in data:
            if not line: continue
            line = line.replace("\r", "")   # replace \r in case of linux files
            line = line.replace("\n", "")
            if not line: continue

            if line[0] == "+":
                updateGroups.append(line[1:])
            elif line[0] == "-":
                dontUpdateGroups.append(line[1:])
            else:
                fnd = self.re_vLocalLine.match(line)
                if fnd:
                    fname, version, md = fnd.group("fname", "version", "md5")
                    fname = fname.replace("\\", "/")
                    versions[fname] = ([int(x) for x in version.split(".")], md)

                    # add widget category if not already in updateGroups
                    dirs = splitDirs(fname)
                    if len(dirs) >= 2 and dirs[0].lower() == "orangewidgets" and dirs[1] not in updateGroups + dontUpdateGroups and dirs[1].lower() != "icons":
                        updateGroups.append(dirs[1])
                    if len(dirs) >= 1 and dirs[0].lower() == "doc" and "Documentation" not in updateGroups + dontUpdateGroups: updateGroups.append("Documentation")
                    if len(dirs) >= 1 and dirs[0].lower() == "orangecanvas" and "Orange Canvas" not in updateGroups + dontUpdateGroups: updateGroups.append("Orange Canvas")
                    if len(dirs) == 1 and "Orange Root" not in updateGroups + dontUpdateGroups: updateGroups.append("Orange Root")

        return versions, updateGroups, dontUpdateGroups

    def readInternetVersionFile(self, updateGroups = 1):
        try:
            f = urllib.urlopen(self.whatsupUrl)
        except IOError:
            self.addText('Unable to download current status file. Check your internet connection.')
            return {}, [], []

        data = f.read().split("\n")
        versions = {}
        updateGroups = []; dontUpdateGroups = []
        for line in data:
            if not line: continue
            line = line.replace("\r", "")   # replace \r in case of linux files
            line = line.replace("\n", "")
            if not line: continue

            if line[0] == "+":
                updateGroups.append(line[1:])
            elif line[0] == "-":
                dontUpdateGroups.append(line[1:])
            else:
                fnd = self.re_vInternetLine.match(line)
                if fnd:
                    fname, version, location = fnd.group("fname", "version", "location")
                    fname = fname.replace("\\", "/")
                    versions[fname] = ([int(x) for x in version.split(".")], location)

                    # add widget category if not already in updateGroups
                    dirs = splitDirs(fname)
                    if len(dirs) >= 2 and dirs[0].lower() == "orangewidgets" and dirs[1] not in updateGroups and dirs[1].lower() != "icons":
                        updateGroups.append(dirs[1])

        return versions, updateGroups, dontUpdateGroups

    def writeVersionFile(self):
        vf = open(self.downfile, "wt")
        itms = self.downstuff.items()
        itms.sort(lambda x,y:cmp(x[0], y[0]))

        for g in self.dontUpdateGroups:
            vf.write("-%s\n" % g)
        for fname, (version, md) in itms:
            vf.write("%s=%s:%s\n" % (fname, reduce(lambda x,y:x+"."+y, [`x` for x in version]), md))
        vf.close()


    def executeUpdate(self):
        updatedFiles = 0
        newFiles = 0

        if self.settings["scripts"]:
            self.addText("Reading file status from web server")

            self.updateGroups = [];  self.dontUpdateGroups = []; self.newGroups = []
            self.downstuff = {}

            upstuff, upUpdateGroups, upDontUpdateGroups = self.readInternetVersionFile(updateGroups = 0)
            if upstuff == {}: return
            try:
                vf = open(self.downfile)
                self.addText("Reading local file status")
                self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readLocalVersionFile(vf.readlines(), updateGroups = 1)
                vf.close()
            except:
                res = QMessageBox.information(self, 'Update Orange', "The 'whatsdown.txt' file if missing (most likely because you downloaded Orange from CVS).\nThis file contains information about versions of your local Orange files.\n\nIf you press 'Replace Local Files' you will not replace only updated files, but will \noverwrite all your local Orange files with the latest versions from the web.\n", 'Replace Local Files', "Cancel", "", 0, 1)
                if res != 0: return

            itms = upstuff.items()
            itms.sort(lambda x,y:cmp(x[0], y[0]))

            for category in upUpdateGroups: #+ upDontUpdateGroups:
                if category not in self.updateGroups + self.dontUpdateGroups:
                    self.newGroups.append(category)

            # show dialog with new groups
            if self.newGroups != []:
                dlg = FoldersDlg("Select new categories you wish to download")
                dlg.setWindowIcon(QIcon(self.foldersIcon))
                for group in self.newGroups: dlg.addCategory(group)
                dlg.finishedAdding(cancel = 0)

                res = dlg.exec_()
                for i in range(len(dlg.checkBoxes)):
                    if dlg.checkBoxes[i].isChecked():
                        self.updateGroups.append(dlg.folders[i])
                    else:
                        self.dontUpdateGroups.append(dlg.folders[i])
                self.newGroups = []

            # update new files
            self.addText("Updating scripts...")
            self.statusBar.showMessage("Updating scripts")

            for fname, (version, location) in itms:
                qApp.processEvents()

                # check if it is a widget directory that we don't want to update
                dirs = splitDirs(fname)
                if len(dirs) >= 2 and dirs[0].lower() == "orangewidgets" and dirs[1] in self.dontUpdateGroups: continue
                if len(dirs) >= 1 and dirs[0].lower() == "doc" and "Documentation" in self.dontUpdateGroups: continue
                if len(dirs) >= 1 and dirs[0].lower() == "orangecanvas" and "Orange Canvas" in self.dontUpdateGroups: continue
                if len(dirs) == 1 and "Orange Root" in self.dontUpdateGroups: continue

                if os.path.exists(fname) and self.downstuff.has_key(fname) and self.downstuff[fname][0] < upstuff[fname][0]:      # there is a newer version
                    updatedFiles += self.updatefile(self.updateUrl + fname, location, version, self.downstuff[fname][1], "Updating")
                elif not os.path.exists(fname) or not self.downstuff.has_key(fname):
                    if self.settings["new"]:
                        updatedFiles += self.updatefile(self.updateUrl + fname, location, version, "", "Downloading new file")
                    else:
                        self.addText("Skipping new file %s" % (fname))
            self.writeVersionFile()
        else:
            self.addText("Skipping updating scripts...")

        if self.settings["binary"]:
            self.addText("Updating binaries...")
            updatedFiles += self.updatePyd()
        else:
            self.addText("Skipping updateing binaries...")

        self.addText("Update finished. New files: <b>%d</b>. Updated files: <b>%d</b>\n" %(newFiles, updatedFiles))

        self.statusBar.showMessage("Update finished.")

    # update binary files
    def updatePyd(self):
        files = "orange", "corn", "statc", "orangeom", "orangene", "_orngCRS"

        baseurl = "http://orange.biolab.si/download/binaries/%i%i/" % sys.version_info[:2]
        repository_stamps = dict([tuple(x.split()) for x in urllib.urlopen(baseurl + "stamps_pyd.txt") if x.strip()])
        updated = 0

        for fle in files:
            if not os.path.exists(fle+".pyd") or repository_stamps[fle+".pyd"] != md5.md5(file(fle+".pyd", "rb").read()).hexdigest().upper():
                updated += self.updatefile(baseurl + fle + ".pyd", fle + ".pyd", "", "", "Updating")
        return updated

    # #########################################################
    # get new file from the internet and overwrite the old file
    # webName = complete path to the file on the web
    # localName = path and name of the file on the local disk
    # version = the newest file version
    # md = hash value of the local file when it was downloaded from the internet - needed to compare if the user has changed the local version of the file
    def updatefile(self, webName, localName, version, md, type = "Downloading"):
        self.addText(type + " %s ... " % localName, addBreak = 0)
        qApp.processEvents()

        try:
            urllib.urlretrieve(webName, localName + ".temp", self.updateDownloadStatus)
        except IOError, inst:
            self.addText('<font color="#FF0000">Failed</font> (%s)' % (inst[1]))
            return 0

        self.statusBar.showMessage("")
        dname = os.path.dirname(localName)
        if dname and not os.path.exists(dname):
            os.makedirs(dname)

        isBinaryFile = localName[-3:].lower() in ["pyd"]
        if not isBinaryFile:
            # read existing file
            if md != "" and os.path.exists(localName):
                currmd = self.computeFileMd(localName)
                if currmd.hexdigest() != md:   # the local file has changed
                    if self.settings["conflicts"] == CONFLICT_OVERWRITE:
                        res = 0
                    elif self.settings["conflicts"] == CONFLICT_KEEP:
                        res = 1
                    elif self.settings["conflicts"] == CONFLICT_ASK:
                        res = QMessageBox.information(self,'Update Orange',"Your local file '%s' was edited, but a newer version of this file is available on the web.\nDo you wish to overwrite local copy with newest version (a backup of current file will be created) or keep your current file?" % (os.path.split(localName)[1]), 'Overwrite with newest', 'Keep current file')

                    if res == 0:    # overwrite
                        currmd = self.computeFileMd(localName+".temp")
                        try:
                            ext = ".bak"
                            if os.path.exists(localName + ext):
                                i = 1
                                while os.path.exists(localName + ext + str(i)): i += 1
                                ext = ext+str(i)
                            os.rename(localName, localName + ext)  # create backup
                        except OSError, inst:
                            self.addText('<font color="#FF0000">Failed</font> (%s)' % (inst[1]))
                            self.addText('Unable to update file <font color="#FF0000">%s</font>. Please close all programs that are using it.' % (os.path.split(localName)[1]))
                            return 0
                    elif res == 1:    # keep local
                        self.addText('<font color="#0000FF">Skipping</font>')
                        return 0
            else:
                currmd = self.computeFileMd(localName + ".temp")

        try:
            if os.path.exists(localName):
                os.remove(localName)
            os.rename(localName + ".temp", localName)
            if not isBinaryFile:
                self.downstuff[localName[2:]] = (version, currmd.hexdigest())       # remove "./" from localName
            self.addText('<font color="#0000FF">OK</font>')
            return 1
        except OSError, inst:
            self.addText('<font color="#FF0000">Failed</font> (%s)' % (inst[1]))
            self.addText('Unable to update file <font color="#FF0000">%s</font>. Please close all programs that are using it.' % (os.path.split(localName)[1]))
            return 0

    # show percent of finished download
    def updateDownloadStatus(self, blk_cnt, blk_size, tot_size):
        self.statusBar.showMessage("Downloaded %.1f%%" % (100*min(tot_size, blk_cnt*blk_size) / (tot_size or 1)))

    def computeFileMd(self, fname):
        f = open(fname, "rb")
        md = md5.new()
        md.update(f.read())
        f.close()
        return md

    def addText(self, text, nobr = 1, addBreak = 1):
        cursor = QTextCursor(self.text.textCursor())                # clear the current text selection so that
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # the text will be appended to the end of the
        self.text.setTextCursor(cursor)                             # existing text
                
        if nobr: self.text.insertHtml('<nobr>' + text + '</nobr>')
        else:    self.text.insertHtml(text)
        
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # and then scroll down to the end of the text
        self.text.setTextCursor(cursor)
        if addBreak: self.text.insertHtml("<br>")
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        else: QMainWindow.keyPressEvent(self, e)

    def closeEvent(self, e):
        f = open("updateOrange.set", "wt")
        cPickle.dump(self.settings, f)
        f.close()
        QMainWindow.closeEvent(self, e)


# show application dlg
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = updateOrangeDlg()
    dlg.show()
    app.exec_()
