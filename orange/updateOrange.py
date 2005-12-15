import os, re, httplib, urllib, sys
from qt import *
import md5


defaultIcon = ['16 13 5 1', '. c #040404', '# c #808304', 'a c None', 'b c #f3f704', 'c c #f3f7f3',  'aaaaaaaaa...aaaa',  'aaaaaaaa.aaa.a.a',  'aaaaaaaaaaaaa..a',
    'a...aaaaaaaa...a', '.bcb.......aaaaa', '.cbcbcbcbc.aaaaa', '.bcbcbcbcb.aaaaa', '.cbcb...........', '.bcb.#########.a', '.cb.#########.aa', '.b.#########.aaa', '..#########.aaaa', '...........aaaaa']


def splitDirs(path):
    dirs, filename = os.path.split(path)
    listOfDirs = []
    while dirs != "":
        dirs, dir = os.path.split(dirs)
        listOfDirs.insert(0, dir)
    return listOfDirs
    

class foldersDlg(QDialog):
    def __init__(self, caption, *args):
        apply(QDialog.__init__,(self,) + args)

        self.layout = QVBoxLayout( self, 10 )
        self.topLayout = QVGroupBox(self)
        self.layout.addWidget(self.topLayout)
        label = QLabel(caption, self.topLayout)
        self.setCaption("Qt Select Folders")
        self.resize(300,100)
        
        self.folders = []
        self.checkBoxes = []

    def addCategory(self, text, checked = 1, indent = 0):
        if indent:
            box = QHBox(self.topLayout)
            QWidget(box).setFixedSize(19, 8)
            check = QCheckBox(text, box)
        else:
            check = QCheckBox(text, self.topLayout)
        check.setChecked(checked)
        self.checkBoxes.append(check)
        self.folders.append(text)

    def addLabel(self, text):
        label = QLabel(text, self.topLayout)
        

    def finishedAdding(self, ok = 1, cancel = 1):
        if ok:
            okButton = QPushButton('OK', self.topLayout)
            self.connect(okButton, SIGNAL('clicked()'),self,SLOT('accept()'))
        if cancel:
            cancelButton = QPushButton('Cancel', self.topLayout)
            self.connect(cancelButton, SIGNAL('clicked()'),self,SLOT('reject()'))

class updateOrangeDlg(QMainWindow):
    def __init__(self,*args):
        apply(QMainWindow.__init__,(self,) + args)
        self.resize(600,600)
        self.setCaption("Qt Orange Update")
        self.toolbar = QToolBar(self, 'toolbar')
        self.statusBar = QStatusBar(self)
        self.text = QTextView (self)
        font = self.text.font(); font.setPointSize(11); self.text.setFont(font)
        self.setCentralWidget(self.text)
        self.statusBar.message('Ready')

        import updateOrange
        self.orangeDir = os.path.split(os.path.abspath(updateOrange.__file__))[0]
        iconsDir = os.path.join(self.orangeDir, "OrangeCanvas/icons")
        updateIcon = os.path.join(iconsDir, "update.png")
        foldersIcon = os.path.join(iconsDir, "folders.png")
        if not os.path.exists(updateIcon): updateIcon = defaultIcon
        if not os.path.exists(foldersIcon): foldersIcon = defaultIcon

        self.re_vLocalLine = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<md5>.*)')
        self.re_vInternetLine = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<location>.*)')
        self.re_widget = re.compile(r'(?P<category>.*)[/,\\].*')
        self.re_documentation = re.compile(r'doc[/,\\].*')

        self.downfile = os.path.join(self.orangeDir, "whatsdown.txt")
        self.httpconnection = httplib.HTTPConnection('www.ailab.si')

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
            #self.addText("Versions of local Orange files were successfully located.")
        except:
            #self.addText("Orange Update failed to locate '%s' file, which should contain information about current versions of Orange files." %(self.downfile), nobr = 0)
            pass
        self.addText("To check for newer versions of files click the 'Update Files' button.", nobr = 0)

        # create buttons
        self.toolUpdate  = QToolButton(QPixmap(updateIcon), "Update Files" , QString.null, self.executeUpdate, self.toolbar, 'Update Files')
        self.toolUpdate.setUsesTextLabel (1)
        self.toolbar.addSeparator()
        self.toolFolders = QToolButton(QPixmap(foldersIcon), "Folders" , QString.null, self.showFolders, self.toolbar, 'Show Folders')
        self.toolFolders.setUsesTextLabel (1)
        self.toolbar.addSeparator()
        self.downloadNewFilesCB = QCheckBox("Download new files", self.toolbar)
        self.downloadNewFilesCB.setChecked(1)
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
            self.addText("Failed to locate file '%s'. There is no information on installed Orange files." %(self.downfile), nobr = 0)
            #self.addText("No folders found.")
            return


        groups = [(name, 1) for name in self.updateGroups] + [(name, 0) for name in self.dontUpdateGroups]
        groups.sort()
        groupDict = dict(groups)
            
        dlg = foldersDlg("Check Orange folders that you wish to update:", None, "", 1)

        dlg.addCategory("Orange Canvas", groupDict.get("Orange Canvas", 1))
        dlg.addCategory("Documentation", groupDict.get("Documentation", 1))
        dlg.addCategory("Orange Root", groupDict.get("Orange Root", 1))
        dlg.addLabel("Orange Widgets:")
        for (group, sel) in groups:
            if group in ["Orange Canvas", "Documentation", "Orange Root"]: continue
            dlg.addCategory(group, sel, indent = 1)
        
        dlg.finishedAdding(cancel = 1)
        dlg.move((qApp.desktop().width()-dlg.width())/2, (qApp.desktop().height()-400)/2)   # center dlg window
        
        res = dlg.exec_loop()
        if res == 1:
            self.updateGroups = []
            self.dontUpdateGroups = []
            for i in range(len(dlg.checkBoxes)):
                if dlg.checkBoxes[i].isChecked(): self.updateGroups.append(dlg.folders[i])
                else:                             self.dontUpdateGroups.append(dlg.folders[i])
            self.writeVersionFile()
        return                

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

    def readInternetVersionFile(self, data, updateGroups = 1):
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
        #for g in self.updateGroups:
        #    vf.write("+%s\n" % g)
        for g in self.dontUpdateGroups:
            vf.write("-%s\n" % g)
        for fname, (version, md) in itms:
            vf.write("%s=%s:%s\n" % (fname, reduce(lambda x,y:x+"."+y, [`x` for x in version]), md))
        vf.close()


    def executeUpdate(self):
        #self.addText("Starting updating new files")
        self.addText("Reading file status from web server")

        self.updateGroups = [];  self.dontUpdateGroups = []; self.newGroups = []
        self.downstuff = {}
        
        upstuff, upUpdateGroups, upDontUpdateGroups = self.readInternetVersionFile(self.download("/orange/download/whatsup.txt").split("\n"), updateGroups = 0)
        try:
            vf = open(self.downfile)
            self.addText("Reading local file status")
            self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readLocalVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
        except:
            #self.addText("Failed to locate file '%s'." %(self.downfile))
            #res = QMessageBox.information(self,'Update Orange',"There is no 'whatsdown.txt' file. This file contains information about versions of your local Orange files.\n\nIf you press 'Replace all' you will replace your local files with the latest versions from the web.\nIf you press 'Keep all' you will keep all your local files (this will not update any files).\n\nWe suggest that you press 'Replace all' button.\n",'Replace all','Keep all', "Cancel", 0, 2)
            res = QMessageBox.information(self,'Update Orange',"There is no 'whatsdown.txt' file. This file contains information about versions of your local Orange files.\nIf you press 'Download Latest Files' you will replace all your local Orange files with the latest versions from the web.\n",'Download Latest Files', "Cancel", 0, 1)
##            if res == 0:    # replace all
##                self.addText("User chose to replace all files")
##            elif res == 1:
##                self.downstuff = upstuff
##                self.addText("User chose to keep all local files")
##                return
##            elif res == 2: return
            if res == 1: return

        itms = upstuff.items()
        itms.sort(lambda x,y:cmp(x[0], y[0]))

        for category in upUpdateGroups: #+ upDontUpdateGroups:
            if category not in self.updateGroups + self.dontUpdateGroups:
                self.newGroups.append(category)

        # show dialog with new groups
        if self.newGroups != []:
            dlg = foldersDlg("New folders have been found. \nPlease check the categories, that you would like to update:\n", None, "", 1)
            for group in self.newGroups: dlg.addCategory(group)
            dlg.finishedAdding(cancel = 0)

            res = dlg.exec_loop()
            for i in range(len(dlg.checkBoxes)):
                if dlg.checkBoxes[i].isChecked():
                    self.updateGroups.append(dlg.folders[i])
                else:
                    self.dontUpdateGroups.append(dlg.folders[i])
            self.newGroups = []

        # update new files
        updatedFiles = 0; newFiles = 0
        self.addText("<hr>Updating files...")
        self.statusBar.message("Updating files")

        for fname, (version, location) in itms:
            qApp.processEvents()
            
            # check if it is a widget directory that we don't want to update
            dirs = splitDirs(fname)
            if len(dirs) >= 2 and dirs[0].lower() == "orangewidgets" and dirs[1] in self.dontUpdateGroups: continue
            if len(dirs) >= 1 and dirs[0].lower() == "doc" and "Documentation" in self.dontUpdateGroups: continue
            if len(dirs) >= 1 and dirs[0].lower() == "orangecanvas" and "Orange Canvas" in self.dontUpdateGroups: continue
            if len(dirs) == 1 and "Orange Root" in self.dontUpdateGroups: continue

            if self.downstuff.has_key(fname) and self.downstuff[fname][0] < upstuff[fname][0]:      # there is a newer version
                updatedFiles += self.updatefile(fname, location, version, self.downstuff[fname][1], "Updating")
            elif not os.path.exists(fname):
                if self.downloadNewFilesCB.isChecked():
                    updatedFiles += self.updatefile(fname, location, version, "", "Downloading new file")
                else:
                    self.addText("Skipping new file %s" % (fname))

        self.writeVersionFile()
        self.addText("Finished updating new files. New files: <b>%d</b>. Updated files: <b>%d</b>\n<hr>" %(newFiles, updatedFiles), addBreak = 0)

        # remove widgetregistry.xml in orangeCanvas directory
        if os.path.exists(os.path.join(self.orangeDir, "OrangeCanvas/widgetregistry.xml")) and newFiles + updatedFiles > 0:
            os.remove(os.path.join(self.orangeDir, "OrangeCanvas/widgetregistry.xml"))
        
        self.statusBar.message("Finished...")
        
    # #####################################################
    # download a file with filename fname from the internet
    def download(self, fname):
        try:
            self.httpconnection.request("GET", urllib.quote(fname))
        except: # in case of exception "connection reset by peer"
            self.httpconnection = httplib.HTTPConnection('www.ailab.si')
            self.httpconnection.request("GET", urllib.quote(fname))
            
        r = self.httpconnection.getresponse()
        resp = r.read()
        if r.status != 200:
            #self.addText("Got '%s' while downloading '%s'" % (r.reason, fname))
            raise Exception("Got '%s'" % (r.reason))
        return resp

    # #########################################################
    # get new file from the internet and overwrite the old file
    # fname = name of the file
    # location = location on the web, where the file can be found
    # version = the newest file version
    # md = hash value of the local file when it was downloaded from the internet - needed to compare if the user has changed the local version of the file
    def updatefile(self, fname, location, version, md, type = "Downloading"):
        self.addText(type + " %s ... " % fname, addBreak = 0)
        qApp.processEvents()
        try:
            newscript = self.download("/orange/download/lastStable/" + location)
        except Exception, inst:
            self.addText('<font color="#FF0000">Failed</font> (%s)' % (inst[0]))
            return 0

        dname = os.path.dirname(fname)
        if dname and not os.path.exists(dname):
            os.makedirs(dname)

        # read existing file
        createBackup = 0
        if md != "" and os.path.exists(fname):
            existing = open(fname, "rb")
            currmd = md5.new()
            currmd.update(existing.read())
            existing.close()
            if currmd.hexdigest() != md:   # the local file has changed
                res = QMessageBox.information(self,'Update Orange',"Your local file '%s' was edited. A newer version of this file is available on the web.\nDo you wish to overwrite local copy with newest version (a backup of current file will be created) or keep your current file?" % (os.path.split(fname)[1]),'Overwrite With Newest','Keep Current File')
                if res == 0:    # overwrite
                    createBackup = 1
                    currmd = md5.new()
                    currmd.update(newscript)
                if res == 1:    # keep local
                    self.addText('<font color="#0000FF">Skipping</font>')
                    return 0
        else:
            currmd = md5.new()
            currmd.update(newscript)
        
        if createBackup:
            try:
                if os.path.exists(fname+".bak"):
                    os.remove(fname+".bak")
                os.rename(fname, fname+".bak")  # create backup
            except:
                self.addText('<font color="#FF0000">Failed</font> (%s)' % (inst[0]))
                self.addText('Unable to update file <font color="#FF0000">%s</font>. Please close all programs that are using it.' % (os.path.split(fname)[1]))
                return 0

        try:
            nf = open(fname, "wb")
            nf.write(newscript)
            nf.close()
            self.downstuff[fname] = (version, currmd.hexdigest())
            self.addText('<font color="#0000FF">OK</font>')
            return 1
        except:
            self.addText('<font color="#FF0000">Failed</font> (%s)' % (inst[0]))
            self.addText('Unable to update file <font color="#FF0000">%s</font>. Please close all programs that are using it.' % (os.path.split(fname)[1]))
            return 0


    def addText(self, text, nobr = 1, addBreak = 1):
        if nobr: self.text.setText(str(self.text.text()) + '<nobr>' + text + '</nobr>')
        else:    self.text.setText(str(self.text.text()) + text)
        if addBreak: self.text.setText(str(self.text.text()) + "<br>")
        self.text.ensureVisible(0, self.text.contentsHeight())

        


# show application dlg
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    dlg = updateOrangeDlg()
    app.setMainWidget(dlg)
    dlg.show()
    app.exec_loop() 
