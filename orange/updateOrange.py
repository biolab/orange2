import os, re, orange, httplib, urllib, sys
from qt import *
import md5


orangeDir = os.path.split(os.path.abspath(orange.__file__))[0]
iconsDir = os.path.join(orangeDir, "OrangeCanvas/icons")
updateIcon = os.path.join(iconsDir, "update.png")
foldersIcon = os.path.join(iconsDir, "folders.png")

defaultIcon = ['16 13 5 1', '. c #040404', '# c #808304', 'a c None', 'b c #f3f704', 'c c #f3f7f3',  'aaaaaaaaa...aaaa',  'aaaaaaaa.aaa.a.a',  'aaaaaaaaaaaaa..a',
    'a...aaaaaaaa...a', '.bcb.......aaaaa', '.cbcbcbcbc.aaaaa', '.bcbcbcbcb.aaaaa', '.cbcb...........', '.bcb.#########.a', '.cb.#########.aa', '.b.#########.aaa', '..#########.aaaa', '...........aaaaa']

if not os.path.exists(updateIcon):
    updateIcon = defaultIcon

if not os.path.exists(foldersIcon):
    foldersIcon = defaultIcon

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

    def addCategory(self, text, checked = 1):
        check = QCheckBox(text, self.topLayout)
        self.checkBoxes.append(check)
        check.setChecked(checked)
        self.folders.append(text)

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
        self.resize(500,500)
        self.setCaption("Qt Orange Update")
        self.toolbar = QToolBar(self, 'toolbar')
        self.statusBar = QStatusBar(self)
        self.text = QTextView (self)
        self.setCentralWidget(self.text)
        self.statusBar.message('Ready')

        self.re_vLocalLine = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<md5>.*)')
        self.re_vInternetLine = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<location>.*)')
        self.re_widget = re.compile(r'(?P<category>.*)[/,\\].*')
        self.re_documentation = re.compile(r'doc[/,\\].*')

        self.downfile = os.path.join(os.path.dirname(orange.__file__),"whatsdown.txt")
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
            self.addText("Current versions of Orange files were successfully located.")
        except:
            self.addText("Orange update failed to locate file '%s'. There is no information about current versions of Orange files." %(self.downfile), 0)
        

        # create buttons
        self.toolUpdate  = QToolButton(QPixmap(updateIcon), "Update Files" , QString.null, self.executeUpdate, self.toolbar, 'Update Files')
        self.toolUpdate.setUsesTextLabel (1)
        self.toolbar.addSeparator()
        self.toolFolders = QToolButton(QPixmap(foldersIcon), "Folders" , QString.null, self.showFolders, self.toolbar, 'Show Folders')
        self.toolFolders.setUsesTextLabel (1)
        self.toolbar.addSeparator()
        self.updateMissingFilesCB = QCheckBox("Update missing files", self.toolbar)
        self.updateMissingFilesCB.setChecked(1)
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
            self.addText("Failed to locate file '%s'. There is no information on Orange folders that need to be updated." %(self.downfile), 0)
            self.addText("No folders found.")
            return
            
        dlg = foldersDlg("Check the list of folders you wish to update:", None, "", 1)
        for group in self.updateGroups: dlg.addCategory(group, 1)
        for group in self.dontUpdateGroups: dlg.addCategory(group, 0)
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
    


    def addText(self, text, nobr = 1):
        if nobr:
            self.text.append("<nobr>" + text + "</nobr>\n")
        else:
            self.text.append(text)
        self.text.ensureVisible(0, self.text.contentsHeight())


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
                    if len(dirs) >= 2 and dirs[0].lower() == "orangewidgets" and dirs[1] not in updateGroups and dirs[1].lower() != "icons":
                        updateGroups.append(dirs[1])

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
        self.addText("Starting updating new files")
        self.addText("Reading file status from server")

        self.updateGroups = [];  self.dontUpdateGroups = []; self.newGroups = []
        self.downstuff = {}
        
        upstuff, upUpdateGroups, upDontUpdateGroups = self.readInternetVersionFile(self.download("/orange/download/whatsup.txt").split("\n"), updateGroups = 0)
        try:
            vf = open(self.downfile)
            self.addText("Reading local file status")
            self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readLocalVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
        except:
            self.addText("Failed to locate file '%s'." %(self.downfile))
            res = QMessageBox.information(self,'Update Orange',"We were unable to locate file 'whatsdown.txt'. This file contains information about versions of your local Orange files.\nThere are 2 solutions. \nIf you press 'Replace all' you will replace all your local files with the latest version.\nIf you press 'Keep all' you will keep all your local files (this way you won't get any updated files).\nWe advise you to press 'Replace all' button.",'Replace all','Keep all')
            if res == 0:    # replace all
                self.addText("User chose to replace all files")
            elif res == 1:
                self.downstuff = upstuff
                self.addText("User chose to keep all local files")
                

        itms = upstuff.items()
        itms.sort(lambda x,y:cmp(x[0], y[0]))

        self.addText("Searching for new widget categories...")
        for category in upUpdateGroups: #+ upDontUpdateGroups:
            if category not in self.updateGroups + self.dontUpdateGroups:
                self.newGroups.append(category)
                self.addText("New category found: <b>%s</b>" % (category))

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
        else:
            self.addText("No new categories were found.")

        # update new files
        updatedFiles = 0; newFiles = 0
        self.addText("<hr>\nUpdating files...")
        self.statusBar.message("Updating files")
        for fname, (version, location) in itms:
            qApp.processEvents()
            
            # check if it is a widget directory that we don't want to update
            dirs = splitDirs(fname)
            if len(dirs) >= 2 and dirs[0].lower() == "orangewidgets" and dirs[1] in self.dontUpdateGroups: continue

            if not os.path.exists(fname):
                if self.updateMissingFilesCB.isChecked():
                    updatedFiles += self.updatefile(fname, location, version, "", "updating missing file")
                else:
                    self.addText("Skipping missing file %s" % (fname))
                continue
            
            if self.downstuff.has_key(fname):
                # there is a newer version
                if self.downstuff[fname][0] < upstuff[fname][0]:
                    updatedFiles += self.updatefile(fname, location, version, self.downstuff[fname][1], "updating")
            else:
                self.updatefile(fname, location, version, "")
                newFiles += 1

        self.writeVersionFile()
        self.addText("Finished updating new files. New files: <b>%d</b>. Updated files: <b>%d</b>\n<hr>" %(newFiles, updatedFiles))

        # remove widgetregistry.xml in orangeCanvas directory
        if os.path.exists(os.path.join(orangeDir, "OrangeCanvas/widgetregistry.xml")):
            os.remove(os.path.join(orangeDir, "OrangeCanvas/widgetregistry.xml"))
        
        self.statusBar.message("Finished...")
        
    # #####################################################
    # download a file with filename fname from the internet
    def download(self, fname):
        self.httpconnection.request("GET", urllib.quote(fname))
        r = self.httpconnection.getresponse()
        if r.status != 200:
            self.addText("Got '%s' while downloading '%s'" % (r.reason, fname))
            raise "Got '%s' while downloading '%s'" % (r.reason, fname)
        return r.read()

    # #########################################################
    # get new file from the internet and overwrite the old file
    # fname = name of the file
    # location = location on the web, where the file can be found
    # version = the newest file version
    # md = hash value of the local file when it was downloaded from the internet - needed to compare if the user has changed the local version of the file
    def updatefile(self, fname, location, version, md, type = "downloading"):
        self.addText(type + " <b>%s</b>" % fname)
        try:
            newscript = self.download("/orange/download/lastStable/"+location)
        except:
            return 0

        dname = os.path.dirname(fname)
        if dname and not os.path.exists(dname):
            os.makedirs(dname)

        # read existing file
        saveFile = 1
        if md != "" and os.path.exists(fname):
            existing = open(fname, "rb")
            currmd = md5.new()
            currmd.update(existing.read())
            existing.close()
            if currmd.hexdigest() != md:   # the local file has changed
                res = QMessageBox.information(self,'Update Orange',"Local file '%s' was changed. Do you wish to overwrite local copy \nwith newest version (a backup of current file will be created) or keep current file?" % (os.path.split(fname)[1]),'Overwrite with newest','Keep current file')
                if res == 0:    # overwrite
                    saveFile = 2
                    currmd = md5.new()
                    currmd.update(newscript)
                if res == 1:    # keep local
                    saveFile = 0
        else:
            currmd = md5.new()
            currmd.update(newscript)

        if saveFile == 0:
            return 0
        elif saveFile == 2:
            try:
                if os.path.exists(fname+".bak"):
                    os.remove(fname+".bak")
                os.rename(fname, fname+".bak")  # create backup
            except:
                self.addText("Unable to rename file <b>'%s'</b> to <b>'%s'</b>. Please close all programs that are using it." % (os.path.split(fname)[1], os.path.split(fname)[1]+'.bak'))
                return 0

        try:
            nf = open(fname, "wb")
            nf.write(newscript)
            nf.close()
            self.downstuff[fname] = (version, currmd.hexdigest())
            return 1
        except:
            self.addText("Unable to write file <b>'%s'</b>. Please close all programs that are using it." % (os.path.split(fname)[1]))
            return 0
        


# show application dlg
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    dlg = updateOrangeDlg()
    app.setMainWidget(dlg)
    dlg.show()
    app.exec_loop() 
