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

        self.re_vline = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<md5>.*)')
        self.re_widget = re.compile(r'(?P<category>.*)[/,\\].*')
        self.re_documentation = re.compile(r'doc[/,\\].*')

        self.downfile = os.path.join(os.path.dirname(orange.__file__),"whatsdown.txt")
        self.httpconnection = httplib.HTTPConnection('magix.fri.uni-lj.si')

        self.updateGroups = []
        self.dontUpdateGroups = []
        self.newGroups = []
        self.downstuff = {}

        # read updateGroups and dontUpdateGroups
        self.addText("Welcome to the Orange update.")
        try:
            vf = open(self.downfile)
            self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
            self.addText("Current versions of Orange files were successfully located.")
        except:
            self.addText("Orange update failed to locate file '%s'." %(self.downfile))
        

        # create buttons
        self.toolUpdate  = QToolButton(QPixmap(updateIcon), "Update Files" , QString.null, self.executeUpdate, self.toolbar, 'Update Files')
        self.toolUpdate.setUsesTextLabel (1)
        self.toolbar.addSeparator()
        self.toolFolders = QToolButton(QPixmap(foldersIcon), "Folders" , QString.null, self.showFolders, self.toolbar, 'Show Folders')
        self.toolFolders.setUsesTextLabel (1)
        self.move((qApp.desktop().width()-self.width())/2, (qApp.desktop().height()-self.height())/2)   # center the window
        self.show()
        

    # ####################################
    # show the list of possible folders
    def showFolders(self):
        self.updateGroups = []
        self.dontUpdateGroups = []
        try:
            vf = open(self.downfile)
            self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
        except:
            self.addText("Failed to locate file '%s'." %(self.downfile))
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
    


    def addText(self, text):
        self.text.append("<nobr>" + text + "</nobr>\n")
        self.text.ensureVisible(0, self.text.contentsHeight())


    def readVersionFile(self, data, updateGroups = 1):
        versions = {}
        updateGroups = []; dontUpdateGroups = []
        for line in data:
            if line:
                line = line.replace("\r", "")   # replace \r in case of linux files
                line = line.replace("\n", "")
                if line[0] == "+":
                    updateGroups.append(line[1:])
                elif line[0] == "-":
                    dontUpdateGroups.append(line[1:])
                else:
                    fnd = self.re_vline.match(line)
                    if fnd:
                        fname, version, md = fnd.group("fname", "version", "md5")
                        versions[fname] = ([int(x) for x in version.split(".")], md)
        return versions, updateGroups, dontUpdateGroups

    def writeVersionFile(self):
        vf = open(self.downfile, "wt")
        itms = self.downstuff.items()
        itms.sort(lambda x,y:cmp(x[0], y[0]))
        for g in self.updateGroups:
            vf.write("+%s\n" % g)
        for g in self.dontUpdateGroups:
            vf.write("-%s\n" % g)
        for fname, (version, md) in itms:
            vf.write("%s=%s:%s\n" % (fname, reduce(lambda x,y:x+"."+y, [`x` for x in version]), md))
        vf.close()

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
    # version = the newest file version
    # md = hash value of the local file when it was downloaded from the internet - needed to compare if the user has changed the local version of the file
    def updatefile(self, fname, version, md):
        dname = os.path.dirname(fname)
        if dname and not os.path.exists(dname):
            os.makedirs(dname)

        self.addText("downloading <b>%s</b>" % fname)
        try:
            newscript = self.download("/orangeUpdate/"+fname)
        except:
            return 0
        
        if not os.path.exists("test/"+os.path.dirname(fname)):
            os.makedirs("test/"+os.path.dirname(fname))

        # read existing file
        saveFile = 1
        if md != "" and os.path.exists("test/" + fname):
            existing = open("test/" + fname, "rb")
            currmd = md5.new()
            currmd.update(existing.read())
            existing.close()
            if currmd.hexdigest() != md:   # the local file has changed
                res = QMessageBox.information(self,'Update Orange',"Local file '%s' was changed. Do you wish to overwrite local copy with newest version (a backup of current file will be created) or keep current file?" % (os.path.split(fname)[1]),'Overwrite with newest','Keep current file')
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
                if os.path.exists("test/"+fname+".bak"):
                    os.remove("test/"+fname+".bak")
                os.rename("test/"+fname, "test/"+fname+".bak")  # create backup
            except:
                self.addText("Unable to rename file <b>'%s'</b> to <b>'%s'</b>. Please close all programs that are using it." % (os.path.split(fname)[1], os.path.split(fname)[1]+'.bak'))
                return 0

        try:
            nf = open("test/"+fname, "wb")
            nf.write(newscript)
            nf.close()
            self.downstuff[fname] = (version, currmd.hexdigest())
            return 1
        except:
            self.addText("Unable to write file <b>'%s'</b>. Please close all programs that are using it." % (os.path.split(fname)[1]))
            return 0


    def executeUpdate(self):
        self.addText("Starting updating new files")
        self.addText("Reading file status from server")

        self.updateGroups = [];  self.dontUpdateGroups = []; self.newGroups = []
        self.downstuff = {}
        
        upstuff, upUpdateGroups, upDontUpdateGroups = self.readVersionFile(self.download("/orangeUpdate/whatsup.txt").split("\n"), updateGroups = 0)
        try:
            vf = open(self.downfile)
            self.addText("Reading local file status")
            self.downstuff, self.updateGroups, self.dontUpdateGroups = self.readVersionFile(vf.readlines(), updateGroups = 1)
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

        self.addText("Searching for new folders")
        for category in upUpdateGroups + upDontUpdateGroups:
            if category not in self.updateGroups + self.dontUpdateGroups:
                self.newGroups.append(category)
                self.addText("New category found: <b>%s</b>" % (category))

        # show dialog with new groups
        if self.newGroups != []:
            dlg = foldersDlg("New folders have been found. \nPlease check the folders, that you would like to update:\n", None, "", 1)
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
            self.addText("No new folders were found.")

        # update new files
        updatedFiles = 0; newFiles = 0
        self.addText("<hr>\nUpdating files...")
        self.statusBar.message("Updating files")
        for fname, (version, md) in itms:
            qApp.processEvents()
            cat = self.findFileCategory(fname)
            if self.downstuff.has_key(fname):
                # there is a newer version
                if self.downstuff[fname][0] < upstuff[fname][0] and cat in self.updateGroups:
                    updatedFiles += self.updatefile(fname, version, self.downstuff[fname][1])
            else:
                if cat in self.updateGroups:
                    self.updatefile(fname, version, "")
                    newFiles += 1
                #else: 
                #    self.addText("Ignoring %s. Category %s is in the ignore list" % (fname, cat))
                

        self.writeVersionFile()
        self.addText("Finished updating new files. New files: <b>%d</b>. Updated files: <b>%d</b>\n<hr>" %(newFiles, updatedFiles))

        # remove widgetregistry.xml in orangeCanvas directory
        if os.path.exists(os.path.join(orangeDir, "OrangeCanvas/widgetregistry.xml")):
            os.remove(os.path.join(orangeDir, "OrangeCanvas/widgetregistry.xml"))
        
        self.statusBar.message("Finished...")
        
    def findFileCategory(self, fileName):
        doc = self.re_documentation.match(fileName)
        if doc:
            return "Orange Documentation"
        
        fnd = self.re_widget.match(fileName)
        if fnd: return fnd.group("category")

        return "Orange Root"


# show application dlg
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    dlg = updateOrangeDlg()
    app.setMainWidget(dlg)
    dlg.show()
    app.exec_loop() 
