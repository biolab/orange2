import os, re, orange, httplib, urllib, sys
from qt import *
import md5


fileopen = [
    '16 13 5 1',
    '. c #040404',
    '# c #808304',
    'a c None',
    'b c #f3f704',
    'c c #f3f7f3',
    'aaaaaaaaa...aaaa',
    'aaaaaaaa.aaa.a.a',
    'aaaaaaaaaaaaa..a',
    'a...aaaaaaaa...a',
    '.bcb.......aaaaa',
    '.cbcbcbcbc.aaaaa',
    '.bcbcbcbcb.aaaaa',
    '.cbcb...........',
    '.bcb.#########.a',
    '.cb.#########.aa',
    '.b.#########.aaa',
    '..#########.aaaa',
    '...........aaaaa'
]
 

class categoriesDlg(QDialog):
    def __init__(self, caption, *args):
        apply(QDialog.__init__,(self,) + args)

        self.layout = QVBoxLayout( self, 10 )
        self.topLayout = QVGroupBox(self)
        self.layout.addWidget(self.topLayout)
        label = QLabel(caption, self.topLayout)
        self.setCaption("Qt Select Categories")
        self.resize(300,100)
        
        self.categories = []
        self.checkBoxes = []

    def addCategory(self, text, checked = 1):
        check = QCheckBox(text, self.topLayout)
        self.checkBoxes.append(check)
        check.setChecked(checked)
        self.categories.append(text)

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
        self.resize(400,400)
        self.setCaption("Qt Update Orange")
        self.toolbar = QToolBar(self, 'toolbar')
        self.statusBar = QStatusBar(self)
        self.text = QTextView (self)
        self.setCentralWidget(self.text)
        self.statusBar.message('Ready')

        self.re_vline = re.compile(r'(?P<fname>.*)=(?P<version>[.0-9]*)(:?)(?P<md5>.*)')
        self.re_widget = re.compile(r'OrangeWidgets/(?P<category>.*)/.*')

        self.downfile = os.path.dirname(orange.__file__) + "/whatsdown.txt"
        self.httpconnection = httplib.HTTPConnection('magix.fri.uni-lj.si')

        self.updateGroups = []
        self.dontUpdateGroups = []
        self.newGroups = []
        self.downstuff = {}

        # read updateGroups and dontUpdateGroups
        try:
            vf = open(self.downfile)
            self.downstuff = self.readVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
            self.addText("List of local files was successfully read.")
        except:
            self.addText("Failed to locate file '%s'." %(self.downfile))
        

        # create buttons
        self.toolUpdate  = QToolButton(QPixmap(fileopen), "Update Files" , QString.null, self.executeUpdate, self.toolbar, 'Update Files')
        self.toolUpdate.setUsesTextLabel (1)
        self.toolbar.addSeparator()
        self.toolCategories = QToolButton(QPixmap(fileopen), "Categories" , QString.null, self.showCategories, self.toolbar, 'Show Categories')
        self.toolCategories.setUsesTextLabel (1)
        self.show()

    # ####################################
    # show the list of possible categories
    def showCategories(self):
        self.updateGroups = []
        self.dontUpdateGroups = []
        try:
            vf = open(self.downfile)
            self.downstuff = self.readVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
        except:
            self.addText("Failed to locate file '%s'." %(self.downfile))
            self.addText("No categories found.")
            return
            
        dlg = categoriesDlg("Check the list of folders you wish to update:", None, "", 1)
        for group in self.updateGroups: dlg.addCategory(group, 1)
        for group in self.dontUpdateGroups: dlg.addCategory(group, 0)
        dlg.finishedAdding(cancel = 1)
        
        res = dlg.exec_loop()
        print res
        if res == 1:
            self.updateGroups = []
            self.dontUpdateGroups = []
            for i in range(len(dlg.checkBoxes)):
                if dlg.checkBoxes[i].isChecked(): self.updateGroups.append(dlg.categories[i])
                else:                             self.dontUpdateGroups.append(dlg.categories[i])
            self.writeVersionFile()
        return                
    


    def addText(self, text):
        self.text.append("<nobr>" + text + "</nobr>\n")
        self.text.ensureVisible(0, self.text.contentsHeight())


    def readVersionFile(self, data, updateGroups = 1):
        versions = {}
        for line in data:
            if line:
                line = line.replace("\r", "")   # replace \r in case of linux files
                line = line.replace("\n", "")
                if line[0] == "+":
                    if updateGroups: self.updateGroups.append(line[1:])
                elif line[0] == "-":
                    if updateGroups: self.dontUpdateGroups.append(line[1:])
                else:
                    fnd = self.re_vline.match(line)
                    if fnd:
                        fname, version, md = fnd.group("fname", "version", "md5")
                        versions[fname] = ([int(x) for x in version.split(".")], md)
        return versions

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
            existing = open("test/" + fname, "rt")
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
                self.addText("Unable to rename file <b>'%s'</b> to <b>'%s'</b>. Check if it is open." % (os.path.split(fname)[1], os.path.split(fname)[1]+'.bak'))
                return 0

        try:
            nf = open("test/"+fname, "wt")
            nf.write(newscript)
            nf.close()
            self.downstuff[fname] = (version, currmd.hexdigest())
            return 1
        except:
            self.addText("Unable to write file <b>'%s'</b>. Check if it is open." % (os.path.split(fname)[1]))
            return 0


    def executeUpdate(self):
        self.addText("Starting updating new files")
        self.addText("Reading file status from server")

        self.updateGroups = [];  self.dontUpdateGroups = []; self.newGroups = []
        self.downstuff = {}
        
        upstuff = self.readVersionFile(self.download("/orangeUpdate/whatsup.txt").split("\n"), updateGroups = 0)
        try:
            vf = open(self.downfile)
            self.addText("Reading local file status")
            self.downstuff = self.readVersionFile(vf.readlines(), updateGroups = 1)
            vf.close()
        except:
            self.addText("Failed to locate file '%s'." %(self.downfile))
            res = QMessageBox.information(self,'Update Qrange',"We were unable to locate file 'whatsdown.txt'. This file contains information about versions of your local files.\nThere are 2 solutions. \nIf you press 'Replace all' you will replace all your local files with the latest version.\nIf you press 'Keep all' you will keep all your local files (this way you won't get any updated files).\nWe advise you to press 'Replace all' button.",'Replace all','Keep all')
            if res == 0:    # replace all
                self.addText("User chose to replace all files")
            elif res == 1:
                self.downstuff = upstuff
                self.addText("User chose to keep all local files")
                

        itms = upstuff.items()
        itms.sort(lambda x,y:cmp(x[0], y[0]))

        self.addText("Searching for new categories of widgets")
        # find new categories
        for fname, version in itms:
            if not self.downstuff.has_key(fname):
                fnd = self.re_widget.match(fname)
                if fnd:
                    category = fnd.group("category")
                    if category not in self.updateGroups + self.dontUpdateGroups + self.newGroups:
                        self.newGroups.append(category)
                        self.addText("New category found: <b>%s</b>" % (category))

        # show dialog with new groups
        if self.newGroups != []:
            dlg = categoriesDlg("New widget categories have been found. \nPlease check the categories, that you would like to update:\n", None, "", 1)
            for group in self.newGroups: dlg.addCategory(group)
            dlg.finishedAdding(cancel = 0)

            res = dlg.exec_loop()
            for i in range(len(dlg.checkBoxes)):
                if dlg.checkBoxes[i].isChecked():
                    self.updateGroups.append(dlg.categories[i])
                else:
                    self.dontUpdateGroups.append(dlg.categories[i])
            self.newGroups = []
        else:
            self.addText("No new categories were found.")

        # update new files
        updatedFiles = 0; newFiles = 0
        self.addText("<hr>\nUpdating files...")
        self.statusBar.message("Updating files")
        for fname, (version, md) in itms:
            if self.downstuff.has_key(fname):
                # there is a newer version
                if self.downstuff[fname][0] < upstuff[fname][0]:
                    updatedFiles += self.updatefile(fname, version, self.downstuff[fname][1])
            else:
                fnd = self.re_widget.match(fname)
                if fnd:
                    category = fnd.group("category")
                    if category in self.updateGroups:
                        self.updatefile(fname, version, "")
                        newFiles += 1
                    elif category not in self.dontUpdateGroups:
                        self.addText("Ignoring %s. Category is in the ignore list" % fname)
                else:
                    self.updatefile(fname, version, "")

        self.writeVersionFile()
        self.addText("Finished updating new files. New files: <b>%d</b>. Updated files: <b>%d</b>\n<hr>" %(newFiles, updatedFiles))
        self.statusBar.message("Finished...")
        


# show application dlg
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    dlg = updateOrangeDlg()
    app.setMainWidget(dlg)
    dlg.show()
    app.exec_loop() 
