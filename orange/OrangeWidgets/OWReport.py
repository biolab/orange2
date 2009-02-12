from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *
import os, time, tempfile, shutil, urllib, zipfile, re, shutil
import orngEnviron, OWGUI

reportsDir = orngEnviron.reportsDir
report = None
def escape(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("'", "\\'")
    
class OWebView(QWebView):
    def resizeEvent(self, *e):
        QWebView.resizeEvent(self, *e)
        self.page().currentFrame().evaluateJavaScript("Ext.EventManager.fireResize();")
        
class ReportWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Report")
        self.setWindowIcon(QIcon(os.path.join(orngEnviron.widgetDir, "icons/Unknown.png")))
        global report
        if not self.checkExtLibrary():
            report = None
            return

        report = self
        
        #if not os.path.exists(reportsDir):
         
        self.setLayout(QVBoxLayout())
        self.layout().setMargin(2)        
        self.reportBrowser = OWebView(self)
        self.layout().addWidget(self.reportBrowser)
        self.reportBrowser.setUrl(QUrl.fromLocalFile(os.path.join(reportsDir, "index.html")))
        self.counter = 0
        
        self.tempdir = tempfile.mkdtemp("", "orange-report-")
        hbox = OWGUI.widgetBox(self, orientation=0)
        self.reportButton = OWGUI.button(hbox, self, "&Save", self.saveReport, width=120)
        OWGUI.rubber(hbox)
       
    # this should have been __del__, but it doesn't get called!
    def removeTemp(self):
        try:
            shutil.rmtree(self.tempdir)
        except:
            pass

    def checkExtLibrary(self):
        if os.path.exists(os.path.join(reportsDir, "ext-2.2")):
            return True
        resp = QMessageBox("Additional library download", "Your version of Orange supports creating reports, but it needs to download the Ext library (~330 kB) for JavaScript. Proceed?\n\nIf you answer no, it will ask again next time you run Orange; sorry for the inconvenience ;)", QMessageBox.Question, QMessageBox.Yes|QMessageBox.Default, QMessageBox.No|QMessageBox.Escape, QMessageBox.NoButton).exec_()
        if resp == QMessageBox.No:
            return False
        try:
            import urllib
            zfname = os.path.join(reportsDir, "ext-2.2.zip")
            zf = file(zfname, "wb")
            wf = urllib.urlopen("http://www.ailab.si/orange/download/ext-2.2.zip")
            while True:
                c = wf.read(65536)
                zf.write(c)
                if len(c) < 65536:
                    break
            zf.close()
        except:
            QMessageBox.warning(None, "Additional library download", "Error downloading Ext. Reporting will not be available")
            return False
        
        try:
            zf = zipfile.ZipFile(zfname)
            for name in zf.namelist():
                if name.endswith('/'):
                    os.mkdir(os.path.join(reportsDir, name))
                else:
                    file(os.path.join(reportsDir, name), 'wb').write(zf.read(name))
        except:
            QMessageBox.warning(None, "Additional library download", "Error unzipping Ext. Reporting will not be available")
            return False

        try:
            os.remove(zfname)
        except:
            pass
        
        return True
             
        
    def __call__(self, name, data):
        if not self.isVisible():
            self.show()
            if not self.counter:
                pass
        else:
            self.raise_()
        self.counter += 1
        self.reportBrowser.page().currentFrame().evaluateJavaScript("addContent(%i, '%s', '%s', '%s');" % (
           self.counter, escape(name),
           escape(time.strftime("%a %b %d %y, %H:%M:%S")),
           escape(data)))

    def createDirectory(self):
        tmpPathName = os.tempnam(orange-report)
        os.mkdir(tmpPathName)
        return tmpPathName
    
    def getUniqueFileName(self, patt):
        for i in xrange(1000000):
            fn = os.path.join(self.tempdir, patt % i)
            if not os.path.exists(fn):
                return "file:///"+fn, fn

    img_re = re.compile(r'<IMG.*?\ssrc="(?P<imgname>[^"]*)"', re.DOTALL+re.IGNORECASE)
    def saveReport(self):
        filename = QFileDialog.getSaveFileName(self, "Save Report", reportsDir, "Web page (*.html *.htm)")
        if not filename:
            return

        filename = str(filename)
        path, fname = os.path.split(filename)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                QMessageBox.error(None, "Error", "Cannot create directory "+path)
                
        tt = unicode(self.reportBrowser.page().mainFrame().toHtml())
        tt = tt[tt.index('<div id="center1"'):]
        
        filepref = "file:///"+self.tempdir
        if filepref[-1] != os.sep:
            filepref += os.sep
        print filepref
        lfilepref = len(filepref)
        imspos = -1
        subdir = None
        while True:
            imspos = tt.find(filepref, imspos+1)
            if imspos == -1:
                break
            
            if not subdir:
                subdir = os.path.splitext(fname)[0]
                if subdir == fname:
                    subdir += "_data"
                cnt = 0
                osubdir = subdir
                while os.path.exists(os.path.join(path, subdir)):
                    cnt += 1
                    subdir = "%s%05i" % (osubdir, cnt)
                absubdir = os.path.join(path, subdir)
                os.mkdir(absubdir)

            imname = tt[imspos+lfilepref:tt.find('"', imspos)]
            shutil.copy(os.path.join(filepref[8:], imname), os.path.join(absubdir, imname))
        if subdir:
            tt = tt.replace(filepref, subdir+"/")
        tt = tt.replace('<div class="selected"', '<div class="normal"')
        file(filename, "wb").write('<html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"><style>%s</style></head><body>'
                                   % file(os.path.join(reportsDir, "index.css")).read() + tt.encode("utf8"))
                       

#        file(filename, "wb").write("""
#        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
#    
#    <link rel="stylesheet" type="text/css" href="ext-2.2/resources/css/ext-all.css">
#    <link rel="stylesheet" type="text/css" href="ext-2.2/resources/css/xtheme-gray.css">
#    <link rel="stylesheet" type="text/css" href="index.css">
#""" + tt.encode("utf8"))

def getDepth(item, expanded=True):
    ccount = item.childCount()
    return 1 + (ccount and (not expanded or item.isExpanded()) and max(getDepth(item.child(cc), expanded) for cc in range(ccount)))

# Need to use the tree's columnCount - children may have unattended additional columns
# (this happens, e.g. in the tree viewer)
def printTree(item, level, depthRem, visibleColumns, expanded=True):
    res = '<tr>'+'<td width="16px"></td>'*level + \
          '<td colspan="%i">%s</td>' % (depthRem, item.text(0) or (not level and "<root>") or "") + \
          ''.join('<td style="padding-left:10px">%s</td>' % item.text(i) for i in visibleColumns) + \
          '</tr>\n'
    if not expanded or item.isExpanded():
        for i in range(item.childCount()):
            res += printTree(item.child(i), level+1, depthRem-1, visibleColumns, expanded)
    return res
    
def reportTree(tree, expanded=True):
    tops = tree.topLevelItemCount()
    header = tree.headerItem()
    visibleColumns = [i for i in range(1, tree.columnCount()) if not tree.isColumnHidden(i)] 

    depth = tops and max(getDepth(tree.topLevelItem(cc), expanded) for cc in range(tops))
    res = "<table>\n"
    res += '<tr><th colspan="%i">%s</th>' % (depth, header.text(0))
    res += ''.join('<th>%s</th>' % header.text(i) for i in visibleColumns)
    res += '</tr>\n'
    res += ''.join(printTree(tree.topLevelItem(cc), 0, depth, visibleColumns, expanded) for cc in range(tops))
    res += "</table>\n"
    return res
 
