 # Widgets cannot be reset to the settings they had at the time of reporting.
 # The reason lies in the OWGUI callback mechanism: callbacks are triggered only
 # when the controls are changed by the user. If the related widget's attribute
 # is changed programmatically, the control is updated but the callback is not
 # called. This is done intentionally and with a very solid reason: it enables us
 # to do multiple changes without, for instance, the widget being redrawn every time.
 # Besides, it would probably lead to cycles or at least a great number of redundant calls. 
 # However, since setting attributes does not triger callbacks, setting the attributes
 # here would have not other effect than changing the widget's controls and leaving it
 # in undefined (possibly invalid) state. The reason why we do not have these problems
 # in "normal" use of settings is that the context independent settings are loaded only
 # when the widget is initialized and the context dependent settings are retrieved when
 # the new data is sent and the widget "knows" it has to reconfigure.
 # The only solution would be to require all the widgets have a method for updating
 # everything from scratch according to settings. This would require a lot of work, which
 # could even be unfeasible. For instance, there are widget which get the data, compute
 # something and discard the data. This is good since it is memory efficient, but it
 # may prohibit the widget from implementing the update-from-the-scratch method.  
 
 
from OWWidget import *
from OWWidget import *
from PyQt4.QtWebKit import *

import os, time, tempfile, shutil, re, shutil, pickle

report = None
def escape(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("'", "\\'")


class MyListWidget(QListWidget):
    def __init__(self, parent, widget):
        QListWidget.__init__(self, parent)
        self.widget = widget
        
    def dropEvent(self, ev):
        QListWidget.dropEvent(self, ev)
        self.widget.rebuildHtml()

    def mousePressEvent(self, ev):
        QListWidget.mousePressEvent(self, ev)
        node = self.currentItem() 
        if ev.button() == Qt.RightButton and node:
            self.widget.nodePopup.popup(ev.globalPos())

    
class ReportWindow(OWWidget):
    indexfile = os.path.join(orngEnviron.widgetDir, "report", "index.html")
    
    def __init__(self):
        OWWidget.__init__(self, None, None, "Report")
        self.dontScroll = False
        global report
        report = self
        self.counter = 0
        
        self.tempdir = tempfile.mkdtemp("", "orange-report-")

        self.tree = MyListWidget(self.controlArea, self)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.controlArea.layout().addWidget(self.tree)
        QObject.connect(self.tree, SIGNAL("currentItemChanged(QListWidgetItem *, QListWidgetItem *)"), self.selectionChanged)
        QObject.connect(self.tree, SIGNAL("itemActivated ( QListWidgetItem *)"), self.raiseWidget)
        QObject.connect(self.tree, SIGNAL("itemDoubleClicked ( QListWidgetItem *)"), self.raiseWidget)
        QObject.connect(self.tree, SIGNAL("itemChanged ( QListWidgetItem *)"), self.itemChanged)

        self.treeItems = {}

        self.reportBrowser = QWebView(self.mainArea)
        self.mainArea.layout().addWidget(self.reportBrowser)
        self.reportBrowser.setUrl(QUrl.fromLocalFile(self.indexfile))
        frame = self.reportBrowser.page().mainFrame()
        self.javascript = frame.evaluateJavaScript
        frame.setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAsNeeded)

        saveButton = OWGUI.button(self.controlArea, self, "&Save", self.saveReport)
        saveButton.setAutoDefault(0)
        
        self.nodePopup = QMenu("Widget")
        self.nodePopup.addAction( "Show widget",  self.showActiveNodeWidget)
        self.nodePopup.addSeparator()
#        self.renameAction = self.nodePopup.addAction( "&Rename", self.renameActiveNode, Qt.Key_F2)
        self.deleteAction = self.nodePopup.addAction("Remove", self.removeActiveNode, Qt.Key_Delete)
        self.nodePopup.setEnabled(1)

        self.resize(900, 850)
       
    # this should have been __del__, but it doesn't get called!
    def removeTemp(self):
        try:
            shutil.rmtree(self.tempdir)
        except:
            pass


    entry = """
    <div id="%s" onClick="myself.changeItem(this.id);">
        <a name="%s" />
        <h1>%s<span class="timestamp">%s</span></h1>
        <div class="insideh1">
            %s
        </div>
    </div>
    """

    def __call__(self, name, data, widgetId):
        if not self.isVisible():
            self.show()
        else:
            self.raise_()
        self.counter += 1
        elid = "N%03i" % self.counter

        widnode = QListWidgetItem(name, self.tree)
        widnode.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled | Qt.ItemIsEditable)
        widnode.elementId = elid
        widnode.widgetId = widgetId
        self.tree.addItem(widnode)
        self.treeItems[elid] = widnode
        
        newreport = self.entry % (elid, elid, name, time.strftime("%a %b %d %y, %H:%M:%S"), data)
        widnode.content = newreport
        self.javascript("""
            document.body.innerHTML += '%s';
            document.getElementById('%s').scrollIntoView();
        """ % (escape(newreport), elid))
        self.reportBrowser.page().mainFrame().addToJavaScriptWindowObject("myself", self)
        
    def selectionChanged(self, current, previous):
        if current:
            if self.dontScroll:
                self.javascript("document.getElementById('%s').className = 'selected';" % current.elementId)
            else:
                self.javascript("""
                    var newsel = document.getElementById('%s');
                    newsel.className = 'selected';
                    newsel.scrollIntoView();""" % current.elementId)
#            if not self.dontScroll:
#                self.javascript("newsel.scrollIntoView(document.getElementById('%s'));" % current.elementId)
        if previous:
            self.javascript("document.getElementById('%s').className = '';" % previous.elementId)
        
        
    def rebuildHtml(self):
        tt = "\n".join(self.tree.item(i).content for i in range(self.tree.count()))
        self.javascript("document.body.innerHTML = '%s'" % escape(tt))
        selected = self.tree.selectedItems()
        if selected:
            self.selectionChanged(selected[0], None)
        
        
    @pyqtSignature("QString") 
    def changeItem(self, elid):
        self.dontScroll = True
        item = self.treeItems[str(elid)]
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)
        self.dontScroll = False
 
    def raiseWidget(self, node):
        for widget in self.widgets:
            if widget.instance.widgetId == node.widgetId:
                break
        else:
            return
        widget.instance.reshow()
        
    def showActiveNodeWidget(self):
        node = self.tree.currentItem()
        if node:
            self.raiseWidget(node)
            
    re_h1 = re.compile(r'<h1>(?P<name>.*?)<span class="timestamp">')
    def itemChanged(self, node):
        if hasattr(node, "content"):
            be, en = self.re_h1.search(node.content).span("name")
            node.content = node.content[:be] + str(node.text()) + node.content[en:]
            self.rebuildHtml()

    def removeActiveNode(self):
        node = self.tree.currentItem()
        if node:
            self.tree.takeItem(self.tree.row(node))
        self.rebuildHtml()

        
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
    browser_re = re.compile(r'<!--browsercode(.*?)-->')
    def saveReport(self):
        filename = QFileDialog.getSaveFileName(self, "Save Report", self.saveDir, "Web page (*.html *.htm)")
        if not filename:
            return

        filename = str(filename)
        path, fname = os.path.split(filename)
        self.saveDir = path
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                QMessageBox.error(None, "Error", "Cannot create directory "+path)

        tt = file(self.indexfile, "rt").read()
        
        index = "<br/>".join('<a href="%s">%s</a>' % (self.tree.item(i).elementId, self.re_h1.search(self.tree.item(i).content).group("name"))
                             for i in range(self.tree.count()))
            
        data = "\n".join(self.tree.item(i).content for i in range(self.tree.count()))
        
        tt = tt.replace("<body>", '<body><table width="100%%"><tr><td valign="top"><p style="padding-top:25px;">Index</p>%s</td><td>%s</td></tr></table>' % (index, data))
        tt = self.browser_re.sub("\\1", tt)
        
        filepref = "file:///"+self.tempdir
        if filepref[-1] != os.sep:
            filepref += os.sep
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
        file(filename, "wb").write(tt.encode("utf8"))
 
       
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
 