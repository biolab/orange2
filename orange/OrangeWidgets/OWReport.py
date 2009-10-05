from OWWidget import *
from OWWidget import *
from PyQt4.QtWebKit import *

import os, time, tempfile, shutil, re, shutil, pickle

import orngEnviron
from orngEnviron import reportsDir

report = None
def escape(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("'", "\\'")


class MyTreeWidget(QTreeWidget):
    def __init__(self, parent, widget):
        QTreeWidget.__init__(self, parent)
        self.widget = widget
        
    def dropEvent(self, ev):
        QTreeWidget.dropEvent(self, ev)
        self.widget.rebuildHtml()

    def mousePressEvent(self, ev):
        QTreeWidget.mousePressEvent(self, ev)
        if ev.button() == Qt.RightButton:
            self.widget.nodePopup.popup(ev.globalPos())
        
class ReportWindow(OWWidget):
    def __init__(self):
        OWWidget.__init__(self, None, None, "Report")
        self.dontScroll = False
        global report
        report = self
        self.counter = 0
        
        self.tempdir = tempfile.mkdtemp("", "orange-report-")

        self.tree = MyTreeWidget(self.controlArea, self)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.controlArea.layout().addWidget(self.tree)
        self.tree.setAllColumnsShowFocus(1)
        self.tree.header().hide()
        QObject.connect(self.tree, SIGNAL("currentItemChanged(QTreeWidgetItem *, QTreeWidgetItem *)"), self.itemChanged)
        QObject.connect(self.tree, SIGNAL("itemActivated ( QTreeWidgetItem *, int)"), self.raiseWidget)
        QObject.connect(self.tree, SIGNAL("itemDoubleClicked ( QTreeWidgetItem *, int)"), self.raiseWidget)

        self.treeItems = {}

        self.reportBrowser = QWebView(self.mainArea)
        self.mainArea.layout().addWidget(self.reportBrowser)
        self.reportBrowser.setUrl(QUrl.fromLocalFile(os.path.join(orngEnviron.widgetDir, "report", "index.html")))
        frame = self.reportBrowser.page().mainFrame()
        self.javascript = frame.evaluateJavaScript
        frame.setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAsNeeded)

        saveButton = OWGUI.button(self.controlArea, self, "&Save", self.saveReport)
        saveButton.setAutoDefault(0)
        
        self.nodePopup = QMenu("Widget", self)
        self.nodePopup.addAction( "Show widget",  self.showActiveNodeWidget)
        self.nodePopup.addSeparator()
        rename = self.nodePopup.addAction( "&Rename", self.renameActiveNode, Qt.Key_F2)
        delete = self.nodePopup.addAction("Remove", self.removeActiveNode, Qt.Key_Delete)
        self.nodePopup.setEnabled(1)

        self.resize(900, 850)
       
    # this should have been __del__, but it doesn't get called!
    def removeTemp(self):
        try:
            shutil.rmtree(self.tempdir)
        except:
            pass


    entry = """
    <div id="%s">
        <h1>%s<span class="timestamp">%s</span></h1>
        <div class="insideh1">
            %s
        </div>
    </div>
    """

    re_h = re.compile(r"<\s*[hH](?P<level>[2-5])\s*>\s*(?P<name>.*?)\s*<\s*/[hH]\1\s*>")
    
    class HIdentificator:
        def __init__(self, node, id, treeItems):
            self.id = id
            self.subid = 0
            self.stack = [(1, node)]
            self.treeItems  = treeItems
            
        def __call__(self, mo):
            self.subid += 1
            elid = "N%03iS%03i" % (self.id, self.subid)

            level = int(mo.group("level"))
            beg = ""
            while self.stack[-1][0] >= level:
                beg += "</div>"
                self.stack.pop()
            beg += """<a name="%s"/><div id="%s" onClick="myself.changeItem(this.id);">""" % (elid, elid)
            
            node = QTreeWidgetItem(self.stack[-1][1], [mo.group("name")])
            node.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.stack[-1][1].setExpanded(True)
            node.elementId = elid
            self.treeItems[elid] = node
            self.stack.append((level, node))

            return beg+mo.group(0)
            
            
    def __call__(self, name, data, widgetId):#, settings):
        if not self.isVisible():
            self.show()
        else:
            self.raise_()
        self.counter += 1
        elid = "N%03i" % self.counter

        widnode = QTreeWidgetItem([name])
        widnode.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
        widnode.elementId = elid
        widnode.widgetId = widgetId
#        widnode.settings = settings 
        self.tree.addTopLevelItem(widnode)
        self.treeItems[elid] = widnode
        
        hiden = self.HIdentificator(widnode, self.counter, self.treeItems)
        data = self.re_h.sub(hiden, data)
        data += "</div>" * len(hiden.stack)
        newreport = self.entry % (elid, name, time.strftime("%a %b %d %y, %H:%M:%S"), data)
        widnode.data = newreport
        self.javascript("""
            document.body.innerHTML += '%s';
            document.getElementById('%s').scrollIntoView();
        """ % (escape(newreport), elid))
        self.reportBrowser.page().mainFrame().addToJavaScriptWindowObject("myself", self)
        widnode.setExpanded(False)
        
    def itemChanged(self, current, previous):
        if current:
            if self.dontScroll:
                self.javascript("document.getElementById('%s').className = 'selected';" % current.elementId)
            else:
                self.javascript("""
                    var newsel = document.getElementById('%s');
                    newsel.className = 'selected';
                    newsel.scrollIntoView();""" % current.elementId)
        if previous:
            self.javascript("document.getElementById('%s').className = '';" % previous.elementId)
        
    def rebuildHtml(self):
        tt = "\n".join(self.tree.topLevelItem(i).data for i in range(self.tree.topLevelItemCount()))
        print "\n".join(self.tree.topLevelItem(i).elementId for i in range(self.tree.topLevelItemCount()))
        self.javascript("document.body.innerHTML = '%s'" % escape(tt))
        selected = self.tree.selectedItems()
        if selected:
            self.itemChanged(selected[0], None)
        
        
    @pyqtSignature("QString") 
    def changeItem(self, elid):
        self.dontScroll = True
        item = self.treeItems[str(elid)]
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)
        self.dontScroll = False
 
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
    
    def raiseWidget(self, node, column):
        for widget in self.widgets:
            if widget.instance.widgetId == node.widgetId:
                break
        else:
            return
#        widget.instance.setSettings(node.settings)
        widget.instance.reshow()
        

    def showActiveNodeWidget(self):
        node = self.tree.currentItem()
        if node:
            self.raiseWidget(node, 0)
            
    def renameActiveNode(self):
        node = self.tree.currentItem()
        if node:
            self.editItem(node)

    def removeActiveNode(self):
        node = self.tree.currentItem()
        while node.parent():
            node = node.parent()
        self.tree.removeItemWidget(node)
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

    def saveTreeView(self, node):
        s = '<li><a href="#%s">%s</a>\n\n' % (node.elementId, node.text(0))
        if node.childCount():
            s += "<ul>\n" + \
                 "\n".join(self.saveTreeView(node.child(i)) for i in range(node.childCount())) + \
                 "</ul>\n"
        s += '</li>'
        return s
        
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

        treeIndex = "<ul>" + \
            "".join(self.saveTreeView(self.tree.topLevelItem(i)) for i in range(self.tree.topLevelItemCount())) + \
            "</ul>"
        
        tt = unicode(self.reportBrowser.page().mainFrame().toHtml())
        tt = tt.replace("<body>", '<body><table width="100%"><tr><td valign="top"><p>Index</p>'+treeIndex+"</td><td>") \
               .replace("</body", '</td></tr></table></body>')
        
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
        tt = tt.replace('<div class="selected"', '<div')
        file(filename, "wb").write(tt.encode("utf8"))
                       

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
 
