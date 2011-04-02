import xml.dom.minidom
import re
from zipfile import ZipFile
import os, glob
import time
import socket
import urllib  # urllib because we need 'urlretrieve'
import urllib2 # urllib2 because it reports HTTP Errors for 'urlopen'
import bisect
import orngEnviron
import widgetParser
import platform
from fileutil import *
from fileutil import _zipOpen

socket.setdefaulttimeout(120)  # In seconds.

class PackingException(Exception):
    pass

def suggestVersion(currentVersion):
    version = time.strftime("%Y.%m.%d")
    try:
        xmlVerInt = map(int, currentVersion.split("."))
    except:
        xmlVerInt = []
    verInt = map(int, version.split("."))
    if xmlVerInt[:3] == verInt[:3]:
        version += ".%d" % ((xmlVerInt[3] if len(xmlVerInt)>3 else 0) +1)
    return version

class OrangeRegisteredAddOn():
    def __init__(self, name, directory, systemWide=False):
        self.name = name
        self.directory = directory
        self.systemWide = systemWide
        
        # Imitate real add-ons behaviour
        self.id = "registered:"+directory

    # Imitate real add-ons behaviour
    def hasSingleWidget(self):
        return False

    def directoryDocumentation(self):
        return os.path.join(self.directory, "doc")

    def uninstall(self, refresh=True):
        try:
            unregisterAddOn(self.name, self.directory, userOnly=True)            
            if refresh:
                refreshAddOns()
            return True
        except Exception, e:
            raise InstallationException("Unable to unregister add-on: %s" % (self.name, e))

    def prepare(self, id=None, name=42, version="auto", description=None, tags=None, authorOrganizations=None, authorCreators=None, authorContributors=None,
             preferredDirectory=None, homePage=None):
        """
        Prepares the add-on for packaging into an .oao ZIP file. Adds the necessary files to the add-on directory (and
        overwrites some!). 
        Usage of parameters:
         - id (default is None): must be a valid GUID; None means it is retained from existing addon.xml if it exists,
           otherwise a new GUID is generated
         - name (default is self.name): name of the add-on; None retains existing value if it exists and raises exception otherwise
         - version (default is "auto"): None retains existing value if it exists and does the same as "auto" otherwise; "auto"
           generates a new version number from the current date in format 'yyyy.mm.dd'; if that is equal to the current version,
           another integer component is appended
         - description (default is None): None retains existing value if it exists and raises an exception otherwise
         - tags (default is None): None retains existing value if it exists, else defaults to []
         - authorOrganizations, authorCreators, authorContributors (default is None): the same as tags
         - preferredDirectory (default is None): None retains existing value, "" removes the tag
         - homePage (default is None): the same as preferredDirectory
        """
        ##########################
        # addon.xml maintenance. #
        ##########################
        addonXmlPath = os.path.join(self.directory, "addon.xml")
        try:
            xmlDoc = xml.dom.minidom.parse(addonXmlPath)
        except Exception, e:
            print "Could not load addon.xml because \"%s\"; a new one will be created." % e
            impl = xml.dom.minidom.getDOMImplementation()
            xmlDoc = impl.createDocument(None, "OrangeAddOn", None)
        xmlDocRoot = xmlDoc.documentElement
        # GUID
        if not id and not xmlTextOf("id", parent=xmlDocRoot):   # GUID needs to be generated
            import uuid
            id = str(uuid.uuid1())
        if id:
            xmlSet(xmlDocRoot, "id", id)
        # name
        if name==42:
            name = self.name
        if name and name.strip():
            xmlSet(xmlDocRoot, "name", name.strip())
        elif not xmlTextOf("name", parent=xmlDocRoot):
            raise PackingException("'name' is a mandatory value!")
        name = xmlTextOf("name", parent=xmlDocRoot)
        # version
        xmlVersion = xmlTextOf("version", parent=xmlDocRoot)
        if not xmlVersion and not version:
            version = "auto"
        if version == "auto":
            version = suggestVersion(xmlVersion)
        if version:
            xmlSet(xmlDocRoot, "version", version)
        # description
        meta = getElementNonRecursive(xmlDocRoot, "meta", create=True)
        if description and description.strip():
            xmlSet(meta, "description", description.strip())
        elif not xmlTextOf("description", parent=meta):
            raise PackingException("'description' is a mandatory value!")
        # tags
        def updateList(root, nodeName, list):
            listNode = getElementNonRecursive(root, nodeName)
            while listNode:
                root.removeChild(listNode)
                listNode = getElementNonRecursive(root, nodeName)
            for value in list:
                root.appendChild(createTextElement(nodeName, value))
        if tags!=None:
            tagsNode = getElementNonRecursive(meta, "tags", create=True)
            updateList(tagsNode, "tag", tags)
        # authors
        if authorOrganizations!=None or authorContributors!=None or authorCreators!=None:
            authorsNode = getElementNonRecursive(meta, "authors", create=True)
            if authorOrganizations!=None: updateList(authorsNode, "organization", authorOrganizations)
            if authorCreators!=None:      updateList(authorsNode, "creator", authorCreators)
            if authorContributors!=None:  updateList(authorsNode, "contributor", authorContributors)
        #  preferredDirectory
        if preferredDirectory != None:
            xmlSet(xmlDocRoot, "preferredDirectory", preferredDirectory if preferredDirectory else None)
        #  homePage
        if homePage != None:
            xmlSet(xmlDocRoot, "homePage", homePage if homePage else None)
            
        import codecs
        xmlDoc.writexml(codecs.open(addonXmlPath, 'w', "utf-8"), encoding="UTF-8")
        print "Updated addon.xml written."

        ##########################
        # style.css creation     #
        ##########################
        localCss = os.path.join(self.directoryDocumentation(), "style.css")
        orangeCss = os.path.join(orngEnviron.orangeDocDir, "style.css")
        if not os.path.isfile(localCss):
            if os.path.isfile(orangeCss):
                import shutil
                shutil.copy(orangeCss, localCss)
                print "doc/style.css created."
            else:
                raise PackingException("Could not find style.css in orange documentation directory.")

        ##########################
        # index.html creation    #
        ##########################
        if not os.path.isdir(self.directoryDocumentation()):
            os.mkdir(self.directoryDocumentation())
        hasIndex = False
        for fname in ["main", "index", "default"]:
            for ext in ["html", "htm"]:
                hasIndex = hasIndex or os.path.isfile(os.path.join(self.directoryDocumentation(), fname+"."+ext))
        if not hasIndex:
            indexFile = open( os.path.join(self.directoryDocumentation(), "index.html"), 'w')
            indexFile.write('<html><head><link rel="stylesheet" href="style.css" type="text/css" /><title>%s</title></head><body><h1>Module Documentation</h1>%s</body></html>' % (name+" Orange Add-on Documentation",
                                                                                            "This is where technical add-on module documentation is. Well, at least it <i>should</i> be."))
            indexFile.close()
            print "doc/index.html written."
            
        ##########################
        # iconlist.html creation #
        ##########################
        wDocDir = os.path.join(self.directoryDocumentation(), "widgets")
        if not os.path.isdir(wDocDir): os.mkdir(wDocDir)
        open(os.path.join(wDocDir, "index.html"), 'w').write(self.iconListHtml())
        print "Widget list (doc/widgets/index.html) written."

        ##########################
        # copying the icons      #
        ##########################
        iconDir = os.path.join(self.directory, "widgets", "icons")
        iconDocDir = os.path.join(wDocDir, "icons")
        protIconDir = os.path.join(self.directory, "widgets", "prototypes", "icons")
        protIconDocDir = os.path.join(wDocDir, "prototypes", "icons")

        import shutil
        iconBgFile = os.path.join(orngEnviron.picsDir, "background_32.png")
        iconUnFile = os.path.join(orngEnviron.picsDir, "Unknown.png")
        if not os.path.isdir(iconDocDir): os.mkdir(iconDocDir)
        if os.path.isfile(iconBgFile): shutil.copy(iconBgFile, iconDocDir)
        if os.path.isfile(iconUnFile): shutil.copy(iconUnFile, iconDocDir)
        
        if os.path.isdir(iconDir):
            import distutils.dir_util
            distutils.dir_util.copy_tree(iconDir, iconDocDir)
        if os.path.isdir(protIconDir):
            import distutils.dir_util
            if not os.path.isdir(os.path.join(wDocDir, "prototypes")): os.mkdir(os.path.join(wDocDir, "prototypes"))
            if not os.path.isdir(protIconDocDir): os.mkdir(protIconDocDir)
            distutils.dir_util.copy_tree(protIconDir, protIconDocDir)
        print "Widget icons copied to doc/widgets/."


    #####################################################
    # What follows are ugly HTML generators.            #
    #####################################################
    def widgetDocSkeleton(self, widget, prototype=False):
        wFile = os.path.splitext(os.path.split(widget.filename)[1])[0][2:]
        pathPrefix = "../" if prototype else ""
        iconCode = '\n<p><img class="screenshot" style="z-index:2; border: none; height: 32px; width: 32px; position: relative" src="%s" title="Widget: %s" width="32" height="32" /><img class="screenshot" style="margin-left:-32px; z-index:1; border: none; height: 32px; width: 32px; position: relative" src="%sicons/background_32.png" width="32" height="32" /></p>' % (widget.icon, widget.name, pathPrefix)
        
        inputsCode = """<DT>(None)</DT>"""
        outputsCode = """<DT>(None)</DT>"""
        il, ol = eval(widget.inputList), eval(widget.outputList)
        if il:
            inputsCode = "\n".join(["<dt>%s (%s)</dt>\n<dd>Describe here, what this input does.</dd>\n" % (p[0], p[1]) for p in il])
        if ol:
            outputsCode = "\n".join(["<dt>%s (%s)</dt>\n<dd>Describe here, what this output does.</dd>\n" % (p[0], p[1]) for p in ol])
        html = """<html>
<head>
<title>%s</title>
<link rel=stylesheet href="%s../style.css" type="text/css" media=screen>
</head>

<body>

<h1>%s</h1>
%s
<p>This widget does this and that..</p>

<h2>Channels</h2>

<h3>Inputs</h3>

<dl class=attributes>
%s
</dl>

<h3>Outputs</h3>
<dl class=attributes>
%s
</dl>

<h2>Description</h2>

<!-- <img class="leftscreenshot" src="%s.png" align="left"> -->

<p>This is a widget which ...</p>

<p>If you press <span class="option">Reload</span>, something will happen. <span class="option">Commit</span> button does something else.</p>

<h2>Examples</h2>

<p>This widget is used in this and that way. It often gets data from
the <a href="Another.htm">Another Widget</a>.</p>

<!-- <img class="schema" src="%s-Example.png" alt="Schema with %s widget"> -->

</body>
</html>""" % (widget.name, pathPrefix, widget.name, iconCode, inputsCode, outputsCode, wFile, wFile, widget.name)
        return html
        
    
    def iconListHtml(self, createSkeletonDocs=True):
        html = """
<style>
div#maininner {
  padding-top: 25px;
}

div.catdiv h2 {
  border-bottom: none;
  padding-left: 20px;
  padding-top: 5px;
  font-size: 14px;
  margin-bottom: 5px;
  margin-top: 0px;
  color: #fe6612;
}

div.catdiv {
  margin-left: 10px;
  margin-right: 10px;
  margin-bottom: 20px;
  background-color: #eeeeee;
}

div.catdiv table {
  width: 98%;
  margin: 10px;
  padding-right: 20px;
}

div.catdiv table td {
  background-color: white;
/*  height: 18px;*/
  margin: 25px;
  vertical-align: center;
  border-left: solid #eeeeee 10px;
  border-bottom: solid #eeeeee 3px;
  font-size: 13px;
}

div.catdiv table td.left {
  width: 3%;
  height: 28px;
  padding: 0;
  margin: 0;
}

div.catdiv table td.left-nodoc {
  width: 3%;
  color: #aaaaaa;
  padding: 0;
  margin: 0
}


div.catdiv table td.right {
  padding-left: 5px;
  border-left: none;
  width: 22%;
  font-size: 11px;
}

div.catdiv table td.right-nodoc {
  width: 22%;
  padding-left: 5px;
  border-left: none;
  color: #aaaaaa;
  font-size: 11px;
}

div.catdiv table td.empty {
  background-color: #eeeeee;
}


.rnd1 {
 height: 1px;
 border-left: solid 3px #ffffff;
 border-right: solid 3px #ffffff;
 margin: 0px;
 padding: 0px;
}

.rnd2 {
 height: 2px;
 border-left: solid 1px #ffffff;
 border-right: solid 1px #ffffff;
 margin: 0px;
 padding: 0px;
}

.rnd11 {
 height: 1px;
 border-left: solid 1px #eeeeee;
 border-right: solid 1px #eeeeee;
 margin: 0px;
 padding: 0px;
}

.rnd1l {
 height: 1px;
 border-left: solid 1px white;
 border-right: solid 1px #eeeeee;
 margin: 0px;
 padding: 0px;
}

div.catdiv table img {
  border: none;
  height: 28px;
  width: 28px;
  position: relative;
}
</style>

<script>
function setElColors(t, id, color) {
  t.style.backgroundColor=document.getElementById('cid'+id).style.backgroundColor = color;
}
</script>

<p style="font-size: 16px; font-weight: bold">Catalog of widgets</p>
        """
        wDir = os.path.join(self.directory, "widgets")
        pDir = os.path.join(wDir, "prototypes")
        widgets = {}
        for (prototype, filename) in [(False, filename) for filename in glob.iglob(os.path.join(wDir, "*.py"))] + [(True, filename) for filename in glob.iglob(os.path.join(pDir, "*.py"))]:
            if os.path.isdir(filename) or os.path.islink(filename):
                continue
            try:
                meta = widgetParser.WidgetMetaData(file(filename).read(), "Prototypes" if prototype else "Uncategorized", enforceDefaultCategory=prototype, filename=filename)
            except:
                continue   # Probably not an Orange Widget module; skip this file.
            if meta.category in widgets:
                widgets[meta.category].append((prototype, meta))
            else:
                widgets[meta.category] = [(prototype, meta)]
        categoryList = [cat for cat in widgets.keys() if cat not in ["Prototypes", "Uncategorized"]]
        categoryList.sort()
        for cat in ["Uncategorized"] + categoryList + ["Prototypes"]:
            if cat not in widgets:
                continue
            html += """    <div class="catdiv">
    <div class="rnd1"></div>
    <div class="rnd2"></div>

    <h2>%s</h2>
    <table><tr>
""" % cat
            for i, (p, w) in enumerate(widgets[cat]):
                if (i>0) and (i%4 == 0):
                    html += "</tr><tr>\n"
                wRelDir = os.path.relpath(os.path.split(w.filename)[0], wDir) if "relpath" in os.path.__dict__ else os.path.split(w.filename)[0].replace(wDir, "")
                docFile = os.path.join(wRelDir, os.path.splitext(os.path.split(w.filename)[1][2:])[0] + ".htm")
                
                iconFile = os.path.join(wRelDir, w.icon)
                if not os.path.isfile(os.path.join(wDir, iconFile)):
                    iconFile = "icons/Unknown.png"
                if os.path.isfile(os.path.join(self.directoryDocumentation(), "widgets", docFile)):
                    html += """<td id="cid%d" class="left"
      onmouseover="this.style.backgroundColor='#fff7df'"
      onmouseout="this.style.backgroundColor=null"
      onclick="this.style.backgroundColor=null; window.location='%s'">
      <div class="rnd11"></div>
      <img style="z-index:2" src="%s" title="Widget: Text File" width="28" height="28" /><img style="margin-left:-28px; z-index:1" src="icons/background_32.png" width="28" height="28" />
      <div class="rnd11"></div>
  </td>

  <td class="right"
    onmouseover="setElColors(this, %d, '#fff7df')"
    onmouseout="setElColors(this, %d, null)"
    onclick="setElColors(this, %d, null); window.location='%s'">
      %s
</td>
""" % (i, docFile, iconFile, i, i, i, docFile, w.name)
                else:
                    skeletonFileName = os.path.join(self.directoryDocumentation(), "widgets", docFile+".skeleton")
                    if not os.path.isdir(os.path.dirname(skeletonFileName)):
                        os.mkdir(os.path.dirname(skeletonFileName))
                    open(skeletonFileName, 'w').write(self.widgetDocSkeleton(w, prototype=p))
                    html += """  <td id="cid%d" class="left-nodoc">
      <div class="rnd11"></div>
      <img style="z-index:2" src="%s" title="Widget: Text File" width="28" height="28" /><img style="margin-left:-28px; z-index:1" src="icons/background_32.png" width="28" height="28" />
      <div class="rnd11"></div>
  </td>
  <td class="right-nodoc">
      <div class="rnd1l"></div>
      %s
      <div class="rnd1l"></div>

  </td>
""" % (i, iconFile, w.name)
            html += '</tr></table>\n<div class="rnd2"></div>\n<div class="rnd1"></div>\n</div>\n'
        return html
    ###########################################################################
    # Here ends the ugly HTML generators. Only beautiful code from now on! ;) #
    ###########################################################################
        

class OrangeAddOn():
    """
    Stores data about an add-on for Orange. 
    """

    def __init__(self, xmlFile=None):
        self.name = None
        self.architecture = None
        self.homePage = None
        self.id = None
        self.versionStr = None
        self.version = None
        
        self.description = None
        self.tags = []
        self.authorOrganizations = []
        self.authorCreators = []
        self.authorContributors = []
        
        self.preferredDirectory = None
        
        self.widgets = []  # List of widgetParser.WidgetMetaData objects
        
        if xmlFile:
            xmlDocRoot = xmlFile if xmlFile.__class__ is xml.dom.minidom.Element else xml.dom.minidom.parse(xmlFile).documentElement
            try:
                self.parseXml(xmlDocRoot)
            finally:
                xmlDocRoot.unlink()

    def clone(self, new=None):
        if not new:
            new = OrangeAddOn()
        new.name = self.name
        new.architecture = self.architecture
        new.homePage = self.homePage
        new.id = self.id
        new.versionStr = self.versionStr
        new.version = list(self.version)
        new.description = self.description
        new.tags = list(self.tags)
        new.authorOrganizations = list(self.authorOrganizations)
        new.authorCreator = list(self.authorCreators)
        new.authorContributors = list(self.authorContributors)
        new.prefferedDirectory = self.preferredDirectory
        new.widgets = [w.clone() for w in self.widgets]
        return new

    def directoryDocumentation(self):
    #TODO This might be redefined in orngConfiguration.
        return os.path.join(self.directory, "doc")

    def parseXml(self, root):
        if root.tagName != "OrangeAddOn":
            raise Exception("Invalid XML add-on descriptor: wrong root element name!")
        
        mandatory = ["id", "architecture", "name", "version", "meta"]
        textNodes = {"id": "id", "architecture": "architecture", "name": "name", "version": "versionStr", "preferredDirectory": "preferredDirectory", "homePage": "homePage"}
        for node in [n for n in root.childNodes if n.nodeType==n.ELEMENT_NODE]:
            if node.tagName in mandatory:
                mandatory.remove(node.tagName)
                
            if node.tagName in textNodes:
                setattr(self, textNodes[node.tagName], widgetParser.xmlTextOf(node))
            elif node.tagName == "meta":
                for node in [n for n in node.childNodes if n.nodeType==n.ELEMENT_NODE]:
                    if node.tagName == "description":
                        self.description = widgetParser.xmlTextOf(node, True)
                    elif node.tagName == "tags":
                        for tagNode in [n for n in node.childNodes if n.nodeType==n.ELEMENT_NODE and n.tagName == "tag"]:
                            self.tags.append(widgetParser.xmlTextOf(tagNode))
                    elif node.tagName == "authors":
                        authorTypes = {"organization": self.authorOrganizations, "creator": self.authorCreators, "contributor": self.authorContributors}
                        for authorNode in [n for n in node.childNodes if n.nodeType==n.ELEMENT_NODE and n.tagName in authorTypes]:
                            authorTypes[authorNode.tagName].append(widgetParser.xmlTextOf(authorNode))
            elif node.tagName == "widgets":
                for node in [n for n in node.childNodes if n.nodeType==n.ELEMENT_NODE]:
                    if node.tagName == "widget":
                        self.widgets.append(widgetParser.WidgetMetaData(node))
        
        if "afterParse" in self.__class__.__dict__:
            self.afterParse(root)
        
        self.validateArchitecture()
        if mandatory:
            raise Exception("Mandatory elements missing: "+", ".join(mandatory))
        self.validateId()
        self.validateName()
        self.validateVersion()
        self.validateDescription()
        if self.preferredDirectory==None:
            self.preferredDirectory = self.name

    def validateArchitecture(self):
        if self.architecture != "1":
            raise Exception("Only architecture '1' is supported by current Orange!")
    
    def validateId(self):
        idPattern = re.compile("[0-9a-z]{8}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{12}")
        if not idPattern.match(self.id):
            raise Exception("Invalid ID!")

    def validateName(self):
        if self.name.strip() == "":
            raise Exception("Name is a mandatory field!")
    
    def validateVersion(self):
        self.version = []  
        for sub in self.versionStr.split("."):
            try:
                self.version.append(int(sub))
            except:
                self.version = []
                raise Exception("Invalid version string: '%s' is not an integer!" % sub)
        self.versionStr = ".".join(map(str,self.version))
            
    def validateDescription(self):
        if self.name.strip() == "":
            raise Exception("Description is a mandatory field!")
        
    def hasSingleWidget(self):
        return len(self.widgets) < 2
        

class OrangeAddOnInRepo(OrangeAddOn):
    def __init__(self, repository, fileName=None, xmlFile=None):
        OrangeAddOn.__init__(self, xmlFile)
        self.repository = repository
        if "fileName" not in self.__dict__:
            self.fileName = fileName
    
    def afterParse(self, xmlRoot):  # Called by OrangeAddOn.parseXml()
        if xmlRoot.hasAttribute("filename"):
            self.fileName = xmlRoot.getAttribute("filename")
            
    def clone(self, new=None):
        if not new:
            new = OrangeAddOnInRepo(self.repository)
        new.fileName = self.fileName
        return OrangeAddOn.clone(self, new)

class OrangeAddOnInstalled(OrangeAddOn):
    def __init__(self, directory):
        OrangeAddOn.__init__(self, os.path.join(directory, "addon.xml") if directory else None)
        self.directory = directory
    
    def uninstall(self, refresh=True):
        try:
            deltree(self.directory)
            del installedAddOns[self.id]
            if refresh:
                refreshAddOns()
            return True
        except Exception, e:
            raise InstallationException("Unable to remove add-on: %s" % (self.name, e))
        
    def clone(self, new=None):
        if not new:
            new = OrangeAddOnInstalled(None)
        new.directory = self.directory
        return OrangeAddOn.clone(self, new)
        
availableAddOns = {}  # RepositoryURL -> OrangeAddOnRepository object 
installedAddOns = {}  # ID -> OrangeAddOnInstalled object
registeredAddOns = [] # OrangeRegisteredAddOn objects


class RepositoryException(Exception):
    pass

global indexRE
indexRE = "[^a-z0-9-']"

class OrangeAddOnRepository:
    def __init__(self, name, url, load=True, force=False):
        self.name = name
        self.url = url
        self.checkUrl()
        self.addOns = {}
        self.index = []
        self.lastRefreshUTC = 0
        self.refreshIndex()
        self.hasWebScript = False
        if load:
            try:
                self.refreshData(True, True)
            except Exception, e:
                if force:
                    print "Couldn't load data from repository '%s': %s" % (self.name, e)
                    return
                raise e
        
    def clone(self, new=None):
        if not new:
            new = OrangeAddOnRepository(self.name, self.url, load=False)
        new.addOns = {}
        for (id, versions) in self.addOns.items():
            new.addOns[id] = [ao.clone() for ao in versions]
        new.index = list(self.index)
        new.lastRefreshUTC = self.lastRefreshUTC
        new.hasWebScript = self.hasWebScript if hasattr(self, 'hasWebScript') else False
        return new

    def checkUrl(self):
        supportedProtocols = ["file", "http"]
        if "://" not in self.url:
            self.url = "file://"+self.url
        protocol = self.url.split("://")[0]
        if protocol not in supportedProtocols:
            raise Exception("Unable to load repository data: protocol '%s' not supported!" % protocol)

    def addAddOn(self, addOn):
        if addOn.id in self.addOns:
            versions = self.addOns[addOn.id]
            for version in versions:
                if version.version == addOn.version:
                    print "Ignoring the second occurence of addon '%s', version '%s'." % (addOn.name, addOn.versionStr)
                    return
            versions.append(addOn)
        else:
            self.addOns[addOn.id] = [addOn]

    def addPackedAddOn(self, oaoFile, fileName=None):
        pack = ZipFile(oaoFile, 'r')
        try:
            manifestFile = _zipOpen(pack, 'addon.xml')
            manifest = xml.dom.minidom.parse(manifestFile).documentElement
            manifest.appendChild(widgetParser.widgetsXml(pack))
            addOn = OrangeAddOnInRepo(self, fileName, xmlFile=manifest)
            self.addAddOn(addOn)
        except Exception, e:
            raise Exception("Unable to load add-on descriptor: %s" % e)
    
    def refreshData(self, force=False, firstLoad=False, interval=3600*24):
        if force or (self.lastRefreshUTC < time.time() - interval):
            self.lastRefreshUTC = time.time()
            self.hasWebScript = False
            try:
                protocol = self.url.split("://")[0]
                if protocol == "http": # A remote repository
                    # Try to invoke a server-side script to retrieve add-on index (and therefore avoid downloading archives)
                    repositoryXmlDoc = None
                    try:
                        repositoryXmlDoc = urllib2.urlopen(self.url+"/addOnServer.py?machine=1")
                        repositoryXml = xml.dom.minidom.parse(repositoryXmlDoc).documentElement
                        if repositoryXml.tagName != "OrangeAddOnRepository":
                            raise Exception("Invalid XML add-on repository descriptor: wrong root element name!")
                        self.addOns = {}
                        for (i, node) in enumerate([n for n in repositoryXml.childNodes if n.nodeType==n.ELEMENT_NODE]):
                            if node.tagName == "OrangeAddOn":
                                try:
                                    addOn = OrangeAddOnInRepo(self, xmlFile = node)
                                    self.addAddOn(addOn)
                                except Exception, e:
                                    print "Ignoring node nr. %d in repository '%s' because of an error: %s" % (i+1, self.name, e)
                        self.hasWebScript = True
                        return True
                    except Exception, e:
                        print "Warning: a problem occurred using server-side script on repository '%s': %s.\nAll add-ons need to be downloaded for their metadata to be extracted!" % (self.name, e)

                    # Invoking script failed - trying to get and parse a directory listing
                    try:
                        repoConn = urllib2.urlopen(self.url+'abc')
                        response = "".join(repoConn.readlines())
                    except Exception, e:
                        raise RepositoryException("Unable to load repository data: %s" % e)
                    addOnFiles = map(lambda x: x.split('"')[1], re.findall(r'href\s*=\s*"[^"/?]*\.oao"', response))
                    if len(addOnFiles)==0:
                        if firstLoad:
                            raise RepositoryException("Unable to load repository data: this is not an Orange add-on repository!")
                        else:
                            print "Repository '%s' is empty ..." % self.name
                    self.addOns = {}
                    for addOnFile in addOnFiles:
                        try:
                            addOnTmpFile = urllib.urlretrieve(self.url+"/"+addOnFile)[0]
                            self.addPackedAddOn(addOnTmpFile, addOnFile)
                        except Exception, e:
                            print "Ignoring '%s' in repository '%s' because of an error: %s" % (addOnFile, self.name, e)
                elif protocol == "file": # A local repository: open each and every archive to obtain data
                    dir = self.url.replace("file://","")
                    if not os.path.isdir(dir):
                        raise RepositoryException("Repository '%s' is not valid: '%s' is not a directory." % (self.name, dir))
                    self.addOns = {}
                    for addOnFile in glob.glob(os.path.join(dir, "*.oao")):
                        try:
                            self.addPackedAddOn(addOnFile, os.path.split(addOnFile)[1])
                        except Exception, e:
                            print "Ignoring '%s' in repository '%s' because of an error: %s" % (addOnFile, self.name, e)
                return True
            finally:
                self.refreshIndex()
        return False
        
    def addToIndex(self, addOn, text):
        words = [word for word in re.split(indexRE, text.lower()) if len(word)>1]
        for word in words:
            bisect.insort_right(self.index, (word, addOn.id) )
                
    def refreshIndex(self):
        self.index = []
        for addOnVersions in self.addOns.values():
            for addOn in addOnVersions:
                for str in [addOn.name, addOn.description] + addOn.authorCreators + addOn.authorContributors + addOn.authorOrganizations + addOn.tags +\
                           [" ".join([w.name, w.contact, w.description, w.category, w.tags]) for w in addOn.widgets]:
                    self.addToIndex(addOn, str)
        self.lastSearchPhrase = None
        self.lastSearchResult = None
                    
    def searchIndex(self, phrase):
        if phrase == self.lastSearchPhrase:
            return self.lastSearchResult
        
        words = [word for word in re.split(indexRE, phrase.lower()) if word!=""]
        result = set(self.addOns.keys())
        for word in words:
            subset = set()
            i = bisect.bisect_left(self.index, (word, ""))
            while self.index[i][0][:len(word)] == word:
                subset.add(self.index[i][1])
                i += 1
                if i>= len(self.index): break
            result = result.intersection(subset)
        lastSearchPhrase = phrase
        lastSearchResult = result
        return result
        
class OrangeDefaultAddOnRepository(OrangeAddOnRepository):
    def __init__(self, **args):
        OrangeAddOnRepository.__init__(self, "Default Orange Repository (orange.biolab.si)", "http://orange.biolab.si/add-ons/", force=True, **args)
        
    def clone(self, new=None):
        if not new:
            new = OrangeDefaultAddOnRepository(load=False)
        new.name = self.name
        new.url = self.url
        return OrangeAddOnRepository.clone(self, new)
        
def loadInstalledAddOnsFromDir(dir):
    if os.path.isdir(dir):
        for name in os.listdir(dir):
            addOnDir = os.path.join(dir, name)
            if not os.path.isdir(addOnDir):
                continue
            try:
                addOn = OrangeAddOnInstalled(addOnDir)
            except Exception, e:
                print "Add-on in directory '%s' has no valid descriptor (addon.xml): %s" % (addOnDir, e)
                continue
            if addOn.id in installedAddOns:
                print "Add-on in directory '%s' has the same ID as the addon in '%s'!" % (addOnDir, installedAddOns[addOn.id].directory)
                continue
            installedAddOns[addOn.id] = addOn

def repositoryListFileName():
    canvasSettingsDir = os.path.realpath(orngEnviron.directoryNames["canvasSettingsDir"])
    listFileName = os.path.join(canvasSettingsDir, "repositoryList.pickle")
    return listFileName

availableRepositories = None
            
def loadRepositories(refresh=True):
    listFileName = repositoryListFileName()
    global availableRepositories
    availableRepositories = []
    if os.path.isfile(listFileName):
        try:
            import cPickle
            file = open(listFileName, 'rb')
            availableRepositories = [repo.clone() for repo in cPickle.load(file)]
            file.close()
        except Exception, e:
            print "Unable to load repository list! Error: %s" % e
    try:
        updateDefaultRepositories(loadList=refresh)
    except Exception, e:
        print "Unable to refresh default repositories: %s" % (e)

    if refresh:
        for r in availableRepositories:
            #TODO: # Should show some progress (and enable cancellation)
            try:
                r.refreshData(force=False)
            except Exception, e:
                print "Unable to refresh repository %s! Error: %s" % (r.name, e)
    saveRepositories()

def saveRepositories():
    listFileName = repositoryListFileName()
    try:
        import cPickle
        global availableRepositories
        cPickle.dump(availableRepositories, open(listFileName, 'wb'))
    except Exception, e:
        print "Unable to save repository list! Error: %s" % e
    

def updateDefaultRepositories(loadList=True):
    global availableRepositories
    default = [OrangeDefaultAddOnRepository(load=False)]
    defaultKeys = [(repo.url, repo.name) for repo in default]
    existingKeys = [(repo.url, repo.name) for repo in availableRepositories]
    
    for i, key in enumerate(defaultKeys):
        if key not in existingKeys:
            availableRepositories.append(default[i])
            if loadList:
                default[i].refreshData(firstLoad=True)
    
    to_remove = []
    for i, key in enumerate(existingKeys):
        if isinstance(availableRepositories[i], OrangeDefaultAddOnRepository) and \
           key not in defaultKeys:
            to_remove.append(availableRepositories[i])
    for tr in to_remove:
        availableRepositories.remove(tr)
    
    
    
    
    
    
addOnDirectories = []
def addAddOnDirectoriesToPath():
    import os, sys
    global addOnDirectories, registeredAddOns
    sys.path = [dir for dir in sys.path if dir not in addOnDirectories]
    for addOn in installedAddOns.values() + registeredAddOns:
        path = addOn.directory
        for p in [path, os.path.join(path, "widgets"), os.path.join(path, "widgets", "prototypes"), os.path.join(path, "lib-%s" % "-".join(( sys.platform, "x86" if (platform.machine()=="") else platform.machine(), ".".join(map(str, sys.version_info[:2])) )) )]:
            if os.path.isdir(p) and not any([orngEnviron.samepath(p, x) for x in sys.path]):
                if p not in sys.path:
                    addOnDirectories.append(p)
                    sys.path.insert(0, p)

def deltree(dirname):
     if os.path.exists(dirname):
        for root,dirs,files in os.walk(dirname):
                for dir in dirs:
                        deltree(os.path.join(root,dir))
                for file in files:
                        os.remove(os.path.join(root,file))     
        os.rmdir(dirname)

class InstallationException(Exception):
    pass

def installAddOn(oaoFile, globalInstall=False, refresh=True):
    try:
        pack = ZipFile(oaoFile, 'r')
    except Exception, e:
        raise Exception("Unable to unpack the add-on '%s': %s" % (oaoFile, e))
        
    try:
        for filename in pack.namelist():
            if filename[0]=="\\" or filename[0]=="/" or filename[:2]=="..":
                raise InstallationException("Refusing to install unsafe package: it contains file named '%s'!" % filename)
        
        root = orngEnviron.addOnsDirSys if globalInstall else orngEnviron.addOnsDirUser
        
        try:
            manifest = _zipOpen(pack, 'addon.xml')
            addOn = OrangeAddOn(manifest)
        except Exception, e:
            raise Exception("Unable to load add-on descriptor: %s" % e)
        
        if addOn.id in installedAddOns:
            raise InstallationException("An add-on with this ID is already installed!")
        
        # Find appropriate directory name for the new add-on.
        i = 1
        while True:
            addOnDir = os.path.join(root, addOn.preferredDirectory + ("" if i<2 else " (%d)"%i))
            if not os.path.exists(addOnDir):
                break
            i += 1
            if i>1000:  # Avoid infinite loop if something goes wrong.
                raise InstallationException("Cannot find an appropriate directory name for the new add-on.")
        
        # Install (unpack) add-on.
        try:
            os.makedirs(addOnDir)
        except OSError, e:
            if e.errno==13:  # Permission Denied
                raise InstallationException("No write permission for the add-ons directory!")
        except Exception, e:
                raise Exception("Cannot create a new add-on directory: %s" % e)

        try:
            if hasattr(pack, "extractall"):
                pack.extractall(addOnDir)
            else: # Python 2.5
                import shutil
                for filename in pack.namelist():
                    # don't include leading "/" from file name if present
                    if filename[0] == '/':
                        targetpath = os.path.join(addOnDir, filename[1:])
                    else:
                        targetpath = os.path.join(addOnDir, filename)                    
                    upperdirs = os.path.dirname(targetpath)
                    if upperdirs and not os.path.exists(upperdirs):
                        os.makedirs(upperdirs)
            
                    if filename[-1] == '/':
                        if not os.path.isdir(targetpath):
                            os.mkdir(targetpath)
                        continue
            
                    source = _zipOpen(pack, filename)
                    target = file(targetpath, "wb")
                    shutil.copyfileobj(source, target)
                    source.close()
                    target.close()

            addOn = OrangeAddOnInstalled(addOnDir)
            installedAddOns[addOn.id] = addOn
        except Exception, e:
            try:
                deltree(addOnDir)
            except:
                pass
            raise Exception("Cannot install add-on: %s"%e)
        
        if refresh:
            refreshAddOns()
    finally:
        pack.close()

def installAddOnFromRepo(addOnInRepo, globalInstall=False, refresh=True):
    try:
        tmpFile = urllib.urlretrieve(addOnInRepo.repository.url+"/"+addOnInRepo.fileName)[0]
    except Exception, e:
        raise InstallationException("Unable to download add-on from repository: %s" % e)
    installAddOn(tmpFile, globalInstall, refresh)

def loadAddOns():
    loadInstalledAddOnsFromDir(orngEnviron.addOnsDirSys)
    loadInstalledAddOnsFromDir(orngEnviron.addOnsDirUser)

def refreshAddOns(reloadPath=False):
    if reloadPath:
        addAddOnDirectoriesToPath()
    for func in addOnRefreshCallback:
        func()
        
        
        
# Registered add-ons support        
def __readAddOnsList(addonsFile, systemWide):
    if os.path.isfile(addonsFile):
        namePathList = [tuple([x.strip() for x in lne.split("\t")]) for lne in file(addonsFile, "rt")]
        return [OrangeRegisteredAddOn(name, path, systemWide) for (name, path) in namePathList]
    else:
        return []
    
def __readAddOnLists(userOnly=False):
    return __readAddOnsList(os.path.join(orngEnviron.orangeSettingsDir, "add-ons.txt"), False) + ([] if userOnly else __readAddOnsList(os.path.join(orngEnviron.orangeDir, "add-ons.txt"), True))

def __writeAddOnLists(addons, userOnly=False):
    file(os.path.join(orngEnviron.orangeSettingsDir, "add-ons.txt"), "wt").write("\n".join(["%s\t%s" % (a.name, a.directory) for a in addons if not a.systemWide]))
    if not userOnly:
        file(os.path.join(orngEnviron.orangeDir        , "add-ons.txt"), "wt").write("\n".join(["%s\t%s" % (a.name, a.directory) for a in addons if     a.systemWide]))

def registerAddOn(name, path, add = True, refresh=True, systemWide=False):
    if not add:
        unregisredAddOn(name, path, userOnly=not systemWide)
    else:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        __writeAddOnLists([a for a in __readAddOnLists(userOnly=not systemWide) if a.name != name and a.directory != path] + ([OrangeRegisteredAddOn(name, path, systemWide)] or []), userOnly=not systemWide)
    
        global registeredAddOns
        registeredAddOns.append( OrangeRegisteredAddOn(name, path, systemWide) )
    if refresh:
        refreshAddOns()

def unregisterAddOn(name, path, userOnly=False):
    global registeredAddOns
    registeredAddOns = [ao for ao in registeredAddOns if (ao.name!=name) or (ao.directory!=path) or (userOnly and ao.systemWide)]
    __writeAddOnLists([a for a in __readAddOnLists(userOnly=userOnly) if a.name != name and a.directory != path], userOnly=userOnly)


def __getRegisteredAddOns():
    return {'registeredAddOns': __readAddOnLists()}

loadAddOns()
globals().update(__getRegisteredAddOns())

addOnRefreshCallback = []
globals().update({'addOnRefreshCallback': addOnRefreshCallback})

addAddOnDirectoriesToPath()

loadRepositories(refresh=False)