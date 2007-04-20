# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    parse widget information to a registry file (.xml) - info is then used inside orngTabs.py
#
import os, sys, string, re, user
import xml.dom.minidom

class WidgetsToXML:

    # read all installed widgets, build a registry and store widgets.pth with directory names in python dir
    def ParseWidgetRoot(self, widgetDirName, outputDir):
        widgetDirName = os.path.realpath(widgetDirName)
        outputDir = os.path.realpath(outputDir)

        # create xml document
        doc = xml.dom.minidom.Document()
        canvas = doc.createElement("orangecanvas")
        categories = doc.createElement("widget-categories")
        doc.appendChild(canvas)
        canvas.appendChild(categories)

        for filename in os.listdir(widgetDirName):
            full_filename = os.path.join(widgetDirName, filename)
            if os.path.isdir(full_filename):
                self.ParseDirectory(doc, categories, full_filename, filename)

        additionalFile = os.path.join(outputDir, "additionalCategories")
        if os.path.exists(additionalFile):
            for lne in open(additionalFile, "rt"):
                try:
                    catName, dirName = [x.strip() for x in lne.split("\t")]
                    self.ParseDirectory(doc, categories, dirName, catName, True)
                except:
                    pass


        # we put widgets that are in the root dir of widget directory to category "Other"
        self.ParseDirectory(doc, categories, widgetDirName, "Other")

        xmlText = doc.toprettyxml()
        file = open(os.path.join(outputDir, "widgetregistry.xml"), "wt")
        file.write(xmlText)
        file.flush()
        file.close()
        doc.unlink()

    # parse all widgets in directory widgetDirName\categoryName into new category named categoryName
    def ParseDirectory(self, doc, categories, full_dirname, categoryName, plugin = False):
        if sys.path.count(full_dirname) == 0:       # add directory to orange path
            sys.path.append(full_dirname)          # this doesn't save the path when you close the canvas, so we have to save also to widgets.pth

        for filename in os.listdir(full_dirname):
            full_filename = os.path.join(full_dirname, filename)
            if os.path.isdir(full_filename) or os.path.islink(full_filename) or os.path.splitext(full_filename)[1] != ".py":
                continue

            file = open(full_filename)
            data = file.read()
            file.close()

            name        = self.GetCustomText(data, '<name>.*</name>')
            #category    = self.GetCustomText(data, '<category>.*</category>')
            author      = self.GetCustomText(data, '<author>.*</author>')
            contact     = self.GetCustomText(data, '<contact>.*</contact>') or ""
            icon        = self.GetCustomText(data, '<icon>.*</icon>')
            priorityStr = self.GetCustomText(data, '<priority>.*</priority>')
            if priorityStr == None:    priorityStr = "5000"
            if author      == None: author = ""
            if icon        == None: icon = "icon/Unknown.png"

            description = self.GetDescription(data)
            inputList   = self.GetAllInputs(data)
            outputList  = self.GetAllOutputs(data)

            if (name == None):      # if the file doesn't have a name, we treat it as a non-widget file
                continue

            # create XML node for the widget
            child = categories.firstChild
            while (child != None and child.attributes.get("name").nodeValue != categoryName):
                child= child.nextSibling

            if (child == None):
                child = doc.createElement("category")
                child.setAttribute("name", categoryName)
                if plugin:
                    child.setAttribute("directory", full_dirname)
                categories.appendChild(child)

            widget = doc.createElement("widget")
            widget.setAttribute("file", filename[:-3])
            widget.setAttribute("name", name)
            widget.setAttribute("in", str(inputList))
            widget.setAttribute("out", str(outputList))
            widget.setAttribute("icon", icon)
            widget.setAttribute("priority", priorityStr)
            widget.setAttribute("author", author)
            widget.setAttribute("contact", contact)

            # description
            if (description != ""):
                desc = doc.createElement("description")
                descText = doc.createTextNode(description)
                desc.appendChild(descText)
                widget.appendChild(desc)

            child.appendChild(widget)


    def GetDescription(self, data):
        #read the description from widget
        search = re.search('<description>.*</description>', data, re.DOTALL)
        if (search == None):
            return ""

        description = search.group(0)[13:-14]    #delete the <...> </...>
        description = re.sub("#", "", description)  # if description is in multiple lines, delete the comment char
        return string.strip(description)

    def GetCustomText(self, data, searchString):
        #read the description from widget
        search = re.search(searchString, data)
        if (search == None):
            return None

        text = search.group(0)
        text = text[text.find(">")+1:-text[::-1].find("<")-1]    #delete the <...> </...>
        return text.strip()


    def GetAllInputs(self, data):
        #result = re.search('self.inputs *= *[[].*]', data)
        result = re.search('[ \t]+self.inputs *= *[[].*]', data)
        if not result: return []
        text = data[result.start():result.end()]
        text = text[text.index("[")+1:text.index("]")]
        text= text.replace('"',"'")
        #inputs = re.findall("\(.*?\)", text)
        inputs = re.findall("\(.*?[\"\'].*?[\"\'].*?\)", text)
        inputList = []
        for input in inputs:
            inputList.append(self.GetAllValues(input))
        return inputList

    def GetAllOutputs(self, data):
        #result = re.search('self.outputs *= *[[].*]', data)
        result = re.search('[ \t]+self.outputs *= *[[].*]', data)
        if not result: return []
        text = data[result.start():result.end()]
        text = text[text.index("[")+1:text.index("]")]
        text= text.replace('"',"'")
        outputs = re.findall("\(.*?\)", text)
        outputList = []
        for output in outputs:
            outputList.append(self.GetAllValues(output))
        return outputList

    def GetAllValues(self, text):
        text = text[1:-1]
        vals = text.split(",")
        vals[0] = vals[0][1:-1]
        for i in range(len(vals)):
            vals[i] = vals[i].strip()
        return tuple(vals)


def rebuildRegistry():
    dirs = __getDirectoryNames()
    parse = WidgetsToXML()
    widgetDir = os.path.join(dirs["canvasDir"], "../OrangeWidgets")
    parse.ParseWidgetRoot(widgetDir, dirs["outputDir"])

def readAdditionalCategories():
    dirs = __getDirectoryNames()
    addCatFile = os.path.join(dirs["outputDir"], "additionalCategories")
    if os.path.exists(addCatFile):
        return [tuple([x.strip() for x in lne.split("\t")]) for lne in open(addCatFile, "r")]
    else:
        return []

def writeAdditionalCategories(categories):
    dirs = __getDirectoryNames()
    open(os.path.join(dirs["outputDir"], "additionalCategories"), "w").write("\n".join(["\t".join(l) for l in categories]))

def addWidgetCategory(category, directory, add = True):
    if os.path.isfile(directory):
        directory = os.path.dirname(directory)
    writeAdditionalCategories([x for x in readAdditionalCategories() if x[0] != category and x[1] != directory] + (add and [(category, directory)] or []))
    rebuildRegistry()



def __getDirectoryNames():
    try:
        canvasDir = os.path.split(os.path.abspath(__file__))[0]
        orangeDir = canvasDir[:-13]
    except:
        import orange
        orangeDir = os.path.split(os.path.abspath(orange.__file__))[0]
        canvasDir = os.path.join(orangeDir, "OrangeCanvas")

    widgetDir = os.path.join(orangeDir, "OrangeWidgets")
    if not os.path.exists(widgetDir):
        print "Error. Directory %s not found. Unable to locate widgets." % widgetDir

    reportsDir = os.path.join(orangeDir, "report")
    if not os.path.exists(reportsDir):
        try: os.mkdir(reportsDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass

    picsDir = os.path.join(widgetDir, "icons")
    if not os.path.exists(picsDir):
        print "Error. Directory %s not found. Unable to locate widget icons." % picsDir

    if os.name == "nt":
        applicationDir = os.path.join(user.home, "Application Data")
        if not os.path.exists(applicationDir):
            try: os.mkdir(applicationDir)
            except: pass
        outputDir = os.path.join(applicationDir, "Orange")                  # directory for saving settings and stuff
    elif sys.platform == "darwin":
        applicationDir = os.path.join(user.home, "Library")
        applicationDir = os.path.join(applicationDir, "Application Support")
        outputDir = os.path.join(applicationDir, "Orange")
    else:
        outputDir = os.path.join(user.home, "Orange")                  # directory for saving settings and stuff
    if not os.path.exists(outputDir):
        try: os.mkdir(outputDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass
    outputDir = os.path.join(outputDir, "OrangeCanvas")
    if not os.path.exists(outputDir):
        try: os.mkdir(outputDir)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except: pass

    registryFileName = os.path.join(outputDir, "widgetregistry.xml")
    if not os.path.exists(registryFileName):
        WidgetsToXML().ParseWidgetRoot(widgetDir, outputDir)

    return dict([(name, vars()[name]) for name in ["canvasDir", "orangeDir", "widgetDir", "reportsDir", "picsDir", "outputDir", "registryFileName"]])


def addWidgetDirectories():
    if os.path.exists(widgetDir):
        for name in os.listdir(widgetDir):
            fullName = os.path.join(widgetDir, name)
            if os.path.isdir(fullName):
                sys.path.append(fullName)

    doc = xml.dom.minidom.parse(registryFileName)
    for category in doc.getElementsByTagName("category"):
        directory = category.getAttribute("directory")
        if directory:
            sys.path.append(directory)


directoryNames = __getDirectoryNames()
vars().update(directoryNames)

if __name__=="__main__":
    rebuildRegistry()
