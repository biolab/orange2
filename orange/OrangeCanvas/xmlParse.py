# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	parse widget information to a registry file (.xml) - info is then used inside orngTabs.py
#
import os
import sys
import string
import re
from xml.dom.minidom import Document

class WidgetsToXML:
    def ParseDirectory(self, widgetDirName, canvasDir):
        # create xml document
        doc = Document()
        canvas = doc.createElement("orangecanvas")
        categories = doc.createElement("widget-categories")
        doc.appendChild(canvas)
        canvas.appendChild(categories)
        
        #os.chdir(widgetDirName)
        for filename in os.listdir(widgetDirName):
            full_filename = os.path.join(widgetDirName, filename)
            if os.path.isdir(full_filename) or os.path.islink(full_filename) or not (full_filename[-2:] == "py"):
                continue

            file = open(full_filename)
            data = file.read()
            file.close()

            name        = self.GetCustomText(data, '<name>.*</name>', 6, -7)
            category    = self.GetCustomText(data, '<category>.*</category>', 10, -11)
            icon        = self.GetCustomText(data, '<icon>.*</icon>', 6, -7)
            priorityStr = self.GetCustomText(data, '<priority>.*</priority>', 10, -11)
            if priorityStr == None:
                priorityStr = "5000"

            description = self.GetDescription(data)
            inputList   = self.GetAllInputs(data)
            outputList  = self.GetAllOutputs(data)

            if (name == None):      # if the file doesn't have a name, we treat it as a non-widget file
                continue
            
            if (category == None):
                category = "Unknown"

            # create XML node for the widget
            child = categories.firstChild
            while (child != None and child.attributes.get("name").nodeValue != category):
                child= child.nextSibling
    
            if (child == None):
                child = doc.createElement("category")
                child.setAttribute("name", category)
                categories.appendChild(child)
    
            widget = doc.createElement("widget")
            widget.setAttribute("file", filename[:-3])
            widget.setAttribute("name", name)
            widget.setAttribute("in", str(inputList))
            widget.setAttribute("out", str(outputList))
            widget.setAttribute("icon", icon)
            widget.setAttribute("priority", priorityStr)
            """
            # inputlist
            ins = doc.createElement("inputs")
            insText = doc.createTextNode(inputList)
            ins.appendChild(insText)
            widget.appendChild(ins)

            # outlist
            outs = doc.createElement("outputs")
            outsText = doc.createTextNode(outputList)
            ins.appendChild(outsText)
            widget.appendChild(outs)
            """
            # description            
            if (description != ""):
                desc = doc.createElement("description")
                descText = doc.createTextNode(description)
                desc.appendChild(descText)
                widget.appendChild(desc)

            child.appendChild(widget)

        xmlText = doc.toprettyxml()
        file = open(canvasDir + "widgetregistry.xml", "wt")
        file.write(xmlText)
        file.flush()
        file.close()
        doc.unlink()

    def GetDescription(self, data):
        #read the description from widget
        search = re.search('<description>.*</description>', data, re.DOTALL)
        if (search == None):
            return ""
         
        description = search.group(0)[13:-14]    #delete the <...> </...>
        description = re.sub("#", "", description)  # if description is in multiple lines, delete the comment char
        return string.strip(description)

    def GetCustomText(self, data, searchString, index1, index2):
        #read the description from widget
        search = re.search(searchString, data)
        if (search == None):
            return None
        
        text = search.group(0)[index1:index2]    #delete the <...> </...>
        return string.strip(text)

        
    def GetAllInputs(self, data):
        result = re.search('self.inputs *= *[[].*]', data)
        if not result: return []
        text = data[result.start():result.end()]
        text = text[text.index("[")+1:text.index("]")]
        text= text.replace('"',"'")
        inputs = re.findall("\(.*?\)", text)
        inputList = []
        for input in inputs:
            inputList.append(self.GetAllValues(input))
        return inputList

    def GetAllOutputs(self, data):
        result = re.search('self.outputs *= *[[].*]', data)
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


if __name__=="__main__":
    parse = WidgetsToXML()
    canvasDir = sys.prefix + "./"
    widgetDir = sys.prefix + "../orangeWidgets/"
    parse.ParseDirectory(widgetDir, canvasDir)