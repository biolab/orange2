import xml.dom.minidom
import re, os
from fileutil import xmlTextOf, createTextElement, _zipOpen

re_inputs = re.compile(r'[ \t]+self.inputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)
re_outputs = re.compile(r'[ \t]+self.outputs\s*=\s*(?P<signals>\[[^]]*\])', re.DOTALL)
re_tuple = re.compile(r"\(([^)]+)\)")

def _getSignalList(regex, data):
    inmo = regex.search(data)
    if inmo:
        return str([tuple([y[0] in "'\"" and y[1:-1] or str(y) for y in (x.strip() for x in ttext.group(1).split(","))])
               for ttext in re_tuple.finditer(inmo.group("signals"))])
    else:
        return "[]"

class WidgetMetaData:
    xmlAttrs = ["name", "description", "contact", "category", "icon", "hasDoc"]
    
    def __init__(self, data, defaultCategory="Prototypes", enforceDefaultCategory=False, filename=None, hasDoc="0"):  # data can either be a string with module (.py file) contents or an xml.dom.Element
        if not filename and data.__class__ is xml.dom.minidom.Element:  # XML (as returned by toXml())
            self.name=""
            self.icon=""
            self.priority=""
            self.description=""
            self.category=""
            self.tags=""
            self.filename=None
            self.hasDoc = hasDoc
            for attr in self.xmlAttrs:
                nodes = data.getElementsByTagName(attr)
                if nodes:
                    setattr(self, attr, xmlTextOf(nodes[0]))
            if data.hasAttribute("filename"):
                self.filename = data.getAttribute("filename")
        else:   # python module
            setattr(self, "filename", filename)
            for attr, deflt in (("name", None), ("contact", "") , ("icon", "icons/Unknown.png"), ("priority", "5000"), ("description", ""), ("category", defaultCategory), ("tags", "")):
                istart, iend = data.find("<"+attr+">"), data.find("</"+attr+">")
                setattr(self, attr, istart >= 0 and iend >= 0 and data[istart+2+len(attr):iend].strip() or deflt)
            if enforceDefaultCategory:
                self.category = defaultCategory
            if not self.name:
                raise Exception("Not an Orange widget module.")
            self.hasDoc = hasDoc
            self.inputList = _getSignalList(re_inputs, data)
            self.outputList = _getSignalList(re_outputs, data)

    def clone(self):
        return WidgetMetaData(self.toXml())
    
    def toXml(self):
        widgetTag = xml.dom.minidom.Element("widget")
        for attr in self.xmlAttrs:
            if hasattr(self, attr):
                widgetTag.appendChild(createTextElement(attr, getattr(self, attr)))
        if "filename" in self.__dict__ and self.filename:
            widgetTag.setAttribute("filename", self.filename)
        return widgetTag

_widgetModuleName = re.compile(r"widgets/(prototypes/|)[^/]*\.py$")
def widgetsXml(oaoZip):
    files = oaoZip.namelist()
    widgetsTag = xml.dom.minidom.Element("widgets")
    for file in files:
        if re.match(_widgetModuleName, file):
            try:
                filename = file[8:]
                meta = WidgetMetaData(_zipOpen(oaoZip, file).read(), "None", filename=filename, hasDoc="1" if ("doc/widgets/%s.htm" % (os.path.splitext(filename)[0][2:]) in files) else "0")
            except:
                continue   # Probably not an Orange widget module
            widgetsTag.appendChild(meta.toXml())
    return widgetsTag
