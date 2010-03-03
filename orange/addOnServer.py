from mod_python import apache
from mod_python import util
from mod_python.util import FieldStorage
import os
import codecs
import glob
import xml.dom.minidom
import time
from zipfile import ZipFile

def _webMakeBody(args):
    return """<p>%s</p>""" % str(addOnsInCache)

def _reqArgs(req):
    """
    Returns a dictionary of name->value pairs for request's arguments and fields.
    """
    result = {}
    data = FieldStorage(req, keep_blank_values=1)
    for fieldName in data.keys():
        fieldData = data[fieldName] 
        if type(fieldData) is list:  # on POST requests if there is name collision between a parameter and a field
            result[fieldName] = fieldData[-1].value
        else:
            result[fieldName] = fieldData.value
    return result

def handler(req):
    args = _reqArgs(req)
    if ("machine" in args) and (args["machine"] != "0"):
        # Orange Canvas is calling home and needs data!
        req.content_type = 'application/xml'
        _updateXml()
        util.redirect(req, ".cache.xml")
    else:
        # Web page: only a test for now.
        req.content_type = 'text/html'
        req.write("""<!doctype html>
<html>
  <head>
    <title>Orange Add-ons</title>
  </head>
  <body>%s</body>
</html>""" % _webMakeBody(args))
        return apache.OK

addOnsInCache = {}
repositoryDir = os.path.dirname(__file__)  
repositoryXmlCache = os.path.join(repositoryDir, ".cache.xml")
globals().update( {"addOnsInCache": addOnsInCache, "repositoryDir": repositoryDir, "repositoryXmlCache": repositoryXmlCache} )

def _addOnsInRepo():
    result = {}
    for addOn in glob.glob(os.path.join(repositoryDir, "*.oao")):
        fileTime = os.stat(addOn).st_mtime
        result[addOn] = fileTime
    return result

def _updateXml():
    addOnsInRepo = _addOnsInRepo()
    global addOnsInCache 
    if (addOnsInRepo != addOnsInCache) or (not os.path.isfile(repositoryXmlCache)):
        impl = xml.dom.minidom.getDOMImplementation()
        cacheDoc = impl.createDocument(None, "OrangeAddOnRepository", None)
        xmlRepo = cacheDoc.documentElement
        for addOn in addOnsInRepo.keys():
            try:
                pack = ZipFile(addOn, 'r')
                try:
                    manifest = pack.open('addon.xml', 'r')
                    addOnXmlDoc = xml.dom.minidom.parse(manifest)
                    addOnXml = addOnXmlDoc.documentElement                    
                    addOnXml.setAttribute("filename", os.path.split(addOn)[1])
                    addOnXml.normalize()
                    xmlRepo.appendChild(addOnXml)
                finally:
                    pack.close()
            except Exception, e:
                print "Ignoring add-on %s: %s" % (addOn, e)
        cacheDoc.writexml(codecs.open(repositoryXmlCache, 'w', "utf-8"), encoding="UTF-8")
        addOnsInCache = addOnsInRepo


_updateXml()