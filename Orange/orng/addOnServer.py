from mod_python import apache
from mod_python import util
from mod_python.util import FieldStorage
import os
import codecs
import glob
import xml.dom.minidom
import time
import re
import widgetParser
import mimetypes
from zipfile import ZipFile
import fileutil

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

def _handlerMachine(req, args, repositoryDir):
    # Orange Canvas is calling home and needs data!
    (ok, cache) = _updateXml(repositoryDir)
    req.content_type = 'application/xml; charset=utf-8'
    if ok:
        req.sendfile(os.path.join(repositoryDir, ".cache.xml"))
        return apache.OK
    else:
        req.write(cache.toxml("UTF-8"))
        return apache.OK

#global cntr
#cntr = 0

def _logTmp(s):
    import time
    tmpf=open('/tmp/server%d.tmp' % os.getpid(), 'a')
    tmpf.write('%s %s %s\n' % (time.strftime('%a, %d %b %Y %H:%M:%S', time.gmtime()), str(os.getpid()), s))
    tmpf.close()

def _handlerContent(req, args, repositoryDir):
    # Documents, extracted from archive -- or a simple default page
    #_logTmp("Got request for %s." % req.uri)
    try:
        suburi = req.uri.split(".py/", 1)
        if len(suburi)>1:
            if "/" in suburi[1]:   # we need to extract .oao contents
                (oao, path) = suburi[1].split("/", 1)
                path = re.sub(r"[\\/]+", "/", path)
                path = re.sub(r"^/", "", path)
                try:
                    zipPath = os.path.join(repositoryDir, oao)
                    global _zipCache
                    if zipPath not in _zipCache:
                        _zipCache[zipPath] = ZipFile(open(zipPath, 'rb'))
                    pack = _zipCache[zipPath]
                    fileName = path.split("/")[-1]
                    
                    filelist = pack.namelist()
                    if path not in filelist and "icon" in args:
                        req.internal_redirect("/"+"/".join(suburi[0].split("/")[:-1])+"/Unknown.png")
                        return apache.OK
                    elif path not in filelist and "doc" in args:
                        import orange_headfoot
                        return orange_headfoot._handleStatic(req, "<head><title>Orange - Documentation not found</title></head><body>Sorry, the documentation you look for is not available.</body>")
                    if fileName=="" or path not in filelist:
                        if path[-1]=="/": path=path[:-1]
                        indexPages = ["main.htm", "main.html", "index.html", "index.htm", "default.html", "default.htm"]
                        for indexPage in indexPages:
                            if path+"/"+indexPage in filelist:
                                path = path+"/"+indexPage
                                break
                        if path not in filelist:
                            return apache.HTTP_NOT_FOUND
                        fileName = path.split("/")[-1]
                    
                    type = mimetypes.guess_type(fileName)[0]
                    if not type:
                        return apache.NO_CONTENT
                    content = pack.read(path)
                    open('/tmp/addons.tmp', 'a').write("%s: %s\n" % (path, type))
                    if type.startswith("text/html"):
                        try:
                            import orange_headfoot
                            return orange_headfoot._handleStatic(req, content)
                        except Exception, e:
                            pass
                    req.content_type = type
                    req.write(content)
                    return apache.OK
                except Exception, e:
                    return apache.INTERNAL_ERROR
            else:
                return apache.DECLINED
        else:
#            content = """<!doctype html>
#            <html>
#              <head>
#                <title>Orange Add-on Repository</title>
#              </head>
#              <body><h1>Orange Add-on Repository %s</h1>
#                <p>This is an <a href="http://orange.biolab.si">Orange</a> add-on repository. Would you like to <a href="upload.html">upload</a> new add-ons?</p>
#              </body>
#            </html>""" % req.uri
#            try: 
#                import orange_headfoot
#                orange_headfoot._handleStatic(req, content)
#            except:
#                req.content_type = 'text/html; charset=utf-8'
#                req.write(content)
#                return apache.OK
            util.redirect(req, ".")
    finally:
        pass           

def handler(req):
    repositoryDir = os.path.dirname(req.filename)
    args = _reqArgs(req)
    return (_handlerMachine if "machine" in args else _handlerContent) (req, args, repositoryDir)



_addOnsInCache = {}
_zipCache = {}
globals().update( {"_addOnsInCache": _addOnsInCache, "_zipCache": _zipCache} )

def _addOnsInRepo(repositoryDir):
    result = {}
    for addOn in glob.glob(os.path.join(repositoryDir, "*.oao")):
        if os.path.isfile(addOn):
            fileTime = os.stat(addOn).st_mtime
            result[addOn] = fileTime
    return result

def _updateXml(repositoryDir, returnCachedDoc=False):
    addOnsInRepo = _addOnsInRepo(repositoryDir)
    cacheFile = os.path.join(repositoryDir, ".cache.xml")
    global _addOnsInCache
    if (repositoryDir not in _addOnsInCache) or (addOnsInRepo != _addOnsInCache[repositoryDir]) or (not os.path.isfile(cacheFile)):
        impl = xml.dom.minidom.getDOMImplementation()
        cacheDoc = impl.createDocument(None, "OrangeAddOnRepository", None)
        xmlRepo = cacheDoc.documentElement
        for addOn in addOnsInRepo.keys():
            try:
                pack = ZipFile(addOn, 'r')
                try:
                    manifest = fileutil._zip_open(pack, 'addon.xml')
                    addOnXmlDoc = xml.dom.minidom.parse(manifest)
                    addOnXml = addOnXmlDoc.documentElement
                    addOnXml.setAttribute("filename", os.path.split(addOn)[1])
                    addOnXml.appendChild(widgetParser.widgetsXml(pack))   # Temporary: this should be done at upload.
                    addOnXml.normalize()
                    xmlRepo.appendChild(addOnXml)
                finally:
                    pack.close()
            except Exception, e:
                xmlRepo.appendChild(cacheDoc.createComment("Ignoring add-on %s: %s" % (addOn, e)))
        try:
            cacheDoc.writexml(codecs.open(cacheFile, 'w', "utf-8"), encoding="UTF-8")
            _addOnsInCache[repositoryDir] = addOnsInRepo
            return (True, cacheDoc)
        except Exception, e:
            print "Cannot write add-on cache! %s" % e
            return (False, cacheDoc)
    return (True, None if not returnCachedDoc else xml.dom.minidom.parse(open(cacheFile, 'r')))