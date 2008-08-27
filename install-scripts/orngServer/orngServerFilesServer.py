import sys
sys.path.insert(0,"./CherryPy-3.1.0")

import cherrypy
print "Loaded CherryPy version", cherrypy.__version__

import os
import shutil
import re
import hashlib
import threading
import cgi
pj = os.path.join
import datetime

basedir = pj(os.getcwd(), "data")

def noBodyProcess():
    """Sets cherrypy.request.process_request_body = False, giving
    us direct control of the file upload destination. By default
    cherrypy loads it to memory, we are directing it to disk."""
    #cherrypy.request.process_request_body = False
    print "noBodyProcess"
    print "LOGIN", cherrypy.request.login
    print "PROCESS RB", cherrypy.request.process_request_body
    if not cherrypy.request.login:
        cherrypy.request.process_request_body = False
    
cherrypy.tools.noBodyProcess = cherrypy.Tool('on_start_resource', noBodyProcess, priority=20)

class FileInfo(object):

    separ = '|||||'

    def __init__(self, fname):
        self.fname = fname
        self.set()

    def set(self, name=None, protection=None, datetime=None, title=None, tags=[]):
        self.name = name
        self.protection = protection
        self.datetime = datetime
        self.title = title
        self.tags = tags
    
    def load(self):
        f = open(self.fname, 'rb')
        cont = f.read()
        f.close()
        #print "CONT", cont
        name, protection, datetime, title, tags = cont.split("\n")
        tags = tags.split(";")
        self.set(name, int(protection), datetime, title, tags)

    def userInfo(self):
        return self.separ.join([\
            str(os.stat(self.fname + ".file").st_size), \
            str(self.datetime), \
            self.title, \
            ";".join(self.tags) \
            ])

    def save(self, fname=None):
        if not fname:
            fname = self.fname
        f = open(fname, 'wb')
        cont = '\n'.join([self.name, str(self.protection), str(self.datetime), \
            self.title, ";".join(self.tags)])
        #print "WRITING", cont
        f.write(cont)
        f.close()

    def exists(self):
        """
        If file info already exists as a file.
        """
        return self.protection != None


"""
Only one client can edit data for now!
FIXME: allow multiple clients changing different files.
Try locking a specific basename!
"""
sem = threading.BoundedSemaphore()

def lock(**kwargs):
    #print "locking"
    sem.acquire()

def unlock(**kwargs):
    sem.release()
    #print "unlocking"

cherrypy.tools.lock = cherrypy.Tool('on_start_resource', lock, priority=2)
cherrypy.tools.unlock = cherrypy.Tool('on_end_request', unlock)
"""
End of simple locking tools
"""

rec = re.compile("[^A-Za-z0-9\-\.\_]")
def safeFilename(s):
    return rec.sub("", s)

def hash(s):
    """
    May hashing function.
    """
    return hashlib.sha256(s).hexdigest()

def baseDomain(domain):
    domain = safeFilename(domain) #force safe domain
    return pj(basedir, domain)

def baseFilename(domain, filename):
    """
    Return base filename for saving on disk: composed of only
    lowercase characters. First part are first 100 alphanumeric
    characters from the filename, next its hash.
    """
    return pj(baseDomain(domain), \
        safeFilename(filename.lower())[:100] + "." + hash(filename))

def fileInfo(domain, filename):
    """
    Each file is saved in two files: its index and its data.
    It is possible that multiple files get the same name. Therefore
    enumerate them. If filename is the same also in index, then this
    is the same file.
    Returns file's FileInfo. If file does not exists, then its fileinfo
    has only attribute fname.
    """
    basename = baseFilename(domain, filename)
    candidate = 1
    filei = None
    while 1:
        try: 
            fileit = FileInfo(basename + "." + str(candidate))
            fileit.load()
            if fileit.name == filename:
                filei = fileit
                break
        except IOError:
            break # no file - file is free to be taken
        candidate += 1
    if not filei:
        filei = FileInfo(basename + "." + str(candidate))
    return filei

def userFileInfo(domain, filename, protected=False):
    fi = fileInfo(domain, filename)
    if fi.protection == 0 or protected:
        return fi.userInfo()
    else:
        return "None"

def downloadFile(domain, filename, protected=False):
    fi = fileInfo(domain, filename)
    if fi.protection == 0 or protected:
        return cherrypy.lib.static.serve_file(fi.fname + ".file", "application/x-download", "attachment", filename)
    else:
        raise cherrypy.HTTPError(500, "File not available!")

def listFiles(domain, protected=False):
    dir = baseDomain(domain)
    files = [ a for a in os.listdir(dir) if a[-1].isdigit() ]
    okfiles = []
    for file in files:
        fi = FileInfo(pj(dir, file))
        try:
            fi.load() 
            if fi.exists() and (fi.protection == 0 or protected):
                okfiles.append(fi.name)

        except:
            pass
    return "|||||".join(okfiles)

class RootServer:
    @cherrypy.expose
    def index(self):
        return """"""

class PublicServer:
    @cherrypy.expose
    def index(self):
        return """"""

    @cherrypy.expose
    def info(self, domain, filename):
        return userFileInfo(domain, filename)

    @cherrypy.expose
    def download(self, domain, filename):
        return downloadFile(domain, filename)

    @cherrypy.expose
    def list(self, domain):
        return listFiles(domain)

class SecureServer:

    @cherrypy.expose
    def index(self):
        return """"""

    @cherrypy.expose
    def info(self, domain, filename):
        return userFileInfo(domain, filename, protected=True)

    @cherrypy.expose
    def download(self, domain, filename):
        return downloadFile(domain, filename, protected=True)

    @cherrypy.expose
    def list(self, domain):
        return listFiles(domain, protected=True)

    @cherrypy.expose
    def createdomain(self, domain):
        """
        Creates domain. If creation is successful, return 0, else
        return error. 
        """
        dir = baseDomain(domain)
        os.mkdir(dir)
        return "0"

    @cherrypy.expose
    def removedomain(self, domain, force=False):
        """
        Removes domain. If successful return 0, else
        return error.
        """
        dir = baseDomain(domain)
        if not force:
            os.rmdir(dir)
        else:
            shutil.rmtree(dir)
        return '0'

    @cherrypy.expose
    @cherrypy.tools.lock()
    @cherrypy.tools.unlock()
    def remove(self, domain, filename):
        fi = fileInfo(domain, filename)
        if fi.exists(): #valid file
            os.remove(fi.fname)
            os.remove(fi.fname+".file")
            return "0"
        else:
            raise cherrypy.HTTPError(500, "File does not exists.")
            
    @cherrypy.expose
    @cherrypy.tools.lock()
    @cherrypy.tools.unlock()
    def upload(self, domain, filename, title, tags, data):

        fi = fileInfo(domain, filename)
        #print data.file.name

        fupl = open(fi.fname + ".uploading", 'wb')
        shutil.copyfileobj(data.file, fupl, 1024*8) #copy with buffer
        fupl.close()
        #print "transfer successful?" #TODO check this - MD5?

        #TODO is there any difference in those files?

        fupl = open(fi.fname + ".uploading", 'rb')
        ffin = open(fi.fname + ".file", 'wb')
        shutil.copyfileobj(fupl, ffin) #copy with buffer
        ffin.close()
        fupl.close()

        #remove file copy
        os.remove(fi.fname + ".uploading")

        datetime_now = str(datetime.datetime.utcnow())
        fi.datetime = datetime_now
        fi.name = filename
        if fi.protection == None:
            fi.protection = 1
        fi.title = title
        fi.tags = tags.split(";")

        fi.save()

    @cherrypy.expose
    @cherrypy.tools.lock()
    @cherrypy.tools.unlock()
    def protect(self, domain, filename):
        fi = fileInfo(domain, filename)
        if fi.exists():
            fi.protection = 1
            fi.save()

    @cherrypy.expose
    @cherrypy.tools.lock()
    @cherrypy.tools.unlock()
    def unprotect(self, domain, filename):
        fi = fileInfo(domain, filename)
        if fi.exists():
            fi.protection = 0
            fi.save()

"""
Tools for enforcing security measures.
Also, 
"""

def force_secure(header="Secure"):
    secure = cherrypy.request.headers.get(header, False)
    if not secure:
        raise cherrypy.HTTPError(500, "Use ssl!")

def force_insecure(header="Secure"):
    secure = cherrypy.request.headers.get(header, False)
    if secure:
        raise cherrypy.HTTPError(500, "Do not use ssl!")

cherrypy.tools.secure = cherrypy.Tool('on_start_resource', force_secure, priority=1)
cherrypy.tools.insecure = cherrypy.Tool('on_start_resource', force_insecure, priority=1)

# remove any limit on the request body size; cherrypy's default is 100MB
cherrypy.server.max_request_body_size = 0

def buildServer():
    users = {"osf": "edit123p"}

    conf = {'global': { 'log.screen': False,
                        'log.access_file': 'af.log',
                        'log.error_file': 'ef.log' },
            '/public': { 'tools.insecure.on': True},
            '/private': { 'tools.secure.on': True,
                       'tools.basic_auth.on': True,
                       'tools.basic_auth.realm': 'orngFileServer',
                       'tools.basic_auth.users': users,
                       'tools.basic_auth.encrypt': lambda x: x,
                      }}

    root = RootServer()
    root.public = PublicServer()
    root.private = SecureServer()

    return root, conf

if __name__ == '__main__':
    
    root, conf = buildServer()
    cherrypy.tree.mount(root, '/', conf)
    cherrypy.engine.start()
    cherrypy.engine.block()

