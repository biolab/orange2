"""
orngServerFiles is a module which enables users to simply access files in
a repository. 

Each file is specified with by two parameters: domain and filename. A domain
is something like a filesystem directory -- a container holding files.

Domain should consist of less than 255 alphanumeric ASCII characters, whereas 
filenames can be arbitary long and can contain any ASCII character (including
" ~ . \ / { }). Please, refrain from using not-ASCII character both in
domain and filenames. Files can be protected or not. Protected file names can 
only be accessed by authenticated users!

orngServerFiles can be used by creating a ServerFiles object. Username
and password need to be passed by object initialization. All password
protected operations and transfers are secured by SSL: this secures both
password and content. Creating SSL connection on Windows platforms are
not tested yes. Maybe they will require additional modules (Ales, please
report).

An un-authenticated user can list files in a domain ("list"), download 
individual file ("download") or get individual file information ("info") --
bytesize and datetime. Datetimes can be compared as strings.

An authenticated user can create and remove domains ("create_domain", "remove 
domain"), upload files ("upload"), delete them ("remove") and manage their 
protection ("protect", "unprotect"). Note, that newly uploaded files,
which name did not exists in domain before, are protected. They can be made  
public by unprotecting them.

USAGE EXAMPLE

def example(myusername, mypassword):

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)
    
    #create domain
    try: 
        s.create_domain("test") 
    except: 
        pass

    #upload this file - save it by a different name
    s.upload('test', 'osf-/test/.py', open("orngServerFiles.py", 'rb'))

    #make it public
    s.unprotect('test', 'osf-/test/.py')

    #login anonymously
    s = ServerFiles()

    #list files in the domain "test"
    files = s.list('test')
    print files

    for f in files:
        size, datetime = s.info('test', f) 
        print "---", f, "---", "size", size, "datetime", datetime
        print s.download('test', f).read()[:100] #show first 100 characters
        print "---"

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)

    print "removing"
    s.remove('test', 'osf-/test/.py')

SERVER

Files are stored in a filesystem. Each domain is a filesystem directory in
which files are stored. Each saved file also has a corresponding information
file.

Current performace limitation: only one concurrent upload. This can be overcome
with smarter locking. 

Todo: checksum after transfer.
"""

import sys
import socket

# timeout in seconds
timeout = 120
socket.setdefaulttimeout(timeout)

import urllib2_file
import urllib2

import os
import shutil

#defserver = "localhost/"
defserver = "asterix.fri.uni-lj.si/orngServerFiles/"

def _parseFileInfo(fir, separ="|||||"):
    """
    Parses file info from server.
    """
    try:
        l= fir.split(separ)
        fi = {}
        fi["size"] = l[0]
        fi["datetime"] = l[1]
        fi["title"] = l[2]
        fi["tags"] = l[3].split(";")
        return fi
    except:
        return None

def openFileInfo(fname):
    f = open(fname, 'rt')
    info = _parseFileInfo(f.read(), separ='\n')
    f.close()
    return info

def saveFileInfo(fname, info):
    f = open(fname, 'wt')
    f.write('\n'.join([info['size'], info['datetime'], info['title'], ';'.join(info['tags'])]))
    f.close()

def _parseList(fl):
    return fl.split("|||||")

def createPathForFile(target):
    try:
        os.makedirs(os.path.dirname(target))
    except OSError:
        pass

def localName(domain, filename):
    import orngRegistry
    return os.path.join(orngRegistry.directoryNames["bufferDir"], "bigfiles", domain, filename)

class ServerFiles(object):

    def __init__(self, server=None, username=None, password=None):
        if not server:
            server = defserver
        self.server = server
        self.secureroot = 'https://' + self.server + 'private/'
        self.publicroot = 'http://' + self.server + 'public/'
        self.username = username
        self.password = password

    def installOpener(self):
        #import time; t = time.time()
        passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
        passman.add_password("orngFileServer", self.secureroot, str(self.username), str(self.password))
        authhandler = urllib2.HTTPBasicAuthHandler(passman)
        opener = urllib2.build_opener(authhandler)
        urllib2.install_opener(opener)
        #print "TIME", time.time() - t
 
    def upload(self, domain, filename, file, title="", tags=[]):
        """
        Uploads file to the server. File can be an open file or a filename.
        """
        if isinstance(file, basestring):
            file = open(file, 'rb')

        data = {'filename': filename, 'domain': domain, 'title':title, 'tags': ";".join(tags), 'data':  file}
        return self._secopen('upload', data)

    def create_domain(self, domain):
        return self._secopen('createdomain', { 'domain': domain })

    def remove_domain(self, domain, force=False):
        data = { 'domain': domain }
        if force:
            data['force'] = True
        return self._secopen('removedomain', data)

    def remove(self, domain, filename):
        return self._secopen('remove', { 'domain': domain, 'filename': filename })

    def unprotect(self, domain, filename):
        return self._secopen('unprotect', { 'domain': domain, 'filename': filename })

    def protect(self, domain, filename):
        return self._secopen('protect', { 'domain': domain, 'filename': filename })

    def list(self, *args, **kwargs):
        if self._authen(): return self.seclist(*args, **kwargs)
        else: return self.publist(*args, **kwargs)

    def downloadFH(self, *args, **kwargs):
        """
        Returns open file handle of requested file.
        Parameters: domain and filename.
        """
        if self._authen(): return self.secdownloadFH(*args, **kwargs)
        else: return self.pubdownloadFH(*args, **kwargs)

    def download(self, domain, filename, target=None):
        """
        Downloads a file into target name. If target is not present,
        file is downloaded into [bufferDir]/bigfiles/domain/filename
        """
        if not target:
            target = localName(domain, filename)

        createPathForFile(target)

        fdown = self.downloadFH(domain, filename)
        f = open(target + '.tmp', 'wb')
        shutil.copyfileobj(fdown, f, 1024*8)
        fdown.close()
        f.close()
        os.rename(target + '.tmp', target)

        #file saved, now save info file
        info = self.info(domain, filename)
        saveFileInfo(target + '.info', info)

    def info(self, *args, **kwargs):
        if self._authen(): return self.secinfo(*args, **kwargs)
        else: return self.pubinfo(*args, **kwargs)

    def pubinfo(self, domain, filename):
        return _parseFileInfo(self._pubopen('info', { 'domain': domain, 'filename': filename }))

    def pubdownloadFH(self, domain, filename):
        return self._pubhandle('download', { 'domain': domain, 'filename': filename })

    def publist(self, domain):
        return _parseList(self._pubopen('list', { 'domain': domain }))

    def secinfo(self, domain, filename):
        return _parseFileInfo(self._secopen('info', { 'domain': domain, 'filename': filename }))

    def secdownloadFH(self, domain, filename):
        return self._sechandle('download', { 'domain': domain, 'filename': filename })

    def seclist(self, domain):
        return _parseList(self._secopen('list', { 'domain': domain }))
 
    def _authen(self):
        """
        Did the user choose authentication?
        """
        if self.username and self.password:
            return True
        else:
            return False

    def _sechandle(self, command, data):
        self.installOpener()
        return urllib2.urlopen(self.secureroot + command, data)
 
    def _pubhandle(self, command, data):
        self.installOpener()
        return urllib2.urlopen(self.publicroot + command, data)

    def _secopen(self, command, data):
        return self._sechandle(command, data).read()

    def _pubopen(self, command, data):
        return self._pubhandle(command, data).read()

def example(myusername, mypassword):

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)
    
    #create domain
    try: 
        s.create_domain("test") 
    except: 
        pass

    #upload this file - save it by a different name
    s.upload('test', 'osf-test.py', open("orngServerFiles.py", 'rb'), title="NT", tags=["fkdl","fdl"])

    #make it public
    s.unprotect('test', 'osf-test.py')

    #login anonymously
    s = ServerFiles()

    #list files in the domain "test"
    files = s.list('test')
    print "ALL FILES:", files

    for f in files:
        fi = s.info('test', f) 
        print "--------------------------------------", f
        print "INFO", fi
        print s.downloadFH('test', f).read()[:100] #show first 100 characters
        print "--------------------------------------"
        s.download('test', f)

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)

    s.remove('test', 'osf-test.py')

if __name__ == '__main__':

    example(sys.argv[1], sys.argv[2])
