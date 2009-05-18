"""
orngServerFiles is a module which enables users to simply access files in
a repository. 

Each file is specified with by two parameters: domain and filename. A domain
is something like a filesystem directory -- a container holding files.

Domain should consist of less than 255 alphanumeric ASCII characters, whereas 
filenames can be arbitary long and can contain any ASCII character (including
"" ~ . \ / { }). Please, refrain from using not-ASCII character both in
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
import glob
import datetime

#defserver = "localhost/"
defserver = "asterix.fri.uni-lj.si/orngServerFiles/"

def _parseFileInfo(fir, separ="|||||"):
    """
    Parses file info from server.
    """
    l= fir.split(separ)
    fi = {}
    fi["size"] = l[0]
    fi["datetime"] = l[1]
    fi["title"] = l[2]
    fi["tags"] = l[3].split(";")
    return fi

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

def _parseAllFileInfo(afi):
    separf = "[[[[["
    separn = "====="
    fis = afi.split(separf)
    out = []
    for entry in fis:
        if entry != "":
            name, info = entry.split(separn)
            out.append((name, _parseFileInfo(info)))

    return dict(out)

def createPathForFile(target):
    try:
        os.makedirs(os.path.dirname(target))
    except OSError:
        pass

def createPath(target):
    try:
        os.makedirs(target)
    except OSError:
        pass

def localpath(domain=None, filename=None):
    """Return a path for the file in the local repository."""
    import orngEnviron
    if not domain:
        return os.path.join(orngEnviron.directoryNames["bufferDir"], "bigfiles")
    if filename:
        return os.path.join(orngEnviron.directoryNames["bufferDir"], "bigfiles", domain, filename)
    else:
        return os.path.join(orngEnviron.directoryNames["bufferDir"], "bigfiles", domain)

class ServerFiles(object):

    def __init__(self, username=None, password=None, server=None, access_code=None):
        if not server:
            server = defserver
        self.server = server
        self.secureroot = 'https://' + self.server + 'private/'
        self.publicroot = 'http://' + self.server + 'public/'
        self.username = username
        self.password = password
        self.access_code = access_code
        self.searchinfo = None

    def installOpener(self):
        #import time; t = time.time()
        passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
        passman.add_password("orngFileServer", self.secureroot, str(self.username), str(self.password))
        authhandler = urllib2.HTTPBasicAuthHandler(passman)
        opener = urllib2.build_opener(authhandler)
        urllib2.install_opener(opener)
        #print "TIME", time.time() - t
 
    def upload(self, domain, filename, file, title="", tags=[]):
        """Upload the file to the server.
        File can be an open file or a filename."""
        if isinstance(file, basestring):
            file = open(file, 'rb')

        data = {'filename': filename, 'domain': domain, 'title':title, 'tags': ";".join(tags), 'data':  file}
        return self._secopen('upload', data)

    def create_domain(self, domain):
        """Create the domain in the server repository."""
        return self._secopen('createdomain', { 'domain': domain })

    def remove_domain(self, domain, force=False):
        """Remove the domain from the server repository. If force=True
        remove if the domain is not empty (includes files)."""
        data = { 'domain': domain }
        if force:
            data['force'] = True
        return self._secopen('removedomain', data)

    def remove(self, domain, filename):
        """Remove a file from the server repository."""
        return self._secopen('remove', { 'domain': domain, 'filename': filename })

    def unprotect(self, domain, filename):
        """Unprotect a file in the server repository."""
        return self._secopen('protect', { 'domain': domain, 'filename': filename, 'access_code': '0' })

    def protect(self, domain, filename, access_code="1"):
        """Protect a file in the server repository."""
        return self._secopen('protect', { 'domain': domain, 'filename': filename, 'access_code': access_code })

    def protection(self, domain, filename):
        """Return 1 if the file in the server is protected, else return 0."""
        return self._secopen('protection', { 'domain': domain, 'filename': filename })

    def listfiles(self, *args, **kwargs):
        """List all files from the server repositotory that reside in specific domain."""
        if self._authen(): return self.seclist(*args, **kwargs)
        else: return self.publist(*args, **kwargs)

    def listdomains(self, *args, **kwargs):
        """List the domains from the server repository."""
        if self._authen(): return self.seclistdomains(*args, **kwargs)
        else: return self.publistdomains(*args, **kwargs)

    def downloadFH(self, *args, **kwargs):
        """Return open file handle of requested file from the server repository given the domain and the filename."""
        if self._authen(): return self.secdownloadFH(*args, **kwargs)
        else: return self.pubdownloadFH(*args, **kwargs)

    def download(self, domain, filename, target, callback=None):
        """Download a file into target name. If target is not present,
        file is downloaded into [bufferDir]/bigfiles/domain/filename."""
        createPathForFile(target)

        fdown = self.downloadFH(domain, filename)
        size = int(fdown.headers.getheader('content-length'))

        f = os.tmpfile() #open(target + '.tmp', 'wb')
 
        chunksize = 1024*8
        lastchunkreport= 0.0001

        readb = 0
        while 1:
            buf = fdown.read(chunksize)
            readb += len(buf)

            while float(readb)/size > lastchunkreport+0.01:
                #print float(readb)/size, lastchunkreport + 0.01, float(readb)/size - lastchunkreport 
                lastchunkreport += 0.01
                if callback:
                    callback()
            if not buf:
                break
            f.write(buf)

        #retired to enable tracin
        #shutil.copyfileobj(fdown, f, 1024*8) 

        fdown.close()
##        f.close()
        f.seek(0)

##        os.rename(target + '.tmp', target)
        shutil.copyfileobj(f, open(target, "wb"))

        if callback:
            callback()

    def _searchinfo(self):
        domains = self.listdomains()
        infos = {}
        for dom in domains:
            dominfo = self.allinfo(dom)
            for a,b in dominfo.items():
                infos[(dom, a)] = b
        return infos

    def search(self, sstrings, **kwargs):
        if not self.searchinfo:
            self.searchinfo = self._searchinfo()
        return _search(self.searchinfo, sstrings, **kwargs)

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

    def seclistdomains(self):
        return _parseList(self._secopen('listdomains', {}))

    def publistdomains(self):
        return _parseList(self._pubopen('listdomains', {}))

    def allinfo(self, *args, **kwargs):
        if self._authen(): return self.secallinfo(*args, **kwargs)
        else: return self.puballinfo(*args, **kwargs)

    def puballinfo(self, domain):
        return _parseAllFileInfo(self._pubopen('allinfo', { 'domain': domain }))

    def secallinfo(self, domain):
        return _parseAllFileInfo(self._secopen('allinfo', { 'domain': domain }))

    def _authen(self):
        """
        Did the user choose authentication?
        """
        if self.username and self.password:
            return True
        else:
            return False


    def _server_request(self, root, command, data, repeat=2):
        def do():
            self.installOpener()
            if data:
                return urllib2.urlopen(root + command, data)
            else:
                return urllib2.urlopen(root + command)

        if repeat <= 0:
            do()
        else:
            try:
                return do()
            except:
                return self._server_request(root, command, data, repeat=repeat-1)

    def _sechandle(self, command, data):
        return self._server_request(self.secureroot, command, data)
 
    def _pubhandle(self, command, data):
        data2 = self.addAccessCode(data)
        return self._server_request(self.publicroot, command, data2)

    def _secopen(self, command, data):
        return self._sechandle(command, data).read()

    def addAccessCode(self, data):
        if self.access_code != None:
            data = data.copy()
            data["access_code"] = self.access_code
        return data

    def _pubopen(self, command, data):
        return self._pubhandle(command, data).read()


def download(domain, filename, serverfiles=None, callback=None, extract=True, verbose=True):
    """Download a file from a server placing it in a local repository."""
    if not serverfiles:
        serverfiles = ServerFiles()

    info = serverfiles.info(domain, filename)
    extract = extract and any(tag.startswith("#uncompressed:") for tag in info["tags"])
    target = localpath(domain, filename)
    callback = DownloadProgress(filename, int(info["size"])) if verbose and not callback else callback    
    serverfiles.download(domain, filename, target + "tmp" if extract else target, callback=callback)
    
    #file saved, now save info file

    saveFileInfo(target + '.info', info)
    
    if extract:
        import tarfile
        f = tarfile.open(target + "tmp")
        try:
            os.mkdir(target)
        except Exception:
            pass
        f.extractall(target)
        f.close()
        os.remove(target + "tmp")

    if type(callback) == DownloadProgress:
        callback.finish()        

def localpath_download(domain, filename, **kwargs):
    """ Returns a location of the given file. If file is not on available yet,
    download it. """
    pathname = localpath(domain, filename)
    if not os.path.exists(pathname):
        download(domain, filename, **kwargs)
    return pathname

def listfiles(domain):
    """Return a list of filenames in a given domain on local Orange
    installation with a valid info file: useful ones."""
    dir = localpath(domain)
    try:
        files = [a for a in os.listdir(dir) if a[-5:] == '.info' ]
    except:
        files = []
    okfiles = []

    for file in files:
        #if file to exists without info
        if os.path.exists(os.path.join(dir,file[:-5])):
            #check info format - needs to be valid
            try:
                openFileInfo(os.path.join(dir,file))
                okfiles.append(file[:-5])
            except:
                pass

    return okfiles

def remove(domain, filename):
    """Remove a file from a local repository."""
    filename = localpath(domain, filename)
    os.remove(filename)
    os.remove(filename + ".info")
    
def remove_domain(domain, force=False):
    """Remove a domain (directory) from the local repository."""
    directory = localpath(domain)
    if force:
        import shutil
        shutil.rmtree(directory)
    else:
        os.rmdir(directory)

def listdomains():
    """Return a list of domains from a local repository."""
    dir = localpath()
    createPath(dir)
    files = [ a for a in os.listdir(dir) ]
    ok = []
    for file in files:
        if os.path.isdir(os.path.join(dir, file)):
            ok.append(file)
    return ok

def info(domain, filename):
    """Returns info of a file in a local repository."""
    target = localpath(domain, filename)
    return openFileInfo(target + '.info')

def allinfo(domain):
    """Returns info of all files in a specific domain in a local reposiotory."""
    files = listfiles(domain)
    dic = {}
    for filename in files:
        target = localpath(domain, filename)
        dic[filename] = info(domain, target)
    return dic

def needs_update(domain, filename, access_code=None):
    """Returns true if a file does not exist in the local repository
    or if there is a newer version on the server."""
    if filename not in listfiles(domain):
        return True
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    dt_local = datetime.datetime.strptime(
        info(domain, filename)["datetime"][:19], dt_fmt)
    server = ServerFiles(access_code=access_code)
    dt_server = datetime.datetime.strptime(
        server.info(domain, filename)["datetime"][:19], dt_fmt)
    return dt_server > dt_local

def update(domain, filename, access_code=None, verbose=True):
    """Downloads a file from a server placing it in a local repository
    if a file on a server is newer or its local version does not exist."""
    if needs_update(domain, filename, access_code=access_code):
        if not access_code:
            download(domain, filename, verbose=verbose)
        else:
            server = orngServerFiles.ServerFiles(access_code=access_code)
            download(domain, filename, serverfiles=server)
        
def _searchinfo():
    domains = listdomains()
    infos = {}
    for dom in domains:
        dominfo = allinfo(dom)
        for a,b in dominfo.items():
            infos[(dom, a)] = b
    return infos

def _search(si, sstrings, caseSensitive=False, inTag=True, inTitle=True, inName=True):
    """
    sstrings contain a list of search strings
    """
    found = []

    for (dom,fn),info in si.items():
        target = ""
        if inTag: target += " ".join(info['tags'])
        if inTitle: target += info['title']
        if inName: target += fn
        if not caseSensitive: target = target.lower()

        match = True
        for s in sstrings:
            if not caseSensitive:
                s = s.lower()
            if s not in target:
                match= False
                break
                
        if match:
            found.append((dom,fn))    
        
    return found

def search(sstrings, **kwargs):
    si = _searchinfo()
    return _search(si, sstrings, **kwargs)

from orngMisc import ConsoleProgressBar
import time, threading

class DownloadProgress(ConsoleProgressBar):
    redirect = None
    lock = threading.Lock()
    def sizeof_fmt(num):
        for x in ['bytes','KB','MB','GB','TB']:
            if num < 1024.0:
                return "%3.1f %s" % (num, x) if x <> 'bytes' else "%1.0f %s" % (num, x)
            num /= 1024.0
            
    def __init__(self, filename, size):
        print "Downloading", filename
        ConsoleProgressBar.__init__(self, "progress:", 20)
        self.size = size
        self.starttime = time.time()
        self.speed = 0.0

    def sizeof_fmt(self, num):
        for x in ['bytes','KB','MB','GB','TB']:
            if num < 1024.0:
                return "%3.1f %s" % (num, x) if x <> 'bytes' else "%1.0f %s" % (num, x)
            num /= 1024.0

    def getstring(self):
        speed = int(self.state * self.size / 100.0 / (time.time() - self.starttime))
        eta = (100 - self.state) * self.size / 100.0 / speed
        return ConsoleProgressBar.getstring(self) + "  %s  %12s/s  %3i:%02i ETA" % (self.sizeof_fmt(self.size), self.sizeof_fmt(speed), eta/60, eta%60)
        
    def __call__(self, *args, **kwargs):
        ret = ConsoleProgressBar.__call__(self, *args, **kwargs)
        if self.redirect:
            self.redirect(self.state)
        return ret
    
    @classmethod
    def setredirect(cls, redirect):
        cls.redirect = redirect
        return cls
    
    @classmethod
    def __enter__(cls):
        cls.lock.acquire()
        return cls
    
    @classmethod
    def __exit__(cls, exc_type, exc_value, traceback):
        cls.redirect = None
        cls.lock.release()
        return False

def consoleupdate(domains=None, searchstr="essential"):
    domains = domains or listdomains()
    sf = ServerFiles()
    info = dict((d, sf.allinfo(d)) for d in domains)
    def searchmenu():
        def printmenu():
            print "\tSearch tags:", search
            print "\t1. Add tag."
            print "\t2. Clear tags."
            print "\t0. Return to main menu."
            return raw_input("\tSelect option:")
        search = searchstr
        while True:
            response = printmenu().strip()
            if response == "1":
                search += " " + raw_input("\tType new tag/tags:")
            elif response == "2":
                search = ""
            elif response == "0":
                break
            else:
                print "\tUnknown option!"
        return search

    def filemenu(searchstr=""):
        files = [None]
        for i, (dom, file) in enumerate(sf.search(searchstr.split())):
            print "\t%i." % (i + 1), info[dom][file]["title"]
            files.append((dom, file))
        print "\t0. Return to main menu."
        print "\tAction: d-download (e.g. 'd 1' downloads first file)"
        while True:
            response = raw_input("\tAction:").strip()
            if response == "0":
                break
            try:
                action, num = response.split(None, 1)
                num = int(num)
            except Exception, ex:
                print "Unknown option!"
                continue
            try:
                if action.lower() == "d":
                    download(*(files[num]))
                    print "\tSuccsessfully downloaded", files[num][-1]
            except Exception, ex:
                print "Error occured!", ex

    def printmenu():
        print "Update database main menu:"
        print "1. Enter search tags (refine search)."
        print "2. Print matching available files."
        print "3. Print all available files."
        print "4. Update all local files."
        print "0. Exit."
        return raw_input("Select option:")
    
    while True:
        try:
            response = printmenu().strip()
            if response == "1":
                searchstr = searchmenu()
            elif response == "2":
                filemenu(searchstr)
            elif response == "3":
                filemenu("")
            elif response == "4":
                update_local_files()
            elif response == "0":
                break
            else:
                print "Unknown option!"
        except Exception, ex:
            print "Error occured:", ex

def update_local_files(verbose=True):
    sf = ServerFiles()
    for domain, filename in search(""):
        uptodate = sf.info(domain, filename)["datetime"] <= info(domain, filename)["datetime"]
        if not uptodate:
            download(domain, filename, sf)
        if verbose:
            print filename, "Ok" if uptodate else "Updated"

def update_by_tags(tags=["essential"], domains=[], verbose=True):
    sf = ServerFiles()
    for domain, filename in sf.search(tags + domains, inTitle=False, inName=False):
        if domains and domain not in domain:
            continue
        if os.path.exists(localpath(domain, filename)+".info"):
            uptodate = sf.info(domain, filename)["datetime"] <= info(domain, filename)["datetime"]
        else:
            uptodate = False
        if not uptodate:
            download(domain, filename, sf)
        if verbose:
            print filename, "Ok" if uptodate else "Updated"
            
def example(myusername, mypassword):

    locallist = listfiles('test')
    for l in locallist:
        print info('test', l)

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)
    
    print "Server search 1"
    import time
    t = time.time()
    print s.search(["rat"])
    print time.time() - t

    t = time.time()
    print s.search(["human", "ke"])
    print time.time() - t 

    #create domain
    try: 
        s.create_domain("test") 
    except:
        print "Failed to create the domain"
        pass

    #upload this file - save it by a different name
    s.upload('test', 'osf-test.py', 'orngServerFiles.py', title="NT", tags=["fkdl","fdl"])
    #make it public

    s.unprotect('test', 'osf-test.py')

    #login anonymously
    s = ServerFiles()

    #list files in the domain "test"
    files = s.listfiles('test')
    print "ALL FILES:", files

    for f in files:
        fi = s.info('test', f) 
        print "--------------------------------------", f
        print "INFO", fi
        print s.downloadFH('test', f).read()[:100] #show first 100 characters
        print "--------------------------------------"
        s.download('test', f, 'a.mkv')

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)

    print s.listdomains()

    s.remove('test', 'osf-test.py')

    s = ServerFiles()
    print s.allinfo("demo2")

    print s.listdomains()

if __name__ == '__main__':
    example(sys.argv[1], sys.argv[2])
