"""
Server files allows users to download files from a common
repository residing on the Orange server. It was designed to simplify
the download and updates of external data sources for Orange Genomics add-on.
Furthermore, an authenticated user can also manage the repository files with
this module.

Orange server file repository was created to store large files that do not
come with Orange installation, but may be required from the user when
running specific Orange functions. A typical example is Orange Bioinformatics
package, which relies on large data files storing genome information.
These do not come pre-installed, but are rather downloaded from the server
when needed and stored in the local repository. The module provides low-level
functionality to manage these files, and is used by Orange modules to locate
the files from the local repository and update/download them when and if needed.

Each managed file is described by domain and the file name.
Domains are like directories - a place where files are put in.

Domain should consist of less than 255 alphanumeric ASCII characters, whereas 
filenames can be arbitary long and can contain any ASCII character (including
"" ~ . \ / { }). Please, refrain from using not-ASCII character both in
domain and filenames. Files can be protected or not. Protected files can 
only be accessed by authenticated users

Local file management
=====================

The files are saved under Orange's settings directory, 
subdirectory buffer/bigfiles. Each domain is a subdirectory. 
A corresponding info 
file bearing the same name and an extension ".info" is created
with every download. Info files
contain title, tags, size and date and time of the file. 

.. autofunction:: allinfo

.. autofunction:: download

.. autofunction:: info

.. autofunction:: listdomains

.. autofunction:: listfiles

.. autofunction:: localpath

.. autofunction:: localpath_download

.. autofunction:: needs_update

.. autofunction:: remove

.. autofunction:: remove_domain

.. autofunction:: search

.. autofunction:: update


Remote file management
======================

.. autoclass:: ServerFiles
    :members:

Examples
========

.. _serverfiles1.py: code/serverfiles1.py
.. _serverfiles2.py: code/serverfiles2.py

Listing local files, files from the repository and downloading all available files from domain "demo" (`serverfiles1.py`_).

.. literalinclude:: code/serverfiles1.py

A possible output (it depends on the current repository state)::

    My files []
    Repository files ['orngServerFiles.py', 'urllib2_file.py']
    Downloading all files in domain 'test'
    Datetime for orngServerFiles.py 2008-08-20 12:25:54.624000
    Downloading orngServerFiles.py
    progress: ===============>100%  10.7 KB       47.0 KB/s    0:00 ETA
    Datetime for urllib2_file.py 2008-08-20 12:25:54.827000
    Downloading urllib2_file.py
    progress: ===============>100%  8.5 KB       37.4 KB/s    0:00 ETA
    My files after download ['urllib2_file.py', 'orngServerFiles.py']
    My domains ['KEGG', 'gene_sets', 'dictybase', 'NCBI_geneinfo', 'GO', 'miRNA', 'demo', 'Taxonomy', 'GEO']

A domain with a simple file can be built as follows (`serverfiles2.py`_). Of course,
the username and password should be valid.

.. literalinclude:: code/serverfiles2.py

A possible output::

    Uploaded.
    Non-authenticated users see: ['']
    Authenticated users see: ['titanic.tab']
    Non-authenticated users now see: ['titanic.tab']
    orngServerFiles.py file info:
    {'datetime': '2011-03-15 13:18:53.029000',
     'size': '45112',
     'tags': ['basic', 'data set'],
     'title': 'A sample .tab file'}

"""

import sys
import socket

# timeout in seconds
timeout = 120
socket.setdefaulttimeout(timeout)

import urllib2
import base64

import urllib2_file 
#switch to poster in the future
#import poster.streaminghttp as psh
#import poster.encode

from orngMisc import ConsoleProgressBar
import time, threading

import os
import shutil
import glob
import datetime
import tempfile

#defserver = "localhost:9999/"
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

def _open_file_info(fname): #no outer usage
    f = open(fname, 'rt')
    info = _parseFileInfo(f.read(), separ='\n')
    f.close()
    return info

def _save_file_info(fname, info): #no outer usage
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

def _create_path_for_file(target):
    try:
        os.makedirs(os.path.dirname(target))
    except OSError:
        pass

def _create_path(target):
    try:
        os.makedirs(target)
    except OSError:
        pass

def localpath(domain=None, filename=None):
    """Return a path for the domain in the local repository. If 
    filename is given, return a path to corresponding file."""
    import orngEnviron
    if not domain:
        return os.path.join(orngEnviron.directoryNames["bufferDir"],
            "bigfiles")
    if filename:
        return os.path.join(orngEnviron.directoryNames["bufferDir"],
            "bigfiles", domain, filename)
    else:
        return os.path.join(orngEnviron.directoryNames["bufferDir"],
            "bigfiles", domain)

class ServerFiles(object):
    """
    To work with the repository, you need to create an instance of
    ServerFiles object. To access the repository as an authenticated user, a
    username and password should be passed to the constructor. All password
    protected operations and transfers are secured by SSL; this secures
    both password and content.

    Repository files are set as protected when first uploaded: only
    authenticated users can see them. They need to be unprotected for
    public use.
    """

    def __init__(self, username=None, password=None, server=None, access_code=None):
        """
        Creates a ServerFiles instance. Pass your username and password
        to use the repository as an authenticated user. If you want to use
        your access code (as an non-authenticated user), pass it also.
        """
        if not server:
            server = defserver
        self.server = server
        self.secureroot = 'https://' + self.server + 'private/'
        self.publicroot = 'http://' + self.server + 'public/'
        self.username = username
        self.password = password
        self.access_code = access_code
        self.searchinfo = None

    def _getOpener(self):
        #commented lines are for poster 0.6
        #handlers = [psh.StreamingHTTPHandler, psh.StreamingHTTPRedirectHandler, psh.StreamingHTTPSHandler]
        #opener = urllib2.build_opener(*handlers)
        opener = urllib2.build_opener()
        return opener
 
    def upload(self, domain, filename, file, title="", tags=[]):
        """ Uploads a file "file" to the domain where it is saved with filename
        "filename". If file does not exist yet, set it as protected. Parameter
        file can be a file handle open for reading or a file name.
        """
        if isinstance(file, basestring):
            file = open(file, 'rb')

        data = {'filename': filename, 'domain': domain, 'title':title, 'tags': ";".join(tags), 'data':  file}
        return self._open('upload', data)

    def create_domain(self, domain):
        """Create a server domain."""
        return self._open('createdomain', { 'domain': domain })

    def remove_domain(self, domain, force=False):
        """Remove a domain. If force is True, domain is removed even
        if it is not empty (contains files)."""
        data = { 'domain': domain }
        if force:
            data['force'] = True
        return self._open('removedomain', data)

    def remove(self, domain, filename):
        """Remove a file from the server repository."""
        return self._open('remove', { 'domain': domain, 'filename': filename })

    def unprotect(self, domain, filename):
        """Put a file into public use."""
        return self._open('protect', { 'domain': domain, 'filename': filename, 'access_code': '0' })

    def protect(self, domain, filename, access_code="1"):
        """Hide file from non-authenticated users. If an access code (string)
        is passed, the file will be available to authenticated users and
        non-authenticated users with that access code."""
        return self._open('protect', { 'domain': domain, 'filename': filename, 'access_code': access_code })

    def protection(self, domain, filename):
        """Return file protection. Legend: "0" - public use,
        "1" - for authenticated users only, anything else
        represents a specific access code.
        """
        return self._open('protection', { 'domain': domain, 'filename': filename })
    
    def listfiles(self, domain):
        """List all files in a repository domain."""
        return _parseList(self._open('list', { 'domain': domain }))

    def listdomains(self):
        """List all domains on repository."""
        return _parseList(self._open('listdomains', {}))

    def downloadFH(self, *args, **kwargs):
        """Return open file handle of requested file from the server repository given the domain and the filename."""
        if self._authen(): return self.secdownloadFH(*args, **kwargs)
        else: return self.pubdownloadFH(*args, **kwargs)

    def download(self, domain, filename, target, callback=None):
        """
        Downloads file from the repository to a given target name. Callback
        can be a function without arguments. It will be called once for each
        downloaded percent of file: 100 times for the whole file.
        """
        _create_path_for_file(target)

        fdown = self.downloadFH(domain, filename)
        size = int(fdown.headers.getheader('content-length'))

        f = tempfile.TemporaryFile()
 
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

        fdown.close()
        f.seek(0)

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
        """
        Search for files on the repository where all substrings in a list
        are contained in at least one choosen field (tag, title, name). Return
        a list of tuples: first tuple element is the file's domain, second its
        name. As for now the search is performed locally, therefore
        information on files in repository is transfered on first call of
        this function. 
        """
        if not self.searchinfo:
            self.searchinfo = self._searchinfo()
        return _search(self.searchinfo, sstrings, **kwargs)

    def info(self, domain, filename):
        """Return a dictionary containing repository file info. 
        Keys: title, tags, size, datetime."""
        return _parseFileInfo(self._open('info', { 'domain': domain, 'filename': filename }))

    def downloadFH(self, domain, filename):
        """Return a file handle to the file that we would like to download."""
        return self._handle('download', { 'domain': domain, 'filename': filename })

    def list(self, domain):
        return _parseList(self._open('list', { 'domain': domain }))

    def listdomains(self):
        """List all domains on repository."""
        return _parseList(self._open('listdomains', {}))

    def allinfo(self, domain):
        """Go through all accessible files in a given domain and return a
        dictionary, where key is file's name and value its info.
        """
        return _parseAllFileInfo(self._open('allinfo', { 'domain': domain }))

    def index(self):
        return self._open('index', {})

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
            opener = self._getOpener()
            #the next lines work for poster 0.6.0
            #datagen, headers = poster.encode.multipart_encode(data)
            #request = urllib2.Request(root+command, datagen, headers)

            if data:
                request = urllib2.Request(root+command, data)
            else:
                request = urllib2.Request(root+command)

            #directy add authorization headers
            if self._authen():
                auth = base64.encodestring('%s:%s' % (self.username, self.password))[:-1] 
                request.add_header('Authorization', 'Basic %s' % auth ) # Add Auth header to request
            
            return opener.open(request)
        if repeat <= 0:
            return do()
        else:
            try:
                return do()
            except:
                return self._server_request(root, command, data, repeat=repeat-1)
    
    def _handle(self, command, data):
        data2 = self._addAccessCode(data)
        addr = self.publicroot
        if self._authen():
            addr = self.secureroot
        return self._server_request(addr, command, data)

    def _open(self, command, data):
        return self._handle(command, data).read()

    def _addAccessCode(self, data):
        if self.access_code != None:
            data = data.copy()
            data["access_code"] = self.access_code
        return data

def download(domain, filename, serverfiles=None, callback=None, 
    extract=True, verbose=True):
    """Downloads file from the repository to local orange installation.
    To download files as an authenticated user you should also pass an
    instance of ServerFiles class. Callback can be a function without
    arguments. It will be called once for each downloaded percent of
    file: 100 times for the whole file."""

    if not serverfiles:
        serverfiles = ServerFiles()

    info = serverfiles.info(domain, filename)
    specialtags = dict([tag.split(":") for tag in info["tags"] if tag.startswith("#") and ":" in tag])
    extract = extract and ("#uncompressed" in specialtags or "#compression" in specialtags)
    target = localpath(domain, filename)
    callback = DownloadProgress(filename, int(info["size"])) if verbose and not callback else callback    
    serverfiles.download(domain, filename, target + ".tmp" if extract else target, callback=callback)
    
    #file saved, now save info file

    _save_file_info(target + '.info', info)
    
    if extract:
        import tarfile, gzip, shutil
        if specialtags.get("#compression") == "tar.gz" and specialtags.get("#files"):
            f = tarfile.open(target + ".tmp")
            f.extractall(localpath(domain))
            shutil.copyfile(target + ".tmp", target)
        if filename.endswith(".tar.gz"):
            f = tarfile.open(target + ".tmp")
            try:
                os.mkdir(target)
            except Exception:
                pass
            f.extractall(target)
        elif specialtags.get("#compression") == "gz":
            f = gzip.open(target + ".tmp")
            shutil.copyfileobj(f, open(target, "wb"))
        f.close()
        os.remove(target + ".tmp")

    if type(callback) == DownloadProgress:
        callback.finish()

def localpath_download(domain, filename, **kwargs):
    """ 
    Return local path for the given domain and file. If file does not exist, 
    download it. Additional arguments are passed to the :obj:`download` function.
    """
    pathname = localpath(domain, filename)
    if not os.path.exists(pathname):
        download(domain, filename, **kwargs)
    return pathname

def listfiles(domain):
    """List all files from a domain in a local repository."""
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
                _open_file_info(os.path.join(dir,file))
                okfiles.append(file[:-5])
            except:
                pass

    return okfiles

def remove(domain, filename):
    """Remove a file from local repository."""
    filename = localpath(domain, filename)
    import shutil
    
    specialtags = dict([tag.split(":") for tag in info(domain, filename)["tags"] if tag.startswith("#") and ":" in tag])
    todelete = [filename, filename + ".info"] 
    if "#files" in specialtags:
        todelete.extend([os.path.join(localpath(domain), path) for path in specialtags.get("#files").split("!@")])
#    print todelete
    for path in todelete:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
        except OSError, ex:
            print "Failed to delete", path, "due to:", ex
    
def remove_domain(domain, force=False):
    """Remove a domain. If force is True, domain is removed even 
    if it is not empty (contains files)."""
    directory = localpath(domain)
    if force:
        import shutil
        shutil.rmtree(directory)
    else:
        os.rmdir(directory)

def listdomains():
    """List all file domains in the local repository."""
    dir = localpath()
    _create_path(dir)
    files = [ a for a in os.listdir(dir) ]
    ok = []
    for file in files:
        if os.path.isdir(os.path.join(dir, file)):
            ok.append(file)
    return ok

def info(domain, filename):
    """Returns info of a file in a local repository."""
    target = localpath(domain, filename)
    return _open_file_info(target + '.info')

def allinfo(domain):
    """Goes through all files in a domain on a local repository and returns a 
    dictionary, where keys are names of the files and values are their 
    information."""
    files = listfiles(domain)
    dic = {}
    for filename in files:
        target = localpath(domain, filename)
        dic[filename] = info(domain, target)
    return dic

def needs_update(domain, filename, serverfiles=None):
    """True if a file does not exist in the local repository
    or if there is a newer version on the server."""
    if serverfiles == None: serverfiles = ServerFiles()
    if filename not in listfiles(domain):
        return True
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    dt_local = datetime.datetime.strptime(
        info(domain, filename)["datetime"][:19], dt_fmt)
    dt_server = datetime.datetime.strptime(
        serverfiles.info(domain, filename)["datetime"][:19], dt_fmt)
    return dt_server > dt_local

def update(domain, filename, serverfiles=None, **kwargs):
    """Downloads the corresponding file from the server and places it in 
    the local repository, but only if the server copy of the file is newer 
    or the local copy does not exist. An optional  :class:`ServerFiles` object
    can be passed for authenticated access.
    """
    if serverfiles == None: serverfiles = ServerFiles()
    if needs_update(domain, filename, serverfiles=serverfiles):
        download(domain, filename, serverfiles=serverfiles, **kwargs)
        
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
    """Search for files in the local repository where all substrings in a list 
    are contained in at least one chosen field (tag, title, name). Return a 
    list of tuples: first tuple element is the domain of the file, second 
    its name."""
    si = _searchinfo()
    return _search(si, sstrings, **kwargs)

class DownloadProgress(ConsoleProgressBar):
    redirect = None
    lock = threading.RLock()
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
    
    class RedirectContext(object):
        def __enter__(self):
            DownloadProgress.lock.acquire()
            return DownloadProgress
        
        def __exit__(self, ex_type, value, tb):
            DownloadProgress.redirect = None
            DownloadProgress.lock.release()
            return False
        
    @classmethod
    def setredirect(cls, redirect):
        cls.redirect = staticmethod(redirect)
        return cls.RedirectContext()
    
    @classmethod
    def __enter__(cls):
        cls.lock.acquire()
        return cls
    
    @classmethod
    def __exit__(cls, exc_type, exc_value, traceback):
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
            
def _example(myusername, mypassword):

    locallist = listfiles('test')
    for l in locallist:
        print info('test', l)

    s = ServerFiles()

    print "testing connection - public"
    print "AN", s.index()

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)
    
    """
    print "Server search 1"
    import time
    t = time.time()
    print s.search(["rat"])
    print time.time() - t

    t = time.time()
    print s.search(["human", "ke"])
    print time.time() - t 
    """

    print "testing connection - private"
    print "AN", s.index()

    #create domain
    try: 
        s.create_domain("test") 
    except:
        print "Failed to create the domain"
        pass

    files = s.listfiles('test')
    print "Files in test", files

    print "uploading"

    #upload this file - save it by a different name
    s.upload('test', 'osf-test.py', 'serverfiles.py', title="NT", tags=["fkdl","fdl"])
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

    #login as an authenticated user
    s = ServerFiles(username=myusername, password=mypassword)

    print s.listdomains()

    s.remove('test', 'osf-test.py')

    s = ServerFiles()

    print s.listdomains()


if __name__ == '__main__':
    _example(sys.argv[1], sys.argv[2])
