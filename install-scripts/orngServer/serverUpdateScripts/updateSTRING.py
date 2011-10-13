##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiPPI, orngServerFiles
import os, sys, shutil, urllib2, tarfile
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

serverFiles = orngServerFiles.ServerFiles(username, password)

filename = orngServerFiles.localpath("PPI", "string-protein.sqlite")
if os.path.exists(filename):
    os.remove(filename)
    
import obiPPI
obiPPI.STRING.download_data("v9.0")

gzfile = gzip.GzipFile(filename + ".gz", "wb")
shutil.copyfileobj(open(fileaname, "rb"), gzfile)

serverFiles.upload("PPI", "string-protein.sqlite", filename + ".gz", "STRING Protein interactions",
                   tags=["protein interaction", "STRING", "#compression:gz", "#version:%s" % obiPPI.STRING.VERSION]
                   )
serverFiles.unprotect("PPI", "string-protein.sqlite")