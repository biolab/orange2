##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiPPI, orngServerFiles
import os, sys, shutil, urllib2, gzip
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

serverFiles = orngServerFiles.ServerFiles(username, password)

import obiPPI

filename = orngServerFiles.localpath("PPI", obiPPI.STRING.FILENAME)

if os.path.exists(filename):
    os.remove(filename)

obiPPI.STRING.download_data("v9.0")

gzfile = gzip.GzipFile(filename + ".gz", "wb")
shutil.copyfileobj(open(filename, "rb"), gzfile)

serverFiles.upload("PPI", obiPPI.STRING.FILENAME, filename + ".gz", 
                   "STRING Protein interactions (Creative Commons Attribution 3.0 License)",
                   tags=["protein interaction", "STRING", 
                         "#compression:gz", "#version:%s" % obiPPI.STRING.VERSION]
                   )
serverFiles.unprotect("PPI", obiPPI.STRING.FILENAME)

# The second part
filename = orngServerFiles.localpath("PPI", obiPPI.STRINGDetailed.FILENAME_DETAILED)

if os.path.exists(filename):
    os.remove(filename)

obiPPI.STRINGDetailed.download_data("v9.0")

gzfile = gzip.GzipFile(filename + ".gz", "wb")
shutil.copyfileobj(open(filename, "rb"), gzfile)

serverFiles.upload("PPI", obiPPI.STRINGDetailed.FILENAME_DETAILED, filename + ".gz", 
                   "STRING Protein interactions (Creative Commons Attribution-Noncommercial-Share Alike 3.0 License)" ,
                   tags=["protein interaction", "STRING",
                         "#compression:gz", "#version:%s" % obiPPI.STRINGDetailed.VERSION]
                   )
serverFiles.unprotect("PPI", obiPPI.STRINGDetailed.FILENAME_DETAILED)
    