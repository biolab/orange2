##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiPPI, orngServerFiles
import os, sys, shutil, urllib2, tarfile
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

serverFiles = orngServerFiles.ServerFiles(username, password)

obiPPI.MIPS.download()

try:
    serverFiles.create_domain("PPI")
except Exception, ex:
    print ex
filename = orngServerFiles.localpath("PPI", "mppi.gz")
serverFiles.upload("PPI", "allppis.xml", filename, "MIPS Protein interactions",
                   tags=["protain interaction", "MIPS", "#compression:gz", "#version:%i" % obiPPI.MIPS.VERSION]
                   )
serverFiles.unprotect("PPI", "allppis.xml") 

