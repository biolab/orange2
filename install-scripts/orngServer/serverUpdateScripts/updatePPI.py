##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiPPI, orngServerFiles
import os, sys, shutil, urllib2, tarfile
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

serverFiles = orngServerFiles.ServerFiles(username, password)

try:
    os.mkdir(orngServerFiles.localpath("PPI"))
except OSError:
    pass

obiPPI.MIPS.download()

try:
    serverFiles.create_domain("PPI")
except Exception, ex:
    print ex
filename = orngServerFiles.localpath("PPI", "mppi.gz")
serverFiles.upload("PPI", "allppis.xml", filename, "MIPS Protein interactions",
                   tags=["protein interaction", "MIPS", "#compression:gz", "#version:%i" % obiPPI.MIPS.VERSION]
                   )
serverFiles.unprotect("PPI", "allppis.xml") 

if False: ## download BIOGRID-ALL manually
    import gzip
    filename = orngServerFiles.localpath("PPI", "BIOGRID-ALL.tab")
    gz = gzip.GzipFile(filename + ".gz", "wb")
    gz.write(open(filename, "rb").read())
    gz.close()
    serverFiles.upload("PPI", "BIOGRID-ALL.tab", filename + ".gz", title="BioGRID Protein interactions", 
                       tags=["protein interaction", "BioGrid", "#compression:gz", "#version:%i" % obiPPI.BioGRID.VERSION]
                       )
    serverFiles.unprotect("PPI", "BIOGRID-ALL.tab")

