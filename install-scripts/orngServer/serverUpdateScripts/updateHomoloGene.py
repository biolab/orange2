##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiHomoloGene
import orngServerFiles

import orngEnviron
import os, sys

from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

path = os.path.join(orngEnviron.bufferDir, "tmp_HomoloGene")
serverFiles = orngServerFiles.ServerFiles(username, password)

try:
    os.mkdir(path)
except OSError:
    pass
filename = os.path.join(path, "homologene.data")
obiHomoloGene.HomoloGene.download_from_NCBI(filename)
uncompressed = os.stat(filename).st_size
import gzip, shutil
f = gzip.open(filename + ".gz", "wb")
shutil.copyfileobj(open(filename), f)
f.close()

#serverFiles.create_domain("HomoloGene")
print "Uploading homologene.data"
serverFiles.upload("HomoloGene", "homologene.data", filename + ".gz", title="HomoloGene",
                   tags=["genes", "homologs", "HomoloGene", "#compression:gz",
                         "#uncompressed:%i" % uncompressed, 
                         "#version:%i" % obiHomoloGene.HomoloGene.VERSION])
serverFiles.unprotect("HomoloGene", "homologene.data")