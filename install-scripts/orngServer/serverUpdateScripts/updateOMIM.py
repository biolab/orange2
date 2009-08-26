##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiOMIM
import orngServerFiles

import orngEnviron
import os, sys

from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

path = os.path.join(orngEnviron.bufferDir, "tmp_OMIM")
serverFiles = orngServerFiles.ServerFiles(username, password)

try:
    os.mkdir(path)
except OSError:
    pass
filename = os.path.join(path, "morbidmap")
obiOMIM.OMIM.download_from_NCBI(path)

serverFiles.upload("OMIM", "morbidmap", filename, title="Online Mendelian Inheritance in Man (OMIM)",
                   tags=["genes", "diseases", "human", "OMIM" "#version:%i" % obiOMIM.OMIM.VERSION])
serverFiles.unprotect("OMIM", "morbidmap")
