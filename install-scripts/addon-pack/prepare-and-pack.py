#!/usr/bin/python

import sys
sys.path += ["/root/orange"]

import Orange.misc.addons
rao = Orange.misc.addons.OrangeRegisteredAddOn(None, sys.argv[1])
rao.prepare(None, None, None, None, None, None, None, None, None, None)

import zipfile, os
oao = zipfile.ZipFile(sys.argv[2], 'w')
dirs = os.walk(".")
for (dir, subdirs, files) in dirs:
    for file in files:
        fileRelPath = os.path.join(dir, file)
        if fileRelPath.startswith("source/") or fileRelPath.startswith(".svn/") or "/.svn/" in fileRelPath:
            continue
        oao.write(fileRelPath, fileRelPath)
