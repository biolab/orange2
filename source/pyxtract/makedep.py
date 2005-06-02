#! /usr/bin/env python

import os, re, sys

filenamedef=re.compile(r".*\..pp$")
includedef=re.compile(r'\s*#include\s*"(?P<filename>.*(([chpij]pp)|(px)))"')
filestemdef=re.compile(r'(?P<filestem>.*)\.(?P<fileextension>[^.]*)$')

def findfiles():
  files = {}
  for filename in os.listdir("."):
    if filenamedef.match(filename):
      files[filename] = []
  return files

ppp_timestamp_dep = []
px_timestamp_dep = []
px_files = []

def finddeps(filename):
  file = open(filename)
  lineno = 0
  mydeps = files[filename]
  for line in file:
    lineno = lineno+1
    found = includedef.match(line)
    if not found:
      continue

    depname = os.path.split(found.group("filename"))[1]
    if (depname[-4:]==".ppp"):
      ppp_timestamp_dep.append(depname[:-4]+".hpp")
      mydeps.append(depname[:-4]+".hpp")
    else:
      if (depname[-3:]==".px"):
        if depname not in px_files:
          px_files.append( depname)
          print depname
        if not filename in px_timestamp_dep:
          px_timestamp_dep.append(filename)

      if not depname in files and depname!="Python.h" and depname[-3:]!=".px" and depname[-2:]!=".i" and depname[-2:]!=".h" and depname[-4:]!=".jpp" and depname[-4:]!=".ipp":
#        print "%s:%i: Warning: included file %s not found" % (filename, lineno, depname)
        continue
      else:
        mydeps.append(depname)    
    

def recdeps(rootname, filename):
  if not filename in deps[rootname]:
    if deps.has_key(filename) and (rootname!=filename):
      deps[rootname].update(deps[filename])
    else:
      deps[rootname][filename]=None
      for dep in files.get(filename, []):
        recdeps(rootname, dep)



def readArguments(args):
  global filenames, verbose, recreate, action, libraries, modulename
  filenames, libraries, verbose, recreate, modulename = [], [], 0, 0, ""
  action = []
  i = 0
  while(i<len(args)):
    if args[i][0]=="-":
      opt = args[i][1:]
      if opt=="n":
        i += 1
        modulename = args[i]
      elif opt=="d":
        import os
        i += 1
        os.chdir(args[i])
      elif opt=="px":
        i += 1
        px_timestamp_dep.append(args[i])
      else:
        print "Unrecognized option %s" % args[i]
    elif not "makedep" in args[i]:
      print "Unrecognized option %s" % args[i]
    i += 1

  if not modulename:
    print "Module name (-n) missing"
    sys.exit()

        
args = sys.argv

readArguments(args)

files = findfiles()
for file in files:
  finddeps(file)

deps = {}  
for file in files:
  deps[file]={}
  recdeps(file, file)
  del deps[file][file]


deplist = deps.items()
deplist.sort(lambda x, y: cmp(x[0], y[0]))

makedepsfile=open("makefile.deps", "wt")

makedepsfile.write("%s_OBJECTS =" % modulename.upper())
cnt = 0

for file in deplist:
    if file[0][-4:]==".cpp" and file[0] != modulename+"_mac.cpp":
        if not cnt:
            makedepsfile.write("\\\n\t")
        cnt = (cnt+1) % 6
        makedepsfile.write(" obj/%s.o" % file[0][:-4])

makedepsfile.write("\n\n")

for (file, filedeps) in deplist:
  if file[-4:]==".cpp":
    dl = filedeps.keys()
    dl.sort()
    makedepsfile.write("obj/%s.o : %s.cpp %s\n" % (file[:-4], file[:-4], dl and reduce(lambda a, b: a+" "+b, dl) or ""))
makedepsfile.write("\n\n")

if ppp_timestamp_dep:
  makedepsfile.write("ppp/stamp: %s\n" % reduce(lambda a, b: a+" "+b, ppp_timestamp_dep))
  makedepsfile.write("\tpython ../pyxtract/pyprops.py -q -n %s" % modulename)
  if modulename != "ORANGE":
    makedepsfile.write(" -l ../orange/ppp/stamp")
  makedepsfile.write("\n\n")

if px_timestamp_dep:
  makedepsfile.write("px/stamp: %s\n" % reduce(lambda a, b: a+" "+b, px_timestamp_dep))
  makedepsfile.write("\tpython ../pyxtract/pyxtract.py -m -q -n %s %s" % (modulename, reduce(lambda x,y: x+" "+y, px_timestamp_dep)))
  if modulename != "ORANGE":
    makedepsfile.write(" -l ../orange/px/stamp")
  makedepsfile.write("\n\n")

  for filename in px_files:
    makedepsfile.write("%s: \n\n" % filename)

makedepsfile.close()
