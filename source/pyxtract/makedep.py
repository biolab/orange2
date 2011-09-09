#! /usr/bin/env python

import os, re, sys

filenamedef=re.compile(r".*\..pp$")
includedef=re.compile(r'\s*#include\s*"(?P<filename>.*(([chpij]pp)|(px)))"')
filestemdef=re.compile(r'(?P<filestem>.*)\.(?P<fileextension>[^.]*)$')

def findfiles():
  files = {}
  for dir in dirs:
    for filename in os.listdir(dir):
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
      mydeps.append("ppp/stamp")
      if "../pyxtract/pyprops.py" not in mydeps:
        mydeps.append("../pyxtract/pyprops.py")
    else:
      if (depname[-3:]==".px") or (os.path.split(found.group("filename"))[0]=="px"):
        mydeps.append("px/stamp")
        if "../pyxtract/pyxtract.py" not in mydeps:
          mydeps.append("../pyxtract/pyxtract.py")
      if depname[-3:]==".px" and filename[-4:]!=".hpp":
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
  global filenames, verbose, recreate, action, libraries, modulename, dirs
  filenames, libraries, verbose, recreate, modulename = [], [], 0, 0, ""
  action = []
  dirs = ["."]
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
      dirs.append(args[i])
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
    stamps = [dep for dep in dl if "stamp" in dep]
    dl = [dep for dep in dl if "stamp" not in dep]
    dl.sort()
    stamps.sort()
    makedepsfile.write("obj/%s.o : %s.cpp %s%s\n" % (file[:-4], file[:-4], " ".join(dl), "" if not stamps else " | "+" ".join(stamps)))
makedepsfile.write("\n\n")

if ppp_timestamp_dep:
  makedepsfile.write("../orange/ppp/lists: ../pyxtract/defvectors.py\n")
  makedepsfile.write("\t%s ../pyxtract/defvectors.py\n" % (sys.executable,))
  if modulename != "ORANGE":
    ppp_timestamp_dep.extend(["../orange/ppp/lists", "../orange/ppp/stamp"])
  makedepsfile.write("ppp/stamp: ../pyxtract/pyprops.py %s\n" % " ".join(ppp_timestamp_dep))
  makedepsfile.write("\t%s ../pyxtract/pyprops.py -q -n %s" % (sys.executable, modulename))
  if modulename != "ORANGE":
    makedepsfile.write(" -l ../orange/ppp/stamp -l ../orange/ppp/lists")
  makedepsfile.write("\n\n")

if px_timestamp_dep:
  makedepsfile.write("px/stamp: ../pyxtract/pyxtract.py %s | ppp/stamp\n" % " ".join(px_timestamp_dep))
  short = {"ORANGEOM": "-w OM", "ORANGENE": "-w OG"}.get(modulename, "")
  makedepsfile.write("\t%s ../pyxtract/pyxtract.py -m -q -n %s %s %s" % (sys.executable, modulename, short, " ".join(px_timestamp_dep)))
  if modulename != "ORANGE":
    makedepsfile.write(" -l ../orange/px/stamp")
  makedepsfile.write("\n\n")

  for filename in px_files:
    makedepsfile.write("%s: \n\n" % filename)

makedepsfile.close()
