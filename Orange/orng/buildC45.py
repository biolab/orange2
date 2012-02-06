import os, os.path, re, string, sys
from sys import argv

# Prepare .h files that wrap .i files and prevent multiple inclusion
for incf in ["buildex", "defns", "extern", "types"]:
    fle = open(incf+".h", "wt")
    fle.write('#ifndef __%s_H\n#define __%s_H\n#include "%s.i"\n#endif' % (incf, incf, incf))
    fle.close()

# Iterate through all .c and .i files and replace include *.i with include *.h
incre = re.compile('\s*#include\s+"([^.]+)\.i"')
mathre = re.compile('\s*#include\s+<math\.h>\s*')
stdiore = re.compile('\s*#include\s+<stdio.h>\s*')

for filename in filter(lambda x: x[-2:] in [".c", ".i"], os.listdir(".")):
    if filename == "ensemble.c":
        continue
    
    inf = open(filename, "rt")
    str = inf.read()
    inf.close()

    str = incre.sub('\n#include "\\1.h"', str)
    str = mathre.sub("\n", str)
    str = stdiore.sub("\n", str)
    
    inf = open(filename, "wt")
    inf.write(str+"\n")
    inf.close()

# Compile ensemble.c and prepare dynamic library

import orange
orangedir, orange = os.path.split(orange.__file__)

if sys.platform == "win32":
    def findprog(name):
        path = os.environ["PATH"]
        for p in path.split(";"):
            if os.path.exists(p+"\\"+name):
                return p+"\\"+name

    compiler = findprog("cl.exe")
    if not compiler:
        print "Compiler (cl.exe) could not be found on PATH.\nCorrect the PATH or compile ensemble.c into c45.dll yourself."
        sys.exit(1)

    linker = findprog("link.exe")
    if not linker:
        print "Linker (link.exe) could not be found on PATH.\nCorrect the PATH or compile ensemble.c into c45.dll yourself."
        sys.exit(2)
        
    if len(argv)>1 and argv[1]=="-d":
        print "DEBUG version"
        addargs_compile = ['/MTd', '/D', '"_DEBUG"', '/Od', '/Zi']
        addargs_link = ['/debug', '/out:c:\\temp\\orange\\debug\\c45_d.dll']
    else:
        addargs_compile = ['/MT', '/D', '"NDEBUG"']
        addargs_link = ['/out:%s\\c45.dll ' % orangedir]
    
    ret = os.spawnv(os.P_WAIT, compiler, string.split('/nologo /W0 /Od /Zi /D "WIN32" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /FD /c ensemble.c ') + addargs_compile)
    if ret:
        print "C compiler exited abnormally"

    ret = os.spawnv(os.P_WAIT, linker, string.split('/nologo /dll /incremental:no /machine:I386 ensemble.obj') + addargs_link)
    if ret:
        print "linker exited abnormally"

    

elif sys.platform == "linux2":
    ret = os.system('gcc ensemble.c -o %s/c45.so -shared -lstdc++' % orangedir)
    if ret:
        print "compiler/linker exited abnormally"

elif sys.platform == "darwin":
    ret = os.system('gcc -F. -bundle -O3 -arch ppc -arch i386 -arch x86_64 ensemble.c -o %s/c45.so' % orangedir)
    if ret:
        print "compiler/linker exiter abnormally"

else:
    print "C4.5 is not supported for this platform"
    exit(3)

import orange    
