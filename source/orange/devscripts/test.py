import os, os.path, sys

startdir = os.getcwd()
os.chdir(r"d:\ai\orange\doc\reference")

newscripts, modified, errors = [], [], []

#skip = ["DATA_INFO.py", "RUNDEMOS.py"]
skip = []

outputdir = "outputs"
newname = outputdir + "\\%s.new.txt"
oldname = outputdir + "\\%s.txt"

if not os.path.exists(outputdir):
    os.mkdir(outputdir)

for name in os.listdir("."):
    if name[-3:]==".py" and not name in skip:
        print name
        fnew = open(newname % name, "wt")
        sout, serr = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = fnew
            execfile(name)
            sys.stdout, sys.stderr = sout, serr
        except:
            sys.stdout, sys.stderr = sout, serr
            fnew.close()
            errors.append(name)
            continue
        fnew.close()
        if os.path.exists(oldname % name):
            fnew, fold = open(newname % name, "rt"), open(oldname % name, "rt")
            equal = fnew.read() == fold.read()
            fnew.close()
            fold.close()
            if equal:
                os.remove(newname % name)
            else:
                modified.append(name)
        else:
            newscripts.append(name)
            os.rename(newname % name, oldname % name)

os.chdir(startdir)

flog = open("log.txt", "wt")
for sname in ["modified", "newscripts", "errors"]:
    mf = vars()[sname]
    if mf:
        mf.sort()
        flog.write("--- %s ---\n" % sname)
        for name in mf:
            flog.write(name+"\n")
        flog.write("\n")

flog.close()