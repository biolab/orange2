import os, os.path, sys, traceback

startdir = os.getcwd()
os.chdir(r"d:\ai\orange\doc\reference")

newscripts, modified, israndom, errors = [], [], [], []

skip = []

outputdir = "outputs"
newname = outputdir + "\\%s.new.txt"
oldname = outputdir + "\\%s.txt"
oldoldname = outputdir + "\\%s.old.txt"

def samefiles(name):
    fnew, fold = open(newname % name, "rt"), open(oldname % name, "rt")
    equal = fnew.read() == fold.read()
    fnew.close()
    fold.close()
    return equal

    
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

for name in os.listdir("."):
    if name[-3:]==".py" and not name in skip:
        print name
        if os.path.exists(oldoldname % name):
            os.remove(oldoldname % name)
        for iteration in range(5):

            fnew = open(newname % name, "wt")
            sout = sys.stdout
            serr = sys.stderr
            try:
                sys.stdout = sys.stderr = fnew
                execfile(name)
            except Exception, e:
                print "*** ERROR ***\n\n"
                apply(traceback.print_exception, sys.exc_info())
                sys.stdout = sout
                sys.stderr = serr
                fnew.close()
                errors.append(name)
                break # next file, won't iterate on this, errors should be fixed first

            sys.stdout = sout
            sys.stderr = serr
            fnew.close()
            if iteration:
                if samefiles(name):
                    os.remove(newname % name) # iterate on with same file
                else:
                    israndom.append(name)
                    break # next file
            elif os.path.exists(oldname % name):
                if samefiles(name):
                    os.remove(newname % name) # iterate on with same file
                else:
                    os.rename(oldname % name, oldoldname % name)
                    os.rename(newname % name, oldname % name)
                    modified.append(name)
                    break # next file
            else:
                newscripts.append(name)
                os.rename(newname % name, oldname % name) # new file, but iterate

os.chdir(startdir)

flog = open("test-log.txt", "wt")
for sname, mf in [("new scripts", newscripts),
                  ("files with errors", errors),
                  ("modified files (either due to version or, possibly, due to randomness)", modified),
                  ("random outputs", israndom)]:
    if mf:
        mf.sort()
        flog.write("--- %s ---\n" % sname)
        for name in mf:
            flog.write(name+"\n")
        flog.write("\n")

flog.close()