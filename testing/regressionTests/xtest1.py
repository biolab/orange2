import sys as t__sys
import traceback as t__traceback
import string as t__string
import os as t__os

NO_RANDOMNESS = 1 # prevent random parts of scripts to run

def t__samefiles(name1, name2):
    equal = 1
    fnew, fold = open(name1, "rt"), open(name2, "rt")
    equal = [t__string.rstrip(x) for x in fnew.readlines()] == [t__string.rstrip(x) for x in fold.readlines()]
    fnew.close()
    fold.close()
    return equal

def t__copyfile(src, dst):
    srcf = open(src, "rt")
    dstf = open(dst, "wt")
    dstf.write(srcf.read())
    srcf.close()
    dstf.close()

t__sys.path.append(".")

# Arguments: name, #iterations, runNo, isNewFile, outputsdir

t__name = t__sys.argv[1]
t__iterations = int(t__sys.argv[2])
t__runNo = int(t__sys.argv[3])
t__outputsdir = t__sys.argv[5]

t__crashname, t__errorname, t__newname, t__changedname, t__random1name, t__random2name = ["%s/%s.%s.%s.txt" % (t__outputsdir, t__name, t__string.zfill(t__runNo, 5), t) for t in ["crash", "error", "new", "changed", "random1", "random2"]]
t__officialname = "%s/%s.txt" % (t__outputsdir, t__name)

t__isNewFile = not t__os.path.exists(t__officialname)

t__message = open("xtest1_report", "wt")

t__isChanged = False

for t__iteration in range(t__iterations):
    if t__iterations>1:
        print t__iteration+1,

    t__fnew = open(t__crashname, "wt")
    t__sout = t__sys.stdout
    t__serr = t__sys.stderr
    try:
        t__sys.stdout = t__sys.stderr = t__fnew
        execfile(t__name)

    except Exception, e:
        # execution ended with an error
        apply(t__traceback.print_exception, t__sys.exc_info())
        t__sys.stdout = t__sout
        t__sys.stderr = t__serr
        t__fnew.close()

        t__message.write("error\n%i\n" % t__iteration)
        print "error"
        t__message.write(reduce(lambda x,y: x+y, apply(t__traceback.format_exception, t__sys.exc_info())))
        t__message.close()
        t__sys.exit(1)

    t__sys.stdout = t__sout
    t__sys.stderr = t__serr
    t__fnew.close()

    if not t__iteration:
        if t__isNewFile:
            # the file is a new files and this has been the first iteration
            t__os.rename(t__crashname, t__newname)
            t__copyfile(t__newname, t__officialname)
            t__prevname = t__newname
        elif not t__samefiles(t__crashname, t__officialname):
            # it's an old file and it has changed
            t__os.rename(t__crashname, t__changedname)
            t__prevname = t__changedname
            t__isChanged = True
        else:
            # file is OK
            t__os.remove(t__crashname)
            t__prevname = t__officialname
    else:
        if not t__samefiles(t__crashname, t__prevname):
            # random file (either new or old)
            t__copyfile(t__prevname, t__random1name)
            t__os.rename(t__crashname, t__random2name)
            t__message.write("random\n")
            print "random"
            t__message.close()
            t__sys.exit(2)
        else:
            t__os.remove(t__crashname) # iterate on with same file

if t__isChanged:
    t__message.write("changed")
    print "changed"
    t__sys.exit(3)

t__message.write("OK")
print "OK"
