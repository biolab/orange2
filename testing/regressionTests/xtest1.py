import sys as t__sys
import traceback as t__traceback
import string as t__string
import os as t__os

NO_RANDOMNESS = 1 # prevent random parts of scripts to run

def t__isdigit(c):
    return c in "0123456789"

def t__samefiles(name1, name2):
    equal = 1
    fnew, fold = open(name1, "rt"), open(name2, "rt")
    lines1 = [t__string.rstrip(x) for x in fnew.readlines()]
    lines2 = [t__string.rstrip(x) for x in fold.readlines()]
    fnew.close()
    fold.close()
    if lines1 == lines2:
        return 1
    if len(lines1) != len(lines2):
        return 0
    for l in range(len(lines1)):
        line1, line2 = lines1[l], lines2[l]
        if line1 != line2:
            if len(line1) != len(line2):
                return 0
            i = 0
            while i < len(line1):
                if line1[i] != line2[i]:
                    j = i
                    while i<len(line1) and t__isdigit(line1[i]):
                        i += 1
                    if i==j:
                        return 0
                    while j>=0 and t__isdigit(line1[j]):
                        j -= 1
                    if j<0 or line1[j] != ".":
                        return 0
                    j -= 1
                    while j>=0 and t__isdigit(line1[j]):
                        j -= 1
                    if (j >= 0) and (line1[j] in "+-"):
                        j -= 1
                    n1, n2 = line1[j+1:i], line2[j+1:i]
                    if n1.count(".") != n2.count("."):
                        return 0
                    for c in n2:
                        if not c in "0123456789.+- ":
                            return 0
                    maxdiff = 1.5 * (.1 ** (len(n1) - n1.find(".") - 1))
                    if abs(float(n1) - float(n2)) > maxdiff:
                        return 0
                else:
                    i += 1
    return 1


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
t__outputsdir = t__sys.argv[3]

t__crashname, t__errorname, t__newname, t__changedname, t__random1name, t__random2name = ["%s/%s.%s.%s.%s.txt" % (t__outputsdir, t__name, t__sys.platform, t__sys.version[:3], t) for t in ["crash", "error", "new", "changed", "random1", "random2"]]
t__officialname = "%s/%s.%s.txt" % (t__outputsdir, t__name, t__sys.platform)
if not t__os.path.exists(t__officialname):
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
t__message.close()
print "OK"
