#! usr/bin/env python

import os, re, sys, time

regtestdir = os.getcwd().replace("\\", "/")
re_israndom = re.compile(r"#\s*xtest\s*:\s*RANDOM")

date = "%2.2i-%2.2i-%2.2i" % time.localtime()[:3]

platform = sys.platform
pyversion = sys.version[:3]
states = ["OK", "changed", "random", "error", "crash"]

def testScripts(complete, just_print):
    global error_status
    if sys.platform == "win32" and sys.executable[-6:].upper() != "_D.EXE":
        import win32process, win32api
        win32process.SetPriorityClass(win32api.GetCurrentProcess(), 64)

    for dir in os.listdir("."):
        if not os.path.isdir(dir) or dir in ["cvs", "datasets", "widgets", "processed"] or (directories and not dir in directories):
            continue
        
        print "\nDirectory '%s'\n" % dir

        os.chdir(dir)
        outputsdir = "%s/%s-output" % (regtestdir, dir)
        if not os.path.exists(outputsdir):
            os.mkdir(outputsdir)

        if os.path.exists("exclude-from-regression.txt"):
            dont_test = [x.strip() for x in file("exclude-from-regression.txt").readlines()]
        else:
            dont_test = []
        test_set = []

        names = [name for name in os.listdir('.') if (testFiles and name in testFiles) or (not testFiles and name[-3:]==".py") and (not name in dont_test)]
        names.sort()
        for name in names:
            if not os.path.exists("%s/%s.txt" % (outputsdir, name)):
                test_set.append((name, "new"))
            else:
                for state in states:
                    if os.path.exists("%s/%s.%s.%s.%s.txt" % (outputsdir, name, platform, pyversion, state)):
                        test_set.append((name, state))
                        break
                else:
                    if os.path.exists("%s/%s.%s.%s.random1.txt" % (outputsdir, name, platform, pyversion)):
                        test_set.append((name, "random"))
                    elif complete:
                        test_set.append((name, "OK"))
                    else:
                        dont_test.append(name)

        if just_print:
            for name, lastResult in test_set:
                print "%s: %s" % (name, lastResult)
                
        else:
            if dont_test:
                print "Skipped: %s\n" % reduce(lambda x,y: "%s, %s" % (x,y), dont_test)

            for name, lastResult in test_set:
                print "%s (%s): " % (name, lastResult == "new" and lastResult or ("last: %s" % lastResult)),

                for state in ["crash", "error", "new", "changed", "random1", "random2"]:
                    remname = "%s/%s.%s.%s.%s.txt" % (outputsdir, name, platform, pyversion, state)
                    if os.path.exists(remname):
                        os.remove(remname)
                    
                titerations = re_israndom.search(open(name, "rt").read()) and 1 or iterations
                os.spawnl(os.P_WAIT, sys.executable, "-c", regtestdir+"/xtest1.py", name, `titerations`, outputsdir)

                result = open("xtest1_report", "rt").readline().rstrip() or "crash"
                error_status = max(error_status, states.index(result))
                os.remove("xtest1_report")

        os.chdir("..")


if len(sys.argv) == 1 or sys.argv[1][0] == "-":
    command = "update"
    ind = 1
else:
    command = sys.argv[1]
    if command not in ["update", "test", "report"]:
        print "Unrecognized command ('%s')" % command
        sys.exit(1)
    ind = 2


iterations = 3
testFiles = []
directories = []

while ind < len(sys.argv):
    flag = sys.argv[ind]
    ind += 1
    if flag == "-single":
        iterations = 1
        
    elif flag == "-dir":
        if ind >= len(sys.argv) or sys.argv[ind][0]=="-":
            print "Missing argument for -dir"
            sys.exit(1)
        dir = sys.argv[ind]
        ind += 1
        if not dir in directories:
            directories.append(dir)

    elif flag[0] == "-":
        print "Unrecognized option: %s" % flag
    else:
        testFiles = sys.argv[ind:]
        break

os.chdir("../doc")
error_status = 0
testScripts(command=="test", command=="report")
sys.exit(error_status)