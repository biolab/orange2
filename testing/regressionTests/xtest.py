#! usr/bin/env python

import os, re, sys, time
import getopt
import orngEnviron

regtestdir = os.getcwd().replace("\\", "/")
re_israndom = re.compile(r"#\s*xtest\s*:\s*RANDOM")

date = "%2.2i-%2.2i-%2.2i" % time.localtime()[:3]

platform = sys.platform
pyversion = sys.version[:3]
states = ["OK", "changed", "random", "error", "crash"]

def testScripts(complete, just_print, module="orange", directory=".", test_files = None):
    """Test the scripts in the given directory."""
    global error_status
    if sys.platform == "win32" and sys.executable[-6:].upper() != "_D.EXE":
        import win32process, win32api
        win32process.SetPriorityClass(win32api.GetCurrentProcess(), 64)

    caller_directory = os.getcwd()
    os.chdir(directory)
    for dir in os.listdir("."):
        if not os.path.isdir(dir) or dir in [".svn", "cvs", "datasets", "widgets", "processed"] or (directories and not dir in directories):
            continue
        
        os.chdir(dir)
        outputsdir = "%s/results/%s/%s" % (regtestdir, module, dir)
        if not os.path.exists(outputsdir):
            os.mkdir(outputsdir)

        if os.path.exists("exclude-from-regression.txt"):
            dont_test = [x.strip() for x in file("exclude-from-regression.txt").readlines()]
        else:
            dont_test = []
        test_set = []

        names = [name for name in os.listdir('.') \
                 if (test_files and name in test_files) or 
                    (not test_files and name[-3:]==".py") and (not name in dont_test)]
        names.sort()

        if names or True:
            print "-" * 79
            print "Directory '%s'" % dir
            print

        # test_set includes all the scripts (file, status) to be tested
        for name in names:
            if not os.path.exists("%s/%s.txt" % (outputsdir, name)):
                # past result not available
                test_set.append((name, "new"))
            else:
                # past result available
                for state in states:
                    if os.path.exists("%s/%s.%s.%s.%s.txt" % \
                                      (outputsdir, name, platform, pyversion, state)):
                        test_set.append((name, state))
                        # current result already on disk
                        break
                else:
                    if os.path.exists("%s/%s.%s.%s.random1.txt" % \
                                      (outputsdir, name, platform, pyversion)):
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
                print "Skipped: %s\n" % ", ".join(dont_test)

            for name, lastResult in test_set:
                print "%s (%s): " % (name, lastResult == "new" and lastResult or ("last: %s" % lastResult)),
                sys.stdout.flush()

                for state in ["crash", "error", "new", "changed", "random1", "random2"]:
                    remname = "%s/%s.%s.%s.%s.txt" % \
                              (outputsdir, name, platform, pyversion, state)
                    if os.path.exists(remname):
                        os.remove(remname)
                    
                titerations = re_israndom.search(open(name, "rt").read()) and 1 or iterations
                os.spawnl(os.P_WAIT, sys.executable, "-c", regtestdir+"/xtest1.py", name, `titerations`, outputsdir)

                result = open("xtest1_report", "rt").readline().rstrip() or "crash"
                error_status = max(error_status, states.index(result))
                os.remove("xtest1_report")

        os.chdir("..")

    os.chdir(caller_directory)


iterations = 3
directories = []
error_status = 0

def usage():
    """Print out help."""
    print "%s update|test|report|errors [-s|-m|-d] [--single|--module m|--dir sd]" % sys.argv[0]
    print "  test:   regression tests on all scripts"
    print "  update: regression tests on all previously failed scripts (default)"
    print "  report: report on testing results"
    print "  errors: report on errors from regression tests"
    
    
def main(argv):
    """Process the argument list and run the regression test."""
    global iterations
    
    if not argv:
        command = "update"
    else:
        if argv[0] not in ["update", "test", "report", "errors", "help"]:
            print "Error: Wrong command"
            usage()
            sys.exit(1)
        command = argv[0]

    try:
        opts, test_files = getopt.getopt(argv[1:], [], ["single", "module=", "dir=", "files=", "verbose="])
    except getopt.GetoptError:
        print "Warning: Wrong argument"
        usage()
        sys.exit(1)
    opts = dict(opts) if opts else {}
    if "--single" in opts:
        iterations = 1
    module = opts.get("--module", "orange").split(",") 
    if "--dir" in opts:
        directories = opts["--dir"].split(",")
    test_files = [tf if ".py" in tf else tf+".py" for tf in test_files]

    testScripts(command=="test", command=="report", module="orange", directory="%s/doc" % orngEnviron.orangeDir, 
                test_files=test_files)
    # sys.exit(error_status)
    
main(sys.argv[1:])
