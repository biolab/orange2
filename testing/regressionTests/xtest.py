#! usr/bin/env python

import os, re, sys, time, subprocess
import getopt
import orngEnviron

regtestdir = os.getcwd().replace("\\", "/")
re_israndom = re.compile(r"#\s*xtest\s*:\s*RANDOM")

date = "%2.2i-%2.2i-%2.2i" % time.localtime()[:3]

platform = sys.platform
pyversion = sys.version[:3]
states = ["OK", "changed", "random", "error", "crash"]

def file_name_match(name, patterns):
    """Is any of the string in patterns a substring of name?"""
    for p in patterns:
        if p in name:
            return True
    return False

def test_scripts(complete, just_print, module="orange", root_directory=".", 
                test_files=None, directories=None):
    """Test the scripts in the given directory."""
    global error_status
    if sys.platform == "win32" and sys.executable[-6:].upper() != "_D.EXE":
        import win32process, win32api
        win32process.SetPriorityClass(win32api.GetCurrentProcess(), 64)

    caller_directory = os.getcwd()
    os.chdir(root_directory) # directory to start the testing in
    for dirname, dir in directories:
        os.chdir(dir)
        if module <> dirname:
            outputsdir = "%s/results/%s/%s" % (regtestdir, module, dirname)
        else:
            outputsdir = "%s/results/%s" % (regtestdir, module)
            
        print "DIR %s (%s)" % (dirname, dir)
        if not os.path.exists(outputsdir):
            os.mkdir(outputsdir)

        if os.path.exists("exclude-from-regression.txt"):
            dont_test = [x.strip() for x in file("exclude-from-regression.txt").readlines()]
        else:
            dont_test = []
        test_set = []

        # file name filtering
        names = [name for name in os.listdir('.') if name[-3:]==".py"]
        if test_files:
            names = [name for name in names if file_name_match(name, test_files)]
        names = [name for name in names if name not in dont_test]
        names.sort()

        if names or True:
            if just_print == "report-html":
                print "<h2>Directory '%s'</h2>" % dir
                print '<table class="xtest_report">'
            elif just_print:
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
                    elif complete or just_print:
                        test_set.append((name, "OK"))
                    else:
                        dont_test.append(name)

        if just_print == "report-html":
            for name, lastResult in test_set:
                if lastResult =="OK":
                    print '  <tr><td><a href="results/%s/%s/%s.txt">%s</a></td><td>%s</td></tr>' % (module, dirname, name, name, lastResult)
                else:            
                    print '  <tr><td><a href="results/%s/%s/%s.%s.%s.%s.txt">%s</a></td><td>%s</td></tr>' % (module, dirname, name, platform, pyversion, lastResult, name, lastResult)
            print "</table>"
        elif just_print:
            for name, lastResult in test_set:
                print "%-30s %s" % (name, lastResult)
                
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
                os.spawnl(os.P_WAIT, sys.executable, "-c", regtestdir+"/xtest_one.py", name, str(titerations), outputsdir)
                
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
    print "%s [update|test|report|report-html|errors] -[h|s] [--single|--module=[orange|obi|text]|--dir=<dir>|] <files>" % sys.argv[0]
    print "  test:   regression tests on all scripts"
    print "  update: regression tests on all previously failed scripts (default)"
    print "  report: report on testing results"
    print "  errors: report on errors from regression tests"
    print
    print "-s, --single: runs a single test on each script"
    print "--module=<module>: defines a module to test"
    print "--dir=<dir>: a comma-separated list of names where any should match the directory to be tested"
    print "<files>: space separated list of string matching the file names to be tested"
    
    
def main(argv):
    """Process the argument list and run the regression test."""
    global iterations
    
    command = "update"
    if argv:
        if argv[0] in ["update", "test", "report", "report-html", "errors", "help"]:
            command = argv[0]
            del argv[0]

    try:
        opts, test_files = getopt.getopt(argv, "hs", ["single", "module=", "help", "files=", "verbose="])
    except getopt.GetoptError:
        print "Warning: Wrong argument"
        usage()
        sys.exit(1)
    opts = dict(opts) if opts else {}
    if "--single" in opts or "-s" in opts:
        iterations = 1
    if "--help" in opts or '-h' in opts:
        usage()
        sys.exit(0)
    
    module = opts.get("--module", "orange")
    if module in ["orange"]:
        root = "%s/doc" % orngEnviron.orangeDir
        module = "orange"
        dirs = [("modules", "modules"), ("reference", "reference"), ("ofb", "ofb-rst/code")]
    elif module in ["ofb-rst"]:
        root = "%s/doc" % orngEnviron.orangeDir
        module = "orange"
        dirs = [("ofb", "ofb-rst/code")]
    elif module in ["orange25"]:
        root = "%s/doc" % orngEnviron.orangeDir
        module = "orange25"
        dirs = [("orange25", "Orange/rst/code")]
    elif module == "obi":
        root = orngEnviron.addOnsDirSys + "/Bioinformatics/doc"
    elif module == "text":
        root = orngEnviron.addOnsDirSys + "/Text/doc"
    else:
        print "Error: %s is wrong name of the module, should be in [orange|obi|text]" % module
        sys.exit(1)
    
    test_scripts(command=="test", command=="report" or (command=="report-html" and command or False), 
                 module=module, root_directory=root,
                 test_files=test_files, directories=dirs)
    # sys.exit(error_status)
    
main(sys.argv[1:])
