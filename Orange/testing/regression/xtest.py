#! usr/bin/env python

import os, re, sys, time, subprocess
import getopt
from Orange.misc import environ

regtestdir = os.getcwd().replace("\\", "/")
re_israndom = re.compile(r"#\s*xtest\s*:\s*RANDOM")

date = "%2.2i-%2.2i-%2.2i" % time.localtime()[:3]

platform = sys.platform
pyversion = sys.version[:3]
states = ["OK", "timedout", "changed", "random", "error", "crash"]

def file_name_match(name, patterns):
    """Is any of the string in patterns a substring of name?"""
    for p in patterns:
        if p in name:
            return True
    return False

def test_scripts(complete, just_print, module="orange", root_directory=".",
                test_files=None, directories=None, timeout=5):
    """Test the scripts in the given directory."""
    global error_status
    if sys.platform == "win32" and sys.executable[-6:].upper() != "_D.EXE":
        import win32process, win32api
        win32process.SetPriorityClass(win32api.GetCurrentProcess(), 64)

    caller_directory = os.getcwd()
    os.chdir(root_directory) # directory to start the testing in
    for dirname, dir in directories:
        dir = os.path.join(root_directory, dir)
        os.chdir(dir)
        #if module <> dirname:
        #    outputsdir = "%s/results/%s/%s" % (regtestdir, module, dirname)
        #else:
        #    outputsdir = "%s/results/%s" % (regtestdir, module)

        outputsdir = "%s/results_%s" % (regtestdir, dirname)

        print "DIR %s (%s)" % (dirname, dir)
        if not os.path.exists(outputsdir):
            os.mkdir(outputsdir)

        if os.path.exists("exclude-from-regression.txt"):
            dont_test = [x.strip() for x in file("exclude-from-regression.txt").readlines()]
        else:
            dont_test = []
        test_set = []

        # file name filtering
        names = [name for name in os.listdir('.') if name[-3:] == ".py"]
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
                if lastResult == "OK":
                    result = "results/%s/%s/%s.txt" % (module, dirname, name)
                    print '''  <tr><td><a href="http://orange.biolab.si/trac/browser/trunk/orange/doc/%(dir)s/%(name)s">%(name)s</a></td>
    <td><a href="%(result)s">%(lastResult)s</a></td>
  </tr>''' % {"dir":dir, "name":name, "lastResult":lastResult, "result":result}

#                    print '  <tr><td><a href="results/%s/%s/%s.txt">%s</a></td><td>%s</td></tr>' % (module, dirname, name, name, lastResult)
                elif lastResult in ["changed", "crash", "random"]:
#                else:
                    if lastResult == "random":
                        result = "results/%s/%s/%s.%s.%s.%s.txt" % \
                        (module, dirname, name, platform, pyversion, lastResult + "1")
                    else:
                        result = "results/%s/%s/%s.%s.%s.%s.txt" % (module, dirname, name, platform, pyversion, lastResult)
                    original = "results/%s/%s/%s.txt" % (module, dirname, name)
                    print '''  <tr><td><a href="http://orange.biolab.si/trac/browser/trunk/orange/doc/%(dir)s/%(name)s">%(name)s</a>
    </td><td><a href="%(result)s">%(lastResult)s</a></td>
    <td><a href="%(original)s">original</a></td>
  </tr>''' % {"dir":dir, "name":name, "lastResult":lastResult, "result":result, "original":original}

            print "</table>"
        elif just_print:
            for name, lastResult in test_set:
                print "%-30s %s" % (name, lastResult)

        else:
            if dont_test:
                print "Skipped: %s\n" % ", ".join(dont_test)

            for name, lastResult in test_set:
                print "%s (%s): " % (name, lastResult == "new" and lastResult \
                                     or ("last: %s" % lastResult)),
                sys.stdout.flush()

                for state in states:
                    remname = "%s/%s.%s.%s.%s.txt" % \
                              (outputsdir, name, platform, pyversion, state)
                    if os.path.exists(remname):
                        os.remove(remname)

                titerations = re_israndom.search(open(name, "rt").read()) and 1 or iterations
                #os.spawnl(os.P_WAIT, sys.executable, "-c", regtestdir + "/xtest_one.py", name, str(titerations), outputsdir)
                p = subprocess.Popen([sys.executable, regtestdir + "/xtest_one.py", \
                                      name, str(titerations), outputsdir])

                passed_time = 0
                while passed_time < timeout:
                    time.sleep(0.01)
                    passed_time += 0.01

                    if p.poll() is not None:
                        break

                if p.poll() is None:
                    p.kill()
                    result2 = "timedout"
                    print "timedout (use: --timeout #)"
                    # remove output file and change it for *.timedout.*
                    for state in states:
                        remname = "%s/%s.%s.%s.%s.txt" % \
                                  (outputsdir, name, platform, pyversion, state)
                        if os.path.exists(remname):
                            os.remove(remname)

                    timeoutname = "%s/%s.%s.%s.%s.txt" % \
                                    (outputsdir, name, sys.platform, \
                                     sys.version[:3], "timedout")
                    open(timeoutname, "wt").close()
                    result = "timedout"
                else:
                    stdout, stderr = p.communicate()
                    result = open("xtest1_report", "rt").readline().rstrip() or "crash"

                error_status = max(error_status, states.index(result))
                os.remove("xtest1_report")

        os.chdir("..")

    os.chdir(caller_directory)

iterations = 1
directories = []
error_status = 0

def usage():
    """Print out help."""
    print "%s [test|update|report|report-html|errors] -[h|s] [--single|--module=[all|orange|docs]|--timeout=<#>|--dir=<dir>|] <files>" % sys.argv[0]
    print "  test:   regression tests on all scripts (default)"
    print "  update: regression tests on all previously failed scripts"
    print "  report: report on testing results"
    print "  errors: report on errors from regression tests"
    print
    print "-s, --single: runs a single test on each script"
    print "--module=<module>: defines a module to test"
    print "--timeout=<#seconds>: defines max. execution time"
    print "--dir=<dir>: a comma-separated list of names where any should match the directory to be tested"
    print "<files>: space separated list of string matching the file names to be tested"


def main(argv):
    """Process the argument list and run the regression test."""
    global iterations

    command = "test"
    if argv:
        if argv[0] in ["update", "test", "report", "report-html", "errors", "help"]:
            command = argv[0]
            del argv[0]

    try:
        opts, test_files = getopt.getopt(argv, "hs", ["single", "module=", "timeout=", "help", "files=", "verbose="])
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

    module = opts.get("--module", "all")
    if module == "all":
        root = "%s/.." % environ.install_dir
        module = "orange"
        dirs = [("tests", "Orange/testing/regression/tests"),
                ("tests_20", "Orange/testing/regression/tests_20"),
                ("tutorial", "docs/tutorial/rst/code"),
                ("reference", "docs/reference/rst/code")]
    elif module == "orange":
        root = "%s" % environ.install_dir
        module = "orange"
        dirs = [("tests", "testing/regression/tests"),
                ("tests_20", "testing/regression/tests_20")]
    elif module == "docs":
        root = "%s/.." % environ.install_dir
        module = "orange"
        dirs = [("tutorial", "docs/tutorial/rst/code"),
                ("reference", "docs/reference/rst/code")]
    else:
        print "Error: %s is wrong name of the module, should be in [orange|docs]" % module
        sys.exit(1)

    timeout = 5
    try:
        _t = opts.get("--timeout", "5")
        timeout = int(_t)
        if timeout <= 0 or timeout > 300:
            raise AttributeError()
    except AttributeError:
        print "Error: timeout out of range (0 < # < 300)"
        sys.exit(1)
    except:
        print "Error: %s wrong timeout" % opts.get("--timeout", "5")
        sys.exit(1)

    test_scripts(command == "test", command == "report" or (command == "report-html" and command or False),
                 module=module, root_directory=root,
                 test_files=test_files, directories=dirs, timeout=timeout)
    # sys.exit(error_status)

main(sys.argv[1:])
