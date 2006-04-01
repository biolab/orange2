#! usr/bin/env python

import os, os.path, sys, traceback, time
from string import rstrip, zfill
from operator import add
from xml.dom import minidom
import re

regtestdir = os.getcwd().replace("\\", "/")
re_israndom = re.compile(r"#\s*xtest\s*:\s*RANDOM")

date = "%2.2i-%2.2i-%2.2i" % time.localtime()[:3]
error_status = 0

status = []
results = []
platform = sys.platform

def findFileNode(name, dir):
    name = dir + "/" + name
    for node in status.childNodes:
        nodeName = node.getAttribute("name")
        if nodeName == name:
            return node, False
    else:
        node = None
        
    newnode = dom1.createElement("FILE")
    newnode.setAttribute("name", name)
    return newnode, True

    
def readXML(runType = None, filename = regtestdir+"/testresults.xml"):
    global status, run, dom1
    if os.path.exists(filename):
        fle = open(filename, "rt")
        dom1 = minidom.parse(fle)
        fle.close()
        xml = dom1.firstChild
        stati, run = xml.childNodes
    else:
        dom1 = minidom.Document()
        xml = dom1.createElement("XML")
        dom1.appendChild(xml)
        
        stati = dom1.createElement("STATUS")
        xml.appendChild(stati)
        run = dom1.createElement("RUN")
        xml.appendChild(run)

    for status in stati.childNodes:
        if status.getAttribute("name") == platform:
            break
    else:
        status = dom1.createElement("PLATFORM")
        status.setAttribute("name", platform)
        stati.appendChild(status)

    if runType:
        currentRun = dom1.createElement("RUN")
        currentRun.setAttribute("PLATFORM", sys.platform)
        currentRun.setAttribute("DATE", date)
        currentRun.setAttribute("TYPE", runType)
        xml.replaceChild(currentRun, run)
        run = currentRun


def saveXML(filename = regtestdir + "/testresults.xml"):
    fle = open(filename, "wt")
    fle.write(dom1.toxml())
    fle.close()


def testScripts(complete):
    global error_status
    if sys.platform == "win32" and sys.executable[-6:].upper() != "_D.EXE":
        import win32process, win32api
        win32process.SetPriorityClass(win32api.GetCurrentProcess(), 64)

    skip = ["buildC45.py"]
    for dir in os.listdir("."):
        if not os.path.isdir(dir) or dir in ["cvs", "datasets", "widgets"] or (directories and not dir in directories):
            continue
        
        print "\nDirectory '%s'\n" % dir

        os.chdir(dir)
        outputsdir = "%s/%s-output" % (regtestdir, dir)
        if not os.path.exists(outputsdir):
            os.mkdir(outputsdir)

        names = os.listdir(".")
        names = filter(lambda name: (testFiles and name in testFiles)  or  (not testFiles and (name[-3:]==".py") and (not name in skip)), names)
        names.sort()
        test_set, dont_test = [], []
        for name in names:
            node, isNewFile = findFileNode(name, dir)
            if isNewFile:
                lastResult = "new"
                addToTest = True
            else:
                lastResult = node.getAttribute("STATUS")
                addToTest = complete or lastResult != "OK"
            if addToTest:
                test_set.append((name, node, isNewFile, lastResult))
            else:
                dont_test.append(name)
        if dont_test:
            print "Skipped: %s\n" % reduce(lambda x,y: "%s, %s" % (x,y), dont_test)

        for name, node, isNewFile, lastResult in test_set:
            if isNewFile:
                print "%s (new): " % name,
            else:
                print "%s (last: %s): " % (name, lastResult),

            for t in ["crash", "error", "new", "changed", "random1", "random2"]:
                remname = "%s/%s.%s.txt" % (outputsdir, name, t)
                if os.path.exists(remname):
                    os.remove(remname)
                
            sys.stdout.flush()
            titerations = re_israndom.search(open(name, "rt").read()) and 1 or iterations
            os.spawnl(os.P_WAIT, sys.executable, "-c", regtestdir+"/xtest1.py", name, `titerations`, `00001`, `int(isNewFile)`, outputsdir)
            report = open("xtest1_report", "rt")
            
            result = rstrip(report.readline()) or "crash"
            node.setAttribute("STATUS", result)
            while node.firstChild:
                node.removeChild(node.firstChild)

            results = ["OK", "changed", "random", "error", "crash"]
            if result:
                error_status = max(error_status, results.index(result))
                
            if result in ["error", "crash"]:
                err_iter = rstrip(report.readline())
                err_msg = "".join(report.readlines())
                if not err_msg:
                    err_msg = "<no message -- segmentation fault?>"
                    print
                node.setAttribute("xml:space", "preserve")
                msgNode = dom1.createTextNode(err_msg)
                node.appendChild(msgNode)

            run.appendChild(node.cloneNode(True))
            node.setAttribute("DATE", date)
            if isNewFile:
                status.appendChild(node)

            report.close()
            os.remove("xtest1_report")
            saveXML()
        os.chdir("..")


def report(complete):
    readXML(False)

    prevdir = ""
    for node in files.childNodes:
        nodeDir, nodeName = node.getAttribute("directory"), node.getAttribute("name")
        if directories and dir not in directories  or  testFiles and name not in testFiles:
            continue

        totname = "%s/%s" % (nodeDir, nodeName)
        if platform == "all":
            if prevdir != nodeDir:
                print
                prevdir = nodeDir
            reportedplatforms = []
            for testnode in node.childNodes:
                testplat = testnode.getAttribute("PLATFORM")
                if testplat not in reportedplatforms:
                    status = testnode.getAttribute("RESULT")
                    if complete or status != "OK":
                        if reportedplatforms:
                            print "%40s: %8s: %s\t%s (%i)" % ("", testplat, status, testnode.getAttribute("DATE"), int(testnode.getAttribute("RUN")))
                        else:
                            print "%40s: %8s: %s\t%s (%i)" % (totname, testplat, status, testnode.getAttribute("DATE"), int(testnode.getAttribute("RUN")))
                    reportedplatforms.append(testplat)

        else:
            for testnode in node.childNodes:
                if testnode.getAttribute("PLATFORM") == platform:
                    status = testnode.getAttribute("RESULT")
                    if complete or status != "OK":
                        if prevdir != nodeDir:
                            print
                            prevdir = nodeDir
                        print "%40s: %s\t%s (%i)" % (totname, status, testnode.getAttribute("DATE"), int(testnode.getAttribute("RUN")))
                    break
            else:
                print "%40s: not tested on %s" % (totname, platform)
    



def parseArguments():
    global iterations, directories, testFiles, command, argRun, platform

    if len(sys.argv) == 1 or sys.argv[1][0] == "-":
        command = "update"
        ind = 1
    else:
        command = sys.argv[1]
        if command not in ["update", "purge", "test", "report", "errors", "help"]:
            print "Unrecognized command ('%s')" % command
            sys.exit(1)
        ind = 2
    
    if "-h" in sys.argv:
        do_help()
        sys.exit(0)

    iterations = 3
    testFiles = []
    directories = []
    argRun = 0
    platform = sys.platform

    while ind < len(sys.argv):
        flag = sys.argv[ind]
        ind += 1
        if flag == "-single":
            iterations = 1
            
        elif flag == "-run":
            try:
                argRun = int(sys.argv[ind])
                ind += 1
            except:
                print "Missing or invalid argument for -run"
                sys.exit(1)
            
        elif flag == "-dir":
            if ind >= len(sys.argv) or sys.argv[ind][0]=="-":
                print "Missing argument for -dir"
                sys.exit(1)
            dir = sys.argv[ind]
            ind += 1
            if not dir in directories:
                directories.append(dir)

        elif flag == "-platform":
            if ind >= len(sys.argv) or sys.argv[ind][0]=="-":
                print "Missing argument for -platform"
                sys.exit(1)
            platform = sys.argv[ind]
            ind += 1
            
        elif flag[0] == "-":
            print "Unrecognized option: %s" % flag
        else:
            testFiles = sys.argv[ind:]
            break


def do_help():
    print "xtest.py COMMAND OPTIONS files"
    print "  update     tests files with problems  (this is the default command)"
    print "  test       tests all files"
    print "  report     reports the actual status for all files"
    print "  errors     prints the files with errors"
    print "  help       prints this help"
    print
    print "  -single    run each file only once (no check for randomness)"
    print "  -dir       directory to test (default: all) "
    print "  -d         run debug version (python_d, orange_d, ...)"
    print "              (more than one -dir can be given)"

def do_update():
    readXML("update")
    testScripts(False)
    
def do_test():
    readXML("re-test")
    testScripts(True)
    
def do_report():
    report(True)

def do_errors():
    report(False)

os.chdir("../doc")
parseArguments()
vars()["do_"+command]()
sys.exit(error_status)