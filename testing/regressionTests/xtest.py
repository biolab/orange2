#! usr/bin/env python

import os, os.path, sys, traceback, time
from string import rstrip, zfill
from operator import add
from xml.dom import minidom
import re

regtestdir = os.getcwd().replace("\\", "/")

date = "%2.2i-%2.2i-%2.2i" % time.localtime()[:3]

def findFileNode(files, name, dir):
    for node in files.childNodes:
        nodeDir, nodeName = node.getAttribute("directory"), node.getAttribute("name")
        if (nodeDir == dir) and (nodeName == name):
            return node, False, None
    else:
        node = None
        
    newnode = dom1.createElement("FILE")
    newnode.setAttribute("name", name)
    newnode.setAttribute("directory", dir)

    # The newnode will be added when the file is tested.
    # Otherwise, we can run into problems if this script crashes and XML contains
    # entries for files that were not tested yet and thus have no attributes set
    return newnode, True, node

    
def findTestOnPlatform(filenode, platf):
    for node in filenode.childNodes:
        if node.getAttribute("PLATFORM") == platf:
            return node


def readXML(runType = None, filename = regtestdir+"/testresults.xml"):
    global runs, files, runNoNode, runNo, dom1, currentRun
    if os.path.exists(filename):
        fle = open(filename, "rt")
        dom1 = minidom.parse(fle)
        fle.close()
        xml = dom1.firstChild
        runs, files = xml.childNodes
        runNo = int(runs.firstChild.getAttribute("ID")) + 1
    
    else:
        dom1 = minidom.Document()
        doc = dom1.createElement("XML")
        dom1.appendChild(doc)
        
        runs = dom1.createElement("RUNS")
        doc.appendChild(runs)
        files = dom1.createElement("FILES")
        doc.appendChild(files)
        runNo = 1

    if runType:
        currentRun = dom1.createElement("RUN")
        currentRun.setAttribute("ID", zfill(runNo, 5))
        currentRun.setAttribute("DATE", date)
        currentRun.setAttribute("TYPE", runType)
        currentRun.setAttribute("PLATFORM", sys.platform)
        if runs.childNodes:
            runs.insertBefore(currentRun, runs.firstChild)
        else:
            runs.appendChild(currentRun)
        #saveXML()

def saveXML(filename = regtestdir + "/testresults.xml"):
    fle = open(filename, "wt")
    fle.write(dom1.toxml())
    fle.close()


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

    itset = False    
    
    # (can't use enumerate since I increase ind)
    while ind < len(sys.argv):
        flag = sys.argv[ind]
        ind += 1
        if flag == "-single":
            if itset:
                print "Multiple options for setting the number of iterations (-single, -n)"
                sys.exit(1)
            iterations = 1
            itset = True
            
        elif flag == "-n":
            if itset:
                print "Multiple options for setting the number of iterations (-single, -n)"
                exit(1)
            try:
                iterations = int(sys.argv[ind])
                ind += 1
            except:
                print "Missing or invalid argument for number of iterations (-n)"
                sys.exit(1)
            
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
    print "  purge [n]  purges all results prior to n-th run"
    print "  report     reports the actual status for all files"
    print "  errors     prints the files with errors"
    print "  help       print this help"
    print
    print "  -single    run each file only once (no check for randomness"
    print "  -n t       run each script for t times (default: 3)"
    print "  -dir       directory to test (default: all) "
    print "              (more than one -dir can be given)"

def testScripts(complete):
    if sys.platform == "win32":
        import win32process, win32api
        win32process.SetPriorityClass(win32api.GetCurrentProcess(), 64)

    skip = ["buildC45.py"]
    for dir in os.listdir("."):
        if not os.path.isdir(dir) or dir in ["cvs", "datasets"] or (directories and not dir in directories):
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
            node, isNewFile, appendBeforeNode = findFileNode(files, name, dir)
            addToTest = isNewFile
            if not addToTest:
                testnode = findTestOnPlatform(node, sys.platform)
                addToTest = not testnode or (testnode.getAttribute("RESULT") != "OK") or complete
            else:
                testnode = None
            if addToTest:
                test_set.append((name, node, isNewFile, testnode))
            else:
                dont_test.append(name)
        if dont_test:
            print "Skipped: %s\n" % reduce(lambda x,y: "%s, %s" % (x,y), dont_test)

        for name, node, isNewFile, oldtestnode in test_set:
            testNode = dom1.createElement("TEST")
            testNode.setAttribute("RUN", zfill(runNo, 5))
            testNode.setAttribute("DATE", date)
            testNode.setAttribute("PLATFORM", sys.platform)
            testNode.setAttribute("ITERATIONS", `iterations`)

            if isNewFile:
                print "%s (new): " % name,
                node.appendChild(testNode)
                if appendBeforeNode:
                    files.insertBefore(node, appendBeforeNode)
                else:
                    files.appendChild(node)
            else:
                if oldtestnode:            
                    status = oldtestnode.getAttribute("RESULT")
                    print "%s (last outcome: %s): " % (name, status),
                else:
                    print "%s (not yet tested on %s): " % (name, sys.platform),
                node.insertBefore(testNode, node.firstChild)

            runTestNode = dom1.createElement("FILE")
            runTestNode.setAttribute("NAME", name)
            currentRun.appendChild(runTestNode)
            
            for t in ["crash", "error", "new", "changed", "random1", "random2"]:
                remname = "%s/%s.%s.%s.txt" % (outputsdir, name, zfill(runNo, 5), t)
                if os.path.exists(remname):
                    os.remove(remname)
                
            os.spawnl(os.P_WAIT, sys.executable, "-c", regtestdir+"/xtest1.py", name, `iterations`, `runNo`, `int(isNewFile)`, outputsdir)
            report = open("xtest1_report", "rt")
            
            result = rstrip(report.readline())
            testNode.setAttribute("RESULT", result)
            runTestNode.setAttribute("RESULT", result)
            
            if result == "error":
                err_iter = rstrip(report.readline())
                err_msg = reduce(add, report.readlines())
                for nn in [testNode, runTestNode]:
                    nn.setAttribute("ITERATION", err_iter)
                    nn.setAttribute("xml:space", "preserve")
                    msgNode = dom1.createTextNode(err_msg)
                    nn.appendChild(msgNode)

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
    
def do_purge():
    readXML(False)
    cn = runs.childNodes[1:]
    for run in cn:
        if not argRun or argRun >= int(run.getAttribute("ID")):
            runs.removeChild(run)
            run.unlink()
            
    for fle in files.childNodes:
        cn = fle.childNodes[1:]
        platformsLeft = []
        for test in cn:
            platf = test.getAttribute("PLATFORM")
            if (platf in platformsLeft) and (not argRun==-1 or argRun > int(test.getAttribute("RUN"))):
                fle.removeChild(test)
                test.unlink()
                platformsLeft.append(platf)
    saveXML()

    re_lf = re.compile(r".+\.py\.(?P<run>\d{5})\..+\.txt")
    for dir in os.listdir("."):
        if not os.path.isdir(dir) or dir in ["cvs", "datasets"] or (directories and not dir in directories):
            continue
        outputsdir = "%s/%s-output" % (regtestdir, dir)
        for fle in os.listdir(outputsdir):
            m = re_lf.match(fle)
            if m and (not argRun or int(m.group("run")) <= argRun):
                os.remove(outputsdir +"/"+ fle)

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
