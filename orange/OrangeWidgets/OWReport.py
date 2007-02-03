import IEC

ie = IEC.IEController()

def feed(data):
    file("c:\\orangeReport.html", "wt").write(data)
    ie.Navigate("file://c:/orangeReport.html")

def createDirectory():
    tmpPathName = "c:\\temp"
    import os
    if not os.path.exists(tmpPathName):
        os.mkdir(tmpPathName)
    return tmpPathName