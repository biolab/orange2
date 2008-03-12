

def test():
    print "testing IEC"
    iec = IEC.IEController()
    iec.Navigate("file:///c:/Diplomska/workspace/Diplomska/html/index.html")
    addContentToIE(iec.ie, "g1", "Moja diplomska", "je prav fletna rec")
    addContentToIE(iec.ie, "g2", "Tabelca", "<table border='1'><tr><td>c1</td><td>c2</td></table>je prav fletna rec")
        

class IEFeeder:
    def __init__(self, reportsDir):
        self.iec = None
        self.stevec = 1
        self.reportsDir = reportsDir
        self.reportsDir = "c:\\D\\html-report"
        global reportFeeder
        reportFeeder = self

    def addContentToIE(self, title, document):
        import win32com.client, pythoncom
        if not self.iec:
            self.initReport()
            
        res=self.iec.ie.Document.Script._oleobj_.Invoke(self.addNewContent, 0, pythoncom.DISPATCH_METHOD, True, self.stevec, title, document)
        self.stevec += 1


    def initReport(self):
        import os, time
        datestr = "%04i-%02i-%02i" % time.gmtime()[:3]
        if not os.path.exists(self.reportsDir + datestr):
            self.directory = datestr
        else:
            for i in range(1000):
                if not os.path.exists(self.reportsDir + datestr+"%03i" % i):
                    self.directory = datestr + "%03i" % i

        self.directory = ""
        
        self.abspath = self.reportsDir + self.directory 
        #os.mkdir(abspath)

        import IEC
        self.iec = IEC.IEController()
        self.iec.Navigate("file:///" + self.abspath + "/index.html")
        self.addNewContent=self.iec.ie.Document.Script._oleobj_.GetIDsOfNames("addNewContent")

    def createDirectory(self):
        if not self.iec:
            self.initReport()
        import os
        tmpPathName = os.tempnam(self.abspath)
        os.mkdir(tmpPathName)
        return tmpPathName

    def __call__(self, title, document):
        self.addContentToIE(title, document)

