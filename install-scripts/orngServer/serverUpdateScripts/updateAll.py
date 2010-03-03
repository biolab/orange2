import sys, os
import subprocess
import time

from getopt import getopt
from datetime import datetime

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

age = datetime.now() - datetime.fromtimestamp(0) ## age of the universe

files = ["updateTaxonomy.py", "updateGO.py", "updateKEGG.py", "updateMeSH.py", "updateNCBI_geneinfo.py",
         "updateOMIM.py", "updateHomoloGene.py", "updateDictyBase.py", "updatePPI.py"]

for filename in files:
    options = dict([line[3:].split("=") for line in open(filename).readlines() if line.startswith("##!")])
    if age.days % int(options.get("interval", "7")) == 0:
        output = open("log.txt", "w")
        process = subprocess.Popen([sys.executable, filename, "-u", username, "-p", password], stdout=output, stderr=output)
        while process.poll() == None:
            time.sleep(3)
#        print "/sw/bin/python2.5 %s -u %s -p %s" % (filename, username, password)
#        print os.system("/sw/bin/python2.5 %s -u %s -p %s" % (filename, username, password))
        output.close()
        if process.poll() != 0:
            content = open("log.txt", "r").read()
            print content
            toaddr = options.get("contact", "ales.erjavec@fri.uni-lj.si")
            fromaddr = "orange@fri.uni-lj.si"
            msg = "From: %s\r\nTo: %s\r\nSubject: Exception in server update script - %s\r\n\r\n" % (fromaddr, toaddr, filename) + content
            try:
                import smtplib
                s = smtplib.SMTP('212.235.188.18', 25)
                s.sendmail(fromaddr, toaddr, msg)
                s.quit()
            except Exception, ex:
                print "Failed to send error report due to:", ex
                