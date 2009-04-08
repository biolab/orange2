import sys, os

from getopt import getopt
from datetime import datetime

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

age = datetime.now() - datetime.fromtimestamp(0) ## age of the universe

files = ["updateGO.py", "updateKEGG.py", "updateTaxonomy.py", "updateMeSH.py", "updateNCBI_geneinfo.py"]

for filename in files:
    options = dict([line[3:].split("=") for line in open(filename).readlines() if line.startswith("##!")])
    if age.days % int(options.get("interval", "7")) == 0:
        print "python %s -u %s -p %s" % (filename, username, password)
        print os.system("python %s -u %s -p %s" % (filename, username, password))
##        _, stdout, stderr = os.popen3("python %s -u %s -p %s" % (filename, username, password))
##        out = stdout.read()
##        err = stderr.read()
##        print out
##        print err
##        if err:
##            to = options.get("contact", "ales.erjavec@fri.uni-lj.si")
##            from_ = "automation@ailab.si"
##            from email.mime.text import MIMEText
##            msg = MIMEText(err)
##            msg["Subject"] = filename
##            msg["From"] = from_
##            msg["To"] = to
##            try:
##                import smtplib
##                s = smtplib.SMTP()
##                s.connect()
##                s.sendmail(from_, [to], msg.as_string())
##                s.close()
##            except Exception, ex:
##                print "Failed to send error report due to:", ex

    
    

