import sys, os
import subprocess
import time, glob
import optparse

from getopt import getopt
from datetime import datetime

usage="""usage: %prog [options] [update_script ...]

Run update scripts"""

parser = optparse.OptionParser(usage=usage)
parser.add_option("-u", "--user", help="User name")
parser.add_option("-p", "--password", help="Password")
parser.add_option("-l", "--log-dir", dest="log_dir", help="Directory to store the logs", default="./")
parser.add_option("-m", "--mailto", help="e-mail the results to EMAIL", metavar="EMAIL", default=None)

option, args = parser.parse_args()

if not args:
    args = ["updateTaxonomy.py", "updateGO.py", "updateMeSH.py", "updateNCBI_geneinfo.py",
            "updateHomoloGene.py", "updateDictyBase.py", "updatePPI.py"]
    
for script in args:
    log = open(os.path.join(option.log_dir, script + ".log.txt"), "wb")
    p = subprocess.Popen([sys.executable, script, "-u", option.user, "-p", option.password], stdout=log, stderr=log)
    while p.poll() is None:
        time.sleep(3)
    log.write("\n" + script + " exited with exit status %s" % p.poll())
    log.close()
    if option.mailto:
        fromaddr = "orange@fri.uni-lj.si"
        toaddr = option.mailto.split(",")
        msg = open(os.path.join(option.log_dir, script + ".log.txt"), "rb").read()
        msg = "From: %s\r\nTo: %s\r\nSubject: Error running %s update script\r\n\r\n" % (fromaddr, ",".join(toaddr), script) + msg
        try:
            import smtplib
            s = smtplib.SMTP('212.235.188.18', 25)
            s.sendmail(fromaddr, toaddr, msg)
            s.quit()
        except Exception, ex:
            print "Failed to send error report due to:", ex
    

def files_report():
  import orngServerFiles as sf
  sf = sf.ServerFiles()
  html = []
  for domain in sf.listdomains():
      if domain not in ["demo", "demo2", "test", "gad"]:
          allinfo = sf.allinfo(domain)
          html += ["<h2>%s</h2>" % domain,
                   "<table><tr><th>Title</th><th>Date</th><th>Filename</th></tr>"] + \
                  ["<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (info["title"], info["datetime"], file) \
                   for file, info in allinfo.items()] + \
                  ["</table>"]
  return "\n".join(html)
  
open(os.path.join(option.log_dir, "serverFiles.html"), "wb").write(files_report())
