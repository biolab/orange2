import time
time.sleep(70)
import os, popen2
os.chdir(r"c:\inetpub\wwwusers\orange")
err_out, stdin = popen2.popen4('"c:\program files\wincvs 1.3\cvsnt\cvs.exe" -d :sspi:cvso@estelle.fri.uni-lj.si:/CVS update doc')

