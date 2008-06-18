import time
time.sleep(5)
import os
os.chdir(r"c:\inetpub\wwwusers\orange")
os.system('"c:\program files\wincvs 1.3\cvsnt\cvs.exe" -d :sspi:cvso@estelle.fri.uni-lj.si:/CVS update doc')
