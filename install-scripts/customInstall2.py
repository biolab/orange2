f = file("c:\\xxx")
f.close()
import os
os.chdir(ORANGEROOT + "scripts")
os.spawn(os.P_WAIT, "C:\python23\python.exe", ["customInstall.py"])