import os
import shutil

path = r"C:\Documents and Settings\orngServerFiles\orngServerFiles"
svnpath = "http://www.ailab.si/svn/orange/trunk/install-scripts/orngServer/"
servicename = "orngServerFilesServerService2"

os.system("net stop " + servicename)

command = "svn checkout " + svnpath + " " + '"' + path + '"'
print command
os.system(command)

os.system("net start " + servicename)