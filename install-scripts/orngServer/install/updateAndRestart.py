import os
import shutil

path = r"C:\Documents and Settings\orngServerFiles\orngServerFiles"
svnpath = "http://www.ailab.si/svn/orange/trunk/install-scripts/orngServer/"
servicename = "orngServerFilesServerService2"

os.system("net stop " + servicename)

try:
	shutil.rmtree(path)
except:
	pass
	
try:
	os.mkdir(os.path.join(path, "..", "orngServerData"))
except:
	pass
	
command = "svn export " + svnpath + " " + '"' + path + '"'
print command
os.system(command)

os.system("net start " + servicename)