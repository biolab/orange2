import orngServerFiles
import os
reload(orngServerFiles)

domain = "demo"
filename = "orngServerFiles.py"

print "Downloading domain: %s, file: %s" % (domain, filename) 
orngServerFiles.download(domain, filename, verbose=False)
print "Needs update? %s (should be False)" % orngServerFiles.needs_update(domain, filename)

# change the access and modified time of the local file
# this is stored in the .info file (does not depend on the actual datetime
# values that we could access through os.stat(path) or set using us.utime)
path = orngServerFiles.localpath(domain, filename) + ".info"
f = file(path)
str = [f.readline()] # first line
s = "1800" + f.readline()[4:] # second line with date, change it
print "Changing date to", s.strip() 
str += [s]
str += f.readlines() # remaining lines
f.close()
f = file(path, "w")
f.writelines(str)
f.close()

print "Needs update? %s (should be True)" % orngServerFiles.needs_update(domain, filename)
print "Updating ..."
orngServerFiles.update(domain, filename, verbose=False)
print "Needs update? %s (should be False)" % orngServerFiles.needs_update(domain, filename)

