##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

from urllib import urlopen
import orngServerFiles
import os, sys

from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))


ontology = urlopen("ftp://nlmpubs.nlm.nih.gov/online/mesh/.asciimesh/d2008.bin")
size = int(ontology.info().getheader("Content-Length"))
rsize = 0
results = list()
for i in ontology:
	rsize += len(i)
	line = i.rstrip("\t\n")
	if(line == "*NEWRECORD"):
		if(len(results) > 0 and results[-1][1] == []): # we skip nodes with missing mesh id
			results[-1] = ["",[],"No description."]
		else:
			results.append(["",[],"No description."])	
	parts = line.split(" = ")
	if(len(parts) == 2 and len(results)>0):
		if(parts[0] == "MH"):
			results[-1][0] = parts[1].strip("\t ") 

		if(parts[0] == "MN"):
			results[-1][1].append(parts[1].strip("\t "))
		if(parts[0] == "MS"):
			results[-1][2] = parts[1].strip("\t ")
ontology.close()

output = file('mesh-ontology.dat', 'w')

for i in results:
	print i[0] + "\t"
	output.write(i[0] + "\t")
	g=len(i[1])			
	for k in i[1]:
		g -= 1
		if(g > 0):
			output.write(k + ";")
		else:
			output.write(k + "\t" + i[2] + "\n")
output.close()
print "Ontology downloaded."




ordinary = orngServerFiles.ServerFiles()
authenticated = orngServerFiles.ServerFiles(username, password)

authenticated.upload('MeSH', 'mesh-ontology.dat', 'mesh-ontology.dat', title="MeSH ontology", tags=['MeSH', 'ontology', 'orngMeSH'])
#authenticated.upload('MeSH', 'cid-annotation.dat', 'cid-annotation.dat', title="Annotation for chemicals (CIDs)", tags =['CID','MeSH','orngMeSH','annotation'])

authenticated.unprotect('MeSH', 'mesh-ontology.dat')
os.remove('mesh-ontology.dat')
print "Ontology uploaded to server."