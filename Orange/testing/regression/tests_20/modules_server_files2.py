import orngServerFiles
reload(orngServerFiles)

domain = "demo"
# remove the domain from the local repository
if domain in orngServerFiles.listdomains():
    orngServerFiles.remove_domain(domain, force=True)

# download all the files for this domain from the server
server = orngServerFiles.ServerFiles()
for filename in server.listfiles(domain):
    orngServerFiles.download(domain, filename, verbose=False)

# make sure that both file lists are the same
files_on_server = server.listfiles(domain)
print "Domain: %s" % domain
intersection = set(files_on_server).intersection(set(orngServerFiles.listfiles(domain)))
if len(intersection) == len(files_on_server):
    print "Same number of files on server and local repository."
else:
    print "Directories on server and local repository are different." 