import Orange
sf = Orange.misc.serverfiles

repository = sf.ServerFiles() 

print "My files (in demo)", sf.listfiles('demo') 
print "Repository files", repository.listfiles('demo') 
print "Downloading all files in domain 'demo'" 

for file in repository.listfiles('demo'): 
    print "Datetime for", file, repository.info('demo', file)["datetime"] 
    sf.download('demo', file) 
    
print "My files (in demo) after download", sf.listfiles('demo') 
print "My domains", sf.listdomains() 
