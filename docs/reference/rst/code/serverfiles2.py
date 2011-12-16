import Orange, sys
sf = Orange.misc.serverfiles

ordinary = sf.ServerFiles()
authenticated = sf.ServerFiles(sys.argv[1], sys.argv[2])
#authentication is provided as command line arguments

try: 
    authenticated.remove_domain('demo2', force=True)
except: 
    pass 
    
authenticated.create_domain('demo2')
authenticated.upload('demo2', 'titanic.tab', 'titanic.tab', \
    title="A sample .tab file", tags=["basic", "data set"])
print "Uploaded."
print "Non-authenticated users see:", ordinary.listfiles('demo2')
print "Authenticated users see:", authenticated.listfiles('demo2')
authenticated.unprotect('demo2', 'titanic.tab')
print "Non-authenticated users now see:", ordinary.listfiles('demo2')
print "orngServerFiles.py file info:"
import pprint; pprint.pprint(ordinary.info('demo2', 'titanic.tab')) 

