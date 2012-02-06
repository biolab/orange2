from Orange.misc.serverfiles import *
import Orange

#for backward compatibility

def _sf_with_code(access_code=None):
    if not access_code:
        return ServerFiles()
    else:
        return ServerFiles(access_code=access_code)

def needs_update(domain, filename, access_code=None):
    sf = _sf_with_code(access_code=access_code)
    return Orange.misc.serverfiles.needs_update(domain, filename, serverfiles=sf)

def update(domain, filename, access_code=None, verbose=True):
    sf = _sf_with_code(access_code=access_code)
    return Orange.misc.serverfiles.update(domain, filename, serverfiles=sf, verbose=verbose)

def createPathForFile(target):
    try:
        os.makedirs(os.path.dirname(target))
    except OSError:
        pass
 
def createPath(target):
    try:
        os.makedirs(target)
    except OSError:
        pass
