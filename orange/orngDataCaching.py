

def getCached(data, funct, params = (), kwparams = {}):
    if not data: return None
    if getattr(data, "info", None) == None or data.info["__version__"] != data.version: 
        setattr(data, "info", {"__version__": data.version})

    if data.info.has_key(funct):
        return data.info[funct]
    else:
        if type(funct) != str:
            data.info[funct] = funct(*params, **kwparams)
            return data.info[funct]
        else:
            return None 
         

def setCached(data, name, info):
    if not data: return
    if getattr(data, "info", None) == None or data.info["__version__"] != data.version:
        setattr(data, "info", {"__version__": data.version})
    data.info[name] = info

def delCached(data, name):
    if not data: return
    if getattr(data, "info", None) != None and data.info.has_key(name):
        del data.info[name]