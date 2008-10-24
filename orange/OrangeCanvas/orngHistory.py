# Author: Miha Stajdohar (miha.stajdohar@fri.uni-lj.si)
#

import os, sys, re, glob, stat, time
import orngEnviron

logHistory = 1
logFile = os.path.join(orngEnviron.directoryNames["canvasSettingsDir"], "history.log")

if not os.path.exists(logFile):
    file = open(logFile, "w")
    file.close()

lastSchemaID = None
openSchemas = {}

def getLastSchemaID():
    schemaID = None
    fn = None
    try:
        fn = open(logFile, 'r')
        schemaID = 0
        
        for line in fn:
            values = line.split(';')
            if values[2].upper() == 'NEWSCHEMA':
                ID = int(values[1])
                if ID > schemaID:
                    schemaID = ID
    except:
        print "%s: %s" % sys.exc_info()[:2]
    finally:
        if fn != None: fn.close()
        
    return schemaID

def logAppend(schemaID, command, params=""):
    if not logHistory:
        return
    
    if schemaID == None:
        return
    
    fn = None
    try:
        fn = open(logFile, 'a')
        if params == "": fn.write(str(time.localtime()) + ";" + str(schemaID) + ";" + command + ";\n")
        else: fn.write(str(time.localtime()) + ";" + str(schemaID) + ";" + command + ";" + params + ";\n")
    except:
        print "%s: %s" % sys.exc_info()[:2]
    finally:
        if fn != None: fn.close()

def logNewSchema():    
    schemaID = getLastSchemaID()
    
    if schemaID == None:
        return None
     
    schemaID += 1
    
    logAppend(schemaID, "NEWSCHEMA")    
    return schemaID
        
def logCloseSchema(schemaID):
    logAppend(schemaID, "CLOSESCHEMA")
    
def logAddWidget(schemaID, nameKey, x, y):
    logAppend(schemaID, "ADDWIDGET", nameKey + ";" + str(x) + ";" + str(y))
    
def logRemoveWidget(schemaID, nameKey):
    logAppend(schemaID, "REMOVEWIDGET", nameKey)

def logChangeWidgetPosition(schemaID, nameKey, x, y):
    logAppend(schemaID, "MOVEWIDGET", nameKey + ";" + str(x) + ";" + str(y))
    