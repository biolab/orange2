# Author: Miha Stajdohar (miha.stajdohar@fri.uni-lj.si)
#

import os, sys, time, smtplib
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
    logAppend(schemaID, "ADDWIDGET", str(nameKey) + ";" + str(x) + ";" + str(y))
    
def logRemoveWidget(schemaID, nameKey):
    logAppend(schemaID, "REMOVEWIDGET", str(nameKey))

def logChangeWidgetPosition(schemaID, nameKey, x, y):
    logAppend(schemaID, "MOVEWIDGET", str(nameKey) + ";" + str(x) + ";" + str(y))

def sendHistory(username, password, to, host='fri-postar1.fri1.uni-lj.si'):
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEBase import MIMEBase
    from email.MIMEText import MIMEText
    from email.Utils import formatdate
    from email import Encoders

    fro = "Orange user"
    
    msg = MIMEMultipart()
    msg['From'] = fro
    msg['To'] = to
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = "history.log"
       
    msg.attach(MIMEText("history.log from Orange user"))
    
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(logFile,"r").read())
    Encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(logFile))
    msg.attach(part)
   
    smtp = smtplib.SMTP(host)
    smtp.login(username, password)
    smtp.sendmail(fro, to, msg.as_string())
    smtp.close()

