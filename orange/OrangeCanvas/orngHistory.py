# Author: Miha Stajdohar (miha.stajdohar@fri.uni-lj.si)
# This module 

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
    """Sends history file to specified email."""
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

def cmpBySecondValue(x,y):
    """Compares two array by the second value."""
    if x[1] > y[1]:
        return -1
    elif x[1] < y[1]:
        return 1
    else:
        return 0

def historyFileToBasket():
    """Reads history file to a 'basket'.
        
       Each element in a basket list contains a list of widgets in one session. 
    """
    fin = open(logFile, 'r')
    
    session_widgets = {}
    for line in fin:
        vals = line.split(";")
        session = int(vals[1])
        action = vals[2]
        
        if len(vals[3].split(" - ")) == 2:
            group, widget = vals[3].split(" - ")
            group = group.strip() #.replace(' ','')
            widget = widget.strip() #.replace(' ','')
        else:
            vals[3] = vals[3][2:-2].split("', '")
            if len(vals[3]) == 2:
                group, widget = vals[3]    
                group = group.strip() #.replace(' ','')
                widget = widget.strip() #.replace(' ','')
            else:
                group, widget = None, None
        
        if action == "ADDWIDGET":
            if session in session_widgets:
                widgets = session_widgets[session]
                widgets.append(widget)
            else:
                session_widgets[session] = [widget]

    fin.close()
    
    basket = []
    for key in session_widgets:
        if len(session_widgets[key]) > 0:
            widgets = session_widgets[key]
            if len(widgets) > 1:
                basket.append(widgets)
    
    return basket

def buildWidgetProbabilityTree(basket):
    """Builds a widget probability 'tree'
    
       Levels:
          0 - probability of inserting a widget in empty canvas
          1 - probability of inserting a widget after one widget
          2 - probability of inserting a widget after two widgets
          3 - probability of inserting a widget after three widgets
          ...
    """
    firstWidget = {}
    for vals in basket:
        firstWidget[vals[0]] = firstWidget[vals[0]] + 1 if vals[0] in firstWidget else 1
    
    tree = {}
    tree[0] = firstWidget
    for i in range(1,10):
        tree[i] = estimateWidgetProbability(basket, i)
        
    return tree
    
def estimateWidgetProbability(basket, depth):
    """Estimates the probability of inserting the widget after several (depth) widgets.""" 
    widgetProbs = {}
    for widgets in basket:
        if len(widgets) > depth:
            for i in range(len(widgets) - depth):
                c = ''
                for j in range(i, i + depth + 1):
                    c += widgets[j] + ';'
                c = c[:-1]    
                widgetProbs[c] = widgetProbs[c] + 1 if c in widgetProbs else 1
    
    widgetProbs = widgetProbs.items()
    c = 0
    for l in widgetProbs:
        c += l[1]
        
    widgetProbs = [(widgets.split(';'),float(n)/c,n,c) for widgets, n in widgetProbs]
    widgetProbs.sort(cmpBySecondValue)
    return widgetProbs

def nextWidgetProbility(state, tree):
    """Returns a list of candidate widgets and their probability. The list is sorted descending by probability."""
    #state = [w.replace(' ','') for w in state]
    predictions = []
    # calculate probabilities on levels in a tree up to the number of already inserted widgets
    for i in range(1, len(state)+1):
        predictions_tmp = []
        widgetCounts = tree[i]
        count = 0
        for widgets, p, c, n in widgetCounts:
            
            if len(widgets) > i:
                #print widgets[-2], state[-1]
                flag = True
                for j in range(i):
                    if widgets[-j-2] != state[-j-1]:
                        flag = False
                        
                if flag:
                    predictions_tmp.append((widgets, p, c, n))
                    count += n
        
        # compute the probability of next widget in current tree level
        predictions_tmp = [(predictions_tmp[j][0][-1], float(predictions_tmp[j][2]) / count) for j in range(len(predictions_tmp))]
        predictions.extend(predictions_tmp)
    
    predictions.sort(cmpBySecondValue)
    predictions_set = set()
    # remove double widget entries; leave the one with highest probability 
    todel = []
    for i in range(len(predictions)):
        if predictions[i][0] in predictions_set:
            todel.append(i)
        else:
            predictions_set.add(predictions[i][0])

    todel.sort()
    for i in range(len(todel)):
        del predictions[todel[len(todel) - i - 1]]
    
    return predictions

def predictWidgets(state, nWidgets=3, tree=None):
    """Returns the most probable widgets depending on the state and tree."""
    #state = [w.replace(' ','') for w in state]
    if not tree:
        basket = historyFileToBasket()
        tree = buildWidgetProbabilityTree(basket)
        
    widgets = nextWidgetProbility(state, tree)
    return [widgets[i][0] for i in range(min(nWidgets, len(widgets)))]
