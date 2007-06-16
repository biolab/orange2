import orange, orngSQL
import re

def loadSQL(filename, dontCheckStored = False, domain = None):
    f = open(filename)
    lines = f.readlines()
    queryLines = []
    discreteNames = None
    uri = None
    metaNames = None
    className = None
    for i in lines:
        if i.startswith("--orng"):
            (dummy, command, line) = i.split(None, 2)
            if command == 'uri':
                uri = eval(line)
            elif command == 'discrete':
                discreteNames = eval(line)
            elif command == 'meta':
                metaNames = eval(line)
            elif command == 'class':
                className = eval(line)
            else:
                queryLines.append(i)
        else:
            queryLines.append(i)
    query = "\n".join(queryLines)
    r = orngSQL.SQLReader(uri)
    if discreteNames:
        r.discreteNames = discreteNames
    if className:
        r.className = className
    if metaNames:
        r.metaNames = metaNames
    r.execute(query)
    data = r.data()
    return data

def saveSQL():
    pass