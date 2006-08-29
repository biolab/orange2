

def loadWordSet(f):
    f = open(f, 'r')
    setW = []
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.lower()
        if line.find(';') != -1:
            line = line[:line.find(';')]
        line = line.strip(' \t\r\n')
        if len(line):
            setW.append(line)
    return setW
