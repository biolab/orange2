#
# Module: Orange Input/Output
# --------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org) 
# (Copyright (C)2005 Aleks Jakulin)
#
# Purpose: Parsing WEKA's input files, support for output into ARFF (tested) and C5.0 (untested)
#
# Project initiated on 2005/03/10
#
# Notes: 
#
# ChangeLog:
#   2005/04/04:
#       * export for R (source(filename) is to be used for reading the data.frame within R)
#   2005/03/10:
#       * initial version
#       * registration in orange (Janez)

import orange,string

def loadARFF(filename):
    try:
        f = open(filename,'r')
    except:
        f = open(filename+'.arff','r')
    attributes = []
    name = ''
    state = 0 # header
    data = []
    for l in f.readlines():
        l = l.strip() # strip \n
        l = string.replace(l,'\t',' ') # get rid of tabs
        x = string.split(l,sep='%')[0] # strip comments
        if len(string.strip(x)) == 0:
            continue
        if state == 0 and x[0] != '@':
            print "ARFF import ignoring:",x
        if state == 1:
            dd = string.split(x,sep=',')
            r = []
            for xs in dd:
                y = string.strip(xs,chars=" ")
                if len(y) > 0:
                    if y[0]=="'" or y[0]=='"':
                        r.append(string.strip(xs,chars="'\""))
                    else:
                        ns = string.split(xs,sep=' ')
                        for ls in ns:
                            if len(ls) > 0:
                                r.append(ls)
                else:
                    r.append('?')
            data.append(r[:len(attributes)])
        else:
            y = []
            for cy in string.split(x,sep=' '):
                if len(cy) > 0:
                    y.append(cy)
            if string.lower(y[0][1:]) == 'data':
                state = 1
            elif string.lower(y[0][1:]) == 'relation':
                name = string.strip(y[1])
            elif string.lower(y[0][1:]) == 'attribute':
                if y[1][0] == "'":
                    atn = string.strip(y[1],chars="' ")
                    idx = 1
                    while y[idx][-1] != "'":
                        idx += 1
                        atn += ' '+y[idx]
                    atn = string.strip(atn,chars="' ")
                else:
                    atn = y[1]
                z = string.split(x,sep='{')
                w = string.split(z[-1],sep='}')
                if len(z) > 1 and len(w) > 1:
                    # there is a list of values
                    vals = []
                    for y in string.split(w[0],sep=','):
                        sy = string.strip(y,chars=" '\"")
                        if len(sy)>0:
                            vals.append(sy)
                    #print atn,vals
                    a = orange.EnumVariable(name=atn,values=vals)
                else:
                    # real...
                    a = orange.FloatVariable(name=atn)
                attributes.append(a)
    # generate the domain
    d = orange.Domain(attributes)
    lex = []
    for dd in data:
        e = orange.Example(d,dd)
        lex.append(e)
    t = orange.ExampleTable(d,lex)
    t.name = name
    return t

def toARFF(filename,table,try_numericize=0):
    t = table
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    #print filename
    f = open(filename+'.arff','w')
    f.write('@relation %s\n'%t.domain.classVar.name)
    # attributes
    ats = [i for i in t.domain.attributes]
    ats.append(t.domain.classVar)
    for i in ats:
        real = 1
        if i.varType == 1:
            if try_numericize:
                # try if all values numeric
                for j in i.values:
                    try:
                        x = string.atof(j)
                    except:
                        real = 0 # failed
                        break
            else:
                real = 0
        iname = str(i.name)
        if string.find(iname," ") != -1:
            iname = "'%s'"%iname
        if real==1:
            f.write('@attribute %s real\n'%iname)
        else:
            f.write('@attribute %s { '%iname)
            x = []
            for j in i.values:
                s = str(j)
                if string.find(s," ") == -1:
                    x.append("%s"%s)
                else:
                    x.append("'%s'"%s)
            for j in x[:-1]:
                f.write('%s,'%j)
            f.write('%s }\n'%x[-1])

    # examples
    f.write('@data\n')
    for j in t:
        x = []
        for i in range(len(ats)):
            s = str(j[i])
            if string.find(s," ") == -1:
                x.append("%s"%s)
            else:
                x.append("'%s'"%s)
        for i in x[:-1]:
            f.write('%s,'%i)
        f.write('%s\n'%x[-1])

def toC50(filename,table):
    t = table
    # export names
    f = open('%s.names'%filename,'w')
    f.write('%s.\n\n'%t.domain.classVar.name)
    # attributes
    ats = [i for i in t.domain.attributes]
    ats.append(t.domain.classVar)
    for i in ats:
        real = 1
        # try if real
        if i.varType == 1 and try_numericize:
            # try if all values numeric
            for j in i.values:
                try:
                    x = string.atof(j)
                except:
                    real = 0 # failed
                    break
        if real==1:
            f.write('%s: continuous.\n'%i.name)
        else:
            f.write('%s: '%i.name)
            x = []
            for j in i.values:
                x.append('%s'%j)
            for j in x[:-1]:
                f.write('%s,'%j)
            f.write('%s.\n'%x[-1])
    # examples
    f.close()
    
    f = open('%s.data'%n,'w')
    for j in t:
        x = []
        for i in range(len(ats)):
            x.append('%s'%j[i])
        for i in x[:-1]:
            f.write('%s,'%i)
        f.write('%s\n'%x[-1])

def toR(filename,t):
    if string.upper(filename[-2:]) == ".R":
        filename = filename[:-2]
    f = open(filename+'.R','w')

    atyp = []
    aord = []
    labels = []
    as0 = []
    for a in t.domain.attributes:
        as0.append(a)
    as0.append(t.domain.classVar)
    for a in as0:
        labels.append(str(a.name))
        atyp.append(a.varType)
        aord.append(a.ordered)

    f.write('data <- data.frame(\n')
    for i in xrange(len(labels)):
        if atyp[i] == 2: # continuous
            f.write('"%s" = c('%(labels[i]))
            for j in xrange(len(t)):
                if t[j][i].isSpecial():
                    f.write('NA')
                else:
                    f.write(str(t[j][i]))
                if (j == len(t)-1):
                    f.write(')')
                else:
                    f.write(',')
        elif atyp[i] == 1: # discrete
            if aord[i]: # ordered
                f.write('"%s" = ordered('%labels[i])
            else:
                f.write('"%s" = factor('%labels[i])
            f.write('levels=c(')
            for j in xrange(len(as0[i].values)):
                f.write('"x%s"'%(as0[i].values[j]))
                if j == len(as0[i].values)-1:
                    f.write('),c(')
                else:
                    f.write(',')
            for j in xrange(len(t)):
                if t[j][i].isSpecial():
                    f.write('NA')
                else:
                    f.write('"x%s"'%str(t[j][i]))
                if (j == len(t)-1):
                    f.write('))')
                else:
                    f.write(',')
        else:
            raise "Unknown attribute type."
        if (i < len(labels)-1):
            f.write(',\n')
    f.write(')\n')
    
def toLibSVM(filename, example):
    import orngSVM
    orngSVM.exampleTableToSVMFormat(example, open(filename, "wb"))
    
def loadLibSVM(filename):
    data = [line.split() for line in open(filename, "rb").read().splitlines() if line.strip()]
    vars = type("attr", (dict,), {"__missing__": lambda self, key: self.setdefault(key, orange.FloatVariable(key))})()
    item = lambda i, v: (vars[i], vars[i](v))
    values = [dict([item(*val.split(":"))  for val in ex[1:]]) for ex in data]
    classes = [ex[0] for ex in data]
    disc = all(["." not in c for c in classes])
    attributes = sorted(vars.values(), key=lambda var: int(var.name))
    classVar = orange.EnumVariable("class", values=sorted(set(classes))) if disc else orange.FloatVariable("target")
    domain = orange.Domain(attributes, classVar)
    return orange.ExampleTable([orange.Example(domain, [ex.get(attr, attr("?")) for attr in attributes] + [c]) for ex, c in zip(values, classes)])


orange.registerFileType("R", None, toR, ".R")
orange.registerFileType("Weka", loadARFF, toARFF, ".arff")
orange.registerFileType("C50", None, toC50, [".names", ".data", ".test"])
orange.registerFileType("libSVM", loadLibSVM, toLibSVM, ".svm")
