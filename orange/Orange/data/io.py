import os

import Orange
import Orange.data.variable
import Orange.misc
from Orange.core import \
     BasketFeeder, FileExampleGenerator, BasketExampleGenerator, \
     C45ExampleGenerator, TabDelimExampleGenerator, registerFileType

def loadARFF(filename, create_on_new = Orange.data.variable.Variable.MakeStatus.Incompatible, **kwargs):
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    if os.path.exists(filename + ".xml") and os.path.exists(filename + ".arff"):
        xml_name = filename + ".xml" 
        arff_name = filename + ".arff" 
        return Orange.multilabel.mulan.trans_mulan_data(xml_name,arff_name,create_on_new)
    else:
        return loadARFF_Weka(filename, create_on_new, kwargs)
        
def loadARFF_Weka(filename, create_on_new = Orange.data.variable.Variable.MakeStatus.Incompatible, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in Weka ARFF format"""
    if not os.path.exists(filename) and os.path.exists(filename + ".arff"):
        filename = filename + ".arff" 
    f = open(filename,'r')
    
    attributes = []
    attributeLoadStatus = []
    
    name = ''
    state = 0 # header
    data = []
    for l in f.readlines():
        l = l.rstrip("\n") # strip \n
        l = l.replace('\t',' ') # get rid of tabs
        x = l.split('%')[0] # strip comments
        if len(x.strip()) == 0:
            continue
        if state == 0 and x[0] != '@':
            print "ARFF import ignoring:",x
        if state == 1:
            dd = x.split(',')
            r = []
            for xs in dd:
                y = xs.strip(" ")
                if len(y) > 0:
                    if y[0]=="'" or y[0]=='"':
                        r.append(xs.strip("'\""))
                    else:
                        ns = xs.split()
                        for ls in ns:
                            if len(ls) > 0:
                                r.append(ls)
                else:
                    r.append('?')
            data.append(r[:len(attributes)])
        else:
            y = []
            for cy in x.split(' '):
                if len(cy) > 0:
                    y.append(cy)
            if str.lower(y[0][1:]) == 'data':
                state = 1
            elif str.lower(y[0][1:]) == 'relation':
                name = str.strip(y[1])
            elif str.lower(y[0][1:]) == 'attribute':
                if y[1][0] == "'":
                    atn = y[1].strip("' ")
                    idx = 1
                    while y[idx][-1] != "'":
                        idx += 1
                        atn += ' '+y[idx]
                    atn = atn.strip("' ")
                else:
                    atn = y[1]
                z = x.split('{')
                w = z[-1].split('}')
                if len(z) > 1 and len(w) > 1:
                    # there is a list of values
                    vals = []
                    for y in w[0].split(','):
                        sy = y.strip(" '\"")
                        if len(sy)>0:
                            vals.append(sy)
                    a, s = Orange.data.variable.Variable.make(atn, Orange.data.Type.Discrete, vals, [], create_on_new)
                else:
                    # real...
                    a, s = Orange.data.variable.Variable.make(atn, Orange.data.Type.Continuous, [], [], create_on_new)
                    
                attributes.append(a)
                attributeLoadStatus.append(s)
    # generate the domain
    d = Orange.data.Domain(attributes)
    lex = []
    for dd in data:
        e = Orange.data.Instance(d,dd)
        lex.append(e)
    t = Orange.data.Table(d,lex)
    t.name = name
    t.attribute_load_status = attributeLoadStatus
    return t
loadARFF = Orange.misc.deprecated_keywords(
{"createOnNew": "create_on_new"}
)(loadARFF)


def toARFF(filename,table,try_numericize=0):
    """Save class:`Orange.data.Table` to file in Weka's ARFF format"""
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
                        x = float(j)
                    except:
                        real = 0 # failed
                        break
            else:
                real = 0
        iname = str(i.name)
        if iname.find(" ") != -1:
            iname = "'%s'"%iname
        if real==1:
            f.write('@attribute %s real\n'%iname)
        else:
            f.write('@attribute %s { '%iname)
            x = []
            for j in i.values:
                s = str(j)
                if s.find(" ") == -1:
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
            if s.find(" ") == -1:
                x.append("%s"%s)
            else:
                x.append("'%s'"%s)
        for i in x[:-1]:
            f.write('%s,'%i)
        f.write('%s\n'%x[-1])

def loadMULAN(filename, create_on_new = Orange.data.variable.Variable.MakeStatus.Incompatible, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in Mulan ARFF and XML format"""
    if filename[-4:] == ".xml":
        filename = filename[:-4]
    if os.path.exists(filename + ".xml") and os.path.exists(filename + ".arff"):
        xml_name = filename + ".xml" 
        arff_name = filename + ".arff" 
        return Orange.multilabel.mulan.trans_mulan_data(xml_name,arff_name)
    else:
        return None
loadARFF = Orange.misc.deprecated_keywords(
{"createOnNew": "create_on_new"}
)(loadARFF)

def toC50(filename,table):
    """Save class:`Orange.data.Table` to file in C50 format"""
    t = table
    # export names
    f = open('%s.names' % filename,'w')
    f.write('%s.\n\n' % t.domain.class_var.name)
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
                    x = float(j)
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
    """Save class:`Orange.data.Table` to file in R format"""
    if str.upper(filename[-2:]) == ".R":
        filename = filename[:-2]
    f = open(filename+'.R','w')

    atyp = []
    aord = []
    labels = []
    as0 = []
    for a in t.domain.attributes:
        as0.append(a)
    as0.append(t.domain.class_var)
    for a in as0:
        labels.append(str(a.name))
        atyp.append(a.var_type)
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
    """Save class:`Orange.data.Table` to file in LibSVM format"""
    import Orange.classification.svm
    Orange.classification.svm.tableToSVMFormat(example, open(filename, "wb"))
    
def loadLibSVM(filename, create_on_new=Orange.data.variable.Variable.MakeStatus.Incompatible, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in LibSVM format"""
    attributeLoadStatus = {}
    def make_float(name):
        attr, s = Orange.data.variable.make(name, Orange.data.Type.Continuous, [], [], create_on_new)
        attributeLoadStatus[attr] = s
        return attr
    
    def make_disc(name, unordered):
        attr, s = Orange.data.variable.make(name, Orange.data.Type.Discrete, [], unordered, create_on_new)
        attributeLoadStatus[attr] = s
        return attr
    
    data = [line.split() for line in open(filename, "rb").read().splitlines() if line.strip()]
    vars = type("attr", (dict,), {"__missing__": lambda self, key: self.setdefault(key, make_float(key))})()
    item = lambda i, v: (vars[i], vars[i](v))
    values = [dict([item(*val.split(":"))  for val in ex[1:]]) for ex in data]
    classes = [ex[0] for ex in data]
    disc = all(["." not in c for c in classes])
    attributes = sorted(vars.values(), key=lambda var: int(var.name))
    classVar = make_disc("class", sorted(set(classes))) if disc else make_float("target")
    attributeLoadStatus = [attributeLoadStatus[attr] for attr in attributes] + \
                          [attributeLoadStatus[classVar]]
    domain = Orange.data.Domain(attributes, classVar)
    table = Orange.data.Table([Orange.data.Instance(domain, [ex.get(attr, attr("?")) for attr in attributes] + [c]) for ex, c in zip(values, classes)])
    table.attribute_load_status = attributeLoadStatus
    return table
loadLibSVM = Orange.misc.deprecated_keywords(
{"createOnNew": "create_on_new"}
)(loadLibSVM)

registerFileType("R", None, toR, ".R")
registerFileType("Weka", loadARFF, toARFF, ".arff")
registerFileType("Mulan", loadMULAN, None, ".xml")
registerFileType("C50", None, toC50, [".names", ".data", ".test"])
registerFileType("libSVM", loadLibSVM, toLibSVM, ".svm")
