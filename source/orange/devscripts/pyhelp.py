from pyxtractstructures import *
import xml

classdefs = {}
functions = []
constants = []

def reftag(name):
    return '<CODE><A HREF="class_%s.html">%s</A></CODE>' % (name, name)
                                             
def descriptionsFromXML(filename):
  def mergeText(node):
    return reduce(lambda x, y:x+y.toxml(), node.childNodes, "")

  def getChildrenByName(node, name):
      return filter(lambda child:child.nodeName==name, node.childNodes)
    
  def getChildByName(node, name):
    t=getChildrenByName(node, name)
    if len(t)>1:
        raise SystemError, "More than one %s child", name
    elif len(t)==1:
        return t[0]
      
  from xml.dom.minidom import parse, parseString
  try:
    fle=open(filename, "rt")
  except:
    return
  
  dom1=parse(filename)

  for classref in dom1.getElementsByTagName("CLASSREF"):
    t=parseString(reftag(mergeText(classref))).firstChild
    classref.parentNode.replaceChild(t, classref)
  
  dom1.normalize()
  
  domclasses=dom1.getElementsByTagName("CLASS")
  for domclass in domclasses:
    classdefs[domclass.getAttribute("name")] = classdef = ClassDefinition(name=domclass.getAttribute("name"))
    
    desc=getChildByName(domclass, "DESCRIPTION")
    if desc:
        classdef.description=mergeText(desc)

    desc=getChildByName(domclass, "ANCESTORS")
    if desc:
        classdef.ancestors=[]
        for ancestor in getChildrenByName(desc, "BACK"):
            classdef.ancestors.append(mergeText(ancestor))

    prop=domclass.getElementsByTagName("PROPERTY")
    classdef.properties=[]
    for property in prop:
        classdef.properties.append(AttributeDefinition(name=property.getAttribute("name"), description=mergeText(property)))

    meth=domclass.getElementsByTagName("METHOD")
    classdef.methods=[]
    for method in meth:
        classdef.methods.append(MethodDefinition(name=method.getAttribute("name"), description=mergeText(method), arguments=method.getAttribute("arguments")))

    cons=getChildByName(domclass, "CONSTRUCTOR")
    if cons:
      classdef.constructor=ConstructorDefinition(description=mergeText(cons), arguments=cons.getAttribute("arguments"))
      
    call=getChildByName(domclass, "CALL")
    if call:
      classdef.call=CallDefinition(description=mergeText(call), arguments=call.getAttribute("arguments"))

  domfunctions=dom1.getElementsByTagName("FUNCTION")
  for domfunction in domfunctions:
      functions.append(FunctionDefinition(name=domfunction.getAttribute("name"), description=mergeText(domfunction), arguments=domfunction.getAttribute("arguments")))
      
  domconstants=dom1.getElementsByTagName("CONSTANT")
  for domconstant in domconstants:
      constants.append(ConstantDefinition(name=domconstant.getAttribute("name"), description=mergeText(domconstant)))

def derivedClasses():
  for classdef in classdefs.values():
    classdef.derivedClasses=[]

  for classdef in classdefs.values():
    if len(classdef.ancestors):
      classdefs[classdef.ancestors[0]].derivedClasses.append(classdef.name)

  for classdef in classdefs.values():
    classdef.derivedClasses.sort()
  
def writeClassHTML(classdef):
    def wne(how, who):
        if who:
            file.write(how % who)

    def inhtag(cd):
        if cd==classdef:
            return ""
        else:
            return " <FONT size=-1>(inherited from %s)</FONT>" % reftag(cd.name)
          
    this_an_old=[classdef]+[classdefs[name] for name in classdef.ancestors]

    file=open("pyhelp\\class_%s.html" % classdef.name, "wt")
    file.write("<HTML>\n<HEAD><TITLE>Orange class %s</TITLE></HEAD>\n\n" % classdef.name)
    file.write("<BODY>\n")

    trev=[reftag(x.name) for x in this_an_old]
    trev.reverse()
    trev=reduce(lambda x, y: x+" &gt; "+y, trev)
    file.write("<H4>%s</H4>\n" % trev)
    file.write("<H1>%s</H1>\n" % classdef.name)
    
    wne("<P>%s</P>\n\n", classdef.description)
    if classdef.constructor:
        file.write("<H4>Constructor</H4>\n")
        wne("<CODE>%s</CODE>\n", classdef.constructor.arguments)
        wne("<P>%s</P>\n", classdef.constructor.description)

    for cd in this_an_old:
      if cd.call:
        file.write("<H4>Call%s</H4>\n" % inhtag(cd))
        wne("<CODE>%s</CODE>\n", cd.call.arguments)
        wne("<P>%s</P>\n", cd.call.description)
        break
                       
    file.write("<H4>Attributes</H4>\n")
    file.write("<DL>\n")
    for cd in this_an_old:
        for prop in cd.properties:
            file.write("<DT><CODE><B>%s</B></CODE>%s</DT>\n" % (prop.name, inhtag(cd)))
            file.write("<DD>%s</DD>\n" % prop.description)
    file.write("</DL>\n")
        
    file.write("<H4>Methods</H4>\n")
    file.write("<DL>\n")
    for cd in this_an_old:
        for method in cd.methods:
            file.write("<DT><CODE><B>%s %s</B></CODE>%s</DT>\n" % (method.name, method.arguments, inhtag(cd)))
            file.write("<DD>%s</DD>\n" % method.description)            
    file.write("</DL>\n")

    if len(classdef.derivedClasses):
      file.write("<H4>Derived classes</H4>\n")
      file.write(reduce(lambda x, y: x+", "+y, [reftag(x) for x in classdef.derivedClasses]))

    file.close()


descriptionsFromXML("names2.xml")
derivedClasses()
for cl in classdefs.values():
    writeClassHTML(cl)