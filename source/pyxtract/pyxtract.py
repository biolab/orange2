#!/usr/bin/env python
import re, os, sys, os.path, string, pickle
from pyxtractstructures import *

                
if 1: ### Definitions of method slots

  specialmethods=[
                ("dealloc", "tp_dealloc", "destructor"),
                ("", "tp_print", "0"),
                ("", "tp_getattr", "0"),
                ("", "tp_setattr", "0"),
                ("cmp", "tp_compare", "cmpfunc"),
                ("repr", "tp_repr", "reprfunc"),
                ("", "as_number"),
                ("", "as_sequence"),
                ("", "as_mapping"),
                ("hash", "tp_hash", "hashfunc"),
                ("call", "tp_call", "ternaryfunc"),
                ("str", "tp_str", "reprfunc"),
                ("getattr", "tp_getattro", "getattrofunc"),
                ("setattr", "tp_setattro", "setattrofunc"),
                ("", "tp_as_buffer", "0"),
                ("", "FLAGS"),
                ("", "DOC"),
                ("traverse", "tp_traverse", "traverseproc"),
                ("clear", "tp_clear", "inquiry"),
                ("richcmp", "tp_richcmp", "richcmpfunc"),
                ("", "tp_weaklistoffset", "0"),
                ("iter", "tp_iter", "getiterfunc"),
                ("iternext", "tp_iternext", "iternextfunc"),
                ("", "methods",),
                ("", "tp_members", "0"),
                ("", "getset",),
                ("", "BASE",),
                ("", "tp_dict", "0"),
                ("", "tp_descrget", "0"),
                ("", "tp_descrset", "0"),
                ("", "DICTOFFSET"),
                ("init", "tp_init", "initproc"),
                ("", "tp_alloc", "PyType_GenericAlloc"),
                ("new", "tp_new", "newfunc"),
                ("", "tp_free", "_PyObject_GC_Del"),
                ("", "tp_is_gc", "0"),
                ("", "tp_bases", "0"),
                ("", "tp_mro", "0"),
                ("", "tp_cache", "0"),
                ("", "tp_subclasses", "0"),
                ("", "tp_weaklist", "0")
               ]

  specialnumericmethods=[
                ("add", "nb_add", "binaryfunc"),
                ("sub", "nb_subtract", "binaryfunc"),
                ("mul", "nb_multiply", "binaryfunc"),
                ("div", "nb_divide", "binaryfunc"),
                ("mod", "nb_remainder", "binaryfunc"),
                ("divmod", "nb_divmod", "binaryfunc"),
                ("pow", "nb_power", "ternaryfunc"),
                ("neg", "nb_negative", "unaryfunc"),
                ("pos", "nb_positive", "unaryfunc"),
                ("abs", "nb_absolute", "unaryfunc"),
                ("nonzero", "nb_nonzero", "inquiry"),
                ("inv", "nb_invert", "unaryfunc"),
                ("lshift", "nb_lshift", "binaryfunc"),
                ("rshift", "nb_rshift", "binaryfunc"),
                ("and", "nb_and", "binaryfunc"),
                ("xor", "nb_xor", "binaryfunc"),
                ("or", "nb_or", "binaryfunc"),
                ("coerce", "nb_coerce", "coercion"),
                ("int", "nb_int", "unaryfunc"),
                ("long", "nb_long", "unaryfunc"),
                ("float", "nb_float", "unaryfunc"),
                ("oct", "nb_oct", "unaryfunc"),
                ("hex", "nb_hex", "unaryfunc")
                ]

  specialsequencemethods=[
                ("len_sq", "sq_length", "inquiry"),
                ("concat", "sq_concat", "binaryfunc"),
                ("repeat", "sq_repeat", "intargfunc"),
                ("getitem_sq", "sq_item", "intargfunc"),
                ("getslice", "sq_slice", "intintargfunc"),
                ("setitem_sq", "sq_ass_item", "intobjargproc"),
                ("setslice", "sq_ass_slice", "intintobjargproc"),
                ("contains", "sq_contains", "objobjproc")
                ]

  specialmappingmethods=[
                ("len", "mp_length", "inquiry"),
                ("getitem", "mp_subscript", "binaryfunc"),
                ("setitem", "mp_ass_subscript", "objobjargproc")
                ]

  
  genericconstrs = {'C_UNNAMED': 'PyOrType_GenericNew',
                    'C_NAMED'  : 'PyOrType_GenericNamedNew',
                    'C_CALL'   : 'PyOrType_GenericCallableNew',
                    'C_CALL3'  : 'PyOrType_GenericCallableNew'}
               
if 1: ### Definitions of regular expressions

  constrdef_mac=re.compile(r'(?P<constype>ABSTRACT|C_UNNAMED|C_NAMED|C_CALL)\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<basename>\w*\s*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)')
  constrdef_mac_call3=re.compile(r'(?P<constype>C_CALL3)\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<callname>\w*\s*)\s*,\s*(?P<basename>\w*\s*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)')
  constrkeywords = re.compile(r'CONSTRUCTOR_KEYWORDS\s*\(\s*(?P<typename>\w*)\s*, \s*"(?P<keywords>[^"]*)"\s*\)')
  nopickle=re.compile(r"NO_PICKLE\s*\(\s*(?P<typename>\w*)\s*\)")
  constrwarndef=re.compile(r"[^\w](ABSTRACT|CONS|C_UNNAMED|C_NAMED|C_CALL)[^\w]")

  datastructuredef=re.compile(r'DATASTRUCTURE\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<structurename>\w*)\s*,\s*(?P<dictfield>\w*)\s*\)')
  newbasedondef=re.compile(r'(inline)?\s*PyObject\s*\*(?P<typename>\w*)_new\s*\([^)]*\)\s*BASED_ON\s*\(\s*(?P<basename>\w*)\s*,\s*"(?P<doc>[^"]*)"\s*\)\s*(?P<allows_empty_args>ALLOWS_EMPTY)?')
  basedondef=re.compile(r'BASED_ON\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<basename>\w*)\s*\)')
  hiddendef=re.compile(r'HIDDEN\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<basename>\w*)\s*\)')
  

  allspecial=['('+m[0]+')' for m in specialmethods+specialnumericmethods+specialsequencemethods+specialmappingmethods]
  allspecial=filter(lambda x:x!='()', allspecial)
  allspecial=reduce(lambda x,y:x+'|'+y, allspecial)
  specialmethoddef=re.compile(r'((PyObject\s*\*)|(int)|(void)|(Py_ssize_t))\s*(?P<typename>\w*)_(?P<methodname>'+allspecial+r')\s*\(')

  calldocdef=re.compile(r'PyObject\s*\*(?P<typename>\w*)_call\s*\([^)]*\)\s*PYDOC\s*\(\s*"(?P<doc>[^"]*)"\s*\)')
  getdef=re.compile(r'PyObject\s*\*(?P<typename>\w*)_(?P<method>get)_(?P<attrname>\w*)\s*\([^)]*\)\s*(PYDOC\(\s*"(?P<doc>[^"]*)"\s*\))?')
  setdef=re.compile(r'int\s*(?P<typename>\w*)_(?P<method>set)_(?P<attrname>\w*)\s*\([^)]*\)\s*(PYDOC\(\s*"(?P<doc>[^"]*)"\s*\))?')
  methoddef=re.compile(r'PyObject\s*\*(?P<typename>\w\w+)_(?P<cname>\w*)\s*\([^)]*\)\s*PYARGS\((?P<argkw>[^),]*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)(\s*//>(?P<methodname>\w\w+))?')
  reducedef=re.compile(r'PyObject\s*\*(?P<typename>\w\w+)__reduce__')

  funcdef2=re.compile(r'(?P<defpart>PyObject\s*\*(?P<pyname>\w*)\s*\([^)]*\))\s*;?\s*PYARGS\((?P<argkw>[^),]*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)')
  funcwarndef=re.compile("PYARGS")
  keywargsdef=re.compile(r"METH_VARARGS\s*\|\s*METH_KEYWORDS")

  classconstantintdef=re.compile(r"PYCLASSCONSTANT_INT\((?P<typename>\w*)\s*,\s*(?P<constname>\w*)\s*,\s*(?P<constant>.*)\)\s*$")
  classconstantfloatdef=re.compile(r"PYCLASSCONSTANT_FLOAT\((?P<typename>\w*)\s*,\s*(?P<constname>\w*)\s*,\s*(?P<constant>.*)\)\s*$")
  classconstantdef=re.compile(r"PYCLASSCONSTANT\((?P<typename>\w*)\s*,\s*(?P<constname>\w*)\s*,\s*(?P<constant>.*)\)\s*$")
  
  constantintdef=re.compile(r"PYCONSTANT_INT\((?P<pyname>\w*)\s*,\s*(?P<constant>.*)\)\s*$")
  constantfloatdef=re.compile(r"PYCONSTANT_FLOAT\((?P<pyname>\w*)\s*,\s*(?P<constant>.*)\)\s*$")
  constantdef=re.compile(r"PYCONSTANT\((?P<pyname>\w*)\s*,\s*(?P<ccode>.*)\)\s*$")
  constantfuncdef=re.compile(r"PYCONSTANTFUNC\((?P<pyname>\w*)\s*,\s*(?P<cfunc>.*)\)\s*$")
  constantwarndef=re.compile("PYCONSTANT")
  recognizedattrsdef = re.compile(r'RECOGNIZED_ATTRIBUTES\s*\(\s*(?P<typename>\w*)\s*, \s*"(?P<attributes>[^"]*)"\s*\)')

if 1:
  cc_functions = "int cc_func_%(type)s(PyObject *obj, void *ptr) { if (!PyOr%(type)s_Check(obj)) return 0;      *(GCPtr<T%(type)s> *)(ptr) = PyOrange_As%(type)s(obj); return 1; }\n" + \
                 "int ccn_func_%(type)s(PyObject *obj, void *ptr) { if (obj == Py_None) { *(GCPtr<T%(type)s> *)(ptr) = GCPtr<T%(type)s>(); return 1; }      if (!PyOr%(type)s_Check(obj)) return 0;      *(GCPtr<T%(type)s> *)(ptr) = PyOrange_As%(type)s(obj); return 1; }\n\n\n"


f_underscored = open("..\_underscored", "a")
hump = re.compile("([a-z])([A-Z])")

def camelChange(mo):
    if mo.end() == len(mo.string) or not 'A'<=mo.string[mo.end()]<='Z':
        sg = mo.group(2).lower()
    else:
        sg = mo.group(2)
    return "%s_%s" % (mo.group(1), sg)
    
def camel2underscore(s):
    if s[0].isupper():
        return s
    else:
        u = hump.sub(camelChange, s)
        if u != s:
            f_underscored.write("%-40s %s\n" % (s, u))
        return u


def detectConstructors(line, classdefs):
  found=constrdef_mac.search(line)
  if not found:
    found=constrdef_mac_call3.search(line)
  if found:
    typename, basename, constype, doc=found.group("typename", "basename", "constype", "doc")
    printV2("%s (%s): Macro constructor %s", (typename, basename, constype))
    addClassDef(classdefs, typename, parsedFile, "basetype", basename)
    if constype=="ABSTRACT":
      classdefs[typename].abstract = True
    else:
       addClassDef(classdefs, typename, parsedFile, "constructor", ConstructorDefinition(arguments=doc, type=constype, allows_empty_args = True))
    return 1
  
  found = constrkeywords.search(line)
  if found:
    typename, keywords = found.group("typename", "keywords")
    addClassDef(classdefs, typename, parsedFile, "constructor_keywords", keywords.split())
    return 1
  
  if constrwarndef.search(line):
    printV0("Warning: looks like constructor, but syntax is not matching")
    return 1

  found = nopickle.search(line)
  if found:
    typename = found.group("typename")
    addClassDef(classdefs, typename, parsedFile)
    classdefs[typename].methods["__reduce__"] = MethodDefinition(argkw="METH_NOARGS", cname="*yieldNoPickleError")
    return 1

def detectAttrs(line, classdefs):
  found = getdef.search(line) or setdef.search(line)
  if found:
    typename, attrname, method, doc = found.group("typename", "attrname", "method", "doc")
    printV2("%s: definition of %s_%s", (typename, method, attrname))
    addClassDef(classdefs, typename, parsedFile)
    if not classdefs[typename].properties.has_key(attrname):
      classdefs[typename].properties[attrname]=AttributeDefinition()
    setattr(classdefs[typename].properties[attrname], "has"+method, 1)
    if doc:
      classdefs[typename].properties[attrname].description=doc
    return

  found=classconstantintdef.search(line)
  if found:
    typename, constname, constant = found.group("typename", "constname", "constant")
    printV2("%s: constant definition (%s)", (typename, constname))
    addClassDef(classdefs, typename, parsedFile)
    if not classdefs[typename].constants.has_key(constname):
      classdefs[typename].constants[constname] = ConstantDefinition(ccode=("PyInt_FromLong((long)(%s))" % constant))
    else:
      printV0("Warning: constant %s.%s duplicated", (typename, constname))
    return

  found=classconstantfloatdef.search(line)
  if found:
    typename, constname, constant = found.group("typename", "constname", "constant")
    printV2("%s: constant definition (%s)", (typename, constname))
    addClassDef(classdefs, typename, parsedFile)
    if not classdefs[typename].constants.has_key(constname):
      classdefs[typename].constants[constname] = ConstantDefinition(ccode=("PyFloat_FromDouble((double)(%s))" % constant))
    else:
      printV0("Warning: constant %s.%s duplicated", (typename, constname))
    return

  found=classconstantdef.search(line)
  if found:
    typename, constname, constant = found.group("typename", "constname", "constant")
    printV2("%s: constant definition (%s)", (typename, constname))
    addClassDef(classdefs, typename, parsedFile)
    if not classdefs[typename].constants.has_key(constname):
      classdefs[typename].constants[constname] = ConstantDefinition(ccode=constant)
    else:
      printV0("Warning: constant %s.%s duplicated", (typename, constname))
    return

  found=recognizedattrsdef.search(line)
  if found:
    typename, attributes = found.group("typename", "attributes")
    addClassDef(classdefs, typename, parsedFile, "recognized_attributes", attributes.split())


def detectMethods(line, classdefs):
  # The below if is to avoid methods, like, for instance, map's clear to be recognized
  # also as a special method. Special methods never include PYARGS.
  if line.find("PYARGS")<0:
    found=specialmethoddef.search(line)
    if found:
      typename, methodname = found.group("typename", "methodname")
      addClassDef(classdefs, typename, parsedFile)
      classdefs[typename].specialmethods[methodname]=1

  found=reducedef.search(line)
  typename = None
  if found:
    typename = found.group("typename")
    methodname = "__reduce__"
    cname = "_reduce__"
    argkw = "METH_NOARGS"
    doc = "()"
    
  else:
    found=methoddef.search(line)
    if found:
      typename, cname, methodname, argkw, doc = found.group("typename", "cname", "methodname", "argkw", "doc")

  if typename:  
    if not classdefs.has_key(typename) and "_" in typename:
      com = typename.split("_")
      for i in range(1, len(com)):
          subname = "_".join(com[:-i]) 
          if subname in classdefs:
              typename = subname
              cname = "_".join(com[-i:])+"_"+cname
              break

    if not methodname:
      methodname = camel2underscore(cname)

      
    addClassDef(classdefs, typename, parsedFile)
    classdefs[typename].methods[methodname]=MethodDefinition(argkw=argkw, arguments=doc, cname=cname)
    return 1

def detectHierarchy(line, classdefs):
  found=datastructuredef.search(line)
  if found:
    typename, structurename, dictfield = found.group("typename", "structurename", "dictfield")
    addClassDef(classdefs, typename, parsedFile, "datastructure", structurename)
    addClassDef(classdefs, typename, parsedFile, "dictfield", dictfield, 0)
    printV2("%s: definition/declaration of datastructure", typename)
    return 1
  
  found=newbasedondef.match(line)
  if found:
    typename, basename, doc = found.group("typename", "basename", "doc")
    allows_empty_args = bool(found.group("allows_empty_args"))
    addClassDef(classdefs, typename, parsedFile, "basetype", basename, 0)
    addClassDef(classdefs, typename, parsedFile, "constructor", ConstructorDefinition(arguments=doc, type="MANUAL", allows_empty_args=allows_empty_args))
    return 1

  found=basedondef.match(line)
  if found:
    typename, basename = found.group("typename", "basename")
    addClassDef(classdefs, typename, parsedFile, "basetype", basename, 0)
    return 1

  found=hiddendef.match(line)
  if found:
    typename, basename = found.group("typename", "basename")
    addClassDef(classdefs, typename, parsedFile, "basetype", basename, 0)
    classdefs[typename].hidden = 1
    return 1


def detectCallDoc(line, classdefs):
  found=calldocdef.search(line)
  if found:
    typename, doc = found.group("typename", "doc")
    printV2("%s: definition/declaration of description" % typename)
    addClassDef(classdefs, typename, parsedFile, "description", doc)
    return 1



def detectFunctions(line, functiondefs):     
  found=funcdef2.search(line)
  if found:
    defpart, pyname, argkw, doc =found.group("defpart", "pyname", "argkw", "doc")
    printV2("%s: function definition (%s)", (pyname, pyname))
    functiondefs[pyname]=FunctionDefinition(cname=pyname, argkw=argkw, definition=defpart, arguments=doc)
    return 1
  if funcwarndef.search(line):
    printV0("Warning: looks like function, but syntax is not matching")


def detectConstants(line, constantdefs):
  found=constantdef.search(line)
  if found:
    pyname, ccode = found.group("pyname", "ccode")
    printV2("%s: constant definition (%s)", (pyname, ccode))
    constantdefs[pyname]=ConstantDefinition(ccode=ccode)
    #constantdefs.append((pyname, ccode))
    return 1

  found=constantintdef.search(line)
  if found:
    pyname, constant = found.group("pyname", "constant")
    printV2("%s: constant definition (%s)", (pyname, constant))
    constantdefs[pyname]=ConstantDefinition(ccode=("PyInt_FromLong((long)(%s))" % constant))
    #constantdefs.append((pyname, ccode))
    return 1

  found=constantfloatdef.search(line)
  if found:
    pyname, constant = found.group("pyname", "constant")
    printV2("%s: constant definition (%s)", (pyname, constant))
    constantdefs[pyname]=ConstantDefinition(ccode=("PyFloat_FromDouble((double)(%s))" % constant))
    #constantdefs.append((pyname, ccode))
    return 1

  found=constantfuncdef.search(line)
  if found:
    pyname, cfunc = found.group("pyname", "cfunc")
    printV2("%s: constant returning definition (%s)", (pyname, cfunc))
    constantdefs[pyname]=ConstantDefinition(cfunc=cfunc)
    #constantdefs.append((pyname, cfunc, 0))
    return 1

  if constantwarndef.search(line):
    printV0("Warning: looks like constant, but syntax is not matching")



def parseFiles():
  global parsedFile
  functions, constants, classdefs = {}, {}, {}

  for l in libraries:
    f = open(l, "rt")
    cd = pickle.load(f)
    #ignore functions and constants - we don't need them
    for c in cd.values():
      c.imported = True
    classdefs.update(cd)
    
    
  aliases=readAliases()  

  filenamedef=re.compile(r"(?P<stem>.*)\.\w*$")
  for parsedFile in filenames:
    found=filenamedef.match(parsedFile)
    if found:
      filestem=found.group("stem")
    else:
      filestem=parsedFile

    infile=open(parsedFile, "rt")
    printNQ("Parsing " + parsedFile)
    global lineno
    lineno=0

    for line in infile:
      lineno=lineno+1
      if line.strip().startswith("PYXTRACT_IGNORE"):
        continue
      detectHierarchy(line, classdefs) # BASED_ON, DATASTRUCTURE those lines get detected twice!
      # detectMethods must be before detectAttrs to allow for methods named get_something, set_something
      for i in [detectConstructors, detectMethods, detectAttrs, detectCallDoc]:
        if i(line, classdefs):
          break
      else:
        detectFunctions(line, functions)
        detectConstants(line, constants)

    infile.close()

  classdefsEffects(classdefs)


  return functions, constants, classdefs, aliases

def findDataStructure(classdefs, typename):
  rtypename = typename
  while typename and typename!="ROOT" and classdefs.has_key(typename) and not classdefs[typename].datastructure:
    typename = classdefs[typename].basetype
    if not classdefs.has_key(typename):
      return None
    if classdefs[typename].imported:
      classdefs[typename].used = True
  if not typename or not classdefs.has_key(typename) or typename=="ROOT":
    return None
  else:
    return classdefs[typename].datastructure

def findConstructorDoc(classdefs, typename):
  while typename and typename!="ROOT" and classdefs.has_key(typename) and not classdefs[typename].constructor:
    typename=classdefs[typename].basetype
  if not typename or not classdefs.has_key(typename) or typename=="ROOT":
    return "<abstract class>"
  else:
    return classdefs[typename].constructor.arguments

def classdefsEffects(classdefs):
  ks=classdefs.keys()        
  for typename in ks:
    classdef=classdefs[typename]
    classdef.datastructure = findDataStructure(classdefs, typename)
    if not classdefs[typename].datastructure:
      printNQ("Warning: %s looked like a class, but is ignored since no corresponding data structure was found" % typename)
      del classdefs[typename]
      
    scs = pyprops and pyprops.get("T"+typename, None)
    classdef.subconstants = scs and scs.constants
    if scs:
      for consttype, valuelist in classdef.subconstants.items():
        for k, v in valuelist:
          classdef.constants[k]=ConstantDefinition(ccode=("Py%s_%s_FromLong((long)(%s))" % (typename, consttype, v)))
#        classdef.constants[consttype]=ConstantDefinition(ccode=("(PyObject *)&Py%s_%s_Type" % (typename, consttype)))
      
      


def readAliases():
  aliases={}
  if os.path.isfile("_aliases.txt"):
    f=open("_aliases.txt", "rt")
    actClass, aliasList = "", []
    for line in f:
      ss=line.split()
      if len(ss):
        if len(ss)==2:
          aliasList.append(ss)
        if (len(ss)==1) or (len(ss)==3):
          if actClass and len(aliasList):
            aliases[actClass]=aliasList
          actClass=ss[0]
          aliasList=[]

    if actClass and len(aliasList):
      aliases[actClass]=aliasList
    f.close()
    
  return aliases
        
def findSpecialMethod(classdefs, type, name):
  while 1:
    if classdefs[type].specialmethods.has_key(name):
      return type
    elif classdefs[type].basetype:
      type=classdefs[type].basetype
    else:
      return None  


def writeAppendix(filename, targetname, classdefs, aliases):
  if (    not recreate
      and  os.path.isfile(targetname)
      and (os.path.getmtime(targetname)>=os.path.getmtime(filename))
      and (os.path.getmtime(targetname)>=os.path.getmtime("aliases.txt"))):
    printV1("\nFile unchanged, skipping.")
    return

  usedbases={}
  classdefi=classdefs.items()
  classdefi.sort(lambda x,y:cmp(x[0],y[0]))
  basecount=0
  for (type, fields) in classdefi:
    if fields.infile==filename:
      usedbases[fields.basetype]=1
    basecount += 1
  if usedbases.has_key("ROOT"):
    del usedbases["ROOT"]

  if not basecount:
    if os.path.isfile(targetname):
      os.remove(targetname)
      printV1("\nFile does not define any classes, removing.")
    else:
      printV1("\nFile does not define any classes, skipping.")
    return

  printV1("\nConstructing class definitions")
  
  outfile=open("px/"+targetname+".new", "wt")
  newfiles.append(targetname)
  outfile.write("/* This file was generated by pyxtract \n   Do not edit.*/\n\n")

  usedbases=usedbases.keys()
  usedbases.sort()
  #outfile.write("extern TOrangeType PyOrOrangeType_Type;\n")
  for type in usedbases:
    if type:
      if classdefs[type].imported:
        outfile.write("extern IMPORT_DLL TOrangeType PyOr"+type+"_Type;\n")
      else:
        outfile.write("extern %s_API TOrangeType PyOr%s_Type;\n" % (modulename.upper(), type))
  outfile.write("\n\n")

  for (type, fields) in classdefi:
    if fields.infile!=filename:
      continue

    outfile.write('/* -------------- %s --------------*/\n\n' % type)

    # Write PyMethodDef
    if len(fields.methods):
      methodnames=fields.methods.keys()
      methodnames.sort()
      outfile.write("PyMethodDef "+type+"_methods[] = {\n")
      for methodname in methodnames:
        method=fields.methods[methodname]
        cname = method.cname[0] == "*" and method.cname[1:] or type+"_"+method.cname
        if method.arguments:
          outfile.write('     {"'+methodname+'", (binaryfunc)'+cname+", "+method.argkw+", \""+method.arguments+"\"},\n")
        else:
          outfile.write('     {"'+methodname+'", (binaryfunc)'+cname+", "+method.argkw+"},\n")
      outfile.write("     {NULL, NULL}\n};\n\n")
      

    # Write GetSetDef
    properties=filter(lambda (name, definition): not definition.builtin, fields.properties.items())
    if len(properties):
      properties.sort(lambda x,y:cmp(x[0], y[0]))
      outfile.write("PyGetSetDef "+type+"_getset[]=  {\n")
      for (name, definition) in properties:
        camelname = camel2underscore(name)
        outfile.write('  {"%s"' % camelname)
        if definition.hasget:
          outfile.write(", (getter)%s_get_%s" % (type, name))
        else:
          outfile.write(", NULL")
        if definition.hasset:
          outfile.write(", (setter)%s_set_%s" % (type, name))
        else:
          outfile.write(", NULL")
        if definition.description:
          outfile.write(', "'+definition.description+'"')
        outfile.write("},\n")
      outfile.write('  {NULL}};\n\n')

    # Write doc strings
    if fields.call and fields.call.arguments and len(fields.call.arguments):
      outfile.write('char '+type+'[] = "'+fields.call.arguments+'";\n')
    if fields.description:
      outfile.write('char '+type+'_doc[] = "'+fields.description+'";\n')
    outfile.write('\n')

    if fields.subconstants:
      for constname, constvalues in fields.subconstants.items():
        outfile.write("""
TNamedConstantsDef %(wholename)s_values[] = {%(valueslist)s, {0, 0}};
static PyObject *%(wholename)s_repr(PyObject *self) { return stringFromList(self, %(wholename)s_values); }
PyObject *%(wholename)s__reduce__(PyObject *self);
PyMethodDef %(wholename)s_methods[] = { {"__reduce__", (binaryfunc)%(wholename)s__reduce__, METH_NOARGS, "reduce"}, {NULL, NULL}};
PyTypeObject Py%(wholename)s_Type = {PyObject_HEAD_INIT(&PyType_Type) 0, "%(classname)s.%(constname)s", sizeof(PyIntObject), 0, 0, 0, 0, 0, 0, (reprfunc)%(wholename)s_repr, 0, 0, 0, 0, 0, (reprfunc)%(wholename)s_repr, 0, 0, 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES, 0, 0, 0, 0, 0, 0, 0, %(wholename)s_methods, 0, 0, &PyInt_Type, 0, 0, 0, 0, 0, 0, 0, PyObject_Del};
PyObject *Py%(wholename)s_FromLong(long ok) { PyIntObject *r = PyObject_New(PyIntObject, &Py%(wholename)s_Type); r->ob_ival = ok; return (PyObject *)r; }
void *PT%(wholename)s(void *l) { return Py%(wholename)s_FromLong(*(int *)l); }
PyObject *%(wholename)s__reduce__(PyObject *self) { return Py_BuildValue("O(s(i))", getExportedFunction("__pickleLoaderNamedConstants"), "%(wholename)s", ((PyIntObject *)(self))->ob_ival); }

""" % {"wholename": type+"_"+constname, "classname": type, "constname": constname, "valueslist": ", ".join('{"%s", %s}' % k for k in constvalues)})

#PyObject *%(wholename)s__reduce__(PyObject *self) { return Py_BuildValue("O(i)", &PyInt_Type, ((PyIntObject *)(self))->ob_ival); }




    # Write constants
    if fields.constants:
      outfile.write("void %s_addConstants()\n{ PyObject *&dict = PyOr%s_Type.ot_inherited.tp_dict;\n  if (!dict) dict = PyDict_New();\n" % (type, type))
      for name, const in fields.constants.items():
        if const.ccode:
          outfile.write('  PyDict_SetItemString(dict, "%s", %s);\n' % (name, const.ccode))
        else:
          outfile.write('  PyDict_SetItemString(dict, "%s", %s());\n' % (name, const.cfunc))
      outfile.write("}\n\n")

    # Write default constructor
    if fields.constructor:
      if fields.constructor.type!="MANUAL":
        outfile.write('POrange %s_default_constructor(PyTypeObject *type)\n{ return POrange(mlnew T%s(), type); }\n\n' % (type, type))
    else:
      outfile.write('PyObject *%s_abstract_constructor(PyTypeObject *type, PyObject *args, PyObject *kwds)\n{ return PyOrType_GenericAbstract((PyTypeObject *)&PyOr%s_Type, type, args, kwds); }\n\n' % (type, type))

    # Write constructor keywords
    if fields.constructor_keywords:
      outfile.write('char *%s_constructor_keywords[] = {%s, NULL};\n' % (type, reduce(lambda x, y: x + ", " + y, ['"%s"' % x for x in fields.constructor_keywords])))

    if fields.recognized_attributes:
      outfile.write('char *%s_recognized_attributes[] = {%s, NULL};\n' % (type, reduce(lambda x, y: x + ", " + y, ['"%s"' % x for x in fields.recognized_attributes])))
      
    outfile.write('\n')                    
                    
    # Write aliases    
    if aliases.has_key(type):
      outfile.write("TAttributeAlias "+type+"_aliases[] = {\n")
      for alias in aliases[type]:
        outfile.write('    {"%s", "%s"},\n' % tuple(alias))
      outfile.write("    {NULL, NULL}};\n\n")
    
    # Write type object  

    def hasany(methods, fields):
      for smethod in methods:
        if smethod[0] and fields.specialmethods.has_key(smethod[0]):
          return 1
      return 0

    def writeslots(methods, isbase=0):
      def write0(innulls):
        outfile.write(innulls and ' 0,' or '  0,')
        return 1

      innulls=0
      for smethod in methods:
        if not smethod[0]:
          if smethod[1]=="BASE":
            if fields.basetype and (fields.basetype!="ROOT"):
              name='(_typeobject *)&PyOr'+fields.basetype+'_Type,'
              innulls=outfile.write((innulls and '\n' or '') + ('  %-50s /* tp_base */\n' % name))
            else:
              innulls=write0(innulls)

          elif smethod[1]=="DICTOFFSET":
            if fields.dictfield and fields.dictfield!="0":
              innulls=outfile.write((innulls and '\n' or '') + ('  %-50s /* tp_dictoffset */\n' % ("offsetof(%s, %s)," % (fields.datastructure, fields.dictfield))))
            else:
              innulls=write0(innulls)

          elif smethod[1]=="DOC":
            innulls=outfile.write((innulls and '\n' or '') + ('  %-50s /* tp_doc */\n' % ('"'+findConstructorDoc(classdefs, type)+'",')))

          elif smethod[1]=="FLAGS":
            fl = "Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_RICHCOMPARE"
            for cond, flag in [(fields.specialmethods.has_key("traverse"), "Py_TPFLAGS_HAVE_GC")
                              ]:
              if cond:
                fl += " | "+flag
            innulls=outfile.write((innulls and '\n' or '') + ('  %s, /* tp_flags */\n' % fl))
                        
          else:
            otherfields= [("as_number",    hasnumeric),
                          ("as_sequence",  hassequence),
                          ("as_mapping",   hasmapping),
                          ("doc",          fields.description),
                          ("methods",      fields.methods),
                          ("getset",       len(properties))
                         ] 
                      
            for (name, condition) in otherfields:
              if smethod[1]==name:
                if condition:
                  slotcont='%s_%s,' % (type, name)
                  if name[:3]=="as_":
                     slotcont="&"+slotcont
                  innulls=outfile.write((innulls and '\n' or '') + ('  %-50s /* tp_%s */\n' % (slotcont, name)))
                else:
                  innulls=write0(innulls)
                break
            else:
              if len(smethod)==3:
                if smethod[2]=='0':
                  innulls=write0(innulls)
                else:
                  innulls=outfile.write((innulls and '\n' or '') + ('  %-50s /* %s */\n' % (smethod[2]+',', smethod[1])))
              else:
                raise "invalid slot name %s" % smethod[1]
              
        else: # smethod[0]!=""
          if fields.specialmethods.has_key(smethod[0]):
            innulls=outfile.write((innulls and '\n' or '') + ('  %-50s /* %s */\n' % ("(%s)%s_%s," % (smethod[2], type, smethod[0]), smethod[1])))
            
          elif smethod[0]=="new":
            if fields.constructor:
              innulls = outfile.write((innulls and '\n' or '') + ('  %-50s /* tp_new */\n' % ("(newfunc)%s," % genericconstrs[fields.constructor.type])))
            else:
              innulls = outfile.write((innulls and '\n' or '') + ('  %-50s /* tp_new */\n' % ("(newfunc)%s_abstract_constructor," % type)))

          else:
            innulls=write0(innulls)
      
      return innulls


    additional=[]
    for (subtype, submethods, subappendix) in [('PyNumberMethods', specialnumericmethods, '_as_number'),
                                               ('PySequenceMethods', specialsequencemethods, '_as_sequence'),
                                               ('PyMappingMethods', specialmappingmethods, '_as_mapping')]:
      hasit = hasany(submethods, fields)
      additional.append(hasit)
      if hasit:
        outfile.write(subtype+' '+type+subappendix +' = {\n')
        innulls=writeslots(submethods, 0)
        outfile.write((innulls and '\n' or '') + '};\n\n')

    hasnumeric, hassequence, hasmapping = tuple(additional)

    outfile.write('PyTypeObject PyOr'+type+'_Type_inh = {\n')
    outfile.write('  PyObject_HEAD_INIT((_typeobject *)&PyType_Type)\n')
    outfile.write('  0,\n')
    outfile.write('  "%s.%s",\n' % (modulename, type))
    outfile.write('  sizeof(%s), 0,\n' % fields.datastructure)
    innulls=writeslots(specialmethods, 1)
    outfile.write((innulls and '\n' or '') + '};\n\n')

    if fields.datastructure == "TPyOrange":
      outfile.write(cc_functions % {"type": type})

    outfile.write('%(modulename)s_API TOrangeType PyOr%(type)s_Type (PyOr%(type)s_Type_inh, typeid(T%(type)s)' % {"modulename": modulename.upper(), "type": type})
    outfile.write(', ' + (fields.constructor and fields.constructor.type!="MANUAL" and type+'_default_constructor' or '0'))
    if fields.datastructure == "TPyOrange":
      outfile.write(', cc_%s, ccn_%s' % (type, type))
    else:
      outfile.write(', PyOr_noConversion, PyOr_noConversion')
    outfile.write(', ' + (fields.constructor_keywords and type+'_constructor_keywords' or 'NULL'))
    outfile.write(', ' + (fields.constructor and fields.constructor.allows_empty_args and 'true' or 'false'))
    outfile.write(', ' + (fields.recognized_attributes and type+'_recognized_attributes' or 'NULL'))
    outfile.write(', ' + (aliases.has_key(type) and type+'_aliases' or 'NULL'))
    outfile.write(');\n\n\n\n')

    if not (fields.abstract or fields.constructor and fields.constructor.allows_empty_args or fields.methods.has_key("__reduce__")):
      printV0("Warning: class '%s' will not be picklable", type, False)
      
  outfile.close()


def writeAppendices(classdefs):
  filenamedef=re.compile(r"(?P<stem>.*)\.\w*$")
  for filename in filenames:
    found=filenamedef.match(filename)
    if found:
      filestem=found.group("stem")
    else:
      filestem=filename
    writeAppendix(filename, filestem+".px", classdefs, aliases)
    printV1()

def writeExterns():
  externsfile=open("px/externs.px.new", "wt")
  newfiles.append("externs.px")

  externsfile.write("/* This file was generated by pyxtract \n   Do not edit.*/\n\n")

  externsfile.write("#ifdef _MSC_VER\n  #define IMPORT_DLL __declspec(dllimport)\n#else\n  #define IMPORT_DLL\n#endif\n\n")
  ks=classdefs.keys()
  ks.sort()
  for type in ks:

    if not classdefs[type].imported:
      externsfile.write("extern %s_API TOrangeType PyOr%s_Type;\n" % (modulename.upper(), type))
    else:
      externsfile.write("extern IMPORT_DLL TOrangeType PyOr%s_Type;\n" % type)

    externsfile.write('#define PyOr%s_Check(op) PyObject_TypeCheck(op, (PyTypeObject *)&PyOr%s_Type)\n' % (type, type))
    if classdefs[type].datastructure == "TPyOrange":
      externsfile.write('#define PyOrange_As%s(op) (*(GCPtr< T%s > *)(void *)(&PyOrange_AS_Orange(op)))\n' % (type, type))
      externsfile.write('\n')

  classdefi=classdefs.items()
  classdefi.sort(lambda x,y:cmp(x[0],y[0]))
  externsfile.write("#if defined(%s_EXPORTS) || !defined(_MSC_VER)\n\n" % modulename.upper())  
  for (type, fields) in classdefi:
    if fields.datastructure == "TPyOrange":
      if not fields.imported: 
        externsfile.write("  int cc_func_"+type+"(PyObject *, void *);\n")
        externsfile.write("  int ccn_func_"+type+"(PyObject *, void *);\n")
        externsfile.write("  #define cc_%s cc_func_%s\n" % (type, type))
        externsfile.write("  #define ccn_%s ccn_func_%s\n\n" % (type, type))
      else:
        externsfile.write("  #define cc_%s PyOr%s_Type.ot_converter\n" % (type, type))
        externsfile.write("  #define ccn_%s PyOr%s_Type.ot_nconverter\n\n" % (type, type))
  externsfile.write("#else\n\n")
  for (type, fields) in classdefi:
    if not fields.imported and fields.datastructure == "TPyOrange":
      externsfile.write("  #define cc_%s PyOr%s_Type.ot_converter\n" % (type, type))
      externsfile.write("  #define ccn_%s PyOr%s_Type.ot_nconverter\n\n" % (type, type))
  externsfile.write("#endif\n")

  externsfile.write("\n\n")  

  externsfile.close()


def writeInitialization(functions, constants):
  functionsfile=open("px/initialization.px.new", "wt")
  newfiles.append("initialization.px")
  functionsfile.write("/* This file was generated by pyxtract \n   Do not edit.*/\n\n")
  functionsfile.write('#include "externs.px"\n\n')

  myclasses = dict(filter(lambda x:not x[1].imported, classdefs.items()))

  classconstants = [(type, fields.subconstants) for type, fields in myclasses.items() if fields.subconstants]
  if classconstants:
    functions["__pickleLoaderNamedConstants"] = FunctionDefinition(cname="__pickleLoaderNamedConstants", argkw="METH_VARARGS", arguments="")
    
  if len(functions):
    olist=functions.keys()
    olist.sort()
    for functionname in olist:
      function=functions[functionname]
      if function.definition:
        functionsfile.write(function.definition+";\n")
      else:
        if keywargsdef.search(function.argkw):
          functionsfile.write("PyObject *"+function.cname+"(PyObject *, PyObject *, PyObject *);\n")
        else:
          functionsfile.write("PyObject *"+function.cname+"(PyObject *, PyObject *);\n")
  else:
    olist = []

  printV1NoNL("\nFunctions:")
  functionsfile.write("\n\nPyMethodDef %sFunctions[]={\n" % modulename)
  for functionname in olist:
    function=functions[functionname]
    printV1NoNL(functionname)
    if function.arguments:
      functionsfile.write('     {"'+functionname+'", (binaryfunc)'+function.cname+', '+function.argkw+', "'+function.arguments+'"},\n')
    else:
      functionsfile.write('     {"'+functionname+'", (binaryfunc)'+function.cname+', '+function.argkw+'},\n')
  functionsfile.write("     {NULL, NULL}\n};\n\n")
  printV1()
    

  olist=constants.keys()
  olist.sort()

  for constantname in olist:
    constant=constants[constantname]
    if constant.cfunc:
      functionsfile.write("PyObject *"+constant.cfunc+"();\n")

  addconstscalls = ""
  if len(myclasses):
    ks=myclasses.keys()
    ks.sort()
    printV1NoNL("\nClasses:")
    functionsfile.write("int noOf%sClasses=%i;\n\n" % (modulename, len(ks)))
    functionsfile.write("TOrangeType *%sClasses[]={\n" % modulename)
    for i in ks:
      functionsfile.write("    &PyOr%s_Type,\n" % i)
      printV1NoNL(i)
    functionsfile.write("    NULL};\n\n")
    printV1()

    for i in ks:
      if classdefs[i].constants:
        functionsfile.write('\nvoid %s_addConstants();' % i)
        addconstscalls += ('     %s_addConstants();\n' % i)
  functionsfile.write("\n")

  printV1NoNL("\nConstants:")
  if classconstants:
    for type, subconstants in classconstants:
      for constname in subconstants:
        functionsfile.write('extern PyTypeObject Py%s_Type;\n' % (type+"_"+constname))

    functionsfile.write("\nTNamedConstantRecord %sNamedConstants[] = {\n" % modulename)
    for type, subconstants in classconstants:
      for constname in subconstants:
        functionsfile.write('    {"%s_%s", &Py%s_%s_Type},\n' % (type, constname, type, constname))
    functionsfile.write('    {NULL, NULL}\n};\n\n')
    functionsfile.write("PyObject *__pickleLoaderNamedConstants(PyObject *, PyObject *args)\n{ return unpickleConstant(%sNamedConstants, args); }\n\n" % modulename)
    

  functionsfile.write("\nvoid add%sConstants(PyObject *mod) {\n" % modulename)
  if olist:
    for constantname in olist:
      constant=constants[constantname]
      printV1NoNL(constantname)
      if constant.cfunc:
        functionsfile.write('     PyModule_AddObject(mod, "'+constantname+'", '+constant.cfunc+'());\n')
      else:
        functionsfile.write('     PyModule_AddObject(mod, "'+constantname+'", '+constant.ccode+');\n')
    functionsfile.write("\n\n")

  if addconstscalls:
    functionsfile.write(addconstscalls + "\n\n")

  if len(myclasses):
    for classname in ks:
      if not classdefs[i].hidden:
        functionsfile.write('     PyModule_AddObject(mod, "%s", (PyObject *)&PyOr%s_Type);\n' % (classname, classname))

  for type, fields in myclasses.items():
      if fields.subconstants:
        for constname in fields.subconstants:
          functionsfile.write('\n     PyType_Ready(&Py%(wholename)s_Type);\n     Py%(wholename)s_Type.tp_print = 0;\n' %
                              {"wholename": type+"_"+constname, "classname": type, "constname": constname})

  functionsfile.write("}\n\n")

  printV1("\n")

  functionsfile.write("""
#ifdef _MSC_VER
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  BOOL APIENTRY DllMain( HANDLE, DWORD  ul_reason_for_call, LPVOID)  { return TRUE; }
#endif

extern %(MODULENAME)s_API PyObject *%(modulename)sModule;

ORANGE_API void addClassList(TOrangeType **);

extern "C" %(MODULENAME)s_API void init%(modulename)s()
{ 
  if (!init%(modulename)sExceptions())
    return;
""" % {"modulename" : modulename, "MODULENAME": modulename.upper()})

  if len(myclasses):
    functionsfile.write("""
  for(TOrangeType **type=%(modulename)sClasses; *type; type++)
    if (PyType_Ready((PyTypeObject *)*type)<0)
      return;
  addClassList(%(modulename)sClasses);
""" % {"modulename" : modulename})

  functionsfile.write("""
  gc%(modulename)sUnsafeStaticInitialization();
  %(modulename)sModule = Py_InitModule("%(modulename)s", %(modulename)sFunctions);  
  add%(modulename)sConstants(%(modulename)sModule);
}
""" % {"modulename" : modulename, "MODULENAME": modulename.upper()})

  functionsfile.close()


def writeGlobals():
  newfiles.append("%s_globals.hpp" % modulename)

  globalsfile = file("px/%s_globals.hpp.new" % modulename, "wt")
  globalsfile.write(r"""/* This file was generated by pyxtract \n   Do not edit.*/

#ifdef _MSC_VER
    #define %(MODULENAME)s_EXTERN

    #ifdef %(MODULENAME)s_EXPORTS
        #define %(MODULENAME)s_API __declspec(dllexport)

    #else
        #define %(MODULENAME)s_API __declspec(dllimport)
        #ifdef _DEBUG
            #pragma comment(lib, "%(modulename)s_d.lib")
        #else
            #pragma comment(lib, "%(modulename)s.lib")
        #endif

    #endif // exports

    #define %(MODULENAME)s_VWRAPPER(x) \
        %(MODULENAME)s_EXTERN template class %(MODULENAME)s_API T##x; \
        %(MODULENAME)s_EXTERN template class %(MODULENAME)s_API GCPtr< T##x >; \
        typedef GCPtr< T##x > P##x;
   
    #define %(MODULENAME)s_WRAPPER(x) \
        class %(MODULENAME)s_API T##x; \
        %(MODULENAME)s_EXTERN template class %(MODULENAME)s_API GCPtr< T##x >; \
        typedef GCPtr< T##x > P##x;

#else
#ifdef DARWIN
    #define %(MODULENAME)s_API
    
    #ifdef %(MODULENAME)s_EXPORTS
        #define %(MODULENAME)s_EXTERN
    #else
        #define %(MODULENAME)s_EXTERN extern
    #endif

    #define %(MODULENAME)s_VWRAPPER(x) \
        typedef GCPtr< T##x > P##x;
 
    #define %(MODULENAME)s_WRAPPER(x) \
        class %(MODULENAME)s_API T##x; \
        typedef GCPtr< T##x > P##x;

#else // not _MSC_VER
    #define %(MODULENAME)s_API

    #ifdef %(MODULENAME)s_EXPORTS
        #define %(MODULENAME)s_EXTERN
    #else
        #define %(MODULENAME)s_EXTERN extern
    #endif

    #define %(MODULENAME)s_VWRAPPER(x) \
        %(MODULENAME)s_EXTERN template class %(MODULENAME)s_API T##x; \
        %(MODULENAME)s_EXTERN template class %(MODULENAME)s_API GCPtr< T##x >; \
        typedef GCPtr< T##x > P##x;
   
    #define %(MODULENAME)s_WRAPPER(x) \
        class %(MODULENAME)s_API T##x; \
        typedef GCPtr< T##x > P##x;

#endif
#endif
"""
 % {"modulename": modulename, "MODULENAME": modulename.upper()})

  if wrapper_shortcut != None:
    globalsfile.write("""
#define %(SHORTCUT)sVWRAPPER %(MODULENAME)s_VWRAPPER
#define %(SHORTCUT)sWRAPPER %(MODULENAME)s_WRAPPER
""" % {"SHORTCUT": wrapper_shortcut, "MODULENAME": modulename.upper()})

def samefiles(n1, n2):
  f1, f2 = open(n1, "rt"), open(n2, "rt")
  same = (f1.readlines()==f2.readlines())
  f1.close()
  f2.close()
  return same
  
def renewfiles(newfiles):
  for i in newfiles:
    oldexists=os.path.isfile("px/"+i)
    if recreate or not oldexists or not samefiles("px/"+i, "px/"+i+".new"):
      if oldexists:
        os.remove("px/"+i)
        printNQ("Renewing %s" % i)
      else:
        printNQ("Creating %s" % i)
      os.rename("px/"+i+".new", "px/"+i)
    else:
      os.remove("px/"+i+".new")
      printV1("Keeping %s", i)


#args=sys.argv[1:]


def make():
  if not os.path.isdir("px"):
    os.mkdir("px")
  writeAppendices(classdefs)
  writeExterns()
  writeInitialization(functions, constants)
  writeGlobals()
  renewfiles(newfiles)

  f=open("px/stamp", "wt")
  pickle.dump(classdefs, f)
  pickle.dump(functions, f)
  pickle.dump(constants, f)
  f.close()
  

def listOfExports():
  for name, c in classdefs.items():
    print "Class '%s', derived from %s" % (name, getattr(c, "basetype", "none (or unknown)"))
    if getattr(c, "constructor", None):
      print "\tConstructor visible from Python: %s" % c.constructor.arguments
    if getattr(c, "call", None):
      print "\tCallable from Python: %s" % c.call.arguments
    properties = ", ".join(getattr(c, "builtinproperties", {}).keys()) + ", ".join(getattr(c, "properties", {}).keys())
    if hasattr(c, "getattr"):
      if hasattr(c, "setattr"):
        print "\tSpecialized getattr and setattr"
      else:
        print "\tSpecialized getattr"
    elif hasattr(c, "setattr"):
        print "\tSpecialized setattr"
    if len(properties):
      print "\tProperties: %s" % reduce(lambda x, y: x+", "+y, properties)
    print

  print "\n\n\nFunctions"
  for func in functions:
    print "%s %s" % (func[0], func[-1])

  print "\n\n\nConstants"
  for cons in constants:
    print "%s" % cons[0]


def listNode(name, hier, level):
  print "     "*level + name
  for child in sorted(hier.get(name, [])):
    listNode(child, hier, level+1)
    
def listOfClasses():
  hier = {}
  for name, c in classdefs.items():
    base = getattr(c, "basetype", None)
    if base:
      if not base in hier:
        hier[base] = []
      hier[base].append(name)
  listNode("Orange", hier, 0)
  for name, c in classdefs.items():
    if not getattr(c, "basetype", None) or c.basetype=="ROOT":
      listNode(name, hier, 0)
      
    
  

def saferemove(fname):
  if os.path.isfile(fname):
    os.remove(fname)
    
def removeFiles():
  print "Removing externs.px, initialization.px,"
  saferemove("externs.px")
  saferemove("initialization.px")

  for filename in filenames:
    found=filenamedef.match(filename)
    if found:
      filestem=found.group("stem")
    else:
      filestem=filename
    print "%s," % (filestem+".px"),
    saferemove(filestem+".px")


def readArguments(args):
  global filenames, verbose, recreate, action, libraries, modulename, wrapper_shortcut, quiet
  filenames, libraries, verbose, recreate, modulename, wrapper_shortcut = [], [], 0, 0, "", None
  action, quiet = [], 0
  i=0
  while(i<len(args)):
    if args[i][0]=="-":
      opt=args[i][1:]
      if opt=="q":
        quiet = 1
      elif opt=="v":
        verbose=1
      elif opt=="V":
        verbose=2
      elif opt=="r":
        recreate=1
      elif opt=="c":
        action.append("clean")
      elif opt=="m":
        action.append("make")
      elif opt=="i":
        action.append("list")
      elif opt=="e":
        action.append("hierarchy")
      elif opt=="l":
        i=i+1
        libraries.append(args[i])
      elif opt=="n":
        i=i+1
        modulename = args[i].lower()
      elif opt=="d":
        import os
        i+=1
        os.chdir(args[i])
      elif opt=="w":
        i+=1
        wrapper_shortcut = args[i]
      else:
        print "Unrecognized option %s" % args[i]
    else:
      if not "pyxtract.py" in args[i]:
        filenames.append(args[i])
    i=i+1

  if not modulename:
    print "Module name (-n) missing"
    sys.exit()
  if not len(action):
    action=["make"]

def printNQ(str):
  if not quiet:
    print str
    
def printV0(str="", tup=(), printLine = True):
  if printLine:
    print "%20s:%4i:" % (parsedFile, lineno),
  print str % tup

def printV2(str="", tup=(), printLine = True):
  if verbose==2:
    printV0(str, tup, printLine)
    
def printV1(str="", tup=(), printLine = True):
  if verbose>=1:
    printV0(str, tup, printLine)

def printV1NoNL(str="", tup=()):
  if verbose>=1:
    print str % tup,
        
args = sys.argv

readArguments(args)
parsedFile=""

if action.count("clean"):
  removeFiles()
  action.remove("clean")

if len(action):
  pyprops = pickle.load(open("../orange/ppp/stamp", "rt")) #home dir on linux is different from windows; changed ppp/stamp to ../orange/ppp/stamp
  newfiles=[]
  functions, constants, classdefs, aliases = parseFiles()
    
  if action.count("list"):
    listOfExports()
  if action.count("hierarchy"):
    listOfClasses()
  if action.count("make"):
    make()




