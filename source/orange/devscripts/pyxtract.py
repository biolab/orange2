#!/usr/bin/env python
import re, os, sys, os.path, string
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

  specialorangemethods=[
                ]
                
               
if 1: ### Definitions of regular expressions

  constrdef_mac=re.compile(r'(?P<constype>ABSTRACT|C_UNNAMED|C_NAMED|C_CALL)\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<basename>\w*\s*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)')
  constrdef_mac_call3=re.compile(r'(?P<constype>C_CALL3)\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<callname>\w*\s*)\s*,\s*(?P<basename>\w*\s*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)')
  constrkeywords = re.compile(r'CONSTRUCTOR_KEYWORDS\s*\(\s*(?P<typename>\w*)\s*, \s*"(?P<keywords>[^"]*)"\s*\)')
  constrwarndef=re.compile("[^\w](ABSTRACT|CONS|C_UNNAMED|C_NAMED|C_CALL)[^\w]")

  datastructuredef=re.compile(r'DATASTRUCTURE\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<structurename>\w*)\s*,\s*(?P<dictfield>\w*)\s*\)')
  newbasedondef=re.compile(r'(inline)?\s*PyObject\s*\*(?P<typename>\w*)_new\s*\([^)]*\)\s*BASED_ON\s*\(\s*(?P<basename>\w*)\s*,\s*"(?P<doc>[^"]*)"\s*\)')
  basedondef=re.compile(r'BASED_ON\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<basename>\w*)\s*\)')
  hiddendef=re.compile(r'HIDDEN\s*\(\s*(?P<typename>\w*)\s*,\s*(?P<basename>\w*)\s*\)')
  

  allspecial=['('+m[0]+')' for m in specialmethods+specialnumericmethods+specialsequencemethods+specialmappingmethods+specialorangemethods]
  allspecial=filter(lambda x:x!='()', allspecial)
  allspecial=reduce(lambda x,y:x+'|'+y, allspecial)
  specialmethoddef=re.compile(r'((PyObject\s*\*)|(int)|(void))\s*(?P<typename>\w*)_(?P<methodname>'+allspecial+r')\s*\(')

  calldocdef=re.compile(r'PyObject\s*\*(?P<typename>\w*)_call\s*\([^)]*\)\s*PYDOC\s*\(\s*"(?P<doc>[^"]*)"\s*\)')
  getdef=re.compile(r'PyObject\s*\*(?P<typename>\w*)_(?P<method>get)_(?P<attrname>\w*)\s*\([^)]*\)\s*(PYDOC\(\s*"(?P<doc>[^"]*)"\s*\))?')
  setdef=re.compile(r'int\s*(?P<typename>\w*)_(?P<method>set)_(?P<attrname>\w*)\s*\([^)]*\)\s*(PYDOC\(\s*"(?P<doc>[^"]*)"\s*\))?')
  methoddef=re.compile(r'PyObject\s*\*(?P<typename>\w\w+)_(?P<methodname>\w*)\s*\([^)]*\)\s*PYARGS\((?P<argkw>[^),]*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)')

  funcdef=re.compile(r'PYFUNCTION\((?P<pyname>\w*)\s*,\s*(?P<cname>\w*)\s*,\s*(?P<argkw>[^,]*)\s*(,\s*"(?P<doc>[^"]*)"\))?[\s;]*$')
  funcdef2=re.compile(r'(?P<defpart>PyObject\s*\*(?P<pyname>\w*)\s*\([^)]*\))\s*;?\s*PYARGS\((?P<argkw>[^),]*)\s*(,\s*"(?P<doc>[^"]*)")?\s*\)')
  funcwarndef=re.compile("PYFUNCTION|PYARGS")
  keywargsdef=re.compile(r"METH_VARARGS\s*\|\s*METH_KEYWORDS")

  classconstantintdef=re.compile(r"PYCLASSCONSTANT_INT\((?P<typename>\w*)\s*,\s*(?P<constname>\w*)\s*,\s*(?P<constant>.*)\)\s*$")
  classconstantfloatdef=re.compile(r"PYCLASSCONSTANT_FLOAT\((?P<typename>\w*)\s*,\s*(?P<constname>\w*)\s*,\s*(?P<constant>.*)\)\s*$")
  constantdef=re.compile(r"PYCONSTANT\((?P<pyname>\w*)\s*,\s*(?P<ccode>.*)\)\s*$")
  constantfuncdef=re.compile(r"PYCONSTANTFUNC\((?P<pyname>\w*)\s*,\s*(?P<cfunc>.*)\)\s*$")
  constantwarndef=re.compile("PYCONSTANT")


def detectConstructors(line, classdefs):
  found=constrdef_mac.search(line)
  if not found:
    found=constrdef_mac_call3.search(line)
  if found:
    typename, basename, constype, doc=found.group("typename", "basename", "constype", "doc")
    printV2("%s (%s): Macro constructor %s", (typename, basename, constype))
    addClassDef(classdefs, typename, parsedFile, "basetype", basename)
    if (constype!="ABSTRACT"):
       addClassDef(classdefs, typename, parsedFile, "constructor", ConstructorDefinition(arguments=doc, type=constype))
    return 1
  
  found = constrkeywords.search(line)
  if found:
    typename, keywords = found.group("typename", "keywords")
    addClassDef(classdefs, typename, parsedFile, "constructor_keywords", keywords.split())
  
  if constrwarndef.search(line):
    printV0("Warning: looks like constructor, but syntax is not matching")


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
      classdefs[typename].constants[constname] = "PyInt_FromLong((long)(%s))" % constant
    else:
      printV0("Warning: constant %s.%s duplicated", (typename, constname))
    return

  found=classconstantfloatdef.search(line)
  if found:
    typename, constname, constant = found.group("typename", "constname", "constant")
    printV2("%s: constant definition (%s)", (typename, constname))
    addClassDef(classdefs, typename, parsedFile)
    if not classdefs[typename].constants.has_key(constname):
      classdefs[typename].constants[constname] = "PyFloat_FromDouble((double)(%s))" % constant
    else:
      printV0("Warning: constant %s.%s duplicated", (typename, constname))
    return
    
  
def detectMethods(line, classdefs):
  # The below if is to avoid methods, like, for instance, map's clear to be recognized
  # also as a special method. Special methods never include PYARGS...
  if line.find("PYARGS")<0:
    found=specialmethoddef.search(line)
    if found:
      typename, methodname = found.group("typename", "methodname")
      addClassDef(classdefs, typename, parsedFile)
      classdefs[typename].specialmethods[methodname]=1
    
  found=methoddef.search(line)
  if found:
    typename, methodname, argkw, doc = found.group("typename", "methodname", "argkw", "doc")

    if not classdefs.has_key(typename) and "_" in typename:
      com = typename.split("_")
      if len(com)==2 and classdefs.has_key(com[0]):
        typename = com[0]
        methodname = com[1] + "_" + methodname
      
    addClassDef(classdefs, typename, parsedFile)
    classdefs[typename].methods[methodname]=MethodDefinition(argkw=argkw, arguments=doc)
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
    addClassDef(classdefs, typename, parsedFile, "basetype", basename, 0)
    addClassDef(classdefs, typename, parsedFile, "constructor", ConstructorDefinition(arguments=doc, type="MANUAL"))
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
  found=funcdef.search(line)
  if found:
    pyname, cname, argkw, doc = found.group("pyname", "cname", "argkw", "doc")
    printV2("%s: function definition (%s)" , (pyname, cname))
    functiondefs[pyname]=FunctionDefinition(cname=cname, argkw=argkw, arguments=doc)
    return 1
  else:
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

  aliases=readAliases()  

  filenamedef=re.compile(r"(?P<stem>.*)\.\w*$")
  for parsedFile in filenames:
    found=filenamedef.match(parsedFile)
    if found:
      filestem=found.group("stem")
    else:
      filestem=parsedFile

    infile=open(parsedFile, "rt")
    print "Parsing", parsedFile
    global lineno
    lineno=0

    for line in infile:
      lineno=lineno+1
      if line[:15]=="PYXTRACT_IGNORE":
        continue
      detectHierarchy(line, classdefs) # BASED_ON, DATASTRUCTURE those lines get detected twice!
      for i in [detectConstructors, detectAttrs, detectMethods, detectCallDoc]:
        if i(line, classdefs):
          break
      else:
        detectFunctions(line, functions)
        detectConstants(line, constants)

    infile.close()

  classdefsEffects(classdefs)


  return functions, constants, classdefs, aliases

def findDataStructure(classdefs, typename):
  while typename and typename!="ROOT" and classdefs.has_key(typename) and not classdefs[typename].datastructure:
    typename=classdefs[typename].basetype
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
      print "Warning: %s looked like a class, but is ignored since no corresponding data structure was found" % typename
      del classdefs[typename]


def readAliases():
  f=open("devscripts/aliases.txt", "rt")
  aliases={}
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
  outfile.write("extern TOrangeType PyOrOrangeType_Type;\n")
  for type in usedbases:
    outfile.write("extern TOrangeType PyOr"+type+"_Type;\n")
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
        if method.arguments:
          outfile.write('     {"'+methodname+'", (binaryfunc)'+type+"_"+methodname+", "+method.argkw+", \""+method.arguments+"\"},\n")
        else:
          outfile.write('     {"'+methodname+'", (binaryfunc)'+type+"_"+methodname+", "+method.argkw+"},\n")
      outfile.write("     {NULL, NULL}\n};\n\n")
      

    # Write GetSetDef
    properties=filter(lambda (name, definition): not definition.builtin, fields.properties.items())
    if len(properties):
      properties.sort(lambda x,y:cmp(x[0], y[0]))
      outfile.write("PyGetSetDef "+type+"_getset[]=  {\n")
      for (name, definition) in properties:
        outfile.write('  {"%s"' % name)
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
      outfile.write('char '+type+'_call_doc[] = "'+fields.call.arguments+'";\n')
    if fields.description:
      outfile.write('char '+type+'_doc[] = "'+fields.description+'";\n')
    outfile.write('\n')

    # Write constants
    if fields.constants:
      outfile.write("void %s_addConstants()\n{ PyObject *&dict = PyOr%s_Type.ot_inherited.tp_dict;\n  if (!dict) dict = PyDict_New();\n" % (type, type))
      for i in fields.constants.items():
        outfile.write('  PyDict_SetItemString(dict, "%s", %s);\n' % i)
      outfile.write("}\n\n")

    # Write default constructor
    if fields.constructor and fields.constructor.type!="MANUAL":
      outfile.write('POrange %s_default_constructor(PyTypeObject *type)\n{ return POrange(mlnew T%s(), type); }\n' % (type, type))

    # Write constructor keywords
    if fields.constructor_keywords:
      outfile.write('char *%s_constructor_keywords[] = {%s, NULL};\n' % (type, reduce(lambda x, y: x + ", " + y, ['"%s"' % x for x in fields.constructor_keywords])))

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
            if not fields.constructor:
              innulls=write0(innulls)
            else:
              genericconstrs = {'C_UNNAMED': 'PyOrType_GenericNew',
                                'C_NAMED'  : 'PyOrType_GenericNamedNew',
                                'C_CALL'   : 'PyOrType_GenericCallableNew',
                                'C_CALL3'  : 'PyOrType_GenericCallableNew'}
              innulls = outfile.write((innulls and '\n' or '') + ('  %-50s /* tp_new */\n' % ("(newfunc)%s," % genericconstrs[fields.constructor.type])))

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
    outfile.write('  "'+type+'",\n')
    outfile.write('  sizeof(%s), 0,\n' % fields.datastructure)
    innulls=writeslots(specialmethods, 1)
    outfile.write((innulls and '\n' or '') + '};\n\n')

    outfile.write('TOrangeType PyOr'+type+'_Type (PyOr'+type+'_Type_inh, typeid(T'+type+')')
    outfile.write(', ' + (fields.constructor and fields.constructor.type!="MANUAL" and type+'_default_constructor' or '0'))
    outfile.write(', ' + (fields.constructor_keywords and type+'_constructor_keywords' or 'NULL'))
                          
    if aliases.has_key(type):
      outfile.write(', '+type+'_aliases')
    outfile.write(');\n\n')

    if fields.datastructure == "TPyOrange":
      outfile.write('DEFINE_cc('+type+')\n\n\n\n')
    
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

def writeExterns(targetname):
  externsfile=open("px/"+targetname+".new", "wt")
  newfiles.append(targetname)

  externsfile.write("/* This file was generated by pyxtract \n   Do not edit.*/\n\n")

  ks=classdefs.keys()
  ks.sort()
  for type in ks:
    externsfile.write("extern TOrangeType PyOr"+type+"_Type;\n")
    externsfile.write('#define PyOr%s_Check(op) PyObject_TypeCheck(op, (PyTypeObject *)&PyOr%s_Type)\n' % (type, type))
    externsfile.write("int cc_"+type+"(PyObject *, void *);\n")
    externsfile.write("int ccn_"+type+"(PyObject *, void *);\n")
    if classdefs[type].datastructure == "TPyOrange":
      externsfile.write('#define PyOrange_As%s(op) (GCPtr< T%s >(PyOrange_AS_Orange(op)))\n' % (type, type))
    externsfile.write('\n')
  externsfile.write("\n\n")

  externsfile.close()


def writeInitialization(targetname, functions, constants, externsname):
  functionsfile=open("px/"+targetname+".new", "wt")
  newfiles.append(targetname)
  functionsfile.write("/* This file was generated by pyxtract \n   Do not edit.*/\n\n")
  functionsfile.write('#include "'+externsname+'"\n\n')

  if len(classdefs):
    ks=classdefs.keys()
    ks.sort()
    printV1NoNL("\nClasses:")
    functionsfile.write("int noOfOrangeClasses=%i;\n\n" % len(ks))
    functionsfile.write("TOrangeType *orangeClasses[]={\n")
    for i in ks:
      functionsfile.write("    &PyOr%s_Type,\n" % i)
      printV1NoNL(i)
    functionsfile.write("    NULL};\n\n")
    printV1()

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

    printV1NoNL("\nFunctions:")
    functionsfile.write("\n\nPyMethodDef orangeFunctions[]={\n")
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

  for classname in ks:
    if classdefs[classname].constants:
      functionsfile.write('\nvoid %s_addConstants();' % classname)
  functionsfile.write("\n")

  printV1NoNL("\nConstants:")
  functionsfile.write("\nvoid addConstants(PyObject *mod) {\n")
  for constantname in olist:
    constant=constants[constantname]
    printV1NoNL(constantname)
    if constant.cfunc:
      functionsfile.write('     PyModule_AddObject(mod, "'+constantname+'", '+constant.cfunc+'());\n')
    else:
      functionsfile.write('     PyModule_AddObject(mod, "'+constantname+'", '+constant.ccode+');\n')
  functionsfile.write("\n\n")

  for classname in ks:
    if classdefs[classname].constants:
      functionsfile.write('     %s_addConstants();\n' % classname)
  functionsfile.write("\n\n")
  
  for classname in ks:
    if not classdefs[i].hidden:
      functionsfile.write('     PyModule_AddObject(mod, "%s", (PyObject *)&PyOr%s_Type);\n' % (classname, classname))
  functionsfile.write("}\n\n")
  printV1("\n")

  functionsfile.close()


def writeChanges(of, year, month, date, changes):
    of.write("  changesVector.push_back(\n")
    of.write("    TDateChanges(%d, %d, %d,\n" % (year+2000, month, date))

    for i in range(len(changes)):
        if len(changes[i]):
            break
    changes=changes[i:]

    for i in range(len(changes)-1, -1, -1):
        if len(changes[i]):
            break
    changes=changes[:i+1]
        
    for i in changes:
        of.write('        "%s\\n"\n' % i)
    of.write("  ));\n\n")


def parseChanges():
  f = open("devdoc/changes.txt", "rt")
  of= open("px/changes.px.new", "wt")
  newfiles.append("changes.px")

  of.write("/* This file was generated by pyxtract \n   Do not edit.*/\n\n")
  of.write("void initializeChangesVector() {\n")
  
  thisPack=[]
  year=None
  while 1:
    line=f.readline()
    if not len(line):
      break
    
    datedef=re.compile(r"(?P<year>\d\d)-(?P<month>\d\d)-(?P<date>\d\d)")                     
    found=datedef.search(line)
    if found:
      if year and len(thisPack):
        writeChanges(of, year, month, date, thisPack)
        thisPack=[]
      year, month, date = tuple(map(int, found.group("year", "month", "date")))
    else:
      if line[-1]=="\012":
        thisPack.append(line[:-1])
      else:
        thisPack.append(line)
    
  writeChanges(of, year, month, date, thisPack)

  of.write("}\n")
  of.close()
  f.close()

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
        print "Renewing %s" % i
      else:
        print "Creating %s" % i
      os.rename("px/"+i+".new", "px/"+i)
    else:
      os.remove("px/"+i+".new")
      printV1("Keeping %s", i)


#args=sys.argv[1:]


def make():
  if not os.path.isdir("px"):
    os.mkdir("px")
  writeAppendices(classdefs)
  writeExterns(externsname)
  writeInitialization(initializationname, functions, constants, externsname)
  #parseChanges()
  renewfiles(newfiles)

  f=open("px/stamp", "wt")
  f.close()
  

def listOfExports():
  for name, c in classdefs.items():
    print "Class '%s', derived from %s" % (name, c.get("basetype", "none (or unknown)"))
    if c.has_key("constructor"):
      print "\tConstructor visible from Python: %s" % c["constructor"].arguments
    if c.has_key("call"):
      print "\tCallable from Python: %s" % c["call"].arguments
    properties = c.get("builtinproperties", []) + string.split(c.get("properties", ""))
    if c.has_key("getattr"):
      if c.has_key("setattr"):
        print "\tSpecialized getattr and setattr"
      else:
        print "\tSpecialized getattr"
    elif c.has_key("setattr"):
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



def htmlString(s):
  d="%s" %s
  d=d.replace('<', '&lt;')
  d=d.replace('>', '&gt;')
  return d

def exportXML(filename):
  re_lastline=re.compile("\s+\Z", re.MULTILINE)
  def relal(s):
    return re_lastline.sub("", s or "")+"\n"
  
  xml=open(filename, "wt")
  olist=classdefs.items()
  olist.sort(lambda x,y:cmp(x[0], y[0]))

  xml.write('<ORANGEEXPORTS>\n')
  xml.write('<CLASSES>\n')
  for name, c in olist:
    xml.write('<CLASS name="%s">\n' % name)

    if c.description:
      xml.write('  <DESCRIPTION>')
      xml.write('    '+relal(c.description))
      xml.write('  </DESCRIPTION>\n')
    
    xml.write('  <ANCESTORS>\n')
    d=c
    while d and d.basetype:
      xml.write('    <BACK>%s</BACK>\n' % d.basetype)
      d=classdefs.get(d.basetype)
    xml.write('  </ANCESTORS>\n')
    xml.write('\n')

    properties = c.properties.items()
    properties.sort(lambda x,y: cmp(x[0], y[0]))
    xml.write('  <PROPERTIES>\n')
    for name, property in properties:
      xml.write('    <PROPERTY name="%s">' % name)
      xml.write('      '+relal(property.description))
      xml.write('    </PROPERTY>\n')
    xml.write('  </PROPERTIES>\n')
    xml.write('\n')

    methods = c.methods.items()
    methods.sort(lambda x,y: cmp(x[0], y[0]))
    xml.write('  <METHODS>\n')
    for name, method in methods:
      xml.write('    <METHOD name="%s" arguments="%s">' % (name, htmlString(method.arguments)))
      xml.write('      '+relal(method.description))
      xml.write('    </METHOD>\n')
    xml.write('  </METHODS>\n')
    xml.write('\n')

    if c.constructor:
      if c.constructor.arguments:
        xml.write('  <CONSTRUCTOR arguments="%s">' % htmlString(c.constructor.arguments))
      else:
        xml.write('  <CONSTRUCTOR>')
      xml.write('    '+relal(c.constructor.description))
      xml.write('  </CONSTRUCTOR>\n')
      xml.write('\n')

    if c.call:
      if c.call.arguments:
        xml.write('  <CALL arguments="%s">' % htmlString(c.call.arguments))
      else:
        xml.write('  <CALL>')
      xml.write('    '+relal(c.call.description))
      xml.write('  </CALL>\n')
      xml.write('\n')
    xml.write('</CLASS>\n')

    xml.write('\n')
    xml.write('\n')
  xml.write('</CLASSES>\n')

  olist=functions.items()
  olist.sort(lambda x,y:cmp(x[0], y[0]))
  xml.write('<FUNCTIONS>\n')
  for name, c in olist:
    if c.arguments:
      xml.write('<FUNCTION name="%s" arguments="%s">' % (name, htmlString(c.arguments)))
    else:
      xml.write('<FUNCTION name="%s">' % name)
    xml.write('  '+relal(c.description))
    xml.write('</FUNCTION>\n')
    xml.write('\n')
  xml.write('</FUNCTIONS>\n')
  xml.write('\n')

  olist=constants.items()
  olist.sort(lambda x,y:cmp(x[0], y[0]))
  xml.write('<CONSTANTS>\n')
  for name, c in olist:
    xml.write('<CONSTANT name="%s">' % name)
    xml.write('  '+relal(c.description))
    xml.write('</CONSTANT>')
    xml.write('\n')
  xml.write('</CONSTANTS>\n')
  xml.write('\n')
  xml.write('</ORANGEEXPORTS>\n')

  xml.close()  


def descriptionsFromXML(filename):
  def mergeText(node):
    return reduce(lambda x, y:x+y.toxml(), node.childNodes, "")
      
  def getChildrenByName(node, name):
    return filter(lambda child:child.nodeName==name, node.childNodes)
  
  from xml.dom.minidom import parse
  try:
    fle=open(filename, "rt")
  except:
    return
  
  dom1=parse(filename)
  
  domclasses=dom1.getElementsByTagName("CLASS")
  for domclass in domclasses:
    name=domclass.getAttribute("name")
    if classdefs.has_key(name):
      classdef=classdefs[name]
      desc=getChildrenByName(domclass, "DESCRIPTION")
      if len(desc):
        classdef.description=mergeText(desc[0])

      if classdef.constructor:
        cons=getChildrenByName(domclass, "CONSTRUCTOR")
        if len(cons):
          classdef.constructor.description=mergeText(cons[0])

      if classdefs.has_key("call"):
        cons=getChildrenByName(domclass, "CALL")
        if len(cons):
          classdefs.call.description=mergeText(cons[0])

      props=domclass.getElementsByTagName("PROPERTY")
      for prop in props:
        propname=prop.attributes.get("name").value
        if classdefs[name].properties and classdefs[name].properties.has_key(propname):
          classdefs[name].properties[propname].description=mergeText(prop)

  
def saferemove(fname):
  if os.path.isfile(fname):
    os.remove(fname)
    
def removeFiles():
  print "Removing %s, %s," % (externsname, initializationname),
  saferemove(externsname)
  saferemove(initializationname)

  for filename in filenames:
    found=filenamedef.match(filename)
    if found:
      filestem=found.group("stem")
    else:
      filestem=filename
    print "%s," % (filestem+".px"),
    saferemove(filestem+".px")


def readArguments(args):
  global filenames, verbose, recreate, externsname, initializationname, action, listname, xmlname
  filenames, verbose, recreate = [], 0, 0
  externsname, initializationname = "externs.px", "initialization.px"
  listname = xmlname = ""
  action = []
  i=0
  while(i<len(args)):
    if args[i][0]=="-":
      opt=args[i][1:]
      if opt=="v":
        verbose=1
      elif opt=="V":
        verbose=2
      elif opt=="r":
        recreate=1
      elif opt=="c":
        action.append("clean")
      elif opt=="m":
        action.append("make")
      elif opt=="l":
        action.append("list")
        i=i+1
        listname=args[i]
      elif opt=="x":
        action.append("xml")
        i=i+1
        xmlname=args[i]
      elif opt=="e":
        i=i+1
        externsname=args[i]
      elif opt=="i":
        i=i+1
        externsname=args[i]
      else:
        print "Unrecognized option %s" % args[i]
    else:
      filenames.append(args[i])
    i=i+1

  if not len(action):
    action=["make"]
    
def printV0(str="", tup=()):
  print "%20s:%4i:" % (parsedFile, lineno),
  print str % tup

def printV2(str="", tup=()):
  if verbose==2:
    printV0(str, tup)
    
def printV1(str="", tup=()):
  if verbose>=1:
    print str % tup

def printV1NoNL(str="", tup=()):
  if verbose>=1:
    print str % tup,
        

args=["lib_kernel.cpp", "lib_components.cpp", "lib_preprocess.cpp", "lib_learner.cpp", "lib_io.cpp", "lib_vectors.cpp",
      "cls_example.cpp", "cls_value.cpp", "cls_orange.cpp",
      "functions.cpp", "obsolete.cpp",
      "-m" # "-r" if you want to recreate all files
      #, "-x", "names2.xml"
     ]

readArguments(args)
parsedFile=""

orig_dir = os.getcwd()
if len(sys.argv)<2:
    os.chdir("..")
else:
    os.chdir(sys.argv[1])

if action.count("clean"):
  removeFiles()
  action.remove("clean")

if len(action):  
  newfiles=[]
  functions, constants, classdefs, aliases = parseFiles()
  if action.count("xml") or action.count("list"):
    descriptionsFromXML(xmlname)
    
  if action.count("list"):
    listOfExports(listname)
  if action.count("xml"):
    exportXML(xmlname)
  if action.count("make"):
    make()

os.chdir(orig_dir)
