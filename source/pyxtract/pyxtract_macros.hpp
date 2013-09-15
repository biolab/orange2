

#ifndef __PYXTRACT_MACROS_HPP
#define __PYXTRACT_MACROS_HPP

#define PYFUNCTION(pyname,cname,args,doc)
#define PYCONSTANT(pyname,ccode)
#define PYCONSTANT_INT(pyname,ccode)
#define PYCONSTANT_FLOAT(pyname,ccode)
#define PYCONSTANTFUNC(pyname,cname)

#define PYCLASSCONSTANT_INT(classname, constname, intconst)
#define PYCLASSCONSTANT_FLOAT(classname, constname, intconst)
#define PYCLASSCONSTANT(classname, constname, oconst)

#define PYPROPERTIES(x)
#define PYARGS(x,doc)
#define PYDOC(x)

#define PYXTRACT_IGNORE
#define DATASTRUCTURE(type,structure,dictfield)

#define BASED_ON(type, basetype)
#define ABSTRACT(type, basetype)

#define ALLOWS_EMPTY

#define C_UNNAMED(type, basetype, doc)
#define C_NAMED(type, basetype, doc)
#define C_CALL(type, basecall, doc)
#define C_CALL3(type, basecall, basetype,doc)
#define HIDDEN(type, base)

#define CONSTRUCTOR_KEYWORDS(type, keywords)
#define NO_PICKLE(type)
#define RECOGNIZED_ATTRIBUTES(type, keywords)

#endif
