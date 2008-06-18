/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


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
