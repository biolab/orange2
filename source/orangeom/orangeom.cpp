/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "px/orangeom_globals.hpp"
#include "cls_orange.hpp"

#include "mds.hpp"

PyObject *orangeVersion = PyString_FromString("2.0b ("__TIME__", "__DATE__")");

PYCONSTANT(version, orangeVersion)

bool initorangeomExceptions()
{ return true; }

void gcorangeomUnsafeStaticInitialization()
{}

#include "orangeom.px"

ORANGEOM_API PyObject *orangeomModule;

#include "initialization.px"
