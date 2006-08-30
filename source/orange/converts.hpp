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


#ifndef __CONVERTS_HPP
#define __CONVERTS_HPP

#include "Python.h"
#include <vector>
#include <string>
using namespace std;

//WRAPPER(Contingency)
//WRAPPER(Distribution)

bool convertFromPython(PyObject *, string &);
bool convertFromPython(PyObject *, float &);
bool convertFromPython(PyObject *, pair<float, float> &);
bool convertFromPython(PyObject *, int &);
bool convertFromPython(PyObject *, unsigned char &);
bool convertFromPython(PyObject *, bool &);
//bool convertFromPython(PyObject *, PContingency &, bool allowNull=false, PyTypeObject *type=NULL);

PyObject *convertToPython(const string &);
PyObject *convertToPython(const float &);
PyObject *convertToPython(const pair<float, float> &);
PyObject *convertToPython(const int &);
PyObject *convertToPython(const unsigned char &);
PyObject *convertToPython(const bool &);

PyObject *convertToPython(const vector<int> &v);


//string convertToString(const PDistribution &);
string convertToString(const string &);
string convertToString(const float &);
string convertToString(const pair<float, float> &);
string convertToString(const int &);
string convertToString(const unsigned char &);
//string convertToString(const PContingency &);

class TOrangeType;
bool convertFromPythonWithML(PyObject *obj, string &str, const TOrangeType &base);

bool PyNumber_ToFloat(PyObject *o, float &);
bool PyNumber_ToDouble(PyObject *o, double &);

template<class T>
PyObject *convertToPython(const T &);

template<class T>
string convertToString(const T &);

int getBool(PyObject *args, void *isTrue);

// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX

#endif
