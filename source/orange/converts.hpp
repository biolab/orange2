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
PyObject *convertToPython(const pair<int, float> &);
PyObject *convertToPython(const int &);
PyObject *convertToPython(const long &);
PyObject *convertToPython(const unsigned char &);
PyObject *convertToPython(const bool &);

PyObject *convertToPython(const vector<int> &v);


//string convertToString(const PDistribution &);
string convertToString(const string &);
string convertToString(const float &);
string convertToString(const pair<float, float> &);
string convertToString(const pair<int, float> &);
string convertToString(const int &);
string convertToString(const long &);
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
