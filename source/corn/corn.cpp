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


#ifdef _MSC_VER
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
  #include <windows.h>
#endif

#include "c2py.hpp"
#include "pywrapper.hpp"
#include "stladdon.hpp"
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <string>
using namespace std;

#include "corn.hpp"
#include "c2py.hpp"

/* *********** MODULE INITIALIZATION ************/

#define PyTRY try {

#define PYNULL ((PyObject *)NULL)
#define PyCATCH   PyCATCH_r(PYNULL)
#define PyCATCH_1 PyCATCH_r(-1)

#define PyCATCH_r(r) \
  } \
catch (pyexception err)   { err.restore(); return r; } \
catch (exception err) { PYERROR(PyExc_CornKernel, err.what(), r); }


PyObject *PyExc_CornKernel;
PyObject *PyExc_CornWarning;

CORN_API void initcorn()
{ if (   ((PyExc_CornKernel = makeExceptionClass("corn.KernelException", "An error occurred in corn's C++ code")) == NULL)
      || ((PyExc_CornWarning = makeExceptionClass("corn.Warning", "corn warning", PyExc_Warning)) == NULL))
    return;

  PyObject *me;
  me = Py_InitModule("corn", corn_functions);
}


#ifdef _MSC_VER
BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{ switch (ul_reason_for_call)
	{ case DLL_PROCESS_ATTACH:case DLL_THREAD_ATTACH:case DLL_THREAD_DETACH:case DLL_PROCESS_DETACH:break; }
  return TRUE;
}
#endif

/* *********** EXCEPTION CATCHING ETC. ************/


#include <exception>
#include <string>

using namespace std;

#ifdef _MSC_VER

#define cornexception exception

#else

class cornexception : public exception {
public:
   string err_desc;
   cornexception(const string &desc)
   : err_desc(desc)
   {}

   ~cornexception() throw()
   {}

   virtual const char* what () const throw()
   { return err_desc.c_str(); };
};

#endif


exception CornException(const string &anerr)
{ return cornexception(anerr.c_str()); }

exception CornException(const string &anerr, const string &s)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s.c_str());
  return cornexception(buf);
}

exception CornException(const string &anerr, const string &s1, const string &s2)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s1.c_str(), s2.c_str());
  return cornexception(buf);
}

exception CornException(const string &anerr, const string &s1, const string &s2, const string &s3)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s1.c_str(), s2.c_str(), s3.c_str());
  return cornexception(buf);
}

exception CornException(const string &anerr, const long i)
{ char buf[255];
  sprintf(buf, anerr.c_str(), i);
  return cornexception(buf);
}

#undef min
#undef max

#define PyTRY try {
#define PYNULL ((PyObject *)NULL)


int getIntegerAttr(PyObject *self, char *name)
{ PyObject *temp = PyObject_GetAttrString(self, name);
  if (!temp || !PyInt_Check(temp))
    throw CornException("error in attribute '%s': integer expected", name);
  return (int)PyInt_AsLong(temp);
}

float getFloatAttr(PyObject *self, char *name)
{ PyObject *temp = PyObject_GetAttrString(self, name);
  if (!temp || !PyFloat_Check(temp))
    throw CornException("error in attribute '%s': float expected", name);
  return (float)PyFloat_AsDouble(temp);
}



class TestedExample {
public:
  int actualClass;
  int iterationNumber;
  vector<int> classes;
  vector<vector<float> > probabilities;
  float weight;

  TestedExample(const int &ac, const int &it, const vector<int> &c, const vector<vector<float> > &p, const float &w = 1.0);
  TestedExample(PyObject *);
};


class ExperimentResults {
public:
  int numberOfIterations, numberOfLearners;
  vector<TestedExample> results;
  bool weights;
  int baseClass;

  ExperimentResults(const int &ni, const int &nl, const bool &);
  ExperimentResults(PyObject *);
};



TestedExample::TestedExample(const int &ac, const int &it, const vector<int> &c, const vector<vector<float> > &p, const float &w)
: actualClass(ac),
  iterationNumber(it),
  classes(c),
  probabilities(p),
  weight(w)
{}


TestedExample::TestedExample(PyObject *obj)
: actualClass(getIntegerAttr(obj, "actualClass")),
  iterationNumber(getIntegerAttr(obj, "iterationNumber")),
  weight(getFloatAttr(obj, "weight"))

{ PyObject *temp;

  temp = PyObject_GetAttrString(obj, "classes");
  if (!temp || !PyList_Check(temp))
    throw CornException("error in 'classes' attribute");

  int i,e;
  for(i = 0, e = PyList_Size(temp); i<e; i++) {
    PyObject *cp = PyList_GetItem(temp, i);
    if (!cp || !PyInt_Check(cp))
      throw CornException("error in 'classes' attribute");
    classes.push_back((int)PyInt_AsLong(cp));
  }

  temp = PyObject_GetAttrString(obj, "probabilities");
  if (!temp || !PyList_Check(temp))
    throw CornException("error in 'probabilities' attribute");

  for(i = 0, e = PyList_Size(temp); i<e; i++) {
    PyObject *slist = PyList_GetItem(temp, i);
    if (!slist || !PyList_Check(slist))
      throw CornException("error in 'probabilities' attribute");
    probabilities.push_back(vector<float>());
    for(int ii = 0, ee = PyList_Size(slist); ii<ee; ii++) {
      PyObject *fe = PyList_GetItem(slist, ii);
      if (!fe || !PyFloat_Check(fe))
        throw CornException("error in 'probabilities' attribute");
      probabilities.back().push_back((float)PyFloat_AsDouble(fe));
    }
  }
}
        
      

ExperimentResults::ExperimentResults(const int &ni, const int &nl, const bool &w)
: numberOfIterations(ni),
  numberOfLearners(nl),
  weights(w)
{}


ExperimentResults::ExperimentResults(PyObject *obj)
: numberOfIterations(getIntegerAttr(obj, "numberOfIterations")),
  numberOfLearners(getIntegerAttr(obj, "numberOfLearners"))
{ 
  PyObject *temp = PyObject_GetAttrString(obj, "weights");
  weights = temp && (PyObject_IsTrue(temp)!=0);

  temp = PyObject_GetAttrString(obj, "baseClass");
  baseClass = PyInt_AsLong(temp);

  PyObject *pyresults = PyObject_GetAttrString(obj, "results");
  if (!pyresults || !PyList_Check(pyresults))
    throw CornException("error in 'results' attribute");

  for(int i = 0, e = PyList_Size(pyresults); i<e; i++) {
    PyObject *testedExample = PyList_GetItem(pyresults, i);
    results.push_back(TestedExample(testedExample));
  }
}



inline float diff2(const float &abnormal, const float &normal)
{ if (normal<abnormal)
    return 1;
  if (normal>abnormal)
    return 0;
  return 0.5;
}




class pp {
public:
  float normal, abnormal;

  inline pp()
  : normal(0.0),
    abnormal(0.0)
  {}

  inline void add(const bool &b, const float &w = 1.0)
  { *(b ? &abnormal : &normal) += w; }

  pp &operator +=(const pp &other)
  { normal += other.normal;
    abnormal += other.abnormal;
    return *this;
  }

  pp &operator -=(const pp &other)
  { normal -= other.normal;
    abnormal -= other.abnormal;
    return *this;
  }
};

typedef map<float, pp> TCummulativeROC;

void C_computeROCCumulative(const ExperimentResults &results, int classIndex, pp &totals, vector<TCummulativeROC> &cummlists, bool useWeights)
{
  if (classIndex<0)
    classIndex = results.baseClass;
  if (classIndex<0)
    classIndex = 1;

  totals = pp();
  cummlists = vector<TCummulativeROC>(results.numberOfLearners);

  const_ITERATE(vector<TestedExample>, i, results.results) {
    bool ind = (*i).actualClass==classIndex;
    float weight = useWeights ? (*i).weight : 1.0;
    totals.add(ind, weight);

    vector<TCummulativeROC>::iterator ci(cummlists.begin());
    const_ITERATE(vector<vector<float> >, pi, (*i).probabilities) {
      const float &tp = (*pi)[classIndex];
      (*ci)[tp];
      (*ci)[tp].add(ind, weight);
      ci++;
    }
  }
}


class TCDT {
public:
  float C, D, T;
  TCDT()
  : C(0.0),
    D(0.0),
    T(0.0)
  {}
};

void C_computeCDT(const vector<TCummulativeROC> &cummlists, vector<TCDT> &cdts)
{
  cdts = vector<TCDT>(cummlists.size());
  
  vector<TCDT>::iterator cdti (cdts.begin());
  for (vector<TCummulativeROC>::const_iterator ci(cummlists.begin()), ce(cummlists.end()); ci!=ce; ci++, cdti++) {
    pp low;
    for (map<float, pp>::const_iterator cri((*ci).begin()), cre((*ci).end()); cri!=cre; cri++) {
      const pp &thi = (*cri).second;
      (*cdti).C += low.normal   * thi.abnormal;
      (*cdti).D += low.abnormal * thi.normal;
      (*cdti).T += thi.normal   * thi.abnormal;
      low += thi;
    }
  }
}


PyObject *py_computeROCCumulative(PyObject *, PyObject *arg)
{ 
  PyTRY
    PyObject *pyresults;
    int classIndex = -1;
    PyObject *pyuseweights = NULL;
    if (!PyArg_ParseTuple(arg, "O|iO", &pyresults, &classIndex, &pyuseweights))
      PYERROR(PyExc_TypeError, "computeROCCummulative: results and optionally the classIndex and 'useWeights' flag expected", PYNULL);

    bool useweights = pyuseweights && PyObject_IsTrue(pyuseweights)!=0;

    ExperimentResults results(pyresults);
    if (results.numberOfIterations>1)
      PYERROR(PyExc_SystemError, "computeCDT: cannot compute CDT for experiments with multiple iterations", PYNULL);

    pp totals;
    vector<TCummulativeROC> cummlists;
    C_computeROCCumulative(results, classIndex, totals, cummlists, useweights);

    pyresults = PyList_New(results.numberOfLearners);
    int lrn = 0;
    ITERATE(vector<TCummulativeROC>, ci, cummlists) {
      PyObject *pclist = PyList_New((*ci).size());
      int prb = 0;
      ITERATE(TCummulativeROC, si, *ci)
        PyList_SetItem(pclist, prb++, Py_BuildValue("f(ff)", (*si).first, (*si).second.normal, (*si).second.abnormal));
      PyList_SetItem(pyresults, lrn++, pclist);
    }

    return Py_BuildValue("N(ff)", pyresults, totals.normal, totals.abnormal);
  PyCATCH
}


PyObject *py_computeCDT(PyObject *, PyObject *arg)
{
  PyTRY
    PyObject *pyresults;
    int classIndex = -1;
    PyObject *pyuseweights;
    if (!PyArg_ParseTuple(arg, "OiO", &pyresults, &classIndex, &pyuseweights))
      PYERROR(PyExc_TypeError, "computeROCCummulative: results and optionally the classIndex expected", PYNULL);

    bool useweights = PyObject_IsTrue(pyuseweights)!=0;

    ExperimentResults results(pyresults);
    if (results.numberOfIterations>1)
      PYERROR(PyExc_SystemError, "computeCDT: cannot compute CDT for experiments with multiple iterations", PYNULL);

    pp totals;
    vector<TCummulativeROC> cummlists;
    C_computeROCCumulative(results, classIndex, totals, cummlists, useweights);

    vector<TCDT> cdts;
    C_computeCDT(cummlists, cdts);

    PyObject *orngStatModule = PyImport_ImportModule("orngStat");
    if (!orngStatModule)
      return PYNULL;

    // PyModule_GetDict and PyDict_GetItemString return borrowed references
    PyObject *orngStatModuleDict = PyModule_GetDict(orngStatModule);
    Py_DECREF(orngStatModule);

    PyObject *CDTType = PyDict_GetItemString(orngStatModuleDict, "CDT");

    if (!CDTType)
      PYERROR(PyExc_AttributeError, "orngStat does not define CDT class", PYNULL);

    PyObject *res = PyList_New(cdts.size());
    int i = 0;
    ITERATE(vector<TCDT>, cdti, cdts) {
      PyObject *PyCDT = PyInstance_New(CDTType, Py_BuildValue("fff", (*cdti).C, (*cdti).D, (*cdti).T), PyDict_New());
      if (!PyCDT) {
        Py_XDECREF(res);
        return PYNULL;
      }
      PyList_SetItem(res, i++, PyCDT);
    }

    return res;
  PyCATCH
}


PyObject *py_compare2ROCs(PyObject *, PyObject *arg)
{ PyTRY
    PyObject *pyresults;
    int roc1, roc2, classIndex = -1;
    PyObject *pyuseweights;
    if (!PyArg_ParseTuple(arg, "OiiiO", &pyresults, &roc1, &roc2, &classIndex, &pyuseweights))
      PYERROR(PyExc_TypeError, "compare2ROCs: results and two integer indices (optionally also classIndex) expected", PYNULL);

    bool useweights = PyObject_IsTrue(pyuseweights)!=0;
    if (useweights)
      PYERROR(PyExc_SystemError, "compare2ROCs: cannot use weights (weights not implemented yet)", PYNULL);

    ExperimentResults results(pyresults);
    if (results.numberOfIterations>1)
      PYERROR(PyExc_SystemError, "computeCDT: cannot compute CDT for experiments with multiple iterations", PYNULL);

    if (classIndex<0)
      classIndex = results.baseClass;
    if (classIndex<0)
      classIndex = 1;

    float e11[] = {0, 0}, e10[] = {0, 0}, e01[] = {0, 0};
    float e11r = 0, e10r = 0, e01r = 0;
    float th[] ={0, 0};
    int m = 0, n = 0;
    /* m is number of examples with class == classIndex, n are others
       X_* represent example with class == classIndex, Y_* are others
    */

    for (vector<TestedExample>::const_iterator i(results.results.begin()), e(results.results.end()); i!=e; i++)
      if ((*i).actualClass != classIndex) {
        n++;
      }
      else { // (*i).actualClass == classIndex
        m++;

        float X_i[] = {(*i).probabilities[roc1][classIndex], (*i).probabilities[roc2][classIndex]};

        for (vector<TestedExample>::const_iterator j = i+1; j!=e; j++)
          if ((*j).actualClass!=classIndex) {

            float Y_j[] = {(*j).probabilities[roc1][classIndex], (*j).probabilities[roc2][classIndex]};
            float diffs[] = { diff2(X_i[0], Y_j[0]), diff2(X_i[1], Y_j[1]) };

            th[0] += diffs[0];
            th[1] += diffs[1];

            e11[0] += sqr(diffs[0]);
            e11[1] += sqr(diffs[1]);
            e11r   += diffs[0]*diffs[1];

            for (vector<TestedExample>::const_iterator k = j+1; k!=e; k++)
              if ((*k).actualClass == classIndex) { // B_XXY
                float X_k[] = { (*k).probabilities[roc1][classIndex], (*k).probabilities[roc2][classIndex] };
                float diffsk[] = { diff2(X_k[0], Y_j[0]), diff2(X_k[1], Y_j[1]) };
                e01[0] += diffs[0]*diffsk[0];
                e01[1] += diffs[1]*diffsk[1];
                e01r   += diffs[0]*diffsk[1] + diffs[1]*diffsk[0];
              }
              else { // B_XYY
                float Y_k[] = { (*k).probabilities[roc1][classIndex], (*k).probabilities[roc2][classIndex] };
                float diffsk[] = { diff2(X_i[0], Y_k[0]), diff2(X_i[1], Y_k[1]) };
                e10[0] += diffs[0]*diffsk[0];
                e10[1] += diffs[1]*diffsk[1];
                e10r   += diffs[0]*diffsk[1] + diffs[1]*diffsk[0];
              }
          }
      }

    float n11 = float(m)*float(n), n01 = float(m)*float(n)*float(m-1)/2.0, n10 = float(m)*float(n)*float(n-1)/2.0;
  
    th[0] /= n11;
    th[1] /= n11;
    
    e11[0] = e11[0]/n11 - sqr(th[0]);
    e11[1] = e11[1]/n11 - sqr(th[1]);
    e11r   = e11r  /n11 - th[0]*th[1];

    e10[0] = e10[0]/n10 - sqr(th[0]);
    e10[1] = e10[1]/n10 - sqr(th[1]);
    e10r   = e10r  /n10 - th[0]*th[1];

    e01[0] = e01[0]/n01 - sqr(th[0]);
    e01[1] = e01[1]/n01 - sqr(th[1]);
    e01r   = e01r  /n01 - th[0]*th[1];

    float var[] = { ((n-1)*e10[0] + (m-1)*e01[0] + e11[0])/n11, ((n-1)*e10[1] + (m-1)*e01[1] + e11[1])/n11};
    float SE[]  = { sqrt(var[0]), sqrt(var[1]) };

    float covar = ((n-1)*e10r + (m-1)*e01r + e11r) / n11;
    float SEr   = sqrt(var[0]+var[1]-2*covar*SE[0]*SE[1]);

    return Py_BuildValue("(ff)(ff)(ff)", th[0], SE[0], th[1], SE[1], th[0]-th[1], SEr);
  PyCATCH
}




/* *********** AUXILIARY ROUTINES *************/

class CompCallbackLess {
public:
  PyObject *py_compare;

  CompCallbackLess(PyObject *apyc)
    : py_compare(apyc)
    { Py_XINCREF(apyc); }

  ~CompCallbackLess()
    { Py_XDECREF(py_compare); }

  int operator()(PyObject *obj1, PyObject *obj2)
    { PyObject *args = Py_BuildValue("OO", obj1, obj2);
      PyObject *result = PyEval_CallObject(py_compare, args);
      Py_DECREF(args);

      if (!result) 
        throw pyexception();

      bool res = PyInt_AsLong(result)<0;
      Py_DECREF(result);
      return res;
    }
};


class CompCallbackEqual {
public:
  PyObject *py_compare;

  CompCallbackEqual(PyObject *apyc)
    : py_compare(apyc)
    { Py_XINCREF(apyc); }

  ~CompCallbackEqual()
    { Py_XDECREF(py_compare); }

  int operator()(PyObject *obj1, PyObject *obj2)
    { PyObject *args = Py_BuildValue("OO", obj1, obj2);
      PyObject *result = PyEval_CallObject(py_compare, args);
      Py_DECREF(args);

      if (!result) 
        throw pyexception();

      bool res = (PyInt_AsLong(result)==0);
      Py_DECREF(result);
      return res;
    }
};

/*
PyObject *py_sort_random(PyObject *, PyObject *arg)
{ PyTRY
    PyObject *pylist, *compfunc;
    if (   !PyArg_ParseTuple(arg, "OO", &pylist, &compfunc)
        || !PyCallable_Check(compfunc)
        || !PyList_Check(pylist))
      PYERROR(PyExc_TypeError, "sort_random: list and compare function expected", PYNULL);

    vector<PyObject *> toSort;
    int i, e;
    for(i = 0, e = PyList_Size(pylist); i<e; i++) {
      PyObject *item = PyList_GetItem(pylist, i);
      Py_XINCREF(item);
      toSort.push_back(item);
    }

    random_sort(toSort.begin(), toSort.end(), CompCallbackLess(compfunc), CompCallbackEqual(compfunc));

    for(i = 0, e = PyList_Size(pylist); i<e; i++)
      PyList_SetItem(pylist, i, toSort[i]);

    RETURN_NONE;
  PyCATCH
}


PyObject *py_ref(PyObject *, PyObject *arg)
{ return PyInt_FromLong((long)arg); }
*/
    

/* *********** EXPORT DECLARATIONS ************/

#define DECLARE(name) \
 {#name, (binaryfunc)py_##name, METH_VARARGS},

PyMethodDef corn_functions[] = {
     DECLARE(compare2ROCs)
     DECLARE(computeROCCumulative)
//     DECLARE(sort_random)
     DECLARE(computeCDT)

//     {"ref", py_ref, METH_O},

     {NULL, NULL}
};

#undef DECLARE

#undef PyTRY
#undef PyCATCH
#undef PYNULL
