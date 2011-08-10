// QMap<int, TYPE*> is implemented as a Python dictionary.
template<TYPE>
%MappedType QMap<int, TYPE*> /DocType="dict-of-int-TYPE"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, TYPE*>::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        TYPE *t = i.value();

        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj = sipConvertFromType(t, sipType_TYPE, sipTransferObj);

        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);

            if (kobj)
            {
                Py_DECREF(kobj);
            }

            if (tobj)
            {
                Py_DECREF(tobj);
            }
            else
            {
                delete t;
            }

            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End

%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;

        while (PyDict_Next(sipPy, &i, &kobj, &tobj))
            if (!sipCanConvertToType(tobj, sipType_TYPE, SIP_NOT_NONE))
                return 0;

        return 1;
    }

    QMap<int, TYPE*> *qm = new QMap<int, TYPE*>;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int state, k = SIPLong_AsLong(kobj);
        TYPE *t = reinterpret_cast<TYPE *>(sipConvertToType(tobj, sipType_TYPE, sipTransferObj, SIP_NOT_NONE, &state, sipIsErr));
 
        if (*sipIsErr)
        {
            sipReleaseType(t, sipType_TYPE, state);

            delete qm;
            return 0;
        }

        qm->insert(k, t);

        sipReleaseType(t, sipType_TYPE, state);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};

// QMap<int, bool> is implemented as a Python dictionary.
%MappedType QMap<int, bool> /DocType="dict-of-int-bool"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, bool>::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj = Py_False;
        
        if (i.value())
        {
        	tobj = Py_True;
        }

        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);
            if (kobj)
            {
                Py_DECREF(kobj);
            }
            if (tobj)
            {
                Py_DECREF(tobj);
            }
            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End

%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;
        return 1;
    }

    QMap<int, bool> *qm = new QMap<int, bool>;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int k = SIPLong_AsLong(kobj);
        bool t = false;
        if (tobj == Py_True)
        {
        	t = true;
        }
 
        if (*sipIsErr)
        {
            delete qm;
            return 0;
        }

        qm->insert(k, t);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};

// QMap<int, double> is implemented as a Python dictionary.
%MappedType QMap<int, double> /DocType="dict-of-int-float"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, double>::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj = PyFloat_FromDouble(i.value());
        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);
            if (kobj)
            {
                Py_DECREF(kobj);
            }
            if (tobj)
            {
                Py_DECREF(tobj);
            }
            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End


%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;
        return 1;
    }

    QMap<int, double> *qm = new QMap<int, double>;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int k = SIPLong_AsLong(kobj);
        double t = PyFloat_AsDouble(tobj);
 
        if (*sipIsErr)
        {
            delete qm;
            return 0;
        }

        qm->insert(k, t);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};

// QMap<int, QPair<double, double> > is implemented as a Python dictionary of tuples.
%MappedType QMap<int, QPair<double, double> > /DocType="dict-of-int-tuple"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, QPair<double, double> >::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj;
		if ((tobj = PyTuple_New(2)) == NULL)
			return NULL;
		
		PyTuple_SET_ITEM(tobj, 0, PyFloat_FromDouble((double)i.value().first));
		PyTuple_SET_ITEM(tobj, 1, PyFloat_FromDouble((double)i.value().second));
    
        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);
            if (kobj)
            {
                Py_DECREF(kobj);
            }
            if (tobj)
            {
                Py_DECREF(tobj);
            }
            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End


%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;
        return 1;
    }

    QMap<int, QPair<double, double> > *qm = new QMap<int, QPair<double, double> >;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int k = SIPLong_AsLong(kobj);
        double fst = (int)PyInt_AsLong(PyTuple_GET_ITEM(tobj,0));
		double sec = (int)PyInt_AsLong(PyTuple_GET_ITEM(tobj,1));
		
		QPair<double, double> t(fst,sec);
 
        if (*sipIsErr)
        {
            delete qm;
            return 0;
        }

        qm->insert(k, t);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};