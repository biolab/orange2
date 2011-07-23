// QMap<int, TYPE> is implemented as a Python dictionary.
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
        PyObject *tobj = sipConvertFromNewType(t, sipType_TYPE, sipTransferObj);

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
