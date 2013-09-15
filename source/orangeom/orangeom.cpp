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
